import logging
import os
import subprocess
import tempfile
from typing import Optional, List

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from sam2.build_sam import build_sam2_video_predictor

import mediapipe as mp


class Predictor(BasePredictor):
    """
    End-to-end compositor:
      • Generates YAP (portrait with alpha) using MediaPipe + SAM-2
      • Segments arms/hands via Replicate SA2VA (optional, union with static body mask)
      • Rebuilds hex CW reveal assets
      • Composes Background + TOP_mix + YAP + BOT_mix in a single run
    """

    # ─────────────────────────── setup ────────────────────────────
    def setup(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("🏗  Initialising model stack (predict2)")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"ℹ️  Running on {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        # SAM-2 predictor
        self.checkpoint = "/src/sam2_hiera_base_plus.pt"
        self.model_cfg = "sam2.1_hiera_b+.yaml"
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        logging.info("✅ SAM-2 video predictor ready")

        # MediaPipe Selfie for portrait segmentation
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        logging.info("✅ MediaPipe SelfieSegmentation loaded")

        # Optional Replicate client (lazy import later)
        self.replicate_token = os.environ.get("REPLICATE_API_TOKEN")
        self.sa2va_version = os.environ.get("SA2VA_VERSION", "bytedance/sa2va-8b-image")

        # ffmpeg path from vendor
        self.ffmpeg_bin = os.path.join(os.getcwd(), "vendor", "ffmpeg", "bin", "ffmpeg")

    # ────────────────────────── predict ───────────────────────────
    def predict(
        self,
        person_video: Path = Input(description="Input person video (YAP source)"),
        background_video: Path = Input(description="Background video for final composite"),
        hex_image: Path = Input(description="Hex artwork (RGB or RGBA)"),
        mask_sequence_pattern: str = Input(
            description="printf-style path to clockwise mask sequence (e.g. build_hex/mask_cw_%03d.png)",
            default="build_hex/mask_cw_%03d.png",
        ),
        static_body_mask_png: Path = Input(
            description="Static body mask PNG to clip torso (unioned with dynamic arms)",
        ),
        # Sizing/Timing
        out_width: int = Input(description="Output width", default=1080, ge=256, le=4096),
        out_height: int = Input(description="Output height", default=1920, ge=256, le=4096),
        hex_w: int = Input(description="On-canvas hex width", default=600, ge=64, le=2000),
        hex_h: int = Input(description="On-canvas hex height", default=526, ge=64, le=2000),
        yap_scale: float = Input(description="YAP scale as fraction of hex width", default=1.0, ge=0.25, le=2.0),
        yap_y_frac: float = Input(description="Y shift of YAP as fraction of hex height", default=0.156, ge=-2.0, le=2.0),
        reveal_dur: float = Input(description="Reveal duration (s)", default=3.0, ge=0.5, le=30.0),
        mask_total_dur: float = Input(description="Total duration of mask sequence (s)", default=15.0, ge=1.0, le=120.0),
        reveal_fps: int = Input(description="Reveal FPS", default=30, ge=10, le=60),
        # Encoding
        crf_vp9: int = Input(description="Intermediate VP9 CRF", default=19, ge=1, le=32),
        crf_x264: int = Input(description="Final x264 CRF", default=18, ge=12, le=30),
        # SAM/MediaPipe controls
        mask_every_n_frames: int = Input(description="Recompute SAM mask every N frames", default=1, ge=1, le=10),
        jpeg_quality: int = Input(description="Quality for sampled JPEGs", default=94, ge=70, le=100),
        soften_edge: bool = Input(description="Feather YAP edge", default=True),
        # SA2VA controls
        sa2va_every_n_frames: int = Input(description="SA2VA sampling interval (frames)", default=1, ge=1, le=10),
        enable_arms_mask: bool = Input(description="Enable arms/hands overlay; if false, only torso clipped to hex", default=True),
    ) -> Path:
        # 0) decode person video into memory
        cap = cv2.VideoCapture(str(person_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or float(reveal_fps)
        frames: List[np.ndarray] = []
        ok, frm = cap.read()
        while ok:
            frames.append(frm)
            ok, frm = cap.read()
        cap.release()

        n_frames = len(frames)
        if n_frames == 0:
            raise RuntimeError("No decodable frames in person video")
        h, w = frames[0].shape[:2]
        logging.info(f"Loaded person video → {n_frames} frames @ ≈{fps:.2f} fps ({w}x{h})")

        # 1) prepare sampled JPEG directory for SAM-2 propagation
        tmp_root = tempfile.mkdtemp(prefix="predict2_")
        work = os.path.join(tmp_root, "work")
        os.makedirs(work, exist_ok=True)
        jpeg_dir = os.path.join(tmp_root, "sam2_frames")
        os.makedirs(jpeg_dir, exist_ok=True)

        sampled_idxs = list(range(0, n_frames, mask_every_n_frames))
        for j, orig_idx in enumerate(sampled_idxs):
            q = 100 if j < 3 else jpeg_quality
            cv2.imwrite(
                os.path.join(jpeg_dir, f"{j:06d}.jpg"),
                frames[orig_idx],
                [int(cv2.IMWRITE_JPEG_QUALITY), q],
            )
        logging.info(f"Wrote {len(sampled_idxs)} JPEGs for SAM-2 → {jpeg_dir}")

        # 2) seed + propagate SAM-2 masks
        state = self.predictor.init_state(video_path=jpeg_dir)
        seed_mask = self.get_portrait_mask(frames[0], 0.3)
        seed_frames = min(3, len(sampled_idxs))
        for frame_idx in range(seed_frames):
            self.predictor.add_new_mask(state, frame_idx, 1, seed_mask)
        prop_iter = iter(self.predictor.propagate_in_video(state))

        # 3) stream-encode YAP (BGRA) with alpha via VP9
        yap_path = os.path.join(tmp_root, "yap.webm")
        ffmpeg = subprocess.Popen(
            [
                self.ffmpeg_bin,
                "-y",
                "-thread_queue_size",
                "64",
                "-framerate",
                str(fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgra",
                "-s",
                f"{w}x{h}",
                "-i",
                "-",
                "-vf",
                "format=yuva420p",
                "-map",
                "0:v",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                "-auto-alt-ref",
                "0",
                "-crf",
                str(crf_vp9),
                "-b:v",
                "0",
                "-row-mt",
                "1",
                "-tile-columns",
                "2",
                "-threads",
                "0",
                "-metadata:s:v:0",
                "alpha_mode=1",
                yap_path,
            ],
            stdin=subprocess.PIPE,
        )

        # 3b) Prepare adaptive SA2VA arms-mask writer and detection assets
        arms_mask_path = os.path.join(tmp_root, "arms_mask.webm")
        arms_writer = subprocess.Popen(
            [
                self.ffmpeg_bin,
                "-y",
                "-thread_queue_size",
                "64",
                "-framerate",
                str(fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{w}x{h}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "1M",
                "-row-mt",
                "1",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                arms_mask_path,
            ],
            stdin=subprocess.PIPE,
        )

        # Replicate client (optional)
        sa2va_client = None
        arms_enabled = enable_arms_mask
        if arms_enabled and self.replicate_token is not None:
            try:
                import replicate  # lazy import

                sa2va_client = replicate.Client(api_token=self.replicate_token)
            except Exception as e:
                logging.warning(f"SA2VA disabled (replicate import/client failed): {e}")
                sa2va_client = None

        # Load and prepare static torso clip mask for detection (scaled to final YAP width)
        scaled_w = int(hex_w * yap_scale)
        scaled_h = max(1, int(round(h * scaled_w / max(1, w))))
        try:
            m_img = cv2.imread(str(static_body_mask_png), cv2.IMREAD_UNCHANGED)
            if m_img is None:
                raise RuntimeError("Failed to read static_body_mask_png")
            if m_img.ndim == 3 and m_img.shape[2] == 4:
                static_mask_gray = m_img[:, :, 3]
            elif m_img.ndim == 3:
                static_mask_gray = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)
            else:
                static_mask_gray = m_img
            static_mask_scaled = cv2.resize(static_mask_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
            _, static_mask_bin = cv2.threshold(static_mask_scaled, 127, 255, cv2.THRESH_BINARY)
        except Exception as e:
            logging.warning(f"Static torso mask load/scale failed: {e}")
            static_mask_bin = np.zeros((scaled_h, scaled_w), dtype=np.uint8)

        # Adaptive SA2VA state
        arms_active = False
        below_count = 0
        area_hi = 0.005  # 0.5% triggers ON
        area_lo = 0.002  # 0.2% triggers OFF (hysteresis)
        off_after = 3    # require N consecutive frames below to turn OFF
        last_arms_gray = np.zeros((h, w), dtype=np.uint8)

        prev_mask = None
        ptr = 0
        for idx, frame in enumerate(frames):
            if ptr < len(sampled_idxs) and idx == sampled_idxs[ptr]:
                _, _, logits = next(prop_iter)
                mask = logits[0].cpu().numpy()
                ptr += 1
            else:
                mask = prev_mask if prev_mask is not None else np.ones((h, w), np.uint8)
            prev_mask = mask
            bgra = self.apply_alpha_mask(frame, mask, soften_edge)
            # write YAP frame
            ffmpeg.stdin.write(bgra.tobytes())

            # Adaptive arms detection and SA2VA
            try:
                alpha = bgra[:, :, 3]
                # Scale alpha to final YAP width for comparison to static torso clip
                alpha_scaled = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                alpha_bin = (alpha_scaled > 10).astype(np.uint8)
                # Outside = alpha present where torso mask is zero
                outside = (alpha_bin == 1) & (static_mask_bin == 0)
                area_alpha = max(1, int(alpha_bin.sum()))
                area_out = int(outside.sum())
                out_ratio = area_out / float(area_alpha)

                if not arms_active:
                    if out_ratio > area_hi:
                        arms_active = True
                        below_count = 0
                else:
                    if out_ratio < area_lo:
                        below_count += 1
                        if below_count >= off_after:
                            arms_active = False
                            below_count = 0
                    else:
                        below_count = 0

                # Decide whether to call SA2VA for this frame
                do_sa2va = arms_enabled and (sa2va_client is not None) and (
                    arms_active or (idx % max(1, sa2va_every_n_frames) == 0)
                )

                if do_sa2va:
                    mask_sa = self._get_arms_mask_sa2va(sa2va_client, frame) if sa2va_client is not None else None
                    if mask_sa is not None:
                        last_arms_gray = mask_sa
                # write held or updated mask
                arms_writer.stdin.write((last_arms_gray if arms_enabled else np.zeros_like(last_arms_gray)).tobytes())
            except Exception:
                # On any detection error, write last and continue
                arms_writer.stdin.write((last_arms_gray if arms_enabled else np.zeros_like(last_arms_gray)).tobytes())

        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("FFmpeg encoding of YAP failed")
        logging.info(f"✅ YAP ready → {yap_path}")
        # Finalize arms-mask video
        arms_writer.stdin.close()
        if arms_writer.wait() != 0:
            raise RuntimeError("FFmpeg encoding of arms mask failed")
        logging.info(f"Arms mask video → {arms_mask_path}")

        # 5) Build hex assets (TOP/BOT mixes)
        # 5.1 Background → 1080x1920
        bg_1080x1920 = os.path.join(work, "video_1080x1920.mp4")
        self._run_ffmpeg([
            "-y",
            "-i",
            str(background_video),
            "-vf",
            f"scale={out_width}:{out_height}:force_original_aspect_ratio=increase,crop={out_width}:{out_height},setsar=1,format=rgba",
            "-c:v",
            "libx264",
            "-crf",
            str(crf_x264),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            bg_1080x1920,
        ])

        # 5.2 Hex scale + alpha extract
        hex_scaled = os.path.join(work, "hex_scaled.png")
        hex_alpha = os.path.join(work, "hex_alpha.png")
        self._run_ffmpeg([
            "-y",
            "-i",
            str(hex_image),
            "-vf",
            f"scale={hex_w}:-2:flags=lanczos,setsar=1,format=rgba",
            "-frames:v",
            "1",
            "-update",
            "1",
            hex_scaled,
        ])
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_scaled,
            "-vf",
            "alphaextract,format=gray",
            "-frames:v",
            "1",
            "-update",
            "1",
            hex_alpha,
        ])

        # 5.3 Half-plane masks (GRAY)
        half_top = os.path.join(work, "half_top.png")
        half_bot = os.path.join(work, "half_bot.png")
        self._run_ffmpeg([
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={hex_w}x{hex_h}",
            "-vf",
            "format=gray,geq=lum='255*lte(Y,H/2)'",
            "-frames:v",
            "1",
            "-update",
            "1",
            half_top,
        ])
        self._run_ffmpeg([
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={hex_w}x{hex_h}",
            "-vf",
            "format=gray,geq=lum='255*gte(Y,H/2)'",
            "-frames:v",
            "1",
            "-update",
            "1",
            half_bot,
        ])

        # 5.4 Colorize A/B (RGB)
        hexA_rgb = os.path.join(work, "hexA_rgb.png")
        hexB_rgb = os.path.join(work, "hexB_rgb.png")
        A_R, A_G, A_B = 0.1176, 0.5647, 1.0000
        B_R, B_G, B_B = 1.0000, 0.2314, 0.1882
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_scaled,
            "-vf",
            f"format=rgba,colorchannelmixer=rr={A_R}:gg={A_G}:bb={A_B}:aa=1,format=rgb24",
            "-frames:v",
            "1",
            "-update",
            "1",
            hexA_rgb,
        ])
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_scaled,
            "-vf",
            f"format=rgba,colorchannelmixer=rr={B_R}:gg={B_G}:bb={B_B}:aa=1,format=rgb24",
            "-frames:v",
            "1",
            "-update",
            "1",
            hexB_rgb,
        ])

        # 5.5 Static A halves (RGBA)
        A_top = os.path.join(work, "A_top.png")
        A_bot = os.path.join(work, "A_bot.png")
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_alpha,
            "-i",
            half_top,
            "-i",
            hexA_rgb,
            "-filter_complex",
            "[0][1]blend=all_mode=multiply[a];[2][a]alphamerge,format=rgba",
            "-frames:v",
            "1",
            "-update",
            "1",
            A_top,
        ])
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_alpha,
            "-i",
            half_bot,
            "-i",
            hexA_rgb,
            "-filter_complex",
            "[0][1]blend=all_mode=multiply[a];[2][a]alphamerge,format=rgba",
            "-frames:v",
            "1",
            "-update",
            "1",
            A_bot,
        ])

        # 5.6 Animated B halves (RGBA) using CW masks
        B_top = os.path.join(work, "B_top.webm")
        B_bot = os.path.join(work, "B_bot.webm")
        # top
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_alpha,
            "-framerate",
            str(reveal_fps),
            "-start_number",
            "0",
            "-i",
            str(mask_sequence_pattern),
            "-i",
            half_top,
            "-i",
            hexB_rgb,
            "-filter_complex",
            (
                f"[1:v]format=gray,scale={hex_w}:{hex_h}:flags=neighbor,setpts=PTS*({reveal_dur}/{mask_total_dur})[mask];"
                f"[0:v][2:v]blend=all_mode=multiply[a_half];"
                f"[a_half][mask]blend=all_mode=multiply[a_reveal];"
                f"[3:v][a_reveal]alphamerge,format=rgba"
            ),
            "-t",
            str(reveal_dur),
            "-r",
            str(reveal_fps),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-b:v",
            "2M",
            "-row-mt",
            "1",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            B_top,
        ])
        # bot
        self._run_ffmpeg([
            "-y",
            "-i",
            hex_alpha,
            "-framerate",
            str(reveal_fps),
            "-start_number",
            "0",
            "-i",
            str(mask_sequence_pattern),
            "-i",
            half_bot,
            "-i",
            hexB_rgb,
            "-filter_complex",
            (
                f"[1:v]format=gray,scale={hex_w}:{hex_h}:flags=neighbor,setpts=PTS*({reveal_dur}/{mask_total_dur})[mask];"
                f"[0:v][2:v]blend=all_mode=multiply[a_half];"
                f"[a_half][mask]blend=all_mode=multiply[a_reveal];"
                f"[3:v][a_reveal]alphamerge,format=rgba"
            ),
            "-t",
            str(reveal_dur),
            "-r",
            str(reveal_fps),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-b:v",
            "2M",
            "-row-mt",
            "1",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            B_bot,
        ])

        # 5.7 Scale YAP to hex width (keep alpha)
        yap_scaled = os.path.join(work, "yap_scaled.webm")
        self._run_ffmpeg([
            "-y",
            "-i",
            yap_path,
            "-vf",
            f"scale={hex_w}:-2:flags=lanczos,format=yuva420p,setsar=1",
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-auto-alt-ref",
            "0",
            "-pix_fmt",
            "yuva420p",
            "-b:v",
            "2M",
            "-row-mt",
            "1",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            yap_scaled,
        ])

        # 5.8 Build TOP/BOT mixes
        TOP_mix = os.path.join(work, "TOP_mix.webm")
        BOT_mix = os.path.join(work, "BOT_mix.webm")
        # TOP
        self._run_ffmpeg([
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(reveal_fps),
            "-i",
            A_top,
            "-c:v",
            "libvpx-vp9",
            "-i",
            B_top,
            "-filter_complex",
            (
                "[0:v]format=rgba,split[Ac][Aa_src];"
                "[Aa_src]alphaextract[Aa];"
                "[1:v]format=rgba,split[Bc][Ba_src];"
                "[Ba_src]alphaextract[Ba];"
                "[Ac][Bc]overlay=x=0:y=0[c_tmp];"
                "[Aa][Ba]blend=all_mode=lighten[a_union];"
                "[c_tmp][a_union]alphamerge"
            ),
            "-t",
            str(reveal_dur),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-auto-alt-ref",
            "0",
            "-pix_fmt",
            "yuva420p",
            "-b:v",
            "2M",
            "-row-mt",
            "1",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            TOP_mix,
        ])
        # BOT
        self._run_ffmpeg([
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(reveal_fps),
            "-i",
            A_bot,
            "-c:v",
            "libvpx-vp9",
            "-i",
            B_bot,
            "-filter_complex",
            (
                "[0:v]format=rgba,split[Bc0][Ba0_src];"
                "[Ba0_src]alphaextract[Ba0];"
                "[1:v]format=rgba,split[Bc1][Ba1_src];"
                "[Ba1_src]alphaextract[Ba1];"
                "[Bc0][Bc1]overlay=x=0:y=0[c_tmp];"
                "[Ba0][Ba1]blend=all_mode=lighten[a_union];"
                "[c_tmp][a_union]alphamerge"
            ),
            "-t",
            str(reveal_dur),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-auto-alt-ref",
            "0",
            "-pix_fmt",
            "yuva420p",
            "-b:v",
            "2M",
            "-row-mt",
            "1",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            BOT_mix,
        ])

        # 6) FINAL COMPOSITE:
        # BG -> TOP_mix -> YAP_torso(clipped by static hex PNG) -> BOT_mix -> YAP_arms (above all)
        final_out = os.path.join(tmp_root, "final_output.mp4")
        # inputs: 0:bg, 1:TOP_mix, 2:yap_scaled, 3:static body mask, 4:arms_mask, 5:BOT_mix
        overlay_x = f"(main_w-{hex_w})/2"
        overlay_y = f"(main_h-{hex_h})/2"
        overlay_top_y = f"(main_h-{hex_h})/2+1"
        yap_y = f"(main_h+{hex_h})/2-overlay_h-({yap_y_frac}*{hex_h})"
        # Build arms mask branch (feather if soften_edge)
        arms_mask_branch = (
            f"[4:v]format=gray,scale={int(hex_w*yap_scale)}:-2:flags=lanczos"
            + (",gblur=sigma=1.5" if soften_edge else "")
            + "[yap_mask_arms];"
        )

        filter_complex = (
            "[0:v]setsar=1[bg];"
            "[1:v]format=yuva420p,setsar=1[topmix];"
            f"[2:v]scale={int(hex_w*yap_scale)}:-2:flags=lanczos,format=rgba[yap_rgba];"
            "[yap_rgba]split[yap_rgb_src][yap_rgba_for_alpha];"
            "[yap_rgb_src]format=rgb24[yap_rgb];"
            "[yap_rgba_for_alpha]alphaextract[yap_a];"
            f"[3:v]format=gray,scale={int(hex_w*yap_scale)}:-2:flags=lanczos[yap_mask_static];"
            + arms_mask_branch +
            # Torso alpha: clip to static mask only
            "[yap_a][yap_mask_static]blend=all_mode=multiply[yap_a_torso];"
            # Arms alpha: clip to dynamic arms mask only
            "[yap_a][yap_mask_arms]blend=all_mode=multiply[yap_a_arms];"
            # Build two YAP layers
            "[yap_rgb][yap_a_torso]alphamerge,format=yuva420p[yap_torso];"
            "[yap_rgb][yap_a_arms]alphamerge,format=yuva420p[yap_arms];"
            # Prepare BOT mix SAR and overlays: bg -> topmix -> torso -> botmix -> arms
            "[5:v]format=yuva420p,setsar=1[botmix];"
            f"[bg][topmix]overlay=x={overlay_x}:y={overlay_top_y}:alpha=straight[v1];"
            f"[v1][yap_torso]overlay=x='(main_w-overlay_w)/2':y={yap_y}:alpha=straight[v2];"
            f"[v2][botmix]overlay=x={overlay_x}:y={overlay_y}:alpha=straight[v3];"
            f"[v3][yap_arms]overlay=x='(main_w-overlay_w)/2':y={yap_y}:alpha=straight[vout]"
        )
        self._run_ffmpeg([
            "-y",
            "-i",
            bg_1080x1920,
            "-c:v",
            "libvpx-vp9",
            "-i",
            TOP_mix,
            "-c:v",
            "libvpx-vp9",
            "-i",
            yap_scaled,
            "-i",
            str(static_body_mask_png),
            "-i",
            arms_mask_path,
            "-c:v",
            "libvpx-vp9",
            "-i",
            BOT_mix,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            str(crf_x264),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(reveal_fps),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            final_out,
        ])

        logging.info(f"✅ Finished → {final_out}")
        return Path(final_out)

    # ─────────────────────── helpers ─────────────────────────────
    def _run_ffmpeg(self, args: List[str]):
        cmd = [self.ffmpeg_bin] + args
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            logging.error(proc.stderr.decode("utf-8", errors="ignore"))
            raise RuntimeError("FFmpeg step failed")

    def _build_arms_mask_video(
        self,
        frames: List[np.ndarray],
        fps: float,
        out_path: str,
        sample_every: int,
        enable: bool,
    ):
        h, w = frames[0].shape[:2]
        # Always build a video to simplify downstream graph. If disabled, emit zeros.
        writer = subprocess.Popen(
            [
                self.ffmpeg_bin,
                "-y",
                "-thread_queue_size",
                "64",
                "-framerate",
                str(fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{w}x{h}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "1M",
                "-row-mt",
                "1",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                out_path,
            ],
            stdin=subprocess.PIPE,
        )

        sa2va_client = None
        if enable:
            try:
                import replicate  # lazy import

                sa2va_client = replicate.Client(api_token=self.replicate_token)
            except Exception as e:
                logging.warning(f"SA2VA disabled (replicate import/client failed): {e}")
                sa2va_client = None

        last_gray = np.zeros((h, w), dtype=np.uint8)
        for idx, frame in enumerate(frames):
            if sa2va_client is not None and (idx % sample_every == 0):
                mask = self._get_arms_mask_sa2va(sa2va_client, frame)
                if mask is not None:
                    last_gray = mask
            writer.stdin.write(last_gray.tobytes())

        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("FFmpeg encoding of arms mask failed")
        logging.info(f"Arms mask video → {out_path}")

    def _get_arms_mask_sa2va(self, client, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Calls Replicate SA2VA to segment arms/hands and returns a binary uint8 mask.
        Falls back to None on any error.
        """
        try:
            # Write frame to a temp PNG for upload
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                ok = cv2.imwrite(tmp.name, frame)
                if not ok:
                    return None
                tmp_path = tmp.name

            # Prefer explicit version via SA2VA_VERSION if provided; otherwise use model name
            model_spec = self.sa2va_version
            result = client.run(
                model_spec,
                input={
                    "image": open(tmp_path, "rb"),
                    "instruction": "segment just the arms and hands of the person",
                },
            )

            # result can be URL string or list of URLs
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            im = None
            if isinstance(result, str):
                # download image
                try:
                    from urllib.request import urlopen

                    with urlopen(result) as resp:
                        data = resp.read()
                    arr = np.frombuffer(data, dtype=np.uint8)
                    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                except Exception:
                    im = None

            if im is None:
                # Could be returned as bytes-like (rare)
                return None

            # Convert to gray mask; if RGBA/RGB, use luminance; then binarize
            if im.ndim == 3:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            else:
                gray = im

            # Resize to original frame size if needed
            if gray.shape[:2] != frame.shape[:2]:
                gray = cv2.resize(gray, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Binarize and clean
            _, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
            bin_mask = cv2.dilate(bin_mask, kernel, iterations=1)
            return bin_mask.astype(np.uint8)
        except Exception as e:
            logging.warning(f"SA2VA call failed: {e}")
            return None

    def get_portrait_mask(self, frame, thresh: float = 0.3) -> np.ndarray:
        # MediaPipe expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie.process(img)
        if results.segmentation_mask is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        float_mask = (results.segmentation_mask > thresh).astype(np.float32)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin0 = (float_mask > 0.5).astype(np.uint8)
        clean = cv2.morphologyEx(bin0, cv2.MORPH_OPEN, kernel3)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel5)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = (labels == largest_label).astype(np.uint8)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean = cv2.erode(clean, kernel_erode, iterations=2)
        return clean

    def apply_alpha_mask(self, frame, mask, soften_edge):
        prob = 1 / (1 + np.exp(-mask))
        bin_mask = (prob.squeeze() > 0.4).astype(np.uint8)
        bin_mask = cv2.resize(bin_mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)
        alpha = (bin_mask * 255).astype(np.uint8)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        alpha = cv2.erode(alpha, kernel_erode, iterations=2)
        if alpha.max() == alpha.min():
            alpha[0, 0] = 255 if alpha.max() == 0 else 0
        if soften_edge:
            alpha = self.feather_alpha(alpha)
        rgb = cv2.bitwise_and(frame, frame, mask=alpha)
        out = np.dstack([rgb, alpha])
        return out

    @staticmethod
    def feather_alpha(alpha, iterations: int = 0):
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        return np.clip(blurred, 0, 255).astype(np.uint8)


