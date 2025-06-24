import logging
import os
import subprocess
import tempfile
import json
import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path              # ← use cog.Path
from sam2.build_sam import build_sam2_video_predictor

import mediapipe as mp


class Predictor(BasePredictor):
    """
    Background-removal for portrait video using:
      • MediaPipe SelfieSegmentation seed mask
      • SAM-2 "hiera-base-plus" refinement
      • VP9 + alpha single-pass encode
    """

    # ─────────────────────────── setup ────────────────────────────
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("🏗  Initialising model stack")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"ℹ️  Running on {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        # SAM-2 predictor
        self.checkpoint = "/src/sam2_hiera_base_plus.pt"
        self.model_cfg  = "sam2.1_hiera_b+.yaml"
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        logging.info("✅ SAM-2 video predictor ready")

        # MediaPipe Selfie for portrait segmentation
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        logging.info("✅ MediaPipe SelfieSegmentation loaded")

    # ────────────────────────── predict ───────────────────────────
    def predict(
        self,
        input_video: Path = Input(description="Input video file"),
        mask_every_n_frames: int = Input(
            description="Recompute alpha every N frames", default=1, ge=1, le=30
        ),
        jpeg_quality: int = Input(
            description="JPEG quality for intermediate frames", default=94, ge=1, le=100
        ),
        crf: int = Input(description="VP9 CRF", default=19, ge=1, le=32),
        soften_edge: bool = Input(description="Feather mask edge", default=True),
    ) -> Path:                                           # ← return cog.Path
        # warm-up shortcut
        if input_video.name == "warmup.mp4":
            logging.info("Warm-up request – echoing source")
            return input_video

        # 1️⃣ decode the clip into memory
        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []
        ok, frm = cap.read()
        while ok:
            frames.append(frm)
            ok, frm = cap.read()
        cap.release()

        n_frames = len(frames)
        if n_frames == 0:
            raise RuntimeError("No decodable frames in input video")
        h, w = frames[0].shape[:2]
        logging.info(f"Loaded {n_frames} frames (≈{fps:.2f} fps)")

        # 2️⃣ write sampled JPEGs for SAM-2
        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        sampled_idxs = list(range(0, n_frames, mask_every_n_frames))
        for j, orig_idx in enumerate(sampled_idxs):
            # Use max quality for first 3 frames (seed frames) for better SAM-2 results
            q = 100 if j < 3 else jpeg_quality
            cv2.imwrite(
                os.path.join(tmp_dir, f"{j:06d}.jpg"),
                frames[orig_idx],
                [int(cv2.IMWRITE_JPEG_QUALITY), q],
            )
        logging.info(f"Wrote {len(sampled_idxs)} JPEGs → {tmp_dir}")

        # 3️⃣ seed SAM-2 with refined mask on multiple early frames
        state = self.predictor.init_state(video_path=tmp_dir)
        seed_mask = self.get_portrait_mask(frames[0], 0.3)
        
        # Add the same high-quality mask to first 3 frames for better temporal consistency
        seed_frames = min(3, len(sampled_idxs))
        for frame_idx in range(seed_frames):
            self.predictor.add_new_mask(state, frame_idx, 1, seed_mask)
        logging.info(f"Seeded {seed_frames} frames with portrait mask")
                
        prop_iter = iter(self.predictor.propagate_in_video(state))

        ffmpeg_bin = os.path.join(os.getcwd(), "vendor", "ffmpeg", "bin", "ffmpeg")
        ffprobe_bin = os.path.join(os.getcwd(), "vendor", "ffmpeg", "bin", "ffprobe")

        version_proc = subprocess.Popen(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        out, _ = version_proc.communicate()
        logging.info("FFmpeg version info:\n%s", out)

        # 4️⃣ spawn FFmpeg encoder
        output_path = "/output.webm"
        ffmpeg = subprocess.Popen(
            [
                ffmpeg_bin, "-y",
                "-thread_queue_size", "64",
                "-framerate", str(fps),
                "-f", "rawvideo", "-pix_fmt", "bgra",
                "-s", f"{w}x{h}", "-i", "-",
                "-thread_queue_size", "32",
                "-i", str(input_video),
                "-filter_complex", "[0:v]format=yuva420p[vid]",
                "-map", "[vid]", "-map", "1:a?",
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                "-auto-alt-ref", "0",
                "-crf", str(crf), "-b:v", "0",
                "-row-mt", "1", "-tile-columns", "2",
                "-threads", "0",
                "-metadata:s:v:0", "alpha_mode=1",
                "-c:a", "copy",
                output_path,
            ],
            stdin=subprocess.PIPE,
        )

        # 5️⃣ stream frames with live mask propagation
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
            ffmpeg.stdin.write(bgra.tobytes())

        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("FFmpeg encoding failed")

        logging.info(f"✅ Finished → {output_path}")
        
        # ── confirm that the file REALLY has an alpha plane ──────────────
        probe = subprocess.run(
            [
                ffprobe_bin, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt,alpha_mode,width,height",
                "-of", "json", output_path
            ],
            text=True, check=True, capture_output=True,
        )
        info = probe.stdout
        logging.info("ffprobe stream info -> %s", info)
        data = json.loads(info)["streams"][0]
        assert data["pix_fmt"].startswith("yuva"), "❌ alpha plane missing!"
        assert data.get("alpha_mode") == "1",      "❌ alpha_mode tag wrong!"

        return Path(output_path)


    # ─────────────────────── helpers ─────────────────────────────
    def get_portrait_mask(self, frame, thresh: float = 0.3) -> np.ndarray:
        """
        Returns a H×W uint8 mask of the upper-body/head using MediaPipe SelfieSegmentation.
        Now with aggressive morphological refinement to reduce edge halos.
        """
        # MediaPipe expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie.process(img)
        if results.segmentation_mask is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # 1) initial float mask with higher threshold to start tighter
        float_mask = (results.segmentation_mask > thresh).astype(np.float32)

        # 2) clean up binary (for morphology)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin0     = (float_mask > 0.5).astype(np.uint8)
        clean    = cv2.morphologyEx(bin0, cv2.MORPH_OPEN,  kernel3)
        clean    = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel5)

        # 2.5) Keep only the largest connected component (main subject)
        # This filters out people on screens, reflections, photos, etc.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
        if num_labels > 1:  # If we found connected components (excluding background)
            # Find the largest component (excluding background label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = (labels == largest_label).astype(np.uint8)

        # 3) Less aggressive erosion to pull edges inward and reduce halos
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean = cv2.erode(clean, kernel_erode, iterations=2)

        # 4) Light blur with more smoothing on the cleaned float mask
        #blur = cv2.GaussianBlur(clean.astype(np.float32), (3, 3), 0.5)

        # 5) return the clean binary mask directly
        return clean

    def apply_alpha_mask(self, frame, mask, soften_edge):
        # 1) Binary mask from SAM logits
        prob = 1 / (1 + np.exp(-mask))        # sigmoid
        bin_mask = (prob.squeeze() > 0.4).astype(np.uint8)

        # 2) Resize to full frame
        bin_mask = cv2.resize(bin_mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)

        # 3) Build 8-bit alpha
        alpha = (bin_mask * 255).astype(np.uint8)

        # 4) Small erosion to pull edges inward and reduce halos
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        alpha = cv2.erode(alpha, kernel_erode, iterations=2)

        # ── make sure at least ONE pixel is transparent ───────────────
        if alpha.max() == alpha.min():        # constant plane (all 0 or all 255)
            alpha[0, 0] = 255 if alpha.max() ==   0 else 0
            logging.info("🔍 Poked variation into alpha")

        # 5) Optional feathering to soften the now-cleaner edges
        if soften_edge:
            alpha = self.feather_alpha(alpha)

        # 6) Composite
        rgb = cv2.bitwise_and(frame, frame, mask=alpha)
        out  = np.dstack([rgb, alpha])   # shape should be (H, W, 4)

        # ── Debug: print out the shape and channel ranges ───────────────
        logging.info(f"apply_alpha_mask → out.shape = {out.shape}")    # e.g. (600,600,4)
        b, g, r, a = cv2.split(out)
        logging.info(f"  channel mins/maxs: B={b.min()}/{b.max()}, "
                    f"G={g.min()}/{g.max()}, R={r.min()}/{r.max()}, A={a.min()}/{a.max()}")

        return out


    @staticmethod
    def feather_alpha(alpha, iterations: int = 0):
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        return np.clip(blurred, 0, 255).astype(np.uint8)
