import logging
import os
import subprocess
import tempfile

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
        self.checkpoint = "sam2_hiera_base_plus.pt"
        self.model_cfg = "sam2.1_hiera_b+.yaml"
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
        selfie_threshold: float = Input(
            description="MediaPipe selfie segmentation threshold", default=0.3, ge=0.1, le=0.9
        ),
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
            q = 100 if j == 0 else jpeg_quality
            cv2.imwrite(
                os.path.join(tmp_dir, f"{j:06d}.jpg"),
                frames[orig_idx],
                [int(cv2.IMWRITE_JPEG_QUALITY), q],
            )
        logging.info(f"Wrote {len(sampled_idxs)} JPEGs → {tmp_dir}")

        # 3️⃣ seed SAM-2 with refined mask
        state = self.predictor.init_state(video_path=tmp_dir)
        seed_mask = self.get_portrait_mask(frames[0], selfie_threshold)
        self.predictor.add_new_mask(state, 0, 1, seed_mask)
        prop_iter = iter(self.predictor.propagate_in_video(state))

        # 4️⃣ spawn FFmpeg encoder
        output_path = "/output.webm"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-thread_queue_size", "64",
                "-framerate", str(fps),
                "-f", "rawvideo", "-pix_fmt", "bgra",
                "-s", f"{w}x{h}", "-i", "-",
                "-thread_queue_size", "32",
                "-i", str(input_video),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                "-auto-alt-ref", "0",
                "-crf", str(crf), "-b:v", "0",
                "-row-mt", "1", "-tile-columns", "2",
                "-threads", "0",
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
                if idx == 0:
                    logging.info(f"🔍 Frame {idx}: Using SAM-2 refined mask (shape: {mask.shape})")
                ptr += 1
            else:
                mask = prev_mask if prev_mask is not None else np.ones((h, w), np.uint8)
                if idx == 0:
                    logging.info(f"🔍 Frame {idx}: Using fallback mask")
            prev_mask = mask
            bgra = self.apply_alpha_mask(frame, mask, soften_edge)
            ffmpeg.stdin.write(bgra.tobytes())

        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("FFmpeg encoding failed")

        logging.info(f"✅ Finished → {output_path}")
        return Path(output_path)                         # ← cog.Path

    # ─────────────────────── helpers ─────────────────────────────
    def get_portrait_mask(self, frame, thresh: float = 0.35) -> np.ndarray:
        """
        Returns a H×W uint8 mask of the upper-body/head using MediaPipe SelfieSegmentation.
        Now with morphological refinement to reduce edge artifacts.
        """
        # MediaPipe expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie.process(img)
        if results.segmentation_mask is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # 1) initial float mask
        float_mask = (results.segmentation_mask > thresh).astype(np.float32)

        # 2) clean up binary (for morphology)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin0     = (float_mask > 0.5).astype(np.uint8)
        clean    = cv2.morphologyEx(bin0, cv2.MORPH_OPEN,  kernel3)
        clean    = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel5)

        # 3) 1px Gaussian blur on the cleaned float mask
        blur     = cv2.GaussianBlur(clean.astype(np.float32), (5, 5), 1.0)

        # 4) re-threshold at 0.1 to get a softer binary mask
        return (blur > 0.1).astype(np.uint8)

    def apply_alpha_mask(self, frame, mask, soften_edge):
        # 1) Binary mask from SAM logits
        bin_mask = (mask.squeeze() > 0.5).astype(np.uint8)

        # 2) Resize to full frame
        bin_mask = cv2.resize(bin_mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)

        # 3) Build 8-bit alpha
        alpha = (bin_mask * 255).astype(np.uint8)

        # 4) Optional pre-blur (your existing feather_alpha)
        if soften_edge:
            alpha = self.feather_alpha(alpha)

        # 4.5) Small erosion to pull edges inward and reduce halos
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.erode(alpha, kernel_erode, iterations=1)

        # 5) FINAL 1-px Gaussian blur on alpha
        #    kernel (5,5) with sigma=1.0 for softer edges
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)

        # 6) Composite
        rgb = cv2.bitwise_and(frame, frame, mask=alpha)
        return np.dstack([rgb, alpha])

    @staticmethod
    def feather_alpha(alpha, iterations: int = 0):
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        return np.clip(blurred, 0, 255).astype(np.uint8)
