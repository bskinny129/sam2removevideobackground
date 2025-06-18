import logging
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input
from sam2.build_sam import build_sam2_video_predictor
from torchvision.transforms import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


class Predictor(BasePredictor):
    """
    Background-removal for portrait video using:
      • DeepLab V3 (person class) to create a seed mask
      • SAM-2 “hiera-base-plus” to refine & propagate the mask
      • Single-pass piping into FFmpeg (VP9 + alpha)
    """

    # ─────────────────────────── setup ────────────────────────────
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("🏗  Initialising model stack")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"ℹ️  Running on {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            # use FP16 autocast for SAM-2 when available
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        # ── SAM-2 predictor ───────────────────────────────────────
        self.checkpoint = "sam2_hiera_base_plus.pt"
        self.model_cfg = "sam2.1_hiera_b+.yaml"
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        logging.info("✅ SAM-2 video predictor ready")

        # ── DeepLab V3 “person” mask ──────────────────────────────
        self.seg = deeplabv3_resnet50(weights="DEFAULT").eval().to(self.device)
        logging.info("✅ DeepLab V3 loaded")

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
    ) -> Path:
        # warm-up shortcut ------------------------------------------------------
        if input_video.name == "warmup.mp4":
            logging.info("Warm-up request – echoing source")
            return input_video

        # 1️⃣ decode the whole clip into memory ---------------------------------
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

        # 2️⃣ write sampled JPEGs for SAM-2 -------------------------------------
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

        # 3️⃣ initialise & seed SAM-2 with DeepLab mask -------------------------
        state = self.predictor.init_state(video_path=tmp_dir)
        seed_mask = self.get_person_mask(frames[0])  # H×W uint8 {0,1}

        # add the binary mask (object id 1) on frame 0 of the sampled sequence
        self.predictor.add_new_mask(state, 0, 1, seed_mask)

        prop_iter = iter(self.predictor.propagate_in_video(state))

        # 4️⃣ spawn FFmpeg encoder ---------------------------------------------
        output_path = "/output.webm"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
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
                "-",  # video pipe
                "-thread_queue_size",
                "32",
                "-i",
                str(input_video),  # take audio from the source clip
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                "-auto-alt-ref",
                "0",
                "-crf",
                str(crf),
                "-b:v",
                "0",
                "-row-mt",
                "1",
                "-tile-columns",
                "2",
                "-threads",
                "0",
                "-c:a",
                "copy",
                output_path,
            ],
            stdin=subprocess.PIPE,
        )

        # 5️⃣ stream BGRA frames to FFmpeg with live mask propagation -----------
        prev_mask = None
        sampled_ptr = 0
        for idx, frame in enumerate(frames):
            # pull next propagated mask when we hit the next sampled frame
            if sampled_ptr < len(sampled_idxs) and idx == sampled_idxs[sampled_ptr]:
                _, _, logits = next(prop_iter)
                mask = logits[0].cpu().numpy()  # H×W float32  (0…1)
                sampled_ptr += 1
            else:
                mask = prev_mask if prev_mask is not None else np.ones((h, w), np.uint8)
            prev_mask = mask

            bgra = self.apply_alpha_mask(frame, mask, soften_edge)
            ffmpeg.stdin.write(bgra.tobytes())

        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("FFmpeg encoding failed")

        logging.info(f"✅ Finished → {output_path}")
        return Path(output_path)

    # ─────────────────────── helpers ─────────────────────────────
    def get_person_mask(self, frame: np.ndarray, thresh: float = 0.40) -> np.ndarray:
        """Return a binary H×W mask of the ‘person’ class from DeepLab."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ten = F.to_tensor(rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.seg(ten)["out"][0, 15]  # class-id 15 = person
        mask = (logits.sigmoid() > thresh).cpu().numpy().astype(np.uint8)
        return mask  # uint8 {0,1}

    def apply_alpha_mask(
        self, frame: np.ndarray, mask: np.ndarray, soften_edge: bool
    ) -> np.ndarray:
        """Compose BGRA frame where A = mask (optionally feathered)."""
        mask = (mask.squeeze() > 0).astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)

        alpha = (mask * 255).astype(np.uint8)
        if soften_edge:
            alpha = self.feather_alpha(alpha)

        rgb = cv2.bitwise_and(frame, frame, mask=alpha)
        return np.dstack([rgb, alpha])  # BGRA

    @staticmethod
    def feather_alpha(alpha: np.ndarray, iterations: int = 0) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        return np.clip(blurred, 0, 255).astype(np.uint8)
