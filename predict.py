import torch
import numpy as np
import tempfile
import os
import cv2
from sam2.build_sam import build_sam2_video_predictor
from cog import BasePredictor, Input, Path
import logging
import subprocess
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from pathlib import Path


class Predictor(BasePredictor):
    """
    Same logic as the original file, **only** tweaked to
    1. mask every 2 frames (prev‑mask reuse)
    2. skip the PNG detour
    3. pipe raw BGRA frames directly into FFmpeg
    """

    # ─────────────────────────  setup  ────────────────────────────
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("Starting setup")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        self.checkpoint = "/sam2_hiera_base_plus.pt"
        cfg_path = Path(__file__).parent / "sam2.1_hiera_b+.yaml"
        self.model_cfg = f"file://{cfg_path.resolve()}"

        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        logging.info("SAM2 predictor built successfully")

        # pre‑trained Faster R‑CNN for body detection
        self.body_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.body_detector.eval().to(self.device)
        logging.info("Setup completed")

    # ─────────────────────────  predict  ──────────────────────────
    def predict(
        self,
        input_video: Path = Input(description="Input video file"),
        mask_every_n_frames: int = Input(description="Recompute alpha every N frames",
                                        default=2, ge=1, le=30),
    ) -> Path:

        # 1. Decode clip directly into memory (BGR frames)
        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames: list[np.ndarray] = []
        ok, frm = cap.read()
        while ok:
            frames.append(frm)
            ok, frm = cap.read()
        cap.release()
        n_frames = len(frames)
        if n_frames == 0:
            raise RuntimeError("No decodable frames in input video")
        logging.info(f"Loaded {n_frames} frames (fps≈{fps:.2f})")

        h, w = frames[0].shape[:2]

        # 2. JPEG dump for SAM‑2
        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        for i, frm in enumerate(frames):
            cv2.imwrite(os.path.join(tmp_dir, f"{i:06d}.jpg"), frm,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        logging.info(f"Wrote {len(frames)} JPEGs to {tmp_dir}")
        
        state = self.predictor.init_state(video_path=tmp_dir)  # guaranteed to work

        # 3. Seed keypoints on first frame
        keypoints = self.detect_body_keypoints(frames[0])
        self.predictor.add_new_points(state, 0, 1, keypoints,
                                      labels=np.ones(len(keypoints), np.int32))

        # 4. Propagate masks (logits) through entire clip
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for fidx, obj_ids, logits in self.predictor.propagate_in_video(state):
            video_segments[fidx] = {oid: logits[i].cpu().numpy()
                                    for i, oid in enumerate(obj_ids)}
        logging.info("Mask propagation complete – streaming to encoder")

        # 5. Spawn FFmpeg encoder, pipe raw BGRA
        output_path = "/output.webm"
        ffmpeg = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgra",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
            "-crf", "19", "-b:v", "0", "-cpu-used", "4", "-row-mt", "1",
            output_path,
        ], stdin=subprocess.PIPE)

        prev_mask = None
        frame_count = 0
        
        for idx, frame in enumerate(frames):          # use in‑memory frame list
            if idx % mask_every_n_frames == 0 or prev_mask is None:
                try:
                    out_mask = next(iter(video_segments[idx].values()))
                except (KeyError, StopIteration):
                    out_mask = prev_mask if prev_mask is not None else np.ones(frame.shape[:2], np.uint8)
                prev_mask = out_mask
            else:
                out_mask = prev_mask
        
            bgra = self.apply_alpha_mask(frame, out_mask)
            ffmpeg.stdin.write(bgra.tobytes())
            frame_count += 1

        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError("FFmpeg encoding failed")

        logging.info(f"Processed {frame_count} frames → {output_path}")
        return Path(output_path)

    def detect_body_keypoints(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor
        img_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.body_detector(img_tensor)[0]

        # Get the bounding box with the highest score
        if len(prediction['boxes']) > 0:
            best_box = prediction['boxes'][prediction['scores'].argmax()].cpu().numpy()
            x1, y1, x2, y2 = best_box

            # Calculate center of the bounding box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Calculate the dimensions of the bounding box
            width, height = x2 - x1, y2 - y1

            # Define offset for surrounding points (20% of width/height)
            offset_x, offset_y = width * 0.2, height * 0.2

            # Define keypoints
            keypoints = np.array([
                [center_x, center_y],  # Center
                [center_x - offset_x, center_y],  # Left
                [center_x + offset_x, center_y],  # Right
                [center_x, center_y - offset_y],  # Top
                [center_x, center_y + offset_y],  # Bottom
            ], dtype=np.float32)

            # Ensure all points are within the bounding box
            keypoints[:, 0] = np.clip(keypoints[:, 0], x1, x2)
            keypoints[:, 1] = np.clip(keypoints[:, 1], y1, y2)

            return keypoints
        else:
            # If no person is detected, fall back to center point
            height, width = frame.shape[:2]
            center = np.array([[width // 2, height // 2]], dtype=np.float32)
            return np.tile(center, (5, 1))  # Return 5 identical center points as fallback

    
    # --- replace remove_background with this helper -----------------
    def apply_alpha_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Return a 4‑channel BGRA image:
          • RGB channels = original pixels where mask==1, zeros elsewhere  
          • A   channel  = 255 where mask==1, 0 elsewhere
        """
        mask = (mask.squeeze() > 0).astype(np.uint8)            # 1‑channel (0/1)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    
        alpha = (mask * 255).astype(np.uint8)                   # 0/255
        alpha = self.feather_alpha(alpha)
        # Zero‑out RGB where the subject is absent to avoid colored “ghost” halo
        rgb = cv2.bitwise_and(frame, frame, mask=alpha)
    
        bgra = np.dstack([rgb, alpha])                          # (H, W, 4)
        return bgra

    def feather_alpha(self, alpha: np.ndarray, iterations: int = 0) -> np.ndarray:
        """
        Very lightweight mask‑feathering:
        • Dilate a bit, then blur, then rescale to 0‑255.
        • Keeps fine hair edges semi‑transparent instead of chopping them off.
        """
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)          # expand mask
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)                      # soften edge
        # Re‑map 0…255 so that fully‑inside pixels stay opaque
        feathered = np.clip((blurred.astype(np.float32) / 255.0) * 255, 0, 255).astype(np.uint8)
        return feathered

if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Simple CLI:  python predict.py /path/to/video.webm 4
    vid = Path(sys.argv[1])
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    p = Predictor()
    p.setup()
    out = p.predict(input_video=vid, mask_every_n_frames=n)
    print("Output:", out)
