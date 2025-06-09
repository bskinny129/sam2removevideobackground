import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
from cog import BasePredictor, Input, Path
import logging
import shutil
import subprocess
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class Predictor(BasePredictor):
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("Starting setup")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        self.checkpoint = "/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"

        try:
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
            logging.info("SAM2 predictor built successfully")
        except Exception as e:
            logging.exception(f"Error building SAM2 predictor: {e}")
            raise

        # Load a pre-trained Faster R-CNN model for body detection
        self.body_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.body_detector.eval()
        self.body_detector.to(self.device)

        logging.info("Setup completed")

    def detect_body_keypoints(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.body_detector(img_tensor)[0]

        if len(prediction['boxes']) > 0:
            best_box = prediction['boxes'][prediction['scores'].argmax()].cpu().numpy()
            x1, y1, x2, y2 = best_box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            dx, dy = w * 0.2, h * 0.2
            pts = np.array([
                [cx, cy],
                [cx - dx, cy],
                [cx + dx, cy],
                [cx, cy - dy],
                [cx, cy + dy],
            ], dtype=np.float32)
            pts[:, 0] = np.clip(pts[:, 0], x1, x2)
            pts[:, 1] = np.clip(pts[:, 1], y1, y2)
            return pts
        else:
            h, w = frame.shape[:2]
            ctr = np.array([[w // 2, h // 2]], dtype=np.float32)
            return np.tile(ctr, (5, 1))

    def compute_alpha(self, frame, mask_logits):
        # Binary mask interior
        mask_bin = (mask_logits.squeeze() > 0).astype(np.uint8)
        # Hair-edge region
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask_bin * 255, kernel, iterations=2) // 255
        hair_edge = ((dilated - mask_bin) > 0).astype(np.uint8)

        # Background color sample: mean of pixels outside dilated mask
        bg_samples = frame[dilated == 0]
        if bg_samples.size:
            bg_avg = bg_samples.mean(axis=0)
        else:
            bg_avg = frame.reshape(-1, 3).mean(axis=0)

        # Color distance map
        diff = frame.astype(np.float32) - bg_avg
        color_dist = np.linalg.norm(diff, axis=2)
        cd_min, cd_max = color_dist.min(), color_dist.max()
        norm_cd = (color_dist - cd_min) / (cd_max - cd_min + 1e-8)

        # Hair-edge alpha from normalized color distance
        hair_alpha = hair_edge.astype(np.float32) * norm_cd

        # Final alpha: interior fully opaque + hair-edge partial
        alpha = mask_bin.astype(np.float32) + hair_alpha
        alpha = np.clip(alpha, 0.0, 1.0)
        return (alpha * 255).astype(np.uint8)

    def predict(
        self,
        input_video: Path = Input(description="Input video file"),
    ) -> Path:
        frames_dir = "/frames"
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        logging.info(f"Extracting frames from {input_video}")
        subprocess.run([
            "ffmpeg", "-i", str(input_video),
            "-q:v", "2", "-start_number", "0",
            f"{frames_dir}/%05d.jpg"
        ], check=True)

        frame_names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if not frame_names:
            raise RuntimeError("No frames extracted")

        # Initialize SAM2 inference
        state = self.predictor.init_state(video_path=frames_dir)
        first = cv2.imread(os.path.join(frames_dir, frame_names[0]))
        kps = self.detect_body_keypoints(first)
        _, obj_ids, _ = self.predictor.add_new_points(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=kps,
            labels=np.ones(len(kps), dtype=np.int32),
        )

        video_masks = {}
        for idx, out_ids, logits in self.predictor.propagate_in_video(state):
            video_masks[idx] = {oid: logits[i].cpu().numpy() for i, oid in enumerate(out_ids)}

        rgba_dir = "/rgba_frames"
        if os.path.exists(rgba_dir):
            shutil.rmtree(rgba_dir)
        os.makedirs(rgba_dir, exist_ok=True)

        for i, fname in enumerate(frame_names):
            frame_path = os.path.join(frames_dir, fname)
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            logits = next(iter(video_masks[i].values()))

            alpha = self.compute_alpha(frame, logits)
            h, w = frame.shape[:2]
            alpha_resized = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)

            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha_resized

            out_png = os.path.join(rgba_dir, f"{i:05d}.png")
            cv2.imwrite(out_png, bgra)

        out_webm = "/output_with_alpha.webm"
        logging.info("Encoding VP9+alpha WebM…")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", f"{rgba_dir}/%05d.png",
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            out_webm
        ], check=True)

        logging.info(f"Written RGBA WebM (VP9+alpha): {out_webm}")
        return Path(out_webm)
