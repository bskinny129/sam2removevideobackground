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
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s:%(message)s"
        )
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

    def predict(
            self,
            input_video: Path = Input(description="Input video file"),
            bg_color: str = Input(description="Background color (hex code)", default="#00FF00")
    ) -> Path:
        bg_color = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]  # BGR for OpenCV

        frames_dir = "/frames"
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        logging.info(f"Input video path: {input_video}")
        logging.info(f"Input video exists: {os.path.exists(input_video)}")
        logging.info(f"Input video file size: {os.path.getsize(input_video)} bytes")

        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):      # some containers don’t store fps
            fps = 30
        cap.release()
        print(f"Detected FPS: {fps}")

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-i", str(input_video), "-q:v", "2", "-start_number", "0",
                f"{frames_dir}/%05d.jpg"
            ]
            logging.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            logging.info("FFmpeg command executed successfully")
            logging.debug(f"FFmpeg stdout: {result.stdout}")
            logging.debug(f"FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract frames from video: {e.stderr}")

        frame_names = [p for p in os.listdir(frames_dir) if p.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
        logging.info(f"Number of frames extracted: {len(frame_names)}")

        if not frame_names:
            logging.error(f"No frames were extracted. Contents of {frames_dir}: {os.listdir(frames_dir)}")
            raise RuntimeError(
                f"No frames were extracted from the video. The video file may be corrupt or in an unsupported format.")

        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        try:
            inference_state = self.predictor.init_state(video_path=frames_dir)
            logging.info("Inference state initialized successfully")
        except Exception as e:
            logging.exception(f"Error initializing inference state: {e}")
            raise

        first_frame_path = os.path.join(frames_dir, frame_names[0])
        logging.info(f"Attempting to read first frame: {first_frame_path}")
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            logging.error(f"Failed to read the first frame. File exists: {os.path.exists(first_frame_path)}")
            raise RuntimeError(f"Failed to read the first frame: {frame_names[0]}")

        # Detect body keypoints in the first frame
        keypoints = self.detect_body_keypoints(first_frame)
        logging.info(f"Detected {len(keypoints)} keypoints")

        try:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=keypoints,
                labels=np.ones(len(keypoints), dtype=np.int32),  # All points are positive
            )
            logging.info("New points added successfully")
        except Exception as e:
            logging.exception(f"Error adding new points: {e}")
            raise

        video_segments = {}
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i].cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            logging.info("Video segments propagated successfully")
        except Exception as e:
            logging.exception(f"Error propagating video segments: {e}")
            raise

        output_frames_dir = '/output_frames'
        os.makedirs(output_frames_dir, exist_ok=True)

        frame_count = 0
        for out_frame_idx in range(len(frame_names)):
            frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)

            if frame is None:
                logging.error(f"Failed to read frame: {frame_path}")
                continue

            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                frame_with_alpha = self.apply_alpha_mask(frame, out_mask)

            output_frame_path = os.path.join(output_frames_dir, f"{out_frame_idx:05d}.png")
            cv2.imwrite(output_frame_path, frame_with_alpha)
            frame_count += 1

        output_video_path = '/output.webm'

        try:
            final_video_cmd = [
                "ffmpeg", "-y",  # Add -y flag to force overwrite without prompting
                "-framerate", str(fps),
                "-i", f"{output_frames_dir}/%05d.png",
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",      # yuva444p is fine too; must include “a”
                "-auto-alt-ref", "0",        # alpha won’t work if alt‑ref is on
                "-crf", "19",
                "-b:v", "0",
                "-cpu-used", "4",
                "-row-mt", "1",
                output_video_path
            ]
            logging.info(f"Running final FFmpeg command: {' '.join(final_video_cmd)}")

            result = subprocess.run(final_video_cmd, capture_output=True, text=True, check=True)
            logging.info("Final video created successfully")
            logging.debug(f"Final FFmpeg stdout: {result.stdout}")
            logging.debug(f"Final FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating final video: {e.stderr}")
            raise RuntimeError(f"Failed to create final video: {e.stderr}")

        logging.info(f"Processed {frame_count} frames")
        logging.info(f"Background removed video saved as {output_video_path}")

        return Path(output_video_path)

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
        #alpha = self.feather_alpha(alpha)
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
