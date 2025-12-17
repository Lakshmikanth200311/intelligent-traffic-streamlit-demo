import numpy as np
from typing import List
import logging
from app.models.schemas import Detection
from app.config import settings

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self):
        self.model = None
        self.class_names = None
        logger.info("YOLODetector initialized (model not loaded yet)")

    def load_model(self):
        if self.model is None:
            from ultralytics import YOLO  # 🔥 LAZY IMPORT (CRITICAL)
            self.model = YOLO(settings.YOLO_MODEL)
            self.class_names = self.model.names
            logger.info(f"Loaded YOLO model: {settings.YOLO_MODEL}")
        return self.model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in frame"""
        import cv2  # already correct

        try:
            model = self.load_model()  # 🔥 LOAD ONLY WHEN NEEDED

            results = model(
                frame,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.IOU_THRESHOLD,
                classes=settings.VEHICLE_CLASSES,
                verbose=False
            )

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]

                        detection = Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=float(conf),
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


class OpenCVDetector:
    """Fallback detector using OpenCV DNN"""
    def __init__(self):
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []
