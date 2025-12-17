from app.services.detector import YOLODetector
from app.services.counter import LineCrossingCounter
from app.models.schemas import CountingLine, Point, Approach

def process_video(input_video_path: str):
    import cv2  # lazy import (SAFE)

    detector = YOLODetector()

    counting_line = CountingLine(
        start=Point(x=640, y=0),
        end=Point(x=640, y=720)
    )

    counter = LineCrossingCounter(
        counting_line=counting_line,
        approach=Approach.NORTH
    )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        counter.update(detections)

    cap.release()

    # ❌ No output video on cloud
    # ✅ Only return counts
    return "", counter.vehicle_counts
