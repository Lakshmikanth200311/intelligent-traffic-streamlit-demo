import cv2
import os
from app.services.detector import YOLODetector
from app.services.counter import LineCrossingCounter
from app.models.schemas import CountingLine, Point, Approach

def process_video(input_video_path: str, output_dir="uploads/outputs"):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"output_{os.path.basename(input_video_path)}"
    )

    detector = YOLODetector()

    # Simple demo counting line (center vertical)
    counting_line = CountingLine(
        start=Point(x=640, y=0),
        end=Point(x=640, y=720)
    )

    counter = LineCrossingCounter(
        counting_line=counting_line,
        approach=Approach.NORTH
    )

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        vehicle_counts = counter.update(detections)

        # OPTIONAL: draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                det.class_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        out.write(frame)

    cap.release()
    out.release()

    return output_path, vehicle_counts
