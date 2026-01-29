"""
app.py

Usage examples:
  python app.py --source 0 --display-fps --tts --save-log detections.csv
  python app.py --source input.mp4 --model yolov8n.pt --save-log out.csv

This script works in two modes:
 - Live camera (source is '0' or other camera index)
 - Video file (source is a path to .mp4/.avi)

It uses ultralytics YOLOv8 for detection and OpenCV for I/O.
"""
import argparse
import time
import csv
from pathlib import Path

import cv2
import numpy as np

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time/Object-from-file detection with YOLOv8')
    parser.add_argument('--source', type=str, default='0',
                        help="Camera index (0) or path to video file")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model file (yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--display-fps', action='store_true', help='Overlay FPS on frame')
    parser.add_argument('--tts', action='store_true', help='Speak detected labels (requires pyttsx3)')
    parser.add_argument('--save-log', type=str, default=None, help='Path to CSV file to save detection logs')
    parser.add_argument('--output', type=str, default=None, help='Save annotated video to this path (e.g. out.mp4)')
    return parser.parse_args()


def init_tts():
    engine = None
    if TTS_AVAILABLE:
        try:
            engine = pyttsx3.init()
            # optionally set properties here (rate, volume, voice)
            rate = engine.getProperty('rate')
            engine.setProperty('rate', max(120, rate - 20))
        except Exception:
            engine = None
    return engine


def speak(engine, text):
    if engine:
        engine.say(text)
        engine.runAndWait()


def draw_boxes(frame, boxes, scores, class_ids, names):
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{names[int(cls)]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return frame


def save_log_row(csv_writer, timestamp, label, score, bbox):
    x1, y1, x2, y2 = bbox
    csv_writer.writerow({'timestamp': timestamp, 'label': label, 'confidence': score,
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})


def main():
    args = parse_args()

    # Load model
    print(f'Loading model: {args.model} (this may download weights)')
    model = YOLO(args.model)

    # Setup source
    source = args.source
    cap = None
    if source.isdigit() or source == '0':
        source_idx = int(source)
        cap = cv2.VideoCapture(source_idx)
    else:
        # file
        if not Path(source).exists():
            raise FileNotFoundError(f'Source file not found: {source}')
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source: {source}')

    # Prepare output video writer if requested
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # CSV logging
    csv_file = None
    csv_writer = None
    if args.save_log:
        csv_file = open(args.save_log, mode='w', newline='')
        fieldnames = ['timestamp', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    # TTS
    tts_engine = init_tts() if args.tts else None
    if args.tts and not tts_engine:
        print('Warning: pyttsx3 not available or failed to initialize — TTS disabled')

    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {i: str(i) for i in range(1000)}

    prev_time = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No more frames, exiting')
                break

            # Run detection
            results = model.predict(source=frame, conf=args.conf, iou=args.iou, verbose=False)
            r = results[0]

            boxes, scores, class_ids = [], [], []
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    # box.xyxy is tensor-like; convert to list
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    boxes.append(xyxy)
                    scores.append(conf)
                    class_ids.append(cls)

            # Draw boxes
            annotated = frame.copy()
            annotated = draw_boxes(annotated, boxes, scores, class_ids, names)

            # Speak top detection (if enabled)
            if args.tts and tts_engine and class_ids:
                top_idx = int(np.argmax(scores))
                speak_text = f"{names[class_ids[top_idx]]}"
                speak(tts_engine, speak_text)

            # Log detections
            if csv_writer:
                timestamp = time.time()
                for bbox, score, cls in zip(boxes, scores, class_ids):
                    save_log_row(csv_writer, timestamp, names[cls], float(score), bbox)

            # FPS overlay
            if args.display_fps:
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
                prev_time = curr_time
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Show
            cv2.imshow('Detection', annotated)

            # write output video
            if writer is not None:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('User pressed q — exiting')
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
