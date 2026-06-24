#!/usr/bin/env python3
"""
MemoryNav — Day 1 Smoke Test
scripts/day1_yolo_test.py

Phase 1 win: open the webcam, run the YOLOv8-nano Detector, print
detections to the console. Proves the model loads, the configured
device (MPS on M2) works, and frames are flowing end to end before
any of the Risk Engine / Memory / Alert Manager layers exist.

Run from anywhere:
    python scripts/day1_yolo_test.py
Stop with Ctrl+C.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import cv2

from app.config import settings
from app.perception.detector import Detector

detector = Detector()
cap = cv2.VideoCapture(settings.CAMERA_INDEX)
print(f"Webcam open: {cap.isOpened()}  |  device: {detector.device}  |  Ctrl+C to stop")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed — check camera permissions/index.")
            break
        labels = [f"{d.class_name} {d.confidence:.2f}" for d in detector.detect(frame)]
        print(labels or "no detections")
finally:
    cap.release()