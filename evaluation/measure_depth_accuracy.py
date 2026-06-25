#!/usr/bin/env python3
"""
evaluation/measure_depth_accuracy.py

Measures Depth-Anything MAE (Mean Absolute Error) against objects at
known distances. Run this once with a tape measure to get real accuracy numbers.

SETUP:
    1. Place objects at these measured distances from your webcam:
       0.5m, 1.0m, 1.5m, 2.0m, 3.0m
    2. Take a photo of each (or use a webcam snapshot)
    3. Save as evaluation/depth_test_images/dist_0.5m.jpg etc.
    4. Run this script

OR run in webcam mode (--webcam) and press SPACE at each distance.

Usage:
    python evaluation/measure_depth_accuracy.py --images-dir evaluation/depth_test_images
    python evaluation/measure_depth_accuracy.py --webcam
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.perception.depth import DepthEstimator
from evaluation.metrics import compute_mae


# Known true distances to test (metres)
TEST_DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0]


def measure_from_images(images_dir: Path, estimator: DepthEstimator) -> dict:
    """Measure MAE from pre-captured images named dist_<X>m.jpg."""
    predicted = []
    true_vals = []
    results = []

    for dist in TEST_DISTANCES:
        img_path = images_dir / f"dist_{dist}m.jpg"
        if not img_path.exists():
            print(f"  SKIP: {img_path.name} not found")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ERROR: Could not read {img_path.name}")
            continue

        depth_map = estimator.estimate(frame)
        h, w = frame.shape[:2]
        # Sample the centre third of the frame (where the object should be)
        cx1, cy1 = w // 3, h // 3
        cx2, cy2 = 2 * w // 3, 2 * h // 3
        estimated = float(np.median(depth_map[cy1:cy2, cx1:cx2]))

        predicted.append(estimated)
        true_vals.append(dist)
        error = abs(estimated - dist)
        results.append({
            "true_m": dist,
            "estimated_m": round(estimated, 3),
            "error_m": round(error, 3),
        })
        print(f"  {dist:.1f}m true  →  {estimated:.3f}m estimated  (error: {error:.3f}m)")

    mae = compute_mae(predicted, true_vals) if predicted else None
    return {"measurements": results, "mae_m": round(mae, 3) if mae else None}


def measure_from_webcam(estimator: DepthEstimator) -> dict:
    """Interactive webcam mode. Press SPACE to capture at each distance."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    predicted = []
    true_vals = []
    results = []

    print("\nWebcam mode. Press SPACE to capture, Q to quit.\n")

    for dist in TEST_DISTANCES:
        print(f"Place object at {dist:.1f}m and press SPACE to capture...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Capture at {dist}m - press SPACE", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                depth_map = estimator.estimate(frame)
                h, w = frame.shape[:2]
                cx1, cy1 = w // 3, h // 3
                cx2, cy2 = 2 * w // 3, 2 * h // 3
                estimated = float(np.median(depth_map[cy1:cy2, cx1:cx2]))
                predicted.append(estimated)
                true_vals.append(dist)
                error = abs(estimated - dist)
                results.append({
                    "true_m": dist,
                    "estimated_m": round(estimated, 3),
                    "error_m": round(error, 3),
                })
                print(f"  Captured: {estimated:.3f}m estimated  (error: {error:.3f}m)")
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    mae = compute_mae(predicted, true_vals) if predicted else None
    return {"measurements": results, "mae_m": round(mae, 3) if mae else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("evaluation/depth_accuracy.json"))
    args = parser.parse_args()

    if not args.webcam and args.images_dir is None:
        parser.error("Specify --images-dir or --webcam")

    print("Loading Depth-Anything estimator (this takes ~10s first run)...")
    estimator = DepthEstimator()

    print(f"\nMeasuring depth accuracy (metric={estimator.is_metric}):")
    if args.webcam:
        result = measure_from_webcam(estimator)
    else:
        result = measure_from_images(args.images_dir, estimator)

    if result["mae_m"] is not None:
        print(f"\nMAE: {result['mae_m']:.3f} metres across {len(result['measurements'])} measurements")
    else:
        print("\nNo measurements completed.")

    result["model"] = estimator.model_name
    result["is_metric"] = estimator.is_metric

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out}")

    print("\nAdd this to your README:")
    if result["mae_m"] is not None:
        print(f"  Depth-Anything MAE: {result['mae_m']:.3f}m on {len(result['measurements'])} measurements (tape measure ground truth)")


if __name__ == "__main__":
    main()
