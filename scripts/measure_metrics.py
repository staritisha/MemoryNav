"""
Run the full pipeline on a 2-minute video and log real performance metrics.
Output: evaluation/metrics.json

Usage:
    python scripts/measure_metrics.py --video evaluation/videos/living_room_01.mp4
"""
import sys, time, json, argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=3600)  # 2 min @ 30fps
    parser.add_argument("--out", default="evaluation/metrics.json")
    args = parser.parse_args()

    from app.perception.detector import Detector
    from app.perception.depth import DepthEstimator
    from app.perception.frame_quality import check_frame_quality
    from app.perception.tracker import ObjectTracker
    from app.memory_modules.long_term import LongTermMemory
    from app.config import settings

    print("Loading models...")
    detector = Detector()
    depth = DepthEstimator()
    tracker = ObjectTracker()
    memory = LongTermMemory()

    cap = cv2.VideoCapture(args.video)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_times = []
    detect_times = []
    depth_times = []
    memory_times = []
    detections_per_frame = []
    frames_passed_quality = 0
    frames_total = 0

    print("Running pipeline...")
    while frames_total < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames_total += 1

        t_frame_start = time.perf_counter()

        # Frame quality gate
        qr = check_frame_quality(frame)
        if not qr.ok:
            continue
        frames_passed_quality += 1

        # Detection
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        detect_times.append((time.perf_counter() - t0) * 1000)

        # Tracking
        tracked = tracker.update(detections, frame)
        detections_per_frame.append(len(tracked))

        # Depth (every 3rd frame via internal cache)
        t0 = time.perf_counter()
        depth_map = depth.estimate(frame)
        depth_times.append((time.perf_counter() - t0) * 1000)

        # Memory retrieval
        t0 = time.perf_counter()
        if tracked:
            memory.retrieve(tracked[0].class_name, n_results=3)
        memory_times.append((time.perf_counter() - t0) * 1000)

        frame_times.append((time.perf_counter() - t_frame_start) * 1000)

    cap.release()

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    metrics = {
        "video": args.video,
        "frames_total": frames_total,
        "frames_passed_quality_gate": frames_passed_quality,
        "quality_pass_rate_pct": round(100 * frames_passed_quality / max(frames_total, 1), 1),
        "fps_effective": round(1000 / avg(frame_times), 1) if frame_times else 0,
        "avg_frame_latency_ms": avg(frame_times),
        "avg_detection_ms": avg(detect_times),
        "avg_depth_ms": avg(depth_times),
        "avg_memory_retrieval_ms": avg(memory_times),
        "avg_detections_per_frame": avg(detections_per_frame),
        "max_detections_per_frame": max(detections_per_frame) if detections_per_frame else 0,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Frames processed:        {frames_passed_quality}/{frames_total}")
    print(f"Quality pass rate:       {metrics['quality_pass_rate_pct']}%")
    print(f"Effective FPS:           {metrics['fps_effective']}")
    print(f"Avg frame latency:       {metrics['avg_frame_latency_ms']}ms")
    print(f"Avg detection latency:   {metrics['avg_detection_ms']}ms")
    print(f"Avg depth latency:       {metrics['avg_depth_ms']}ms")
    print(f"Avg memory retrieval:    {metrics['avg_memory_retrieval_ms']}ms")
    print(f"Avg detections/frame:    {metrics['avg_detections_per_frame']}")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
