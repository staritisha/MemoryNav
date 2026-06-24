#!/usr/bin/env python3
"""
evaluation/measure_latency.py

Phase 3 item 6 — measure real pipeline latency from a recorded video.

Runs the full pipeline (quality gate → YOLO → ByteTrack → Depth → Risk
→ Memory) on every sampled frame and writes evaluation/latency_metrics.json.

Usage:
    cd /Users/ritishajadhao/Desktop/building/MemoryNav
    source .venv/bin/activate
    python evaluation/measure_latency.py \
        --video evaluation/videos/living_room_01.mp4 \
        --frames 120 \
        --sample-every 3

Output (evaluation/latency_metrics.json):
    {
      "video": "living_room_01.mp4",
      "frames_sampled": 120,
      "fps_sustained": 4.2,
      "avg_latency_ms": 187.3,
      "p50_latency_ms": 182.1,
      "p95_latency_ms": 231.4,
      "avg_yolo_ms": 62.1,
      "avg_depth_ms": 118.4,   (amortised over skip window)
      "avg_memory_ms": 6.8,
      "detections_per_frame": 1.4,
      "memory_retrieval_count": 12,
      "ghost_alerts_fired": 2,
      "component_status": { "detector": "real", ... }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, quantiles

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="evaluation/videos/living_room_01.mp4",
                        type=Path)
    parser.add_argument("--frames", default=120, type=int,
                        help="Max frames to process (default 120 ≈ 4s @ 30fps)")
    parser.add_argument("--sample-every", default=3, type=int,
                        help="Process every Nth frame (default 3)")
    parser.add_argument("--out", default="evaluation/latency_metrics.json", type=Path)
    args = parser.parse_args()

    print(f"Loading pipeline components...")
    from app.perception.detector import Detector
    from app.perception.depth import DepthEstimator
    from app.perception.frame_quality import check_frame_quality
    from app.perception.tracker import ObjectTracker
    from app.memory_modules.long_term import LongTermMemory
    from app.memory_modules.spatial_map import SpatialMap
    from app.risk.engine import RiskEngine
    from app.risk.models import Detection as RiskDetection

    detector = Detector()
    depth_est = DepthEstimator()
    tracker = ObjectTracker(frame_rate=30)
    long_term = LongTermMemory()
    spatial_map = SpatialMap()
    spatial_map.set_room("Test Room")
    risk_engine = RiskEngine(long_term=long_term)

    print(f"Components loaded. Opening {args.video}...")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    total_ms:        list[float] = []
    yolo_ms:         list[float] = []
    depth_ms:        list[float] = []
    memory_ms:       list[float] = []
    detections_counts: list[int] = []
    memory_retrieval_count = 0
    ghost_alerts_fired = 0
    frames_processed = 0
    frame_idx = 0
    wall_start = time.perf_counter()

    print(f"Running pipeline (sample every {args.sample_every} frames, "
          f"max {args.frames} frames)...")

    while frames_processed < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.sample_every != 0:
            continue

        t_frame_start = time.perf_counter()

        # Quality gate
        quality = check_frame_quality(frame)
        if not quality.ok:
            continue

        # YOLO
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        yolo_ms.append((time.perf_counter() - t0) * 1000)

        # ByteTrack
        tracked = tracker.update(dets, frame)

        # Depth
        t0 = time.perf_counter()
        depth_map = depth_est.estimate(frame)
        depth_ms.append((time.perf_counter() - t0) * 1000)

        # Spatial map + risk detections
        risk_dets: list[RiskDetection] = []
        for pd in tracked:
            dist = depth_est.depth_at_bbox(depth_map, pd.bbox)
            if np.isnan(dist) or dist <= 0:
                dist = float("nan")
            rd = RiskDetection.from_perception(pd, distance_metres=dist)
            risk_dets.append(rd)
            spatial_map.update(pd.class_name, pd.bbox, frame.shape[1], dist, pd.confidence)

        # Memory retrieval (via risk engine resolver)
        t0 = time.perf_counter()
        if risk_dets:
            assessments = risk_engine.assess_all(risk_dets, user_context_weight=None)
            for a in assessments:
                if a.context_result and a.context_result.spatial_memory:
                    memory_retrieval_count += 1
        mem_elapsed = (time.perf_counter() - t0) * 1000
        memory_ms.append(mem_elapsed)

        # Ghost alert check
        if long_term.count() > 0:
            detected_classes = {d.class_name for d in tracked}
            try:
                query = ("obstacle near " + ", ".join(list(detected_classes)[:3])
                         if detected_classes else "obstacle")
                results = long_term.retrieve(query, n_results=3)
                for r in results:
                    if r.similarity >= 0.50:
                        remembered = r.metadata.get("class_name", "")
                        if remembered and remembered not in detected_classes:
                            ghost_alerts_fired += 1
                            break
            except Exception:
                pass

        total_ms.append((time.perf_counter() - t_frame_start) * 1000)
        detections_counts.append(len(tracked))
        frames_processed += 1

        if frames_processed % 20 == 0:
            print(f"  {frames_processed}/{args.frames} frames | "
                  f"last={total_ms[-1]:.0f}ms | "
                  f"dets={detections_counts[-1]}")

    cap.release()
    wall_elapsed = time.perf_counter() - wall_start

    if not total_ms:
        print("ERROR: no frames processed.", file=sys.stderr)
        sys.exit(1)

    # Compute p95
    sorted_ms = sorted(total_ms)
    p50 = median(sorted_ms)
    p95_idx = int(len(sorted_ms) * 0.95)
    p95 = sorted_ms[min(p95_idx, len(sorted_ms) - 1)]

    fps_sustained = frames_processed / wall_elapsed

    metrics = {
        "video":                  str(args.video),
        "frames_sampled":         frames_processed,
        "sample_every_n":         args.sample_every,
        "fps_sustained":          round(fps_sustained, 2),
        "avg_latency_ms":         round(mean(total_ms), 1),
        "p50_latency_ms":         round(p50, 1),
        "p95_latency_ms":         round(p95, 1),
        "avg_yolo_ms":            round(mean(yolo_ms), 1) if yolo_ms else None,
        "avg_depth_ms":           round(mean(depth_ms), 1) if depth_ms else None,
        "avg_memory_ms":          round(mean(memory_ms), 1) if memory_ms else None,
        "detections_per_frame":   round(mean(detections_counts), 2),
        "memory_retrieval_count": memory_retrieval_count,
        "ghost_alerts_fired":     ghost_alerts_fired,
        "spatial_map_snapshot":   spatial_map.snapshot(),
        "generated_at":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n── Latency metrics ──────────────────────────────────────")
    print(f"  Frames processed:       {frames_processed}")
    print(f"  Sustained throughput:   {fps_sustained:.1f} fps")
    print(f"  Avg end-to-end latency: {metrics['avg_latency_ms']} ms")
    print(f"  p50 latency:            {metrics['p50_latency_ms']} ms")
    print(f"  p95 latency:            {metrics['p95_latency_ms']} ms")
    print(f"  Avg YOLO:               {metrics['avg_yolo_ms']} ms")
    print(f"  Avg Depth (amortised):  {metrics['avg_depth_ms']} ms")
    print(f"  Avg Memory retrieval:   {metrics['avg_memory_ms']} ms")
    print(f"  Detections/frame:       {metrics['detections_per_frame']}")
    print(f"  Memory hits:            {memory_retrieval_count}")
    print(f"  Ghost alerts:           {ghost_alerts_fired}")
    print(f"  Spatial map: {spatial_map.summary_text()}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
