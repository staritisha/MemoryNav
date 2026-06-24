#!/usr/bin/env python3
"""
MemoryNav — Repeatable Demo Scenario
scripts/demo_scenario.py

Phase 3 item 7: demonstrates the concrete value of long-term memory
by running the same video twice through the pipeline.

Pass 1 — builds memory
    Detects objects, writes observations to ChromaDB.
    No ghost alerts possible yet (nothing in memory).

Pass 2 — uses memory
    Same video. Memory now populated.
    Ghost alerts fire for objects that memory recalls but are not
    currently visible: "Previously observed chair here — proceed carefully."
    Risk scores are higher for objects that have a matching memory entry
    (ContextWeightResolver adds +0.30 spatial boost).

Output: evaluation/demo_results.json with the side-by-side comparison.

Usage:
    cd /Users/ritishajadhao/Desktop/building/MemoryNav
    source .venv/bin/activate
    python scripts/demo_scenario.py \
        --video evaluation/videos/living_room_01.mp4 \
        --frames 90
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


def run_pass(
    video_path: str,
    max_frames: int,
    sample_every: int,
    components: dict,
    write_to_memory: bool,
    label: str,
) -> dict:
    """Run one pass through the video. Returns per-pass stats."""
    detector      = components["detector"]
    depth_est     = components["depth_est"]
    tracker       = components["tracker"]
    risk_engine   = components["risk_engine"]
    long_term     = components["long_term"]
    spatial_map   = components["spatial_map"]

    spatial_map.set_room("Living Room")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    from app.perception.frame_quality import check_frame_quality
    from app.risk.models import Detection as RiskDetection

    high_risk_alerts = []
    ghost_alerts     = []
    memory_boosts    = []
    frame_idx        = 0
    frames_processed = 0
    ghost_cooldown: dict[str, float] = {}

    print(f"\n  [{label}] running on {max_frames} frames (sample every {sample_every})...")

    while frames_processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        t = time.monotonic()

        quality = check_frame_quality(frame)
        if not quality.ok:
            continue

        dets    = detector.detect(frame)
        tracked = tracker.update(dets, frame)
        detected_classes = {d.class_name for d in tracked}

        depth_map = depth_est.estimate(frame)

        risk_dets: list[RiskDetection] = []
        for pd in tracked:
            dist = depth_est.depth_at_bbox(depth_map, pd.bbox)
            if np.isnan(dist) or dist <= 0:
                dist = float("nan")
            rd = RiskDetection.from_perception(pd, distance_metres=dist)
            risk_dets.append(rd)
            spatial_map.update(pd.class_name, pd.bbox, frame.shape[1], dist, pd.confidence)

        # Risk with memory resolver active
        if risk_dets:
            assessments = risk_engine.assess_all(risk_dets, user_context_weight=None)
            for a in assessments:
                if a.level.value == "HIGH":
                    high_risk_alerts.append({
                        "frame": frame_idx,
                        "object": a.detection.class_name,
                        "risk_score": round(a.score, 3),
                        "context_weight": (
                            round(a.context_result.final_weight, 3)
                            if a.context_result else 1.0
                        ),
                    })
                # Record when memory gave a boost
                if (a.context_result
                        and a.context_result.spatial_boost > 0
                        and a.context_result.spatial_memory):
                    memory_boosts.append({
                        "frame":   frame_idx,
                        "object":  a.detection.class_name,
                        "memory":  a.context_result.spatial_memory,
                        "boost":   round(a.context_result.spatial_boost, 2),
                        "sim":     round(a.context_result.spatial_similarity or 0, 3),
                    })

        # Write observations to memory (pass 1 only)
        if write_to_memory:
            for pd, rd in zip(tracked, risk_dets):
                if pd.confidence >= 0.55 and not np.isnan(rd.distance_metres):
                    prox = ("very close" if rd.distance_metres < 0.7
                            else "nearby" if rd.distance_metres < 2.0
                            else "in the area")
                    long_term.add_context(
                        f"{pd.class_name} observed {prox} ({rd.distance_metres:.1f}m)",
                        metadata={"class_name": pd.class_name,
                                  "distance_m": round(rd.distance_metres, 2)},
                    )

        # Ghost alerts (pass 2 only — memory must be populated)
        if not write_to_memory and long_term.count() > 0:
            query = ("obstacle near " + ", ".join(list(detected_classes)[:3])
                     if detected_classes else "obstacle")
            try:
                results = long_term.retrieve(query, n_results=3)
                for r in results:
                    if r.similarity < 0.50:
                        continue
                    remembered = r.metadata.get("class_name", "")
                    if not remembered or remembered in detected_classes:
                        continue
                    last = ghost_cooldown.get(remembered, 0.0)
                    if (t - last) < 15.0:
                        continue
                    ghost_cooldown[remembered] = t
                    ghost_alerts.append({
                        "frame":   frame_idx,
                        "object":  remembered,
                        "memory":  r.text,
                        "sim":     round(r.similarity, 3),
                    })
                    print(f"    GHOST ALERT frame {frame_idx}: "
                          f"'{remembered}' recalled (sim={r.similarity:.2f})")
            except Exception:
                pass

        frames_processed += 1

    cap.release()
    return {
        "label":              label,
        "frames_processed":   frames_processed,
        "high_risk_alerts":   len(high_risk_alerts),
        "ghost_alerts":       len(ghost_alerts),
        "memory_boosts":      len(memory_boosts),
        "ghost_examples":     ghost_alerts[:3],
        "memory_boost_examples": memory_boosts[:3],
        "spatial_map":        spatial_map.snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="evaluation/videos/living_room_01.mp4")
    parser.add_argument("--frames", default=90, type=int,
                        help="Frames per pass (default 90)")
    parser.add_argument("--sample-every", default=3, type=int)
    parser.add_argument("--out",    default="evaluation/demo_results.json")
    args = parser.parse_args()

    print("Loading pipeline components...")
    from app.perception.detector import Detector
    from app.perception.depth import DepthEstimator
    from app.perception.tracker import ObjectTracker
    from app.memory_modules.long_term import LongTermMemory
    from app.memory_modules.spatial_map import SpatialMap
    from app.risk.engine import RiskEngine

    long_term = LongTermMemory()
    memory_count_before = long_term.count()
    print(f"  Memory entries before demo: {memory_count_before}")

    components = {
        "detector":    Detector(),
        "depth_est":   DepthEstimator(),
        "tracker":     ObjectTracker(frame_rate=30),
        "long_term":   long_term,
        "risk_engine": RiskEngine(long_term=long_term),
        "spatial_map": SpatialMap(),
    }

    # Pass 1 — build memory
    pass1 = run_pass(
        video_path=args.video,
        max_frames=args.frames,
        sample_every=args.sample_every,
        components=components,
        write_to_memory=True,
        label="Pass 1 — building memory",
    )
    memory_count_after = long_term.count()
    print(f"  Memory entries after pass 1: {memory_count_after} "
          f"(+{memory_count_after - memory_count_before} new)")

    # Reset tracker between passes (new session, same room)
    components["tracker"].reset()
    components["spatial_map"] = SpatialMap()

    # Pass 2 — use memory
    pass2 = run_pass(
        video_path=args.video,
        max_frames=args.frames,
        sample_every=args.sample_every,
        components=components,
        write_to_memory=False,
        label="Pass 2 — using memory",
    )

    result = {
        "video":               args.video,
        "frames_per_pass":     args.frames,
        "generated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "memory_entries_written": memory_count_after - memory_count_before,
        "pass_1":              pass1,
        "pass_2":              pass2,
        "key_finding": {
            "ghost_alerts_pass1":      pass1["ghost_alerts"],
            "ghost_alerts_pass2":      pass2["ghost_alerts"],
            "memory_boosts_pass1":     pass1["memory_boosts"],
            "memory_boosts_pass2":     pass2["memory_boosts"],
            "high_risk_alerts_pass1":  pass1["high_risk_alerts"],
            "high_risk_alerts_pass2":  pass2["high_risk_alerts"],
            "summary": (
                f"Pass 2 fired {pass2['ghost_alerts']} ghost alert(s) "
                f"and {pass2['memory_boosts']} memory-boosted risk score(s) "
                f"that Pass 1 (no memory) could not produce."
            ),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n── Demo results ─────────────────────────────────────────")
    print(f"  Memory entries written:    {result['memory_entries_written']}")
    print(f"  High-risk alerts pass 1:   {pass1['high_risk_alerts']}")
    print(f"  High-risk alerts pass 2:   {pass2['high_risk_alerts']}")
    print(f"  Memory boosts    pass 1:   {pass1['memory_boosts']}")
    print(f"  Memory boosts    pass 2:   {pass2['memory_boosts']}")
    print(f"  Ghost alerts     pass 1:   {pass1['ghost_alerts']}  (expected: 0)")
    print(f"  Ghost alerts     pass 2:   {pass2['ghost_alerts']}  (expected: >0)")
    print(f"\n  {result['key_finding']['summary']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
