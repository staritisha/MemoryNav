#!/usr/bin/env python3
"""
evaluation/run_ablation.py

Ablation study runner for MemoryNav (design doc Section 6.2).

Runs three system configurations against every video in evaluation/videos/
and writes evaluation/results.json with real, measured numbers — never
estimated ones. If a video has no ground-truth annotation file, it is
skipped and reported as skipped, not silently scored.

  Config A    — YOLO only: detection, no depth, no memory, no suppression.
  Config B    — Detection + depth + risk: distance-aware, no memory,
                no suppression.
  Full system — Detection + depth + risk + memory + alert suppression.

-----------------------------------------------------------------------------
IMPORTANT — wiring this to your real backend
-----------------------------------------------------------------------------
This script tries to import your actual backend components first:

    backend.app.perception.detector      (YOLOv8 wrapper)
    backend.app.perception.frame_quality  (blur/brightness gate)
    backend.app.perception.depth          (distance estimation)
    backend.app.memory.retriever          (ChromaDB home-context lookup)

If any of these can't be imported (wrong path, not built yet, different
class name), this script falls back to a clearly-labeled STUB for that
component so the harness still runs end-to-end. Stub results are NOT
research-grade — check the printed "Component status" block at startup
and fix the import paths in `_load_components()` below to match your
actual module/class names before trusting the numbers.
-----------------------------------------------------------------------------

Ground-truth annotation format
-----------------------------------------------------------------------------
For a video evaluation/videos/kitchen_01.mp4, place an annotation file at
evaluation/videos/kitchen_01.json:

    {
      "events": [
        {"obstacle": "chair", "critical_time_s": 4.2},
        {"obstacle": "rug edge", "critical_time_s": 11.8}
      ]
    }

`critical_time_s` is the last moment a spoken warning would still be
useful — just before the obstacle becomes unavoidable. A configuration
"succeeds" on an event if it raises a HIGH-risk alert at or before that
timestamp. This matches the doc's definition: "navigation success rate
(user warned before reaching obstacle)".

A video can also have an empty `events` list — it then only contributes
to the false-alert-rate measurement (any HIGH alert in such a video is
a false positive by definition).

Usage
-----------------------------------------------------------------------------
    python evaluation/run_ablation.py
    python evaluation/run_ablation.py --videos-dir evaluation/videos \\
        --out evaluation/results.json --conf 0.4 --suppression-window 4.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# --- Make the backend package importable when run from the repo root. ---
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Component loading — real backend modules first, stub fallback otherwise.
# =============================================================================

class ComponentStatus:
    """Tracks which pieces of the pipeline are real vs. stubbed, for the
    honesty banner printed at startup."""

    def __init__(self) -> None:
        self.detector = "stub"
        self.frame_quality = "stub"
        self.depth = "stub"
        self.memory = "stub"

    def banner(self) -> str:
        lines = ["Component status (real backend vs. stub fallback):"]
        for name in ("detector", "frame_quality", "depth", "memory"):
            status = getattr(self, name)
            marker = "REAL" if status == "real" else "STUB"
            lines.append(f"  [{marker:4}] {name}")
        if any(getattr(self, n) == "stub" for n in ("detector", "frame_quality", "depth", "memory")):
            lines.append(
                "  -> STUB components mean these numbers are a harness smoke test,"
            )
            lines.append(
                "     not a credible research result. Fix import paths in"
            )
            lines.append("     _load_components() to wire in your real modules.")
        return "\n".join(lines)


STATUS = ComponentStatus()


def _load_detector():
    """Returns a callable: detect(frame_bgr) -> list[dict] with keys
    label, confidence, box (x, y, w, h normalized 0..1)."""
    try:
        from backend.app.perception.detector import Detector  # type: ignore

        detector = Detector()
        STATUS.detector = "real"

        def run(frame: np.ndarray):
            raw = detector.detect(frame)
            return [_normalize_detection(d, frame.shape) for d in raw]

        return run
    except Exception as exc:  # noqa: BLE001
        print(f"[detector] falling back to stub YOLO wrapper ({exc})", file=sys.stderr)
        return _stub_detector()


def _normalize_detection(raw: dict, frame_shape) -> dict:
    """Best-effort adapter for a few common detection dict shapes, so this
    script doesn't break on minor schema differences in your detector."""
    h, w = frame_shape[0], frame_shape[1]
    label = raw.get("label") or raw.get("class_name") or raw.get("name", "object")
    confidence = float(raw.get("confidence", raw.get("conf", 0.0)))

    if "box" in raw and isinstance(raw["box"], dict):
        box = raw["box"]
        x, y, bw, bh = box["x"], box["y"], box["width"], box["height"]
    elif "xyxy" in raw:
        x1, y1, x2, y2 = raw["xyxy"]
        x, y, bw, bh = x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h
    elif "bbox" in raw:
        x1, y1, x2, y2 = raw["bbox"]
        x, y, bw, bh = x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h
    else:
        x, y, bw, bh = 0.4, 0.4, 0.2, 0.2  # last-resort fallback box

    return {"label": label, "confidence": confidence, "box": (x, y, bw, bh)}


def _stub_detector():
    """Minimal YOLOv8-nano detector using ultralytics directly, so the
    harness runs even without your backend.app.perception.detector module."""
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
    except Exception as exc:  # noqa: BLE001
        print(f"[detector] ultralytics unavailable either ({exc}); "
              "no detections will be produced.", file=sys.stderr)

        def empty_run(_frame: np.ndarray):
            return []

        return empty_run

    def run(frame: np.ndarray):
        h, w = frame.shape[:2]
        results = model.predict(frame, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                out.append(
                    {
                        "label": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "box": (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h),
                    }
                )
        return out

    return run


def _load_frame_quality():
    """Returns a callable: is_usable(frame_bgr) -> bool."""
    try:
        from backend.app.perception.frame_quality import check_frame_quality  # type: ignore

        STATUS.frame_quality = "real"
        return check_frame_quality
    except Exception as exc:  # noqa: BLE001
        print(f"[frame_quality] falling back to stub blur/brightness gate ({exc})", file=sys.stderr)
        return _stub_frame_quality


def _stub_frame_quality(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    return blur_score > 40.0 and 25.0 < brightness < 235.0


def _load_depth():
    """Returns a callable: estimate(frame_bgr, box) -> meters."""
    try:
        from backend.app.perception.depth import DepthEstimator  # type: ignore

        estimator = DepthEstimator()
        STATUS.depth = "real"

        def run(frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
            return float(estimator.estimate(frame, box))

        return run
    except Exception as exc:  # noqa: BLE001
        print(f"[depth] falling back to stub bbox-size heuristic ({exc})", file=sys.stderr)
        return _stub_depth


def _stub_depth(_frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Monocular-depth stand-in: larger normalized bbox area => closer.
    This is a crude proxy, NOT a substitute for Depth-Anything. Only used
    if backend.app.perception.depth can't be imported."""
    _, _, bw, bh = box
    area_ratio = max(bw * bh, 1e-4)
    return float(np.clip(1.2 / np.sqrt(area_ratio), 0.3, 8.0))


def _load_memory():
    """Returns a callable: relevance(label) -> (context_text, score 0..1)."""
    try:
        from backend.app.memory.retriever import MemoryRetriever  # type: ignore

        retriever = MemoryRetriever()
        STATUS.memory = "real"

        def run(label: str):
            result = retriever.query(label, top_k=1)
            if not result:
                return None, 1.0
            text, score = result[0]
            # Map similarity into a 1.0-2.0x risk-context multiplier — a
            # strongly relevant memory (e.g. "known hazard near the stove")
            # should raise risk, not just inform it.
            return text, 1.0 + max(0.0, min(score, 1.0))

        return run
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] falling back to stub (no context boost) ({exc})", file=sys.stderr)
        return lambda label: (None, 1.0)


# =============================================================================
# Risk scoring per configuration
# =============================================================================

RiskLevel = str  # "LOW" | "MEDIUM" | "HIGH"

MEDIUM_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65


def _level_from_score(score: float) -> RiskLevel:
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def score_config_a(detection: dict, frame_shape) -> float:
    """Config A: YOLO only. No depth available, so risk is approximated
    purely from how large/central the detection is in frame — the only
    signal a detection-only system has access to."""
    _, _, bw, bh = detection["box"]
    x, y, bw2, bh2 = detection["box"]
    area = bw * bh
    cx, cy = x + bw2 / 2, y + bh2 / 2
    centrality = 1.0 - min(((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5, 1.0)
    return float(np.clip(area * 2.5 * centrality, 0.0, 1.0))


def score_config_b(distance_m: float, motion_factor: float) -> float:
    """Config B: distance-aware risk = (1 / distance) x motion, no memory
    context. Mirrors the production risk formula minus the context term."""
    raw = (1.0 / max(distance_m, 0.15)) * motion_factor
    return float(np.clip(raw / 4.0, 0.0, 1.0))  # /4.0 keeps scale comparable to A/Full


def score_full(distance_m: float, motion_factor: float, context_multiplier: float) -> float:
    """Full system: (1 / distance) x motion x context."""
    raw = (1.0 / max(distance_m, 0.15)) * motion_factor * context_multiplier
    return float(np.clip(raw / 4.0, 0.0, 1.0))


# =============================================================================
# Ground truth + per-video evaluation
# =============================================================================

@dataclass
class ObstacleEvent:
    obstacle: str
    critical_time_s: float
    warned_at_s: Optional[float] = None

    @property
    def warned(self) -> bool:
        return self.warned_at_s is not None and self.warned_at_s <= self.critical_time_s


@dataclass
class VideoResult:
    video: str
    events: list[ObstacleEvent] = field(default_factory=list)
    false_alerts: int = 0
    frame_count: int = 0
    total_latency_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_s / self.frame_count) * 1000 if self.frame_count else 0.0


def _load_annotation(video_path: Path) -> Optional[dict]:
    ann_path = video_path.with_suffix(".json")
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        return json.load(f)


def _track_motion(prev_centroids: dict, label: str, box) -> tuple[float, dict]:
    """Greedy nearest-centroid tracking by label to estimate normalized
    frame-to-frame displacement as a cheap motion proxy (no optical flow)."""
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    prev = prev_centroids.get(label)
    prev_centroids[label] = (cx, cy)
    if prev is None:
        return 1.0, prev_centroids
    dx, dy = cx - prev[0], cy - prev[1]
    displacement = (dx ** 2 + dy ** 2) ** 0.5
    motion_factor = 1.0 + min(displacement * 5.0, 1.5)  # cap influence
    return motion_factor, prev_centroids


def run_config(
    config: str,
    video_path: Path,
    annotation: dict,
    components: dict,
    suppression_window_s: float,
) -> VideoResult:
    detect, is_usable, estimate_depth, query_memory = (
        components["detector"],
        components["frame_quality"],
        components["depth"],
        components["memory"],
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    events = [ObstacleEvent(e["obstacle"], e["critical_time_s"]) for e in annotation.get("events", [])]
    result = VideoResult(video=video_path.name, events=events)

    prev_centroids: dict = {}
    last_alert_time: dict[str, float] = {}  # per-label suppression tracking

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_s = frame_idx / fps

        t0 = time.perf_counter()

        if not is_usable(frame):
            continue

        detections = detect(frame)
        result.frame_count += 1

        for det in detections:
            label = det["label"]
            box = det["box"]

            if config == "baseline_a":
                score = score_config_a(det, frame.shape)
            else:
                distance_m = estimate_depth(frame, box)
                motion_factor, prev_centroids = _track_motion(prev_centroids, label, box)

                if config == "baseline_b":
                    score = score_config_b(distance_m, motion_factor)
                else:  # full_system
                    _, context_multiplier = query_memory(label)
                    score = score_full(distance_m, motion_factor, context_multiplier)

            level = _level_from_score(score)
            if level != "HIGH":
                continue

            if config == "full_system":
                last = last_alert_time.get(label)
                if last is not None and (t_s - last) < suppression_window_s:
                    continue  # suppressed — would not have spoken
                last_alert_time[label] = t_s

            # Record against the nearest still-unwarned matching event,
            # else count as a false alert.
            matched = False
            for event in events:
                if event.obstacle.lower() in label.lower() or label.lower() in event.obstacle.lower():
                    if event.warned_at_s is None:
                        event.warned_at_s = t_s
                    matched = True
                    break
            if not matched:
                result.false_alerts += 1

        result.total_latency_s += time.perf_counter() - t0

    cap.release()
    return result


# =============================================================================
# Orchestration
# =============================================================================

CONFIG_LABELS = {
    "baseline_a": "Baseline A: YOLO Only",
    "baseline_b": "Baseline B: YOLO + Depth + Risk",
    "full_system": "Full System: + Memory + Suppression",
}


def aggregate(video_results: list[VideoResult]) -> dict:
    total_events = sum(len(v.events) for v in video_results)
    warned_events = sum(sum(1 for e in v.events if e.warned) for v in video_results)
    total_false_alerts = sum(v.false_alerts for v in video_results)
    total_frames = sum(v.frame_count for v in video_results)
    total_latency_s = sum(v.total_latency_s for v in video_results)

    success_rate = (warned_events / total_events) if total_events else None
    false_alert_rate = (
        total_false_alerts / max(total_events, 1) if (total_events or total_false_alerts) else None
    )
    avg_latency_ms = (total_latency_s / total_frames * 1000) if total_frames else None

    return {
        "navigation_success_rate": success_rate,
        "events_total": total_events,
        "events_warned": warned_events,
        "false_alerts_total": total_false_alerts,
        "false_alert_rate": false_alert_rate,
        "avg_latency_ms": avg_latency_ms,
        "frames_processed": total_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default="evaluation/videos", type=Path)
    parser.add_argument("--out", default="evaluation/results.json", type=Path)
    parser.add_argument("--suppression-window", default=4.0, type=float)
    args = parser.parse_args()

    components = {
        "detector": _load_detector(),
        "frame_quality": _load_frame_quality(),
        "depth": _load_depth(),
        "memory": _load_memory(),
    }
    print(STATUS.banner())
    print()

    videos = sorted(args.videos_dir.glob("*.mp4")) + sorted(args.videos_dir.glob("*.mov"))
    if not videos:
        print(f"No videos found in {args.videos_dir}. Nothing to run.", file=sys.stderr)
        sys.exit(1)

    skipped = []
    usable_videos = []
    for v in videos:
        ann = _load_annotation(v)
        if ann is None:
            skipped.append(str(v))
        else:
            usable_videos.append((v, ann))

    if skipped:
        print(f"Skipping {len(skipped)} video(s) with no annotation file:")
        for s in skipped:
            print(f"  - {s}")
        print()

    if not usable_videos:
        print("No annotated videos to evaluate. Add a .json sidecar per video — see the "
              "docstring at the top of this file for the schema.", file=sys.stderr)
        sys.exit(1)

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "videos_dir": str(args.videos_dir),
        "suppression_window_s": args.suppression_window,
        "component_status": {
            "detector": STATUS.detector,
            "frame_quality": STATUS.frame_quality,
            "depth": STATUS.depth,
            "memory": STATUS.memory,
        },
        "configs": {},
        "skipped_videos": skipped,
    }

    for config_key, label in CONFIG_LABELS.items():
        print(f"Running {label} on {len(usable_videos)} video(s)...")
        per_video = []
        for video_path, annotation in usable_videos:
            result = run_config(
                config_key, video_path, annotation, components, args.suppression_window
            )
            per_video.append(result)
            n_warned = sum(1 for e in result.events if e.warned)
            print(
                f"  {video_path.name}: {n_warned}/{len(result.events)} events warned, "
                f"{result.false_alerts} false alert(s), {result.avg_latency_ms:.1f}ms/frame"
            )

        output["configs"][config_key] = {
            "label": label,
            "per_video": [
                {
                    "video": r.video,
                    "events": [
                        {
                            "obstacle": e.obstacle,
                            "critical_time_s": e.critical_time_s,
                            "warned_at_s": e.warned_at_s,
                            "warned": e.warned,
                        }
                        for e in r.events
                    ],
                    "false_alerts": r.false_alerts,
                    "avg_latency_ms": r.avg_latency_ms,
                }
                for r in per_video
            ],
            "aggregate": aggregate(per_video),
        }
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {args.out}")
    print()
    print("Summary (navigation success rate):")
    for config_key, label in CONFIG_LABELS.items():
        rate = output["configs"][config_key]["aggregate"]["navigation_success_rate"]
        rate_str = f"{rate * 100:.1f}%" if rate is not None else "no annotated events"
        print(f"  {label}: {rate_str}")


if __name__ == "__main__":
    main()