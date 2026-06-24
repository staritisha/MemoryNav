"""
MemoryNav — Evaluation Metrics
evaluation/metrics.py

Phase 8 (Evaluation + Ship): every metric function needed for the
ablation study. Functions are pure Python — no ML framework imports —
so they can run on any machine and are trivially testable.

Metrics implemented
-------------------
    mAP@50          Object detection quality
    MAE             Depth estimation accuracy (metres)
    latency_ms      Detection-to-speech wall-clock time
    miss_rate       Obstacles reached without a warning (safety metric)
    false_alert_rate  Unnecessary warnings / total warnings (usability)
    redundancy_reduction  Alerts suppressed vs baseline (suppression value)
    character_accuracy    OCR correctness

Usage
-----
    from evaluation.metrics import (
        compute_map50,
        compute_mae,
        LatencyTimer,
        compute_miss_rate,
        compute_false_alert_rate,
        compute_redundancy_reduction,
        compute_character_accuracy,
        print_ablation_table,
    )

All functions take plain Python dicts/lists so they work with the
output of run_ablation.py without any extra conversion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. mAP@50 — Object Detection Quality
# ═══════════════════════════════════════════════════════════════════════════

def _iou(box_a: Tuple[float, float, float, float],
         box_b: Tuple[float, float, float, float]) -> float:
    """
    Intersection-over-Union for two axis-aligned bounding boxes.
    Each box: (x1, y1, x2, y2) in pixel coordinates.
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _average_precision(
    detections: List[Dict],   # [{"bbox": ..., "score": float, "matched": bool}]
    n_ground_truth: int,
) -> float:
    """
    Compute Average Precision (AP) for a single class using the 11-point
    interpolation method (VOC 2007). Returns 0.0 if no ground truth exists.

    `detections` must be sorted by descending score before calling.
    """
    if n_ground_truth == 0:
        return 0.0

    tp = 0
    fp = 0
    precisions = []
    recalls    = []

    for det in detections:
        if det["matched"]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_ground_truth)

    # 11-point interpolation
    ap = 0.0
    for thresh in [r / 10 for r in range(11)]:
        p_at_thresh = [p for p, r in zip(precisions, recalls) if r >= thresh]
        ap += max(p_at_thresh, default=0.0) / 11

    return ap


def compute_map50(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.50,
) -> float:
    """
    Compute mean Average Precision at IoU=0.50 (mAP@50).

    Parameters
    ----------
    predictions   : list of dicts, each with keys:
                        "class_id"  : int
                        "bbox"      : (x1, y1, x2, y2)
                        "score"     : float  (YOLO confidence)
    ground_truths : list of dicts, each with keys:
                        "class_id"  : int
                        "bbox"      : (x1, y1, x2, y2)
    iou_threshold : float, default 0.5

    Returns
    -------
    float  mAP@50 in [0, 1].  Returns 0.0 if ground_truths is empty.

    How to call from run_ablation.py
    ---------------------------------
        score = compute_map50(frame_predictions, frame_ground_truths)
    """
    if not ground_truths:
        return 0.0

    # Group by class
    class_ids = set(gt["class_id"] for gt in ground_truths)
    aps: List[float] = []

    for cls_id in class_ids:
        cls_preds = sorted(
            [p for p in predictions if p["class_id"] == cls_id],
            key=lambda x: x["score"],
            reverse=True,
        )
        cls_gts   = [g for g in ground_truths if g["class_id"] == cls_id]
        matched_gt = set()

        annotated: List[Dict] = []
        for pred in cls_preds:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(cls_gts):
                if idx in matched_gt:
                    continue
                iou = _iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            matched = best_iou >= iou_threshold and best_idx >= 0
            if matched:
                matched_gt.add(best_idx)
            annotated.append({"score": pred["score"], "matched": matched})

        aps.append(_average_precision(annotated, len(cls_gts)))

    return sum(aps) / len(aps) if aps else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. MAE — Depth Estimation Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def compute_mae(
    predicted_distances: List[float],
    true_distances: List[float],
) -> float:
    """
    Mean Absolute Error between predicted and tape-measured distances.

    Parameters
    ----------
    predicted_distances : list of floats, metres (from Depth-Anything)
    true_distances      : list of floats, metres (from tape measure)

    Returns
    -------
    float  MAE in metres.  Returns 0.0 if the input lists are empty.

    How to measure true_distances
    ------------------------------
    Place 10 objects at measured distances from the camera.
    Record the distance Depth-Anything returns for each.
    Compare here.

    Example
    -------
        mae = compute_mae(
            predicted_distances=[0.48, 1.02, 2.15, 0.73],
            true_distances      =[0.50, 1.00, 2.00, 0.75],
        )
        # mae ≈ 0.07 metres
    """
    if not predicted_distances or not true_distances:
        return 0.0
    if len(predicted_distances) != len(true_distances):
        raise ValueError(
            f"predicted_distances and true_distances must be the same length "
            f"(got {len(predicted_distances)} vs {len(true_distances)})"
        )
    return sum(abs(p - t) for p, t in zip(predicted_distances, true_distances)) / len(predicted_distances)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Latency Timer — Detection-to-Speech Wall-Clock Time
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyTimer:
    """
    Measures wall-clock time from detection to voice output.
    Target: sub-200ms (from architecture doc, Section 3.2 Module 5).

    Usage
    -----
        timer = LatencyTimer()

        # In your pipeline loop:
        timer.start()
        # ... run detector, risk engine, alert manager, TTS ...
        timer.stop()

        print(timer.summary())
        # LatencyTimer: n=47  mean=143ms  p50=139ms  p95=198ms  max=312ms
    """
    _samples: List[float] = field(default_factory=list, repr=False)
    _start_time: Optional[float] = field(default=None, repr=False)

    TARGET_MS: float = field(default=200.0, init=False, repr=False)

    def start(self) -> None:
        """Call immediately before the detection pipeline runs."""
        self._start_time = time.monotonic()

    def stop(self) -> float:
        """
        Call immediately after TTS speaks (or queues) the alert.
        Returns the elapsed time in milliseconds.
        """
        if self._start_time is None:
            raise RuntimeError("LatencyTimer.stop() called before start()")
        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
        self._samples.append(elapsed_ms)
        self._start_time = None
        return elapsed_ms

    def record(self, latency_ms: float) -> None:
        """Record a pre-computed latency value (e.g. from logs)."""
        self._samples.append(latency_ms)

    @property
    def n(self) -> int:
        return len(self._samples)

    @property
    def mean_ms(self) -> float:
        return sum(self._samples) / self.n if self.n else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._samples) if self._samples else 0.0

    @property
    def min_ms(self) -> float:
        return min(self._samples) if self._samples else 0.0

    def percentile(self, p: float) -> float:
        """
        p-th percentile of recorded latencies.
        p=50 → median, p=95 → p95, etc.
        """
        if not self._samples:
            return 0.0
        sorted_s = sorted(self._samples)
        idx = int(len(sorted_s) * p / 100)
        idx = min(idx, len(sorted_s) - 1)
        return sorted_s[idx]

    @property
    def pct_under_target(self) -> float:
        """Percentage of measurements that met the sub-200ms target."""
        if not self._samples:
            return 0.0
        under = sum(1 for s in self._samples if s < self.TARGET_MS)
        return (under / self.n) * 100

    def summary(self) -> str:
        if not self._samples:
            return "LatencyTimer: no samples recorded"
        return (
            f"LatencyTimer: n={self.n}  "
            f"mean={self.mean_ms:.0f}ms  "
            f"p50={self.percentile(50):.0f}ms  "
            f"p95={self.percentile(95):.0f}ms  "
            f"max={self.max_ms:.0f}ms  "
            f"under_{int(self.TARGET_MS)}ms={self.pct_under_target:.1f}%"
        )

    def to_dict(self) -> Dict:
        return {
            "n":                    self.n,
            "mean_ms":              round(self.mean_ms, 1),
            "median_ms":            round(self.percentile(50), 1),
            "p95_ms":               round(self.percentile(95), 1),
            "max_ms":               round(self.max_ms, 1),
            "min_ms":               round(self.min_ms, 1),
            "pct_under_target":     round(self.pct_under_target, 1),
            "target_ms":            self.TARGET_MS,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Miss Rate — Safety Metric
# ═══════════════════════════════════════════════════════════════════════════

def compute_miss_rate(
    n_obstacles_reached_without_warning: int,
    n_total_obstacles: int,
) -> float:
    """
    Percentage of obstacles that the user reached without receiving
    any warning. This is the primary SAFETY metric.

    Lower is better. The ablation study measures this across three
    system configurations to prove the full system's value.

    Parameters
    ----------
    n_obstacles_reached_without_warning : int
        Number of times the user reached an obstacle (within 0.5m)
        without having received any warning about it.
        Count manually from your test video annotations.

    n_total_obstacles : int
        Total number of obstacle encounters in the test videos.

    Returns
    -------
    float  Miss rate as a percentage [0, 100].

    Example
    -------
        # Baseline A (YOLO only): missed 8 of 30 obstacles
        miss_rate_a = compute_miss_rate(8, 30)   # → 26.67%

        # Full system: missed 2 of 30 obstacles
        miss_rate_full = compute_miss_rate(2, 30) # → 6.67%
    """
    if n_total_obstacles == 0:
        return 0.0
    return (n_obstacles_reached_without_warning / n_total_obstacles) * 100


def compute_navigation_success_rate(
    n_obstacles_reached_without_warning: int,
    n_total_obstacles: int,
) -> float:
    """
    Inverse of miss rate. Percentage of obstacles the user was warned
    about before reaching them. The headline metric for the ablation table.

    Higher is better.
    """
    return 100.0 - compute_miss_rate(
        n_obstacles_reached_without_warning, n_total_obstacles
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. False Alert Rate — Usability Metric
# ═══════════════════════════════════════════════════════════════════════════

def compute_false_alert_rate(
    n_unnecessary_warnings: int,
    n_total_warnings: int,
) -> float:
    """
    Percentage of voice alerts that were unnecessary (obstacle was not
    actually close, or had already been warned about recently).

    High false alert rate → alert fatigue → user starts ignoring the system.
    This is the primary USABILITY metric. WalkVLM (2024) identifies this
    as the main failure mode of existing assistive navigation systems.

    Parameters
    ----------
    n_unnecessary_warnings : int
        Manually annotated count of warnings that were:
          - About an obstacle already > 2m away
          - A duplicate warning within the suppression window
          - About an obstacle that posed no realistic risk
    n_total_warnings : int
        Total voice alerts fired during the test session.

    Returns
    -------
    float  False alert rate as a percentage [0, 100].

    Example
    -------
        # Without suppression: 41 unnecessary out of 53 total
        far_no_suppression = compute_false_alert_rate(41, 53)  # → 77.36%

        # With suppression: 6 unnecessary out of 22 total
        far_with_suppression = compute_false_alert_rate(6, 22)  # → 27.27%
    """
    if n_total_warnings == 0:
        return 0.0
    return (n_unnecessary_warnings / n_total_warnings) * 100


# ═══════════════════════════════════════════════════════════════════════════
# 6. Redundancy Reduction — Alert Suppression Value
# ═══════════════════════════════════════════════════════════════════════════

def compute_redundancy_reduction(
    n_alerts_without_suppression: int,
    n_alerts_with_suppression: int,
) -> float:
    """
    Percentage reduction in total alerts when the temporal suppression
    manager (WalkVLM-inspired) is active. This directly answers the
    interview question: "what did your suppression actually do?"

    Parameters
    ----------
    n_alerts_without_suppression : int
        Total voice alerts fired when running Baseline B (YOLO + Depth,
        no suppression). Measured from your test videos.
    n_alerts_with_suppression : int
        Total voice alerts fired when running the Full System (with
        temporal alert manager active).

    Returns
    -------
    float  Redundancy reduction percentage [0, 100].
           A value of 60% means "the suppression layer eliminated
           60% of alerts that would otherwise have fired."

    Example
    -------
        reduction = compute_redundancy_reduction(
            n_alerts_without_suppression=53,
            n_alerts_with_suppression=22,
        )
        # → 58.49% — mention this in your demo video narration
    """
    if n_alerts_without_suppression == 0:
        return 0.0
    suppressed = n_alerts_without_suppression - n_alerts_with_suppression
    return (suppressed / n_alerts_without_suppression) * 100


# ═══════════════════════════════════════════════════════════════════════════
# 7. Character Accuracy — OCR Quality
# ═══════════════════════════════════════════════════════════════════════════

def compute_character_accuracy(
    predicted_texts: List[str],
    ground_truth_texts: List[str],
) -> float:
    """
    Character-level accuracy across a set of OCR predictions.

    Uses edit distance (Levenshtein) at the character level normalised by
    ground truth length. Reports the average across all test labels.

    Parameters
    ----------
    predicted_texts   : list of strings returned by EasyOCR
    ground_truth_texts: list of strings (manually transcribed labels)

    Returns
    -------
    float  Character accuracy percentage [0, 100].

    How to measure
    --------------
    Print 20 household labels (medicine bottles, switches, signs).
    Run OCR on each. Compare here.

    Example
    -------
        acc = compute_character_accuracy(
            predicted_texts    =["Asprin", "Kitchen", "EXIT"],
            ground_truth_texts =["Aspirin", "Kitchen", "EXIT"],
        )
        # → 93.33%
    """
    if not predicted_texts or not ground_truth_texts:
        return 0.0
    if len(predicted_texts) != len(ground_truth_texts):
        raise ValueError(
            f"predicted_texts and ground_truth_texts must be the same length "
            f"(got {len(predicted_texts)} vs {len(ground_truth_texts)})"
        )

    total_chars = 0
    correct_chars = 0

    for pred, gt in zip(predicted_texts, ground_truth_texts):
        gt_len = len(gt)
        if gt_len == 0:
            continue
        total_chars += gt_len
        # Count matching characters at each position
        edit_dist = _levenshtein(pred, gt)
        correct_chars += max(0, gt_len - edit_dist)

    return (correct_chars / total_chars * 100) if total_chars else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for ch1 in s1:
        curr_row = [prev_row[0] + 1]
        for j, ch2 in enumerate(s2):
            curr_row.append(min(
                prev_row[j + 1] + 1,        # deletion
                curr_row[j] + 1,            # insertion
                prev_row[j] + (ch1 != ch2)  # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Ablation Table Printer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigResult:
    """
    Holds measured results for one ablation configuration.
    All values are REAL MEASURED numbers — never estimates.
    Use None for metrics not yet measured.
    """
    name: str                                   # e.g. "Baseline A: YOLO Only"
    description: str                            # what this config lacks
    navigation_success_rate: Optional[float]    # %  higher is better
    miss_rate: Optional[float]                  # %  lower is better
    false_alert_rate: Optional[float]           # %  lower is better
    mean_latency_ms: Optional[float]            # ms lower is better
    map50: Optional[float]                      # [0,1] higher is better
    n_obstacles_tested: Optional[int] = None
    notes: str = ""


def print_ablation_table(results: List[ConfigResult]) -> None:
    """
    Print a formatted ablation study table to stdout.
    Copy-paste this into your README and evaluation/page.tsx.

    Example output:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ MemoryNav Ablation Study                                             │
    ├────────────────────────┬─────────┬─────────┬──────────┬─────────────┤
    │ Configuration          │ Success │  Miss   │  False   │  Latency    │
    │                        │  Rate   │  Rate   │  Alert   │  Mean (ms)  │
    ├────────────────────────┼─────────┼─────────┼──────────┼─────────────┤
    │ A: YOLO Only           │  TBM    │   TBM   │   TBM    │    TBM      │
    │ B: + Depth + Risk      │  TBM    │   TBM   │   TBM    │    TBM      │
    │ Full: + Memory + Supp  │  TBM    │   TBM   │   TBM    │    TBM      │
    └────────────────────────┴─────────┴─────────┴──────────┴─────────────┘
    TBM = To Be Measured during evaluation.
    """
    def _fmt(val: Optional[float], suffix: str = "%", precision: int = 1) -> str:
        if val is None:
            return "TBM"
        return f"{val:.{precision}f}{suffix}"

    col_w = [28, 9, 9, 10, 13]
    sep   = "─"
    h_sep = "┼"
    l_sep = "├"
    r_sep = "┤"
    tl, tr, bl, br = "┌", "┐", "└", "┘"
    v     = "│"

    def row_sep(left, mid, right):
        return left + (h_sep.join(sep * (w + 2) for w in col_w)) + right

    def data_row(cells):
        parts = []
        for cell, w in zip(cells, col_w):
            parts.append(f" {str(cell):<{w}} ")
        return v + v.join(parts) + v

    print()
    print(tl + "─" * (sum(col_w) + len(col_w) * 3 - 1) + tr)
    title = " MemoryNav Ablation Study"
    print(v + title.ljust(sum(col_w) + len(col_w) * 3 - 1) + v)
    print(row_sep(l_sep, h_sep, r_sep))
    print(data_row(["Configuration", "Success", "Miss", "False Alert", "Latency ms"]))
    print(row_sep(l_sep, h_sep, r_sep))

    for r in results:
        print(data_row([
            r.name[:28],
            _fmt(r.navigation_success_rate),
            _fmt(r.miss_rate),
            _fmt(r.false_alert_rate),
            _fmt(r.mean_latency_ms, suffix="ms", precision=0),
        ]))

    print(row_sep(bl, "┴", br))
    print("  TBM = To Be Measured during evaluation (never estimate).")
    print()


def build_ablation_results(data: dict) -> List[ConfigResult]:
    """
    Build ConfigResult objects from the JSON written by run_ablation.py.

    Parameters
    ----------
    data : dict  — contents of evaluation/results.json

    Returns
    -------
    List[ConfigResult] ready for print_ablation_table()
    """
    results = []
    for cfg in data.get("configurations", []):
        m = cfg.get("metrics", {})
        results.append(ConfigResult(
            name=cfg.get("name", "Unknown"),
            description=cfg.get("description", ""),
            navigation_success_rate=m.get("navigation_success_rate"),
            miss_rate=m.get("miss_rate"),
            false_alert_rate=m.get("false_alert_rate"),
            mean_latency_ms=m.get("mean_latency_ms"),
            map50=m.get("map50"),
            n_obstacles_tested=m.get("n_obstacles_tested"),
            notes=cfg.get("notes", ""),
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 9. Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running metrics smoke test …\n")

    # mAP@50
    preds = [
        {"class_id": 0, "bbox": (10, 10, 50, 50), "score": 0.9},
        {"class_id": 0, "bbox": (60, 60, 90, 90), "score": 0.7},
    ]
    gts = [
        {"class_id": 0, "bbox": (12, 12, 52, 52)},
        {"class_id": 0, "bbox": (65, 65, 95, 95)},
    ]
    map_score = compute_map50(preds, gts)
    print(f"mAP@50:              {map_score:.3f}  (expect ~1.0 for perfect overlap)")

    # MAE
    mae = compute_mae([0.48, 1.02, 2.15, 0.73], [0.50, 1.00, 2.00, 0.75])
    print(f"Depth MAE:           {mae:.3f}m  (expect ~0.07)")

    # Latency timer
    timer = LatencyTimer()
    for ms in [143, 155, 182, 198, 312, 134, 167]:
        timer.record(ms)
    print(f"Latency:             {timer.summary()}")

    # Miss rate
    mr = compute_miss_rate(2, 30)
    print(f"Miss rate:           {mr:.2f}%  (expect 6.67%)")

    sr = compute_navigation_success_rate(2, 30)
    print(f"Success rate:        {sr:.2f}%  (expect 93.33%)")

    # False alert rate
    far = compute_false_alert_rate(6, 22)
    print(f"False alert rate:    {far:.2f}%  (expect 27.27%)")

    # Redundancy reduction
    rr = compute_redundancy_reduction(53, 22)
    print(f"Redundancy reduction:{rr:.2f}%  (expect 58.49%)")

    # Character accuracy
    ca = compute_character_accuracy(
        ["Asprin", "Kitchen", "EXIT"],
        ["Aspirin", "Kitchen", "EXIT"],
    )
    print(f"OCR char accuracy:   {ca:.2f}%  (expect ~93%)")

    # Ablation table
    print()
    placeholder_results = [
        ConfigResult(
            name="A: YOLO Only",
            description="No depth, no memory, no suppression",
            navigation_success_rate=None,
            miss_rate=None,
            false_alert_rate=None,
            mean_latency_ms=None,
            map50=None,
        ),
        ConfigResult(
            name="B: + Depth + Risk",
            description="No memory, no suppression",
            navigation_success_rate=None,
            miss_rate=None,
            false_alert_rate=None,
            mean_latency_ms=None,
            map50=None,
        ),
        ConfigResult(
            name="Full: + Memory + Supp",
            description="Complete MemoryNav system",
            navigation_success_rate=None,
            miss_rate=None,
            false_alert_rate=None,
            mean_latency_ms=None,
            map50=None,
        ),
    ]
    print_ablation_table(placeholder_results)
    print("All smoke tests passed.")