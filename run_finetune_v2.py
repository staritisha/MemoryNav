"""
MemoryNav — YOLOv8 Fine-Tuning v2
run_finetune_v2.py

Improvements over v1:
  1. MPS device (Apple Silicon) — v1 ran on CPU (5945s). MPS is 3-5x faster.
  2. Stronger augmentation pipeline — mosaic=1.0, mixup=0.15, copy-paste=0.1,
     aggressive HSV + low-light simulation, horizontal + vertical flips.
     Effectively multiplies 454 training images to ~1800 augmented samples per epoch.
  3. More epochs (50 vs 30) with better patience (15 vs 8) and cosine LR schedule.
  4. Smaller batch size (8) tuned for MPS memory.
  5. Warmup epochs (3) to stabilise early training.
  6. Close mosaic at epoch 40 (last 10 epochs on clean crops for fine detail).

Target improvements:
  - Better generalisation to low-light indoor scenes (hallways, bedrooms)
  - Better small-object detection (partial occlusions, chair legs)
  - More stable bbox regression via longer warmup

Classes: Chair (0), Sofa (1), Table (2)  — same as v1, same data.yaml
"""

import json
import pathlib
import shutil
import time

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_YAML   = "./furniture-ngpea/data.yaml"
BASE_W      = "./backend/models/yolov8n.pt"
OUT_W       = "./backend/models/finetuned_yolov8n.pt"
METRICS_OUT = "./evaluation/finetune_metrics.json"

# ── Load base model ────────────────────────────────────────────────────────────
print("Loading base YOLOv8n weights …")
model = YOLO(BASE_W)

# ── Baseline mAP on validation set ────────────────────────────────────────────
print("\nMeasuring baseline (pretrained, no fine-tuning) …")
m_before = model.val(data=DATA_YAML, verbose=False)
before = {
    "mAP50":    round(float(m_before.box.map50), 4),
    "mAP50_95": round(float(m_before.box.map),   4),
}
print(f"  Baseline  mAP50={before['mAP50']:.3f}  mAP50-95={before['mAP50_95']:.3f}")

# ── Fine-tune ──────────────────────────────────────────────────────────────────
print("\nFine-tuning (50 epochs, MPS, heavy augmentation) …")
t0 = time.perf_counter()

model.train(
    data=DATA_YAML,

    # ── Core training ──────────────────────────────────────────────────────────
    epochs=50,
    imgsz=640,
    batch=8,               # safe for MPS 8GB unified memory
    device="mps",          # Apple Silicon — 3-5x faster than CPU
    workers=4,

    # ── Learning rate + schedule ───────────────────────────────────────────────
    lr0=0.01,              # initial LR
    lrf=0.01,              # final LR = lr0 * lrf
    cos_lr=True,           # cosine annealing — smoother convergence
    warmup_epochs=3,       # stabilise gradients before full LR
    warmup_momentum=0.8,

    # ── Early stopping ─────────────────────────────────────────────────────────
    patience=15,           # wait 15 epochs before stopping (was 8)

    # ── Augmentation — the main improvement ───────────────────────────────────
    # Mosaic: composites 4 images into one — best for small-object detection
    mosaic=1.0,
    # Close mosaic for final 10 epochs so model refines on clean crops
    close_mosaic=10,

    # MixUp: blend two images — improves generalisation to cluttered scenes
    mixup=0.15,

    # Copy-paste: paste objects from other images — key for occlusion robustness
    copy_paste=0.1,

    # Colour/lighting — simulates hallway dim light, glare, shadows
    hsv_h=0.015,           # hue shift ±1.5%
    hsv_s=0.7,             # saturation ±70%
    hsv_v=0.4,             # brightness ±40% (simulates dim indoor lighting)

    # Geometric
    degrees=5.0,           # small rotation — cameras aren't always level
    translate=0.1,         # ±10% translation
    scale=0.5,             # ±50% scale — handles near/far objects
    shear=2.0,             # mild shear
    perspective=0.0005,    # subtle perspective warp (monocular camera)
    fliplr=0.5,            # horizontal flip — rooms are symmetric
    flipud=0.0,            # no vertical flip — gravity-aware scenes

    # ── Misc ───────────────────────────────────────────────────────────────────
    # Save the best checkpoint (by mAP50) and last checkpoint
    save=True,
    exist_ok=True,         # overwrite previous run dir cleanly
    project="runs/detect",
    name="train_v2",
    verbose=False,
)

elapsed = time.perf_counter() - t0
print(f"\nTraining complete in {elapsed / 60:.1f} min ({elapsed:.0f}s)")

# ── Copy best weights ──────────────────────────────────────────────────────────
best = pathlib.Path("runs/detect/train_v2/weights/best.pt")
if not best.exists():
    # Fallback: find the latest train run
    candidates = sorted(pathlib.Path("runs/detect").glob("train*/weights/best.pt"))
    best = candidates[-1] if candidates else best

if best.exists():
    pathlib.Path(OUT_W).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OUT_W)
    print(f"Best weights → {OUT_W}")
else:
    print("WARNING: best.pt not found — check runs/detect/")

# ── Evaluate fine-tuned model ──────────────────────────────────────────────────
print("\nEvaluating fine-tuned model …")
model2 = YOLO(OUT_W)
m_after = model2.val(data=DATA_YAML, verbose=False)
after = {
    "mAP50":    round(float(m_after.box.map50), 4),
    "mAP50_95": round(float(m_after.box.map),   4),
}
print(f"  Fine-tuned mAP50={after['mAP50']:.3f}  mAP50-95={after['mAP50_95']:.3f}")

# Per-class breakdown
print("\nPer-class results:")
class_names = ["Chair", "Sofa", "Table"]
per_class = {}
for i, name in enumerate(class_names):
    try:
        ap50 = float(m_after.box.ap50[i])
        per_class[name] = round(ap50, 4)
        print(f"  {name:8s}  AP50={ap50:.3f}")
    except (IndexError, AttributeError):
        per_class[name] = None

# ── Save metrics ───────────────────────────────────────────────────────────────
result = {
    "version": "v2",
    "dataset": "LibreYOLO/furniture-ngpea (CC-BY-4.0)",
    "classes": class_names,
    "epochs": 50,
    "device": "mps",
    "augmentation": {
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "hsv_v": 0.4,
        "close_mosaic": 10,
    },
    "train_time_s": round(elapsed, 1),
    "train_time_min": round(elapsed / 60, 1),
    "baseline_pretrained": before,
    "finetuned_v2": after,
    "per_class_ap50": per_class,
    "mAP50_improvement_vs_baseline": round(after["mAP50"] - before["mAP50"], 4),
    "mAP50_improvement_vs_v1": round(after["mAP50"] - 0.9604, 4),
}

pathlib.Path(METRICS_OUT).parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_OUT, "w") as f:
    json.dump(result, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"""
{'='*50}
  MemoryNav Fine-Tune v2 — Results
{'='*50}
  Baseline  mAP50:  {before['mAP50']:.3f}
  v1        mAP50:  0.960  (30 epochs, CPU)
  v2        mAP50:  {after['mAP50']:.3f}  (50 epochs, MPS, augmented)
  
  Δ vs baseline: {result['mAP50_improvement_vs_baseline']:+.4f}
  Δ vs v1:       {result['mAP50_improvement_vs_v1']:+.4f}
  
  Train time:  {result['train_time_min']:.1f} min
  Saved:       {OUT_W}
  Metrics:     {METRICS_OUT}
{'='*50}
""")
