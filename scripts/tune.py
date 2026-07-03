#!/usr/bin/env python3
"""
Week 5 - hyperparameter tuning aimed at lifting the weak 'crazing' class.

Baseline was: yolov8n, imgsz 320, 50 epochs  ->  mAP@0.5 0.763, crazing 0.365.

Levers changed here, and why they should help the faint low-contrast class:
  - larger model (yolov8s) -> more capacity for subtle defects
  - higher resolution (imgsz 640) -> crazing is faint hairline cracking, so more
    pixels give the network far more to work with (this is usually the biggest win)
  - longer training + tuned augmentation (flips both axes, brightness/HSV jitter,
    mosaic/mixup) -> more effective data variety

Usage (Colab GPU):
    python scripts/tune.py --data datasets/neu/data.yaml --model yolov8s.pt \
        --imgsz 640 --epochs 100

Then compare the per-class 'crazing' number in results_tuned.json against 0.365.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/neu/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="try yolov8s.pt (bigger) or yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="neu_tuned")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=30,                          # early-stop if val stops improving
        # --- tuned augmentation (helps faint / low-contrast defects) ---
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,    # colour / brightness jitter
        degrees=5.0,                          # small rotations
        translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.5,               # steel defects have no fixed orientation
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
    )

    metrics = model.val()
    summary = {
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    try:
        summary["per_class_mAP50"] = {
            metrics.names[int(i)]: round(float(v), 4)
            for i, v in zip(metrics.box.ap_class_index, metrics.box.ap50)
        }
    except Exception as e:  # attribute names vary slightly across versions
        summary["per_class_mAP50"] = f"unavailable ({e})"

    Path("results_tuned.json").write_text(json.dumps(summary, indent=2))
    print("\n=== TUNED RESULTS (compare 'crazing' against baseline 0.365) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
