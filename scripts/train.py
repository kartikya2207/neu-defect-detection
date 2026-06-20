#!/usr/bin/env python3
"""
Train YOLOv8 on the NEU defect dataset and report mAP.

Usage:
    python scripts/train.py --data datasets/neu/data.yaml --epochs 50
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/neu/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained checkpoint (transfer learning)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="neu_yolov8n")
    args = ap.parse_args()

    model = YOLO(args.model)                       # starts from pretrained weights
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    metrics = model.val()                          # evaluate best weights on val split
    summary = {
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
    except Exception as e:  # API can vary slightly across ultralytics versions
        summary["per_class_mAP50"] = f"unavailable ({e})"

    Path("results_metrics.json").write_text(json.dumps(summary, indent=2))
    print("\n=== RESULTS (paste these into REPORT.md) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
