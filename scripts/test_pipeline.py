#!/usr/bin/env python3
"""
Week 8 - quick smoke test: run the pipeline on a few validation images and
check it returns sane detections. Not a correctness test of accuracy (that's
what mAP is for) - just verifies the end-to-end plumbing works on real files.

Usage:
    python scripts/test_pipeline.py
"""
import glob
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from pipeline import detect, template_report  # noqa: E402

MODEL = "runs/detect/neu_yolov8n/weights/best.pt"


def main():
    imgs = glob.glob("datasets/neu/images/val/*.jpg")
    assert imgs, "No val images found - run scripts/voc_to_yolo.py first."
    assert Path(MODEL).exists(), f"Model not found at {MODEL} - train first."

    for img in random.sample(imgs, min(3, len(imgs))):
        dets, _ = detect(MODEL, img)
        assert isinstance(dets, list)
        report = template_report(dets)
        assert isinstance(report, str) and len(report) > 0
        print(f"{Path(img).name:<28} -> {len(dets)} detection(s)")

    print("\nSmoke test passed: pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
