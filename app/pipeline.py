#!/usr/bin/env python3
"""
Week 7 - end-to-end inference pipeline: image -> YOLO -> maintenance report.

Runs the trained detector on an image, then generates a report. If an OpenAI
API key is present (env var OPENAI_API_KEY), a Vision-Language Model writes a
rich report from the image AND the detections together (genuine multimodal
fusion - the image is never reduced to text first). Otherwise a structured
template report is built from the detections alone, so the pipeline always runs.

Usage:
    python app/pipeline.py --model runs/detect/neu_yolov8n/weights/best.pt \
        --image path/to/steel.jpg
"""
import argparse
import base64
import json
import os
from pathlib import Path

from ultralytics import YOLO

# Rough engineering priority per defect type (tune with domain knowledge).
CLASS_SEVERITY = {
    "crazing": "Medium", "inclusion": "High", "patches": "Low",
    "pitted_surface": "High", "rolled-in_scale": "Medium", "scratches": "Low",
}
_SEV_RANK = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}


def detect(model_path, image_path, conf=0.25):
    """Run YOLO on one image; return (list-of-detections, raw Results)."""
    model = YOLO(model_path)
    result = model(image_path, conf=conf)[0]
    dets = []
    for b in result.boxes:
        cls = result.names[int(b.cls)]
        dets.append({
            "class": cls,
            "confidence": round(float(b.conf), 3),
            "box_xyxy": [round(float(x), 1) for x in b.xyxy[0].tolist()],
            "severity": CLASS_SEVERITY.get(cls, "Unknown"),
        })
    return dets, result


def template_report(dets):
    """Structured report from detections alone (no VLM needed)."""
    if not dets:
        return "No surface defects detected. Component passes visual QA."
    counts = {}
    for d in dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    lines = ["AUTOMATED SURFACE INSPECTION REPORT", "=" * 36, "",
             f"Total defects detected: {len(dets)}", "", "Breakdown by type:"]
    for c, n in sorted(counts.items()):
        lines.append(f"  - {c}: {n}  (severity: {CLASS_SEVERITY.get(c, 'Unknown')})")
    top = max(dets, key=lambda d: _SEV_RANK.get(d["severity"], 0))
    lines += ["",
              f"Highest-priority finding: {top['class']} (severity {top['severity']}).",
              "Recommended action: flag for manual review and schedule maintenance."]
    return "\n".join(lines)


def vlm_report(image_path, dets, api_key):
    """Rich report from a Vision-Language Model reading image + detections."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    prompt = (
        "You are a steel-surface QA inspector. Given this image of a hot-rolled "
        "steel surface and the automated detections below, write a concise "
        "maintenance report: for each defect describe its appearance and severity, "
        "then recommend an action. Keep it factual.\n\nDetections: " + json.dumps(dets)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        max_tokens=400,
    )
    return resp.choices[0].message.content


def run(model_path, image_path, out_dir="pipeline_out"):
    dets, result = detect(model_path, image_path)
    Path(out_dir).mkdir(exist_ok=True)
    result.save(filename=str(Path(out_dir) / "detected.jpg"))   # image with boxes drawn
    key = os.environ.get("OPENAI_API_KEY")
    report = vlm_report(image_path, dets, key) if key else template_report(dets)
    (Path(out_dir) / "report.txt").write_text(report)
    print(report)
    print(f"\n[saved overlay + report to {out_dir}/]")
    return dets, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/neu_yolov8n/weights/best.pt")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    run(args.model, args.image)
