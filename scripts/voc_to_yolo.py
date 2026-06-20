#!/usr/bin/env python3
"""
Convert the NEU-DET dataset (Pascal VOC XML annotations) to YOLO format,
build a stratified train/val split, and write a data.yaml.

Robust to folder layout: it recursively finds every .xml annotation and the
matching image (by filename stem) anywhere under --src, so it works whether
your download is flat or already split into train/validation folders.

Usage:
    python scripts/voc_to_yolo.py --src /path/to/NEU-DET --dst datasets/neu
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def index_images(src: Path):
    """Map filename stem -> image path for every image under src."""
    idx = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.stem, p)
    return idx


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        objs.append((
            name,
            float(bb.findtext("xmin")), float(bb.findtext("ymin")),
            float(bb.findtext("xmax")), float(bb.findtext("ymax")),
        ))
    return w, h, objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="NEU-DET dataset root")
    ap.add_argument("--dst", default=Path("datasets/neu"), type=Path, help="output YOLO dir")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    xmls = sorted(args.src.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No .xml annotations found under {args.src}")
    images = index_images(args.src)
    print(f"Found {len(xmls)} annotations and {len(images)} images under {args.src}")

    records, class_counter = [], Counter()
    for xml in xmls:
        w, h, objs = parse_voc(xml)
        if not objs:
            continue
        img_path = images.get(xml.stem)
        if img_path is None:
            print(f"  ! no image for {xml.stem}, skipping")
            continue
        if not w or not h:                      # fall back to real image size
            from PIL import Image
            with Image.open(img_path) as im:
                w, h = im.size
        records.append((xml.stem, img_path, w, h, objs))
        for o in objs:
            class_counter[o[0]] += 1

    classes = sorted(class_counter)             # deterministic class order
    cls_to_id = {c: i for i, c in enumerate(classes)}
    print("Classes:", classes)

    # Stratified split by each image's majority class
    by_class = defaultdict(list)
    for rec in records:
        maj = Counter(o[0] for o in rec[4]).most_common(1)[0][0]
        by_class[maj].append(rec)

    train, val = [], []
    for recs in by_class.values():
        random.shuffle(recs)
        k = int(round(len(recs) * args.val_split))
        val.extend(recs[:k])
        train.extend(recs[k:])
    random.shuffle(train); random.shuffle(val)

    for split, recs in [("train", train), ("val", val)]:
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, img_path, w, h, objs in recs:
            shutil.copy(img_path, args.dst / "images" / split / img_path.name)
            lines = []
            for name, xmin, ymin, xmax, ymax in objs:
                xc, yc = ((xmin + xmax) / 2) / w, ((ymin + ymax) / 2) / h
                bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
                lines.append(f"{cls_to_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (args.dst / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    (args.dst / "data.yaml").write_text(
        f"path: {args.dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )

    print(f"\nDone. train={len(train)} val={len(val)}")
    print(f"data.yaml -> {args.dst / 'data.yaml'}")
    print("Per-class box counts:", dict(class_counter))


if __name__ == "__main__":
    main()
