#!/usr/bin/env python3
"""
Download the NEU-DET dataset (Pascal VOC bounding-box annotations) from Kaggle.

One-time setup of a Kaggle API token:
  1. kaggle.com  ->  Account  ->  "Create New API Token"  (downloads kaggle.json)
  2. On Colab: upload kaggle.json when prompted (cell below handles it), OR
     place it locally at  ~/.kaggle/kaggle.json  (chmod 600).

Run:
    python scripts/download_data.py
It prints the extracted path; pass that to voc_to_yolo.py via --src.

NOTE: dataset slugs on Kaggle occasionally change. If the default 404s, search
Kaggle for "NEU-DET" / "NEU surface defect", copy any VOC-format mirror's slug,
and pass it with --slug. The converter handles whatever folder layout you get.
"""
import argparse
import kagglehub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="kaustubhdikshit/neu-surface-defect-database")
    args = ap.parse_args()

    path = kagglehub.dataset_download(args.slug)
    print("Dataset downloaded to:", path)
    print("Next:  python scripts/voc_to_yolo.py --src", path, "--dst datasets/neu")


if __name__ == "__main__":
    main()
