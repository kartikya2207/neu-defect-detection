# Steel Surface Defect Detection — NEU-DET (SoS 2026)

Multimodal Vision-Language QA pipeline for hot-rolled steel surface inspection.
**Week-4 midterm:** a YOLOv8 detector trained on the NEU-DET dataset. The VLM
report-generation stage and Streamlit deployment follow in later weeks.

## Repo layout
```
.
├── README.md
├── REPORT.md                 # midterm progress report (for mentor)
├── requirements.txt
├── results_metrics.json      # written by train.py (commit after training)
├── scripts/
│   ├── download_data.py      # fetch NEU-DET from Kaggle
│   ├── voc_to_yolo.py        # Pascal VOC XML -> YOLO txt + split + data.yaml
│   └── train.py              # YOLOv8 training + mAP reporting
└── notebooks/
    └── eda.ipynb             # EDA on the NEU dataset
```
`datasets/` and `runs/` are git-ignored (large / regenerable).

## Setup & run
```bash
pip install -r requirements.txt

# 1. download dataset (needs a Kaggle API token — see download_data.py)
python scripts/download_data.py

# 2. convert VOC -> YOLO, make an 85/15 stratified split, write data.yaml
python scripts/voc_to_yolo.py --src <path-printed-above> --dst datasets/neu

# 3. EDA — open notebooks/eda.ipynb, set DATA_DIR, Run All (commit with outputs)

# 4. train + report mAP
python scripts/train.py --data datasets/neu/data.yaml --epochs 50
```

## Results (preliminary — midterm)
| Metric | Value |
|---|---|
| mAP@0.5 | _[fill from results_metrics.json]_ |
| mAP@0.5:0.95 | _[fill]_ |

## Dataset
NEU-DET: 1,800 grayscale images (200×200 px), 6 defect classes — crazing,
inclusion, patches, pitted surface, rolled-in scale, scratches (~300 images/class),
with Pascal VOC bounding-box annotations.

## References
- Ultralytics YOLOv8 — https://docs.ultralytics.com/
- Song & Yan, "A noise robust method based on completed local binary patterns for
  hot-rolled steel strip surface defects," *Applied Surface Science*, 2013.
