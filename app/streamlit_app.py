#!/usr/bin/env python3
"""
Week 7 - Streamlit demo: upload a steel image -> detection overlay + report.

Run:
    streamlit run app/streamlit_app.py

If OPENAI_API_KEY is set in the environment, the report is written by a VLM;
otherwise a structured template report is shown. Either way the demo works.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import detect, template_report, vlm_report  # same folder

st.set_page_config(page_title="Steel Defect Inspector", layout="wide")
st.title("Steel Surface Defect Inspector")
st.caption("YOLOv8 detection + auto-generated maintenance report (SoS 2026)")

model_path = st.text_input(
    "Model weights path", "runs/detect/neu_yolov8n/weights/best.pt"
)
uploaded = st.file_uploader(
    "Upload a steel surface image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        img_path = tmp.name

    try:
        with st.spinner("Detecting defects..."):
            dets, result = detect(model_path, img_path)
            annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detections")
            st.image(annotated, use_column_width=True)
        with col2:
            st.subheader("Maintenance report")
            key = os.environ.get("OPENAI_API_KEY")
            report = vlm_report(img_path, dets, key) if key else template_report(dets)
            st.text(report)

        st.subheader("Raw detections")
        if dets:
            st.json(dets)
        else:
            st.info("No defects detected.")
    except Exception as e:
        st.error(f"Something went wrong: {e}\n\nCheck that the model path is correct.")
