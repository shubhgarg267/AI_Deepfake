"""Streamlit UI around src/inference/predict.py -- upload a video, get a
real/fake verdict with confidence, attention-weight breakdown, and
face-detection health, with guardrails against showing an over-confident
answer built on too little face data.

Run with: streamlit run src/app/streamlit_app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.predict import predict

MAX_SIZE_MB = 100
ALLOWED_EXTS = (".mp4", ".mov")
DETECTION_RATE_FLOOR = 0.5  # below this, refuse to show a verdict at all
CONFIDENCE_FLOOR = 0.6  # below this, label "Uncertain" instead of Real/Fake

st.set_page_config(page_title="Deepfake Detector (Research Prototype)", layout="centered")

st.title("Deepfake Detection — Research Prototype")

st.warning(
    "**Research prototype, not a production tool.** Trained on a small dataset "
    "(250 videos). Known **~20% false-negative rate** on held-out validation "
    "(5 of 25 fake videos missed). **Not suitable for legal, journalistic, or "
    "content-moderation decisions.**"
)

uploaded = st.file_uploader("Upload a video (.mp4 or .mov, max 100MB)", type=["mp4", "mov"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    size_mb = uploaded.size / 1e6

    if suffix not in ALLOWED_EXTS:
        st.error(f"Unsupported file type '{suffix}'. Please upload a .mp4 or .mov file.")
    elif size_mb > MAX_SIZE_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_SIZE_MB} MB.")
    else:
        tmp_path = None
        result = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Detecting faces and running inference — this can take a few seconds..."):
                result = predict(tmp_path)
        except Exception as e:
            st.error(f"Could not process this video: {e}")
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if result is not None:
            detection_rate = result["frames_used"] / result["frames_sampled"]

            st.subheader("Face detection")
            st.write(
                f"{result['frames_used']} / {result['frames_sampled']} sampled frames "
                f"had a detectable face ({detection_rate * 100:.0f}%)"
            )

            if detection_rate < DETECTION_RATE_FLOOR:
                st.markdown(
                    '<div style="background-color:#e2e3e5;padding:20px;border-radius:8px;'
                    'text-align:center;">'
                    '<h3 style="color:#383d41;margin:0;">No reliable result</h3>'
                    '<p style="margin:6px 0 0 0;color:#383d41;">Too few faces detected in this '
                    "clip to give a reliable result. Try a clearer video with the face more "
                    "visible and centered.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                label = result["label"]
                confidence = result["confidence"]

                if confidence < CONFIDENCE_FLOOR:
                    st.markdown(
                        '<div style="background-color:#e2e3e5;padding:20px;border-radius:8px;'
                        'text-align:center;">'
                        '<h2 style="color:#383d41;margin:0;">Uncertain</h2>'
                        f'<p style="margin:6px 0 0 0;color:#383d41;">Model leans {label.title()} '
                        f"but confidence is only {confidence * 100:.1f}% — too close to call.</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif label == "REAL":
                    st.markdown(
                        '<div style="background-color:#d4edda;padding:20px;border-radius:8px;'
                        'text-align:center;">'
                        f'<h2 style="color:#155724;margin:0;">Likely Real ({confidence * 100:.1f}% '
                        "confidence)</h2></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background-color:#f8d7da;padding:20px;border-radius:8px;'
                        'text-align:center;">'
                        f'<h2 style="color:#721c24;margin:0;">Likely Fake ({confidence * 100:.1f}% '
                        "confidence)</h2></div>",
                        unsafe_allow_html=True,
                    )

                st.subheader("Attention weight breakdown")
                weights_df = pd.DataFrame(
                    {
                        "weight": [
                            result["w_spatial"],
                            result["w_frequency"],
                            result["w_temporal"],
                        ]
                    },
                    index=["spatial", "frequency", "temporal"],
                )
                st.bar_chart(weights_df)
