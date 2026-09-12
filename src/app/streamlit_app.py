"""Streamlit UI around src/inference/predict.py -- upload a video, get a
real/fake verdict with confidence, attention-weight breakdown, and
face-detection health, with guardrails against showing an over-confident
answer built on too little face data.

Run with: streamlit run src/app/streamlit_app.py
"""
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.predict import predict


def html(markup: str):
    """st.markdown(unsafe_allow_html=True) on dedented text -- plain markdown
    treats 4+ space indented lines as a code block, which silently breaks
    <style>/<div> blocks written as indented triple-quoted strings."""
    st.markdown(textwrap.dedent(markup), unsafe_allow_html=True)


MAX_SIZE_MB = 100
ALLOWED_EXTS = (".mp4", ".mov")
DETECTION_RATE_FLOOR = 0.5  # below this, refuse to show a verdict at all
CONFIDENCE_FLOOR = 0.6  # below this, label "Uncertain" instead of Real/Fake

STREAM_COLORS = {
    "spatial": "#22D3EE",     # cyan
    "frequency": "#F472B6",   # pink/magenta
    "temporal": "#FBBF24",    # amber
}

st.set_page_config(page_title="Deepfake Detector (Research Prototype)", layout="centered", page_icon="🎭")

html('<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">')

# NOTE: no blank lines allowed inside this block -- Streamlit's markdown
# renderer treats a blank line as ending the raw-HTML block early, dumping
# everything after it as literal escaped text (cost real debugging time).
html("""
<style>
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(circle at 15% 0%, #241a3d 0%, #0B0B14 45%),
                radial-gradient(circle at 90% 20%, #0f2a3d 0%, #0B0B14 55%);
}
.hero { padding: 8px 0 4px 0; margin-bottom: 6px; }
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #A855F7 0%, #22D3EE 55%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p { color: #9CA3AF; font-size: 1.02rem; margin-top: 6px; }
.disclaimer {
    border-radius: 14px;
    padding: 16px 20px;
    margin: 18px 0 26px 0;
    background: linear-gradient(135deg, rgba(251,191,36,0.14), rgba(239,68,68,0.10));
    border: 1px solid rgba(251,191,36,0.4);
    color: #FDE68A;
    font-size: 0.92rem;
    line-height: 1.5;
}
.disclaimer b { color: #FEF3C7; }
.upload-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #E5E7EB;
    margin-bottom: 2px;
}
.upload-sub { color: #9CA3AF; font-size: 0.88rem; margin-bottom: 10px; }
.verdict-card {
    border-radius: 20px;
    padding: 34px 20px;
    text-align: center;
    margin: 14px 0 24px 0;
    position: relative;
    overflow: hidden;
}
.verdict-card h2 { font-family: 'Space Grotesk', sans-serif; margin: 0; font-size: 2rem; font-weight: 700; }
.verdict-card p { margin: 8px 0 0 0; font-size: 1rem; }
.verdict-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.22), rgba(6,182,212,0.10));
    border: 1px solid rgba(16,185,129,0.55);
    box-shadow: 0 0 40px rgba(16,185,129,0.15);
}
.verdict-real h2 { color: #34D399; }
.verdict-real p { color: #A7F3D0; }
.verdict-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(244,63,94,0.10));
    border: 1px solid rgba(239,68,68,0.55);
    box-shadow: 0 0 40px rgba(239,68,68,0.15);
}
.verdict-fake h2 { color: #F87171; }
.verdict-fake p { color: #FCA5A5; }
.verdict-uncertain, .verdict-none {
    background: linear-gradient(135deg, rgba(168,85,247,0.16), rgba(107,114,128,0.14));
    border: 1px solid rgba(168,85,247,0.4);
    box-shadow: 0 0 40px rgba(168,85,247,0.12);
}
.verdict-uncertain h2, .verdict-none h2 { color: #C4B5FD; font-size: 1.6rem; }
.verdict-uncertain p, .verdict-none p { color: #D1D5DB; }
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #E5E7EB;
    margin: 26px 0 12px 0;
}
.metric-chip {
    display: inline-block;
    background: rgba(168,85,247,0.12);
    border: 1px solid rgba(168,85,247,0.35);
    border-radius: 999px;
    padding: 6px 16px;
    color: #E9D5FF;
    font-size: 0.92rem;
    font-weight: 500;
}
.attn-row { margin-bottom: 14px; }
.attn-label { display: flex; justify-content: space-between; font-size: 0.9rem; color: #D1D5DB; margin-bottom: 4px; }
.attn-track { width: 100%; height: 14px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
.attn-fill { height: 100%; border-radius: 999px; }
footer, #MainMenu { visibility: hidden; }
</style>
""")

html("""
<div class="hero">
    <h1>🎭 Deepfake Detection</h1>
    <p>Multi-scale attention fusion &middot; spatial + frequency + temporal streams</p>
</div>
""")

html("""
<div class="disclaimer">
⚠️ <b>Research prototype, not a production tool.</b> Trained on a small dataset
(250 videos). Known <b>~20% false-negative rate</b> on held-out validation
(5 of 25 fake videos missed). <b>Not suitable for legal, journalistic, or
content-moderation decisions.</b>
</div>
""")

html('<div class="upload-label">Upload a video</div>')
html('<div class="upload-sub">.mp4 or .mov &middot; max 100MB</div>')
uploaded = st.file_uploader("Upload a video", type=["mp4", "mov"], label_visibility="collapsed")


def render_attention_bar(name, weight, color):
    html(f"""
    <div class="attn-row">
        <div class="attn-label"><span>{name}</span><span>{weight * 100:.1f}%</span></div>
        <div class="attn-track">
            <div class="attn-fill" style="width:{max(weight * 100, 2):.1f}%;
                background: linear-gradient(90deg, {color}, {color}AA);"></div>
        </div>
    </div>
    """)


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

            html('<div class="section-title">Face detection</div>')
            html(
                f'<span class="metric-chip">🧑 {result["frames_used"]} / {result["frames_sampled"]} '
                f'frames &middot; {detection_rate * 100:.0f}% detected</span>'
            )

            if detection_rate < DETECTION_RATE_FLOOR:
                html("""
                <div class="verdict-card verdict-none">
                <h2>🚫 No reliable result</h2>
                <p>Too few faces detected in this clip to give a reliable result.
                Try a clearer video with the face more visible and centered.</p>
                </div>
                """)
            else:
                label = result["label"]
                confidence = result["confidence"]

                if confidence < CONFIDENCE_FLOOR:
                    html(f"""
                    <div class="verdict-card verdict-uncertain">
                    <h2>🤔 Uncertain</h2>
                    <p>Model leans {label.title()} but confidence is only
                    {confidence * 100:.1f}% — too close to call.</p>
                    </div>
                    """)
                elif label == "REAL":
                    html(f"""
                    <div class="verdict-card verdict-real">
                    <h2>✅ Likely Real</h2><p>{confidence * 100:.1f}% confidence</p>
                    </div>
                    """)
                else:
                    html(f"""
                    <div class="verdict-card verdict-fake">
                    <h2>🚩 Likely Fake</h2><p>{confidence * 100:.1f}% confidence</p>
                    </div>
                    """)

                html('<div class="section-title">Attention weight breakdown</div>')
                render_attention_bar("Spatial", result["w_spatial"], STREAM_COLORS["spatial"])
                render_attention_bar("Frequency", result["w_frequency"], STREAM_COLORS["frequency"])
                render_attention_bar("Temporal", result["w_temporal"], STREAM_COLORS["temporal"])
