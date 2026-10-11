# AI_Deepfake
Deepfake detection with spatial (ResNet-18), frequency (DCT + CNN) and temporal (Bi-LSTM) streams, combined by attention fusion. Research prototype trained on a 250-video DFD subset. Not for real-world decisions.

## Run
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run src/app/streamlit_app.py
    python src/inference/predict.py path\to\video.mp4

Run from the repo root: model paths are relative (results/checkpoints/...). Python 3.12.
The dataset is not included (too large); it is excluded via .gitignore.
