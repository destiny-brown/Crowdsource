# Crowdsource Fraud Audit

This script audits participant-uploaded math-work photos from a local folder or Google Drive against a survey export and writes `comprehensive_fraud_report.csv`.

## Math2LaTeX setup

Use a Python 3.6+ environment with PyTorch 1.0+ installed. For the recommended local setup:

```bash
conda create -n latexocr python==3.11
conda activate latexocr
pip install -r requirements.txt
```

Set up your local `Math2LaTeX` checkout for fine-tuning:

1. Download Kaggle's Handwritten Mathematical Expressions dataset.
2. Move `archive.zip` into the `Math2LaTeX` directory.
3. From inside `Math2LaTeX`, run:

```bash
bash ./setup.sh
```

The setup is ready when `setup.sh` prints `all checks passed`. Prepared images should be in `Math2LaTeX/img_data`, with image-name/label pairs in `Math2LaTeX/img_data/labels.csv`.

By default, `crowdsource.py` uses the installed local `pix2tex`/LaTeX-OCR Python interface for image-to-LaTeX conversion. If your `Math2LaTeX` checkout exposes a different command-line predictor, set `MATH2LATEX_COMMAND` in `.env` and include `{image}` where the temporary image path should be passed, for example:

```bash
MATH2LATEX_COMMAND="python predict.py --image {image}"
```

## Required inputs

For quick local testing, use the included `test_uploads` folder and `survey_export.csv` fixture. If Google Drive variables are not set, the script automatically falls back to `LOCAL_IMAGE_FOLDER`.

```bash
LOCAL_IMAGE_FOLDER=test_uploads
python crowdsource.py
```

Each local upload can also have an optional sidecar file named `<participant_id>.latex.txt`. When present, the audit uses that text instead of calling Math2LaTeX, which keeps quick tests independent from the model install.

For Google Drive auditing, set these environment variables before running the audit:

```bash
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id_here
GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/service-account.json
MATH2LATEX_PROJECT_DIR=Math2LaTeX
```

The survey export must be saved as `survey_export.csv` and include `Participant ID`, `Start Time`, and `End Time` columns.

AI-style writing is scored locally from OCR/LaTeX text using the keyword flag list in `crowdsource.py`; no AI detector API key is required.

## Optional image embedding anomaly scoring

The audit can optionally score image anomalies using open-source embeddings plus an unsupervised detector:

- Embedding backbone: `open_clip` (default when enabled) or `dinov2`
- Detector: `isolation_forest` (default) or `lof`

Enable it with environment variables:

```bash
ENABLE_IMAGE_ANOMALY_SCORING=true
IMAGE_EMBEDDING_BACKBONE=open_clip
IMAGE_ANOMALY_DETECTOR=isolation_forest
IMAGE_ANOMALY_CONTAMINATION=0.1
IMAGE_ANOMALY_FLAG_THRESHOLD=70
```

Notes:

- Keep `ENABLE_IMAGE_ANOMALY_SCORING=false` for the lightest setup.
- With `IMAGE_EMBEDDING_BACKBONE=dinov2`, the model is loaded via `torch.hub` (`facebookresearch/dinov2`).
- The script needs at least 5 valid images to fit an anomaly detector and produce anomaly scores.
- New output columns in `comprehensive_fraud_report.csv` include `Image Anomaly Score`, `Image Embedding Backbone`, and `Image Anomaly Detector`.

Run the audit with:

```bash
python crowdsource.py
```