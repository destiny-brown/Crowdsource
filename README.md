# Crowdsource Fraud Audit

Dropbox-first fraud audit pipeline for participant math-work photo submissions.

The runtime flow is:

1. Read uploaded images directly from Dropbox.
2. Run OCR through the live SimpleTex API.
3. Cross-check against local `survey_export.csv`.
4. Save audit reports under `outputs/`.

The only required local dataset is `survey_export.csv`.

## Required inputs

1. `survey_export.csv` in the project root with columns:
   - `Participant ID`
   - `Start Time`
   - `End Time`
2. Dropbox API token and source folder.
3. SimpleTex API credentials.

## Environment setup

Create `.env` (or export vars in shell):

```bash
SIMPLETEX_UAT=your_simpletex_uat_here
SIMPLETEX_API_URL=https://server.simpletex.cn/api/simpletex_ocr
SIMPLETEX_REC_MODE=formula
SIMPLETEX_ENABLE_IMG_ROT=true

DROPBOX_ACCESS_TOKEN=your_dropbox_access_token
DROPBOX_SOURCE_FOLDER=/Crowdsource Test/Raw
```

`crowdsource.py` auto-loads `.env` through `math_ocr_api.load_project_env()`.

## Install

Minimal runtime:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
```

Full runtime (optional ML/text/image anomaly features):

```bash
pip install -r requirements.txt
```

## Run

```bash
python crowdsource.py
```

Outputs are written to:

- `outputs/comprehensive_fraud_report.csv`
- `outputs/audit_features.csv`

## Optional advanced features

You can enable optional signals in `.env`:

```bash
ENABLE_TEXT_CLASSIFIER=true
TEXT_CLASSIFIER_MODEL=/absolute/path/or/hf-model-id
TEXT_CLASSIFIER_THRESHOLD=0.7

ENABLE_IMAGE_ANOMALY_SCORING=true
IMAGE_EMBEDDING_BACKBONE=open_clip
IMAGE_ANOMALY_DETECTOR=isolation_forest
IMAGE_ANOMALY_CONTAMINATION=0.1
IMAGE_ANOMALY_FLAG_THRESHOLD=70
```