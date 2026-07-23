"""
Lightweight fraud audit — no local ML models.

Setup:
  python3 -m venv .venv-lite
  source .venv-lite/bin/activate
  pip install -r requirements-minimal.txt

Math OCR (SimpleTex):
    export SIMPLETEX_UAT=your_simpletex_uat
    export SIMPLETEX_API_URL=https://server.simpletex.cn/api/simpletex_ocr
    export SIMPLETEX_REC_MODE=formula
    export SIMPLETEX_ENABLE_IMG_ROT=true

Run:
  python audit.py
"""

import hashlib
import io
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime

import pandas as pd
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

from math_ocr_api import convert_math_photo_to_latex
from audit_report import build_review_row, save_audit_reports

MINIMUM_MINUTES_REQUIRED = 10.0
SURVEY_CSV_PATH = "survey_export.csv"
LOCAL_IMAGE_FOLDER = os.getenv("LOCAL_IMAGE_FOLDER", "test_uploads")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--psm 6")

AI_KEYWORD_FLAG_LIST = {
    "Overly Academic Adjectives": ["meticulous", "comprehensive", "intricate", "straightforward", "rigorous", "profound", "elegant"],
    "Step-by-Step Setup": ["let's break this down", "to solve this step-by-step", "crucially", "importantly", "let's delve into", "it's worth noting"],
    "Over-Explainer Transitions": ["consequently", "hence", "thus", "therefore, we can conclude", "by applying the principles of", "as a result"],
    "Wrapping Up Fluff": ["in summary", "ultimately", "this gives us the final result of", "we successfully determined"],
}
AI_KEYWORD_SCORE_MATCHES_FOR_MAX = 6


def get_participant_id(filename):
    return os.path.splitext(filename)[0] if filename else ""


def load_local_images(folder_path=LOCAL_IMAGE_FOLDER):
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    image_uploads = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        if os.path.splitext(filename)[1].lower() not in supported_extensions:
            continue
        with open(file_path, "rb") as image_file:
            image_uploads.append({"name": filename, "bytes": image_file.read(), "path": file_path})
    if not image_uploads:
        raise ValueError(f"No images found in {folder_path}")
    return image_uploads


def preprocess_image_for_ocr(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.autocontrast(image.convert("L"))
    if min(image.size) < 1200:
        scale = 1200 / min(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def extract_text_from_image(image_bytes):
    if not image_bytes:
        return ""
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return ""
    image = preprocess_image_for_ocr(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name, format="PNG")
        path = tmp.name
    try:
        cmd = [tesseract, path, "stdout"]
        if TESSERACT_CONFIG:
            cmd.extend(shlex.split(TESSERACT_CONFIG))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def load_latex_sidecar(image_path):
    if not image_path:
        return ""
    sidecar_path = f"{os.path.splitext(image_path)[0]}.latex.txt"
    if not os.path.isfile(sidecar_path):
        return ""
    with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
        return sidecar_file.read().strip()


def score_ai_keyword_flags(ocr_text, latex_text):
    extracted_text = "\n".join(part for part in [ocr_text, latex_text] if part).strip()
    if not extracted_text:
        return 0.0, []
    normalized_text = extracted_text.casefold()
    matched_keywords = []
    total_matches = 0
    for category, keywords in AI_KEYWORD_FLAG_LIST.items():
        for keyword in keywords:
            occurrences = normalized_text.count(keyword.casefold())
            if occurrences:
                total_matches += occurrences
                matched_keywords.append(f"{category}: {keyword}")
    score = min(100.0, (total_matches / AI_KEYWORD_SCORE_MATCHES_FOR_MAX) * 100.0)
    return round(score, 1), matched_keywords


def analyze_image_metadata(image_bytes):
    metadata = {
        "MD5 Hash": hashlib.md5(image_bytes or b"").hexdigest(),
        "Device/OS": "Unknown / Stripped Metadata",
        "Software Used": "None Detected",
        "Photo Created Time": None,
    }
    if not image_bytes:
        return metadata
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image._getexif()
        if not exif_data:
            return metadata
        exif = {TAGS.get(key, key): val for key, val in exif_data.items()}
        make = exif.get("Make", "")
        model = exif.get("Model", "")
        if make or model:
            metadata["Device/OS"] = f"{make} {model}".strip()
        software = exif.get("Software", "")
        if software:
            metadata["Software Used"] = software
        photo_time = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if photo_time:
            try:
                metadata["Photo Created Time"] = datetime.strptime(photo_time, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
    except Exception:
        pass
    return metadata


def run_security_audit(image_uploads):
    survey_df = pd.read_csv(SURVEY_CSV_PATH)
    survey_df["Start Time"] = pd.to_datetime(survey_df["Start Time"])
    survey_df["End Time"] = pd.to_datetime(survey_df["End Time"])
    survey_records_by_participant = survey_df.set_index("Participant ID").to_dict("index")

    report_rows = []
    seen_image_hashes = {}

    for upload in image_uploads:
        filename = upload["name"]
        image_bytes = upload["bytes"]
        participant_id = get_participant_id(filename)
        print(f"Auditing Participant: {participant_id}...")

        image_metadata = analyze_image_metadata(image_bytes)
        security_flags = []
        image_hash = image_metadata["MD5 Hash"]

        duplicate_status = "No"
        if image_hash in seen_image_hashes:
            duplicate_status = f"YES (Matches {seen_image_hashes[image_hash]})"
            security_flags.append("Duplicate Image Fingerprint")
        else:
            seen_image_hashes[image_hash] = participant_id

        duration_minutes = None
        participant_missing_from_survey = False
        if participant_id in survey_records_by_participant:
            survey_record = survey_records_by_participant[participant_id]
            survey_start_time = survey_record["Start Time"]
            survey_end_time = survey_record["End Time"]
            duration_minutes = (survey_end_time - survey_start_time).total_seconds() / 60.0
            if duration_minutes < MINIMUM_MINUTES_REQUIRED:
                security_flags.append(
                    f"Speed Running ({duration_minutes:.1f} mins vs min {MINIMUM_MINUTES_REQUIRED})"
                )

            photo_created_time = image_metadata["Photo Created Time"]
            if isinstance(photo_created_time, datetime):
                if photo_created_time < survey_start_time:
                    security_flags.append("Anachronism (Photo taken before survey started)")
                elif photo_created_time > survey_end_time:
                    security_flags.append("Anachronism (Photo taken after survey submitted)")
        else:
            participant_missing_from_survey = True
            security_flags.append("ID mismatch (Participant not found in survey CSV export)")

        editing_software = image_metadata["Software Used"].lower()
        if any(app in editing_software for app in ["photoshop", "canva", "gimp", "illustrator"]):
            security_flags.append(f"Edited with software ({image_metadata['Software Used']})")
        if image_metadata["Device/OS"] == "Unknown / Stripped Metadata":
            security_flags.append("Missing Device Metadata (Likely web download/AI)")

        latex_text = load_latex_sidecar(upload.get("path")) or convert_math_photo_to_latex(
            image_bytes,
            filename=filename,
        )
        ocr_text = extract_text_from_image(image_bytes)
        ai_keyword_score, ai_keyword_matches = score_ai_keyword_flags(ocr_text, latex_text)
        if ai_keyword_score >= 70.0:
            security_flags.append(f"High AI keyword signature ({ai_keyword_score}%)")

        work_preview = latex_text or ocr_text
        report_rows.append(
            build_review_row(
                participant_id=participant_id,
                security_flags=security_flags,
                duration_minutes=duration_minutes,
                participant_in_survey=not participant_missing_from_survey,
                duplicate_status=duplicate_status,
                photo_device=image_metadata["Device/OS"],
                metadata_missing=image_metadata["Device/OS"] == "Unknown / Stripped Metadata",
                math_read_successfully=bool(work_preview.strip()),
                work_preview=work_preview,
                ai_writing_score=ai_keyword_score,
                ai_writing_matches=ai_keyword_matches,
            )
        )

    save_audit_reports(report_rows)


if __name__ == "__main__":
    run_security_audit(load_local_images())
