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
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    items = []
    for filename in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(filename)[1].lower() not in supported:
            continue
        with open(path, "rb") as handle:
            items.append({"name": filename, "bytes": handle.read(), "path": path})
    if not items:
        raise ValueError(f"No images found in {folder_path}")
    return items


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
    sidecar = f"{os.path.splitext(image_path)[0]}.latex.txt"
    if not os.path.isfile(sidecar):
        return ""
    with open(sidecar, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def score_ai_keyword_flags(ocr_text, latex_text):
    combined = "\n".join(part for part in [ocr_text, latex_text] if part).strip()
    if not combined:
        return 0.0, []
    normalized = combined.casefold()
    matches = []
    total = 0
    for category, keywords in AI_KEYWORD_FLAG_LIST.items():
        for keyword in keywords:
            count = normalized.count(keyword.casefold())
            if count:
                total += count
                matches.append(f"{category}: {keyword}")
    return round(min(100.0, (total / AI_KEYWORD_SCORE_MATCHES_FOR_MAX) * 100.0), 1), matches


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


def run_audit(items):
    survey_df = pd.read_csv(SURVEY_CSV_PATH)
    survey_df["Start Time"] = pd.to_datetime(survey_df["Start Time"])
    survey_df["End Time"] = pd.to_datetime(survey_df["End Time"])
    survey_lookup = survey_df.set_index("Participant ID").to_dict("index")

    report = []
    seen_hashes = {}

    for item in items:
        filename = item["name"]
        image_bytes = item["bytes"]
        participant_id = get_participant_id(filename)
        print(f"Auditing Participant: {participant_id}...")

        meta = analyze_image_metadata(image_bytes)
        flags = []
        current_hash = meta["MD5 Hash"]

        if current_hash in seen_hashes:
            is_duplicate = f"YES (Matches {seen_hashes[current_hash]})"
            flags.append("Duplicate Image Fingerprint")
        else:
            is_duplicate = "No"
            seen_hashes[current_hash] = participant_id

        duration_minutes = None
        if participant_id in survey_lookup:
            start_t = survey_lookup[participant_id]["Start Time"]
            end_t = survey_lookup[participant_id]["End Time"]
            duration_minutes = (end_t - start_t).total_seconds() / 60.0
            if duration_minutes < MINIMUM_MINUTES_REQUIRED:
                flags.append(f"Speed Running ({duration_minutes:.1f} mins vs min {MINIMUM_MINUTES_REQUIRED})")
            img_time = meta["Photo Created Time"]
            if isinstance(img_time, datetime):
                if img_time < start_t:
                    flags.append("Anachronism (Photo taken before survey started)")
                elif img_time > end_t:
                    flags.append("Anachronism (Photo taken after survey submitted)")
        else:
            flags.append("ID mismatch (Participant not found in survey CSV export)")

        software_lower = meta["Software Used"].lower()
        if any(app in software_lower for app in ["photoshop", "canva", "gimp", "illustrator"]):
            flags.append(f"Edited with software ({meta['Software Used']})")
        if meta["Device/OS"] == "Unknown / Stripped Metadata":
            flags.append("Missing Device Metadata (Likely web download/AI)")

        latex_text = load_latex_sidecar(item.get("path")) or convert_math_photo_to_latex(
            image_bytes,
            filename=filename,
        )
        ocr_text = extract_text_from_image(image_bytes)
        ai_score, ai_matches = score_ai_keyword_flags(ocr_text, latex_text)
        if ai_score >= 70.0:
            flags.append(f"High AI keyword signature ({ai_score}%)")

        report.append({
            "Participant ID": participant_id,
            "Duration (Mins)": round(duration_minutes, 1) if duration_minutes is not None else "N/A",
            "Device Detected": meta["Device/OS"],
            "Is Duplicate": is_duplicate,
            "OCR Text Preview": (latex_text or ocr_text)[:150],
            "LaTeX Preview": latex_text[:150],
            "AI Keyword Score": ai_score,
            "AI Keyword Matches": "; ".join(ai_matches) if ai_matches else "None",
            "Security Flags": "; ".join(flags) if flags else "Clear",
            "Final Status": "FLAGGED FOR REVIEW" if flags else "APPROVED",
        })

    pd.DataFrame(report).to_csv("comprehensive_fraud_report.csv", index=False)
    print("Done. Open comprehensive_fraud_report.csv")


if __name__ == "__main__":
    run_audit(load_local_images())
