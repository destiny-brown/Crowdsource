import os
import io
import hashlib
import shlex
import subprocess
import tempfile
import pandas as pd
from datetime import datetime
from functools import lru_cache
from PIL import Image
from PIL.ExifTags import TAGS

try:
    import pytesseract
except ImportError:
    pytesseract = None

# --- CONFIGURATION ---
MINIMUM_MINUTES_REQUIRED = 10.0  # Change to survey's minimum threshold
SURVEY_CSV_PATH = "survey_export.csv"
LOCAL_IMAGE_FOLDER = os.getenv("LOCAL_IMAGE_FOLDER", "test_uploads")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
MATH2LATEX_PROJECT_DIR = os.getenv("MATH2LATEX_PROJECT_DIR", "Math2LaTeX")
MATH2LATEX_COMMAND = os.getenv("MATH2LATEX_COMMAND")

AI_KEYWORD_FLAG_LIST = {
    "Overly Academic Adjectives": [
        "meticulous",
        "comprehensive",
        "intricate",
        "straightforward",
        "rigorous",
        "profound",
        "elegant"
    ],
    "Step-by-Step Setup": [
        "let's break this down",
        "to solve this step-by-step",
        "crucially",
        "importantly",
        "let's delve into",
        "it's worth noting"
    ],
    "Over-Explainer Transitions": [
        "consequently",
        "hence",
        "thus",
        "therefore, we can conclude",
        "by applying the principles of",
        "as a result"
    ],
    "Wrapping Up Fluff": [
        "in summary",
        "ultimately",
        "this gives us the final result of",
        "we successfully determined"
    ]
}
AI_KEYWORD_SCORE_MATCHES_FOR_MAX = 6

# --- GOOGLE DRIVE ACCESS ---
def get_google_drive_service():
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        raise ValueError("Set GOOGLE_SERVICE_ACCOUNT_FILE to your service account JSON file path.")

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError("Install Google Drive dependencies with: pip install -r requirements.txt") from exc

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=scopes
    )
    return build("drive", "v3", credentials=credentials)


def load_google_drive_images():
    if not GOOGLE_DRIVE_FOLDER_ID:
        raise ValueError("Set GOOGLE_DRIVE_FOLDER_ID to the Drive folder that receives uploaded photos.")

    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise ImportError("Install Google Drive dependencies with: pip install -r requirements.txt") from exc

    service = get_google_drive_service()
    query = (
        f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents "
        "and trashed = false "
        "and mimeType contains 'image/'"
    )
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()

    drive_items = []
    for file_info in results.get("files", []):
        request = service.files().get_media(fileId=file_info["id"])
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        drive_items.append({
            "name": file_info["name"],
            "bytes": buffer.getvalue()
        })
    return drive_items


def load_local_images(folder_path=LOCAL_IMAGE_FOLDER):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Local image folder not found: {folder_path}")

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    local_items = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        _, extension = os.path.splitext(filename)
        if extension.lower() not in supported_extensions:
            continue

        with open(file_path, "rb") as image_file:
            local_items.append({
                "name": filename,
                "bytes": image_file.read(),
                "path": file_path
            })

    if not local_items:
        raise ValueError(f"No image files found in local folder: {folder_path}")

    return local_items


# --- OCR AND AI ANALYSIS ---
def extract_text_from_image(image_bytes):
    if not image_bytes or pytesseract is None:
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""


@lru_cache(maxsize=1)
def get_math2latex_model():
    try:
        from pix2tex.cli import LatexOCR
    except ImportError as exc:
        raise RuntimeError(
            "Install the local Math2LaTeX/LaTeX-OCR environment first. "
            "For example: conda create -n latexocr python==3.11, "
            "conda activate latexocr, then install the Math2LaTeX requirements."
        ) from exc

    return LatexOCR()


def run_math2latex_command(image_bytes):
    if not MATH2LATEX_COMMAND:
        return ""

    project_dir = MATH2LATEX_PROJECT_DIR if os.path.isdir(MATH2LATEX_PROJECT_DIR) else None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as image_file:
        image_file.write(image_bytes)
        image_file.flush()
        command = [part.format(image=image_file.name) for part in shlex.split(MATH2LATEX_COMMAND)]
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            check=True,
            text=True,
            timeout=120
        )
        return result.stdout.strip()


def convert_math_photo_to_latex(image_bytes):
    if not image_bytes:
        return ""

    try:
        if MATH2LATEX_COMMAND:
            return run_math2latex_command(image_bytes)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model = get_math2latex_model()
        return str(model(image)).strip()
    except Exception as exc:
        print(f"Math2LaTeX conversion failed: {exc}")
        return ""


def load_latex_sidecar(image_path):
    if not image_path:
        return ""

    base_path, _ = os.path.splitext(image_path)
    sidecar_path = f"{base_path}.latex.txt"
    if not os.path.isfile(sidecar_path):
        return ""

    try:
        with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
            return sidecar_file.read().strip()
    except Exception as exc:
        print(f"Could not read LaTeX sidecar {sidecar_path}: {exc}")
        return ""


def score_ai_keyword_flags(ocr_text, latex_text):
    combined_text = "\n".join(part for part in [ocr_text, latex_text] if part).strip()
    if not combined_text:
        return 0.0, []

    normalized_text = combined_text.casefold()
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

# --- METADATA EXTRACTION ---
def analyze_image_metadata(image_bytes):
    if not image_bytes:
        image_bytes = b""

    metadata = {
        "MD5 Hash": hashlib.md5(image_bytes).hexdigest(),
        "Device/OS": "Unknown / Stripped Metadata",
        "Software Used": "None Detected",
        "Photo Created Time": None
    }
    if image_bytes == b"":
        return metadata

    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image._getexif()
        if exif_data:
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

# --- MAIN BATCH EXECUTIVE ---
def run_security_audit(google_drive_items):
    # Load and parse the survey platform times
    try:
        survey_df = pd.read_csv(SURVEY_CSV_PATH)
        # Automatically handle standard datetime formats
        survey_df['Start Time'] = pd.to_datetime(survey_df['Start Time'])
        survey_df['End Time'] = pd.to_datetime(survey_df['End Time'])
        # Set Participant ID as index for fast lookups
        survey_lookup = survey_df.set_index('Participant ID').to_dict('index')
    except Exception as e:
        print(f"Error loading {SURVEY_CSV_PATH}: {e}")
        return

    report_data = []
    seen_hashes = {}

    for item in google_drive_items:
        # Assuming filename structure matches the ID format in your survey CSV
        filename = item.get('name')
        image_bytes = item.get('bytes') # Placeholder for downloaded file data from Drive API

        if not filename:
            print("Skipping Drive item with no filename.")
            continue

        participant_id = os.path.splitext(filename)[0]
        
        print(f"Auditing Participant: {participant_id}...")
        
        # 1. Image Analytics
        meta = analyze_image_metadata(image_bytes)
        current_hash = meta["MD5 Hash"]
        flag_reasons = []
        
        # 2. Check for Duplicate Uploads
        is_duplicate = "No"
        if current_hash in seen_hashes:
            is_duplicate = f"YES (Matches {seen_hashes[current_hash]})"
            flag_reasons.append("Duplicate Image Fingerprint")
        else:
            seen_hashes[current_hash] = participant_id

        # 3. Pull Survey Platform Durations
        duration_minutes = None
        if participant_id in survey_lookup:
            timestamps = survey_lookup[participant_id]
            start_t = timestamps['Start Time']
            end_t = timestamps['End Time']
            
            # Calculate overall survey duration
            duration_minutes = (end_t - start_t).total_seconds() / 60.0
            
            if duration_minutes < MINIMUM_MINUTES_REQUIRED:
                flag_reasons.append(f"Speed Running ({duration_minutes:.1f} mins vs min {MINIMUM_MINUTES_REQUIRED})")
                
            # 4. Image Creation Time vs. Survey Window Verification
            img_time = meta["Photo Created Time"]
            if isinstance(img_time, datetime):
                if img_time < start_t:
                    flag_reasons.append("Anachronism (Photo taken before survey started)")
                elif img_time > end_t:
                    flag_reasons.append("Anachronism (Photo taken after survey submitted)")
        else:
            flag_reasons.append("ID mismatch (Participant not found in survey CSV export)")

        # 5. Metadata Software Flags
        software_lower = meta["Software Used"].lower()
        if any(app in software_lower for app in ["photoshop", "canva", "gimp", "illustrator"]):
            flag_reasons.append(f"Edited with software ({meta['Software Used']})")
        if meta["Device/OS"] == "Unknown / Stripped Metadata":
            flag_reasons.append("Missing Device Metadata (Likely web download/AI)")

        # 6. OCR, Math-to-LaTeX, and AI-style keyword checks
        ocr_text = extract_text_from_image(image_bytes)
        latex_text = load_latex_sidecar(item.get("path")) or convert_math_photo_to_latex(image_bytes)
        ai_keyword_score, ai_keyword_matches = score_ai_keyword_flags(ocr_text, latex_text)
        if ai_keyword_score >= 70.0:
            flag_reasons.append(f"High AI keyword signature ({ai_keyword_score}%)")

        # Determine Final Review State
        status = "FLAGGED FOR REVIEW" if flag_reasons else "APPROVED"

        report_data.append({
            "Participant ID": participant_id,
            "Duration (Mins)": round(duration_minutes, 1) if duration_minutes is not None else "N/A",
            "Device Detected": meta["Device/OS"],
            "Is Duplicate": is_duplicate,
            "OCR Text Preview": ocr_text[:150],
            "LaTeX Preview": latex_text[:150],
            "AI Keyword Score": ai_keyword_score,
            "AI Keyword Matches": "; ".join(ai_keyword_matches) if ai_keyword_matches else "None",
            "Security Flags": "; ".join(flag_reasons) if flag_reasons else "Clear",
            "Final Status": status
        })

    # Save finalized document
    df = pd.DataFrame(report_data)
    df.to_csv("comprehensive_fraud_report.csv", index=False)
    print("Audit completely finished. Open 'comprehensive_fraud_report.csv' to review flags.")


if __name__ == "__main__":
    try:
        if GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_FILE:
            audit_items = load_google_drive_images()
        else:
            print(f"Google Drive is not configured. Loading images from '{LOCAL_IMAGE_FOLDER}' instead.")
            audit_items = load_local_images()
        run_security_audit(audit_items)
    except Exception as exc:
        print(f"Could not run audit: {exc}")
        print("Set Google Drive variables or place test images in LOCAL_IMAGE_FOLDER, then run again.")
