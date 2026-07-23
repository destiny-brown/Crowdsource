"""Shared audit report formatting for reviewer-friendly and technical exports."""

from __future__ import annotations

import re

import pandas as pd

REVIEW_REPORT_FILENAME = "comprehensive_fraud_report.csv"
TECHNICAL_REPORT_FILENAME = "audit_features.csv"

REVIEW_REPORT_COLUMNS = [
    "Participant",
    "Review Status",
    "Priority",
    "Overall Risk Score",
    "Risk Confidence",
    "Time Spent (min)",
    "In Survey",
    "Duplicate Upload",
    "Photo Device",
    "Metadata Missing",
    "Math Read Successfully",
    "Work Preview",
    "AI Writing Score",
    "AI Writing Matches",
    "Text AI Score",
    "Photo Integrity Score",
    "Unusual Photo Score",
    "Top Risk Drivers",
    "Reasons for Review",
    "Recommended Action",
]

EXACT_FLAG_MESSAGES = {
    "Duplicate Image Fingerprint": "Same photo submitted by another participant",
    "Anachronism (Photo taken before survey started)": "Photo taken before the survey started",
    "Anachronism (Photo taken after survey submitted)": "Photo taken after the survey ended",
    "ID mismatch (Participant not found in survey CSV export)": "Participant ID not found in survey records",
    "LaTeX extraction failed": "Could not read math work from the photo",
    "Missing Device Metadata (Likely web download/AI)": "Photo device information is missing",
    "Missing Device Metadata (corroborated by other risk cues)": "Photo device information is missing and other concerns were found",
}

FLAG_PREFIX_MESSAGES = [
    (re.compile(r"^Speed Running"), "Completed the survey too quickly"),
    (re.compile(r"^Edited with software"), "Photo may have been edited"),
    (re.compile(r"^High AI keyword signature"), "Writing looks AI-generated"),
    (re.compile(r"^High text classifier score"), "Text classified as likely AI-generated"),
    (re.compile(r"^Image embedding anomaly"), "Photo looks unusual compared with other submissions"),
    (re.compile(r"^High image forgery cue score"), "Photo may have been manipulated or re-saved"),
]

RISK_DRIVER_LABELS = {
    "Text": "AI writing",
    "Forgery": "Photo integrity",
    "ImageAnomaly": "Unusual photo",
    "Duration": "Survey speed",
    "Metadata": "Missing metadata",
    "Duplicate": "Duplicate upload",
    "IDMismatch": "Survey ID mismatch",
}


def format_report_value(value, missing_label="N/A"):
    return value if value is not None else missing_label


def classify_risk_tier(score):
    if score is None:
        return "Unknown"
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def describe_confidence(confidence_score):
    if confidence_score is None:
        return "N/A"
    if confidence_score >= 75:
        return "High"
    if confidence_score >= 50:
        return "Medium"
    return "Low"


def sanitize_preview_text(text, max_length=200):
    cleaned = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3] + "..."


def format_review_status(security_flags):
    return "Needs Review" if security_flags else "Approved"


def simplify_duplicate_status(duplicate_status):
    duplicate_text = str(duplicate_status or "No")
    return "Yes" if duplicate_text.lower().startswith("yes") else "No"


def humanize_security_flag(flag):
    if flag in EXACT_FLAG_MESSAGES:
        return EXACT_FLAG_MESSAGES[flag]

    for pattern, message in FLAG_PREFIX_MESSAGES:
        if pattern.search(flag):
            return message

    return flag


def humanize_security_flags(security_flags):
    if not security_flags:
        return "None"
    return "; ".join(humanize_security_flag(flag) for flag in security_flags)


def humanize_top_risk_drivers(top_contributors):
    if not top_contributors:
        return "N/A"

    humanized = []
    for part in str(top_contributors).split(";"):
        part = part.strip()
        if not part:
            continue
        label = part.split("(", 1)[0].strip()
        humanized.append(RISK_DRIVER_LABELS.get(label, label))
    return "; ".join(humanized) if humanized else "N/A"


def format_ai_writing_matches(matches):
    if not matches:
        return "None"
    return "; ".join(matches)


def recommended_action(priority, review_status):
    if review_status == "Approved":
        return "No action needed"
    if priority == "High":
        return "Review urgently"
    if priority == "Medium":
        return "Review when possible"
    if priority == "Unknown":
        return "Manual review"
    return "Quick spot check"


def estimate_priority_without_fusion(security_flags, ai_writing_score):
    if not security_flags:
        return "Low"

    high_priority_flags = {
        "Duplicate Image Fingerprint",
        "ID mismatch (Participant not found in survey CSV export)",
    }
    if any(flag in high_priority_flags for flag in security_flags):
        return "High"
    if ai_writing_score >= 70 or len(security_flags) >= 2:
        return "High"
    if security_flags:
        return "Medium"
    return "Low"


def estimate_risk_score_without_fusion(security_flags, ai_writing_score):
    if not security_flags and ai_writing_score <= 0:
        return 0.0
    return round(min(100.0, (len(security_flags) * 20.0) + (ai_writing_score * 0.5)), 1)


def build_review_row(
    *,
    participant_id,
    security_flags,
    duration_minutes=None,
    participant_in_survey=True,
    duplicate_status="No",
    photo_device="Unknown",
    metadata_missing=False,
    math_read_successfully=None,
    work_preview="",
    ai_writing_score=0.0,
    ai_writing_matches=None,
    text_ai_score=None,
    photo_integrity_score=None,
    unusual_photo_score=None,
    overall_risk_score=None,
    risk_confidence=None,
    top_risk_drivers=None,
    priority=None,
):
    review_status = format_review_status(security_flags)
    resolved_priority = priority or estimate_priority_without_fusion(security_flags, ai_writing_score)
    resolved_risk_score = (
        overall_risk_score
        if overall_risk_score is not None
        else estimate_risk_score_without_fusion(security_flags, ai_writing_score)
    )
    if math_read_successfully is None:
        math_read_successfully = bool(str(work_preview or "").strip())

    return {
        "Participant": participant_id,
        "Review Status": review_status,
        "Priority": resolved_priority,
        "Overall Risk Score": format_report_value(resolved_risk_score),
        "Risk Confidence": describe_confidence(risk_confidence),
        "Time Spent (min)": format_report_value(
            round(duration_minutes, 1) if duration_minutes is not None else None
        ),
        "In Survey": "Yes" if participant_in_survey else "No",
        "Duplicate Upload": simplify_duplicate_status(duplicate_status),
        "Photo Device": photo_device,
        "Metadata Missing": "Yes" if metadata_missing else "No",
        "Math Read Successfully": "Yes" if math_read_successfully else "No",
        "Work Preview": sanitize_preview_text(work_preview),
        "AI Writing Score": ai_writing_score,
        "AI Writing Matches": format_ai_writing_matches(ai_writing_matches),
        "Text AI Score": format_report_value(text_ai_score),
        "Photo Integrity Score": format_report_value(photo_integrity_score),
        "Unusual Photo Score": format_report_value(unusual_photo_score),
        "Top Risk Drivers": humanize_top_risk_drivers(top_risk_drivers),
        "Reasons for Review": humanize_security_flags(security_flags),
        "Recommended Action": recommended_action(resolved_priority, review_status),
    }


def save_audit_reports(review_rows, technical_rows=None):
    review_df = pd.DataFrame(review_rows, columns=REVIEW_REPORT_COLUMNS)
    review_df.to_csv(REVIEW_REPORT_FILENAME, index=False)

    if technical_rows is not None:
        technical_df = pd.DataFrame(technical_rows)
        technical_df.to_csv(TECHNICAL_REPORT_FILENAME, index=False)
        print(
            f"Audit finished. Review report: '{REVIEW_REPORT_FILENAME}'. "
            f"Technical features: '{TECHNICAL_REPORT_FILENAME}'."
        )
    else:
        print(f"Audit finished. Review report: '{REVIEW_REPORT_FILENAME}'.")
