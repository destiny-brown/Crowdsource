"""SimpleTex-only math OCR helper."""

import json
import os
import ssl
import urllib.error
import urllib.request

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

SIMPLETEX_UAT = os.getenv("SIMPLETEX_UAT", "").strip()
SIMPLETEX_API_URL = os.getenv("SIMPLETEX_API_URL", "https://server.simpletex.cn/api/simpletex_ocr")
SIMPLETEX_REC_MODE = os.getenv("SIMPLETEX_REC_MODE", "formula").strip().lower()
SIMPLETEX_ENABLE_IMG_ROT = os.getenv("SIMPLETEX_ENABLE_IMG_ROT", "true").strip().lower()


def convert_math_photo_to_latex(image_bytes, filename="upload.jpg"):
    if not image_bytes or not SIMPLETEX_UAT:
        return ""

    content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    boundary = f"----SimpleTexBoundary{os.urandom(12).hex()}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="rec_mode"\r\n\r\n')
    body.extend(f"{SIMPLETEX_REC_MODE}\r\n".encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="enable_img_rot"\r\n\r\n')
    body.extend(f"{SIMPLETEX_ENABLE_IMG_ROT}\r\n".encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(image_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        SIMPLETEX_API_URL,
        data=bytes(body),
        headers={
            "token": SIMPLETEX_UAT,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90, context=SSL_CONTEXT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"SimpleTex OCR failed ({exc.code}): {error_body}")
        return ""
    except Exception as exc:
        print(f"SimpleTex OCR failed: {exc}")
        return ""

    if not payload.get("status"):
        return ""
    res = payload.get("res", {})
    if isinstance(res, dict):
        info = str(res.get("info", "")).strip()
        if info:
            return info
        latex = str(res.get("latex", "")).strip()
        if latex:
            return latex
        return str(res.get("text", "")).strip()
    return ""
