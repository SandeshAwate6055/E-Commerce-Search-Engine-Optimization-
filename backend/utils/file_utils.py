"""
File upload validation utilities.
"""
import hashlib
import imghdr
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.config import settings

ALLOWED_MIME_TYPES = set(settings.ALLOWED_IMAGE_TYPES)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


async def save_upload(file: UploadFile) -> Path:
    """
    Validate and save an uploaded image to the temp upload directory.

    Raises HTTPException on invalid file type or oversized file.
    Returns absolute Path to saved file.
    """
    # Check declared content type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )

    # Read content with size limit
    content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # Validate actual image header (not just the declared MIME type)
    detected = imghdr.what(None, h=content[:32])
    if detected not in ("jpeg", "png", "webp"):
        raise HTTPException(
            status_code=415,
            detail="File content does not appear to be a valid image.",
        )

    # Save with a random name — never trust the original filename
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".jpg"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    dest = Path(settings.UPLOAD_DIR) / unique_name
    dest.write_bytes(content)
    return dest


def cleanup_upload(path: Path) -> None:
    """Remove a temporary upload file after processing."""
    try:
        if path and path.exists():
            path.unlink()
    except OSError:
        pass
