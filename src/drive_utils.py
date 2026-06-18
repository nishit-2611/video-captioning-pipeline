"""Utilities for downloading videos from Google Drive links."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import gdown
import requests

DRIVE_FILE_ID_PATTERN = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
DRIVE_OPEN_ID_PATTERN = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def extract_drive_file_id(url_or_id: str) -> Optional[str]:
    """Extract a Google Drive file ID from a URL or raw ID string."""
    value = (url_or_id or "").strip()
    if not value:
        return None

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", value):
        return value

    match = DRIVE_FILE_ID_PATTERN.search(value)
    if match:
        return match.group(1)

    match = DRIVE_OPEN_ID_PATTERN.search(value)
    if match:
        return match.group(1)

    parsed = urlparse(value)
    if parsed.netloc.endswith("drive.google.com"):
        query_id = parse_qs(parsed.query).get("id", [None])[0]
        if query_id:
            return query_id

    return None


def build_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}&export=download"


def download_from_drive(
    url_or_id: str,
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """Download a video from a public or shared Google Drive link."""
    file_id = extract_drive_file_id(url_or_id)
    if not file_id:
        raise ValueError(f"Could not parse Google Drive file ID from: {url_or_id!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename or f"{file_id}.mp4"
    output_path = output_dir / safe_name

    download_url = build_direct_download_url(file_id)
    try:
        gdown.download(download_url, str(output_path), quiet=False, fuzzy=True)
    except Exception as gdown_error:
        response = requests.get(download_url, stream=True, timeout=120)
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Download failed for {url_or_id!r}. "
                "Ensure the Drive link is shared with 'Anyone with the link'."
            ) from gdown_error

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Download produced an empty file for {url_or_id!r}. "
            "Check that the link is public or use a valid shared link."
        )

    return output_path


def create_temp_download_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="captioning_videos_"))
