"""Batch captioning pipeline for videos listed in a CSV."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from src.drive_utils import create_temp_download_dir, download_from_drive, extract_drive_file_id
from src.gemini_client import (
    GeminiHyperparameters,
    create_client,
    delete_uploaded_file,
    generate_caption,
    upload_video,
)


DRIVE_URL_HINTS = ("drive.google.com", "docs.google.com", "googleusercontent.com")
LINK_COLUMN_HINTS = ("drive", "link", "url", "video", "file")


@dataclass
class PipelineJob:
    row_index: int
    row_id: str
    drive_link: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    row_index: int
    row_id: str
    drive_link: str
    status: str
    caption: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower())


def detect_link_column(columns: list[str]) -> Optional[str]:
    normalized = {_normalize_column_name(col): col for col in columns}

    for hint in LINK_COLUMN_HINTS:
        for norm_name, original in normalized.items():
            if hint in norm_name:
                return original

    for original in columns:
        if "http" in original.lower() or "drive" in original.lower():
            return original

    return columns[0] if columns else None


def detect_id_column(columns: list[str], link_column: str) -> Optional[str]:
    normalized = {_normalize_column_name(col): col for col in columns if col != link_column}

    for hint in ("id", "name", "title", "video_id", "clip"):
        for norm_name, original in normalized.items():
            if hint in norm_name:
                return original

    return None


def looks_like_drive_link(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    if extract_drive_file_id(value):
        return True
    return any(hint in value.lower() for hint in DRIVE_URL_HINTS)


def load_jobs_from_csv(
    csv_bytes: bytes,
    link_column: Optional[str] = None,
    id_column: Optional[str] = None,
) -> tuple[list[PipelineJob], dict[str, Any]]:
    dataframe = pd.read_csv(io.BytesIO(csv_bytes))
    if dataframe.empty:
        raise ValueError("CSV is empty.")

    columns = [str(col) for col in dataframe.columns]
    resolved_link_column = link_column or detect_link_column(columns)
    if not resolved_link_column or resolved_link_column not in columns:
        raise ValueError(
            "Could not detect a drive link column. "
            "Include a column such as 'drive_link' or 'video_url'."
        )

    resolved_id_column = id_column or detect_id_column(columns, resolved_link_column)

    jobs: list[PipelineJob] = []
    for row_index, row in dataframe.iterrows():
        drive_link = str(row.get(resolved_link_column, "")).strip()
        if not drive_link or drive_link.lower() == "nan":
            continue

        if not looks_like_drive_link(drive_link):
            continue

        row_id = str(row.get(resolved_id_column, row_index)) if resolved_id_column else str(row_index)
        metadata = {
            str(col): "" if pd.isna(row[col]) else row[col]
            for col in columns
            if col not in {resolved_link_column}
        }

        jobs.append(
            PipelineJob(
                row_index=int(row_index),
                row_id=row_id,
                drive_link=drive_link,
                metadata=metadata,
            )
        )

    if not jobs:
        raise ValueError(
            f"No valid Google Drive links found in column '{resolved_link_column}'."
        )

    info = {
        "link_column": resolved_link_column,
        "id_column": resolved_id_column,
        "total_rows": len(dataframe),
        "valid_jobs": len(jobs),
        "columns": columns,
    }
    return jobs, info


def results_to_dataframe(results: list[PipelineResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "row_index": result.row_index,
            "row_id": result.row_id,
            "drive_link": result.drive_link,
            "status": result.status,
            "caption": result.caption,
            "error": result.error,
        }
        row.update(result.metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def results_to_csv(results: list[PipelineResult]) -> str:
    dataframe = results_to_dataframe(results)
    return dataframe.to_csv(index=False)


def save_run_artifact(
    output_dir: Path,
    system_prompt: str,
    user_prompt: str,
    hyperparameters: GeminiHyperparameters,
    results: list[PipelineResult],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "hyperparameters": hyperparameters.to_dict(),
        "generated_at_utc": timestamp,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2))

    csv_path = run_dir / "captions.csv"
    csv_path.write_text(results_to_csv(results))
    return run_dir


ProgressCallback = Callable[[int, int, PipelineResult], None]


def run_captioning_pipeline(
    api_key: str,
    jobs: list[PipelineJob],
    system_prompt: str,
    user_prompt: str,
    hyperparameters: GeminiHyperparameters,
    progress_callback: Optional[ProgressCallback] = None,
    cleanup_uploads: bool = True,
) -> list[PipelineResult]:
    client = create_client(api_key)
    download_dir = create_temp_download_dir()
    results: list[PipelineResult] = []

    try:
        for index, job in enumerate(jobs, start=1):
            uploaded_file = None
            try:
                local_video = download_from_drive(
                    job.drive_link,
                    download_dir,
                    filename=f"{job.row_id}.mp4",
                )
                uploaded_file = upload_video(client, local_video, display_name=str(job.row_id))
                caption = generate_caption(
                    client=client,
                    video_path=local_video,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    hyperparameters=hyperparameters,
                    uploaded_file=uploaded_file,
                )
                result = PipelineResult(
                    row_index=job.row_index,
                    row_id=job.row_id,
                    drive_link=job.drive_link,
                    status="success",
                    caption=caption,
                    metadata=job.metadata,
                )
            except Exception as exc:
                result = PipelineResult(
                    row_index=job.row_index,
                    row_id=job.row_id,
                    drive_link=job.drive_link,
                    status="error",
                    error=str(exc),
                    metadata=job.metadata,
                )
            finally:
                if cleanup_uploads and uploaded_file is not None:
                    delete_uploaded_file(client, uploaded_file)

            results.append(result)
            if progress_callback:
                progress_callback(index, len(jobs), result)
    finally:
        for path in download_dir.glob("*"):
            path.unlink(missing_ok=True)
        download_dir.rmdir()

    return results
