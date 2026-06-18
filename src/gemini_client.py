"""Gemini 3.1 Pro Preview client for video captioning."""

from __future__ import annotations

import logging
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MODEL_VARIANTS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
]

THINKING_LEVELS = ["low", "medium", "high"]
MEDIA_RESOLUTIONS = [
    "media_resolution_low",
    "media_resolution_medium",
    "media_resolution_high",
]


@dataclass
class GeminiHyperparameters:
    model: str = DEFAULT_MODEL
    use_defaults: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None
    thinking_level: Optional[str] = None
    include_thoughts: bool = False
    media_resolution: str = "media_resolution_low"
    stop_sequences: list[str] = field(default_factory=list)
    response_mime_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "use_defaults": self.use_defaults,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "thinking_budget": self.thinking_budget,
            "thinking_level": self.thinking_level,
            "include_thoughts": self.include_thoughts,
            "media_resolution": self.media_resolution,
            "stop_sequences": self.stop_sequences,
            "response_mime_type": self.response_mime_type,
        }


def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "video/mp4"


def _media_resolution_level(resolution: str) -> types.PartMediaResolutionLevel:
    mapping = {
        "media_resolution_low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
        "media_resolution_medium": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
        "media_resolution_high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    }
    return mapping.get(
        resolution,
        types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    )


def upload_video(
    client: genai.Client,
    video_path: Path,
    display_name: Optional[str] = None,
    poll_interval_seconds: int = 5,
    max_wait_seconds: int = 600,
) -> types.File:
    """Upload a local video to the Gemini Files API and wait until ACTIVE."""
    uploaded = client.files.upload(
        file=str(video_path),
        config={"display_name": display_name or video_path.name},
    )

    elapsed = 0
    while uploaded.state and uploaded.state.name != "ACTIVE":
        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Video upload failed for {video_path.name}")

        if elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"Timed out waiting for video processing: {video_path.name}"
            )

        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds
        uploaded = client.files.get(name=uploaded.name)

    return uploaded


def build_generation_config(
    hyperparameters: GeminiHyperparameters,
    system_prompt: str = "",
) -> types.GenerateContentConfig:
    config_kwargs: dict[str, Any] = {
        "media_resolution": hyperparameters.media_resolution,
    }

    if not hyperparameters.use_defaults:
        if hyperparameters.temperature is not None:
            config_kwargs["temperature"] = hyperparameters.temperature
        if hyperparameters.top_p is not None:
            config_kwargs["top_p"] = hyperparameters.top_p
        if hyperparameters.top_k is not None:
            config_kwargs["top_k"] = hyperparameters.top_k
        if hyperparameters.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = hyperparameters.max_output_tokens

        thinking_kwargs: dict[str, Any] = {
            "include_thoughts": hyperparameters.include_thoughts,
        }
        if hyperparameters.thinking_level is not None:
            thinking_kwargs["thinking_level"] = hyperparameters.thinking_level
        if hyperparameters.thinking_budget is not None:
            thinking_kwargs["thinking_budget"] = hyperparameters.thinking_budget
        config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

        if hyperparameters.stop_sequences:
            config_kwargs["stop_sequences"] = hyperparameters.stop_sequences
        if hyperparameters.response_mime_type:
            config_kwargs["response_mime_type"] = hyperparameters.response_mime_type

    if system_prompt.strip():
        config_kwargs["system_instruction"] = system_prompt.strip()

    return types.GenerateContentConfig(**config_kwargs)


def _extract_text_from_response(response: types.GenerateContentResponse) -> str:
    """Extract all non-thought text parts from the response candidates."""
    text_parts: list[str] = []
    if not response.candidates:
        return ""
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if part.thought:
                continue
            if part.text:
                text_parts.append(part.text)
    return "\n".join(text_parts)


def generate_caption(
    client: genai.Client,
    video_path: Path,
    system_prompt: str,
    user_prompt: str,
    hyperparameters: GeminiHyperparameters,
    uploaded_file: Optional[types.File] = None,
) -> str:
    """Generate a caption for a single video using Gemini 3.1 Pro Preview."""
    video_file = uploaded_file or upload_video(client, video_path)
    mime_type = _guess_mime_type(video_path)

    video_part = types.Part(
        file_data=types.FileData(
            file_uri=video_file.uri,
            mime_type=mime_type,
        ),
        media_resolution=types.PartMediaResolution(
            level=_media_resolution_level(hyperparameters.media_resolution),
        ),
    )

    try:
        response = client.models.generate_content(
            model=hyperparameters.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        video_part,
                        types.Part(text=user_prompt),
                    ],
                )
            ],
            config=build_generation_config(hyperparameters, system_prompt=system_prompt),
        )
    except Exception as api_error:
        raise RuntimeError(f"Gemini API error: {api_error}") from api_error

    finish_reason = None
    finish_message = None
    safety_ratings = None
    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        finish_message = candidate.finish_message
        safety_ratings = candidate.safety_ratings

    prompt_feedback = response.prompt_feedback

    if finish_reason and finish_reason.name not in ("STOP", "MAX_TOKENS"):
        parts = [f"Gemini refused to generate. finish_reason={finish_reason.name}"]
        if finish_message:
            parts.append(f"message={finish_message}")
        if safety_ratings:
            ratings_str = ", ".join(
                f"{r.category.name}={r.probability.name}"
                for r in safety_ratings
                if hasattr(r, "category") and hasattr(r, "probability")
            )
            if ratings_str:
                parts.append(f"safety_ratings=[{ratings_str}]")
        if prompt_feedback:
            parts.append(f"prompt_feedback={prompt_feedback}")
        raise RuntimeError(" | ".join(parts))

    caption = _extract_text_from_response(response)

    usage = response.usage_metadata
    usage_str = ""
    if usage:
        usage_str = (
            f" (prompt_tokens={usage.prompt_token_count},"
            f" output_tokens={usage.candidates_token_count},"
            f" thoughts_tokens={getattr(usage, 'thoughts_token_count', 'N/A')})"
        )

    if not caption:
        detail = ""
        if finish_reason:
            detail += f" finish_reason={finish_reason.name}"
        detail += usage_str
        raise RuntimeError(f"Gemini returned an empty response.{detail}")

    if finish_reason and finish_reason.name == "MAX_TOKENS":
        log.warning(
            "Response for %s was truncated (MAX_TOKENS).%s",
            video_path.name,
            usage_str,
        )
        caption += "\n\n[TRUNCATED — increase max output tokens]"

    return caption.strip()


def delete_uploaded_file(client: genai.Client, uploaded_file: types.File) -> None:
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass
