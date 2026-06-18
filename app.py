"""Streamlit app for batch video captioning with Gemini 3.1 Pro Preview."""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.gemini_client import (
    DEFAULT_MODEL,
    MEDIA_RESOLUTIONS,
    MODEL_VARIANTS,
    THINKING_LEVELS,
    GeminiHyperparameters,
)
from src.pipeline import (
    PipelineResult,
    detect_id_column,
    detect_link_column,
    load_jobs_from_csv,
    results_to_csv,
    results_to_dataframe,
    run_captioning_pipeline,
    save_run_artifact,
)

DEFAULT_SYSTEM_PROMPT = """You are an expert video captioning assistant.

Watch the provided video carefully and produce a high-quality caption that:
- Describes the main visual content, actions, and context
- Notes important on-screen text, speech, or audio when relevant
- Uses clear, concise language suitable for accessibility or dataset labeling
- Avoids speculation beyond what is visible or audible in the video

Return only the caption text unless the user prompt asks for a specific format."""

DEFAULT_USER_PROMPT = (
    "Generate a detailed caption for this video following the system instructions."
)

OUTPUT_DIR = Path("outputs")

load_dotenv()


def get_api_key() -> str:
    if st.session_state.get("ui_api_key"):
        return st.session_state["ui_api_key"]
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def init_session_state() -> None:
    defaults = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "csv_bytes": None,
        "csv_filename": "",
        "csv_info": None,
        "jobs": [],
        "results": [],
        "last_run_dir": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> GeminiHyperparameters:
    st.sidebar.header("Gemini API")

    env_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            env_key = st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        pass
    if not env_key:
        env_key = os.getenv("GEMINI_API_KEY", "")

    ui_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=env_key,
        placeholder="Paste your Gemini 3.1 Pro Preview API key",
        help="Get one at https://aistudio.google.com/apikey",
    )
    st.session_state["ui_api_key"] = ui_key

    if ui_key:
        st.sidebar.success("API key set.")
    else:
        st.sidebar.warning("Enter your Gemini API key above to proceed.")

    st.sidebar.header("Model Hyperparameters")

    model = st.sidebar.selectbox(
        "Model",
        MODEL_VARIANTS,
        index=MODEL_VARIANTS.index(DEFAULT_MODEL),
        help="Use the customtools variant for agentic workflows with custom tools.",
    )
    thinking_level = st.sidebar.selectbox(
        "Thinking level",
        THINKING_LEVELS,
        index=THINKING_LEVELS.index("high"),
        help="Controls reasoning depth. `high` is the Gemini 3 default.",
    )
    media_resolution = st.sidebar.selectbox(
        "Media resolution",
        MEDIA_RESOLUTIONS,
        index=MEDIA_RESOLUTIONS.index("media_resolution_low"),
        help="Use `high` for text-heavy videos; `low`/`medium` are usually enough for captioning.",
    )
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Gemini 3 recommends keeping this at 1.0.",
    )
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=100, value=40, step=1)
    max_output_tokens = st.sidebar.number_input(
        "Max output tokens",
        min_value=256,
        max_value=65536,
        value=8192,
        step=256,
    )
    include_thoughts = st.sidebar.checkbox(
        "Include thoughts in response",
        value=False,
        help="When enabled, the model may return reasoning traces along with the caption.",
    )
    stop_sequences_raw = st.sidebar.text_input(
        "Stop sequences (comma-separated)",
        value="",
        help="Optional strings that stop generation when encountered.",
    )
    response_mime_type = st.sidebar.selectbox(
        "Response MIME type",
        ["text/plain", "application/json"],
        index=0,
    )

    stop_sequences = [
        item.strip()
        for item in stop_sequences_raw.split(",")
        if item.strip()
    ]

    return GeminiHyperparameters(
        model=model,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_output_tokens=int(max_output_tokens),
        thinking_level=thinking_level,
        include_thoughts=include_thoughts,
        media_resolution=media_resolution,
        stop_sequences=stop_sequences,
        response_mime_type=None if response_mime_type == "text/plain" else response_mime_type,
    )


def render_prompt_section() -> str:
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=220,
        help="Instructions that define caption style, format, and constraints.",
    )
    st.session_state.system_prompt = system_prompt
    return system_prompt.strip()


def render_csv_section() -> None:
    st.subheader("Video CSV Input")
    st.markdown(
        "Upload a CSV containing Google Drive links. "
        "Links must be shared as **Anyone with the link can view**."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Expected columns include a drive link column such as `drive_link` or `video_url`.",
    )

    with st.expander("Example CSV format"):
        example = pd.DataFrame(
            {
                "video_id": ["clip_001", "clip_002"],
                "drive_link": [
                    "https://drive.google.com/file/d/FILE_ID_1/view?usp=sharing",
                    "https://drive.google.com/file/d/FILE_ID_2/view?usp=sharing",
                ],
            }
        )
        st.dataframe(example, use_container_width=True)
        st.download_button(
            "Download example CSV",
            data=example.to_csv(index=False),
            file_name="example_videos.csv",
            mime="text/csv",
        )

    if uploaded is None:
        return

    csv_bytes = uploaded.getvalue()
    st.session_state.csv_bytes = csv_bytes
    st.session_state.csv_filename = uploaded.name

    preview_df = pd.read_csv(io.BytesIO(csv_bytes))
    st.write(f"Preview of `{uploaded.name}`")
    st.dataframe(preview_df.head(20), use_container_width=True)

    columns = [str(col) for col in preview_df.columns]
    default_link = detect_link_column(columns) or columns[0]
    default_id = detect_id_column(columns, default_link)

    col1, col2 = st.columns(2)
    with col1:
        link_column = st.selectbox("Drive link column", columns, index=columns.index(default_link))
    with col2:
        id_options = ["Auto"] + columns
        id_default_index = id_options.index(default_id) if default_id in id_options else 0
        id_choice = st.selectbox("ID column", id_options, index=id_default_index)

    id_column = None if id_choice == "Auto" else id_choice

    try:
        jobs, info = load_jobs_from_csv(csv_bytes, link_column=link_column, id_column=id_column)
        st.session_state.jobs = jobs
        st.session_state.csv_info = info
        st.success(
            f"Found **{info['valid_jobs']}** valid video links out of **{info['total_rows']}** rows."
        )
    except ValueError as exc:
        st.session_state.jobs = []
        st.session_state.csv_info = None
        st.error(str(exc))


def _auto_download_csv(csv_string: str, filename: str) -> None:
    """Inject a small JS snippet that triggers a browser download automatically."""
    b64 = base64.b64encode(csv_string.encode()).decode()
    js = f"""
    <script>
    (function() {{
        var link = document.createElement('a');
        link.href = 'data:text/csv;base64,{b64}';
        link.download = '{filename}';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


def render_run_section(
    hyperparameters: GeminiHyperparameters,
    system_prompt: str,
) -> None:
    st.subheader("Run Captioning Pipeline")

    api_key = get_api_key()
    if not api_key:
        st.info("Enter your Gemini API key in the sidebar before running.")
        return

    if not system_prompt:
        st.warning("System prompt cannot be empty.")
        return

    if not st.session_state.jobs:
        st.warning("Upload a CSV with valid Google Drive links first.")
        return

    job_count = len(st.session_state.jobs)
    st.write(f"Ready to process **{job_count}** videos with `{hyperparameters.model}`.")

    if st.button("Generate Captions", type="primary", use_container_width=True):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        incremental_csv = OUTPUT_DIR / f"captions_live_{timestamp}.csv"

        progress_bar = st.progress(0.0)
        status_box = st.empty()
        live_table_placeholder = st.empty()
        live_captions_container = st.container()
        caption_placeholders: dict[str, object] = {}

        collected_results: list[PipelineResult] = []

        def on_progress(current: int, total: int, result: PipelineResult) -> None:
            collected_results.append(result)

            progress_bar.progress(current / total)
            status_box.write(
                f"Processing {current}/{total}: `{result.row_id}` — **{result.status}**"
            )

            live_table_placeholder.dataframe(
                results_to_dataframe(collected_results),
                use_container_width=True,
            )

            with live_captions_container:
                if result.row_id not in caption_placeholders:
                    caption_placeholders[result.row_id] = st.empty()

                ph = caption_placeholders[result.row_id]
                if result.status == "success":
                    ph.success(f"**{result.row_id}**\n\n{result.caption}")
                else:
                    ph.error(f"**{result.row_id}** — {result.error}")

        with st.spinner("Running captioning pipeline..."):
            results = run_captioning_pipeline(
                api_key=api_key,
                jobs=st.session_state.jobs,
                system_prompt=system_prompt,
                user_prompt=DEFAULT_USER_PROMPT,
                hyperparameters=hyperparameters,
                progress_callback=on_progress,
                incremental_csv_path=incremental_csv,
            )

        st.session_state.results = results
        run_dir = save_run_artifact(
            OUTPUT_DIR,
            system_prompt=system_prompt,
            user_prompt=DEFAULT_USER_PROMPT,
            hyperparameters=hyperparameters,
            results=results,
        )
        st.session_state.last_run_dir = str(run_dir)

        success_count = sum(1 for r in results if r.status == "success")
        st.success(f"Finished: {success_count}/{len(results)} captions generated successfully.")

        final_csv = results_to_csv(results)
        st.session_state["final_csv"] = final_csv
        _auto_download_csv(final_csv, "captions.csv")
        st.toast("CSV downloaded automatically!", icon="\u2705")


def render_results_section() -> None:
    st.subheader("Results")

    if not st.session_state.results:
        st.caption("Run the pipeline to see generated captions here.")
        return

    results_df = results_to_dataframe(st.session_state.results)
    st.dataframe(results_df, use_container_width=True)

    csv_data = st.session_state.get("final_csv") or results_to_csv(st.session_state.results)
    st.download_button(
        "Download captions CSV",
        data=csv_data,
        file_name="captions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Inspect individual captions"):
        for result in st.session_state.results:
            if result.status == "success":
                st.success(f"**{result.row_id}**\n\n{result.caption}")
            else:
                st.error(f"**{result.row_id}** — {result.error}")
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Video Captioning Pipeline",
        page_icon="🎬",
        layout="wide",
    )

    init_session_state()
    hyperparameters = render_sidebar()

    st.title("Video Captioning Pipeline")
    st.caption("Batch caption videos from Google Drive links using Gemini 3.1 Pro Preview.")

    system_prompt = render_prompt_section()
    render_csv_section()
    render_run_section(hyperparameters, system_prompt)
    render_results_section()


if __name__ == "__main__":
    main()
