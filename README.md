# Video Captioning Pipeline

A Streamlit app for batch video captioning using **Gemini 3.1 Pro Preview**. Upload a CSV of Google Drive video links, configure the system prompt and model hyperparameters, and export generated captions as CSV.

## Features

- System prompt and per-video user prompt configuration
- Gemini 3.1 Pro Preview hyperparameters:
  - Model variant (`gemini-3.1-pro-preview` or `customtools`)
  - Thinking level (`low`, `medium`, `high`)
  - Media resolution for video frames
  - Temperature, top-p, top-k, max output tokens
  - Optional stop sequences and JSON response mode
- CSV upload with automatic drive-link column detection
- Downloads public/shared Google Drive videos, uploads to Gemini Files API, generates captions
- Downloadable results CSV and saved run artifacts under `outputs/`

## Prerequisites

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey)
- Google Drive links shared as **Anyone with the link can view**

## Local setup

```bash
cd "/Users/nishitverma/Desktop/Captioning Tasks"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or create a local secrets file at `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Run the app:

```bash
streamlit run app.py
```

## CSV format

Your CSV should include a column with Google Drive links. Column names like `drive_link`, `video_url`, or `link` are auto-detected.

Example:

| video_id | drive_link | category |
|----------|------------|----------|
| clip_001 | https://drive.google.com/file/d/FILE_ID_1/view | sports |
| clip_002 | https://drive.google.com/file/d/FILE_ID_2/view | cooking |

Optional metadata columns are preserved in the output CSV.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10+
4. Add a secret in the app settings:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

5. Deploy and share the public URL with your team.

## How it works

1. Parse the uploaded CSV and extract Google Drive links.
2. Download each video locally from Drive.
3. Upload each video to the Gemini Files API and wait until processing completes.
4. Call `gemini-3.1-pro-preview` with:
   - Your system prompt
   - The video
   - Your user prompt
   - Selected hyperparameters
5. Save captions to CSV and store run config under `outputs/run_<timestamp>/`.

## Notes

- Large videos may take several minutes to upload and process.
- Gemini Files API uploads expire after 48 hours; uploaded files are deleted after each run when cleanup is enabled.
- For private Drive folders, ensure links are shared publicly or switch to a service-account based download flow in a future iteration.

## Project structure

```
app.py                 # Streamlit UI
requirements.txt
src/
  drive_utils.py       # Google Drive download helpers
  gemini_client.py     # Gemini API wrapper
  pipeline.py          # Batch captioning pipeline
outputs/               # Saved run artifacts (created at runtime)
```
