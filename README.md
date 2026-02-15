# Autobahn Control Center

Live Austrian highway camera grid with AI-powered filtering.

## Quick start

```bash
git clone <repo-url> && cd autobahn-control-center
cp .env.example .env
# paste your OpenAI API key into .env
uv run control_room.py
# open http://localhost:8050
```

## How it works

Type a query like "stau" (traffic jam) or "schnee" (snow) into the filter bar. GPT-4.1 vision scans all camera frames and the grid filters down to matching cameras.

Use the highway and direction dropdowns to narrow the scope before running an AI query.

## Architecture

- **`control_room.py`** — FastAPI server + embedded UI. Serves the camera grid, handles frame proxying, and runs AI queries via the OpenAI API (GPT-4.1 vision).
- **`analyze_feeds.py`** — Async vision analysis module. Sends camera frames to GPT-4.1 for categorization. Also works as a standalone CLI tool.
- **`asfinag_cameras.txt`** — Camera list parsed at startup.

## Data source

Austrian Autobahn cameras via the [ASFINAG](https://www.asfinag.at/) public webcam service.
