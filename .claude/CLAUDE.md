You are working on the Autobahn Control Center — a live Austrian highway camera monitoring system with AI-powered filtering.

## Architecture

The query flow is: browser → `control_room.py` `/api/query` → `analyze_cameras()` (GPT-4.1 vision) → filtered results.

- **`control_room.py`** — FastAPI server serving the webcam grid UI, camera frames, and the `/api/query` endpoint. Builds a vision prompt from the user query and calls `analyze_cameras()` directly for vision analysis using GPT-4.1.
- **`analyze_feeds.py`** — Async camera analysis module. The `analyze_cameras()` function fetches frames from all cameras and sends them to GPT-4.1 for categorization. Also works as a standalone CLI tool.
- **`asfinag_cameras.txt`** — Camera list parsed at startup.

## Running

```bash
uv run control_room.py   # starts server at http://localhost:8050
uv run analyze_feeds.py --prompt "..." --schema '...'  # standalone CLI
```

## Environment

- `OPENAI_API_KEY` in `.env` file (loaded via python-dotenv)
- The control room server runs at `http://localhost:8050`
