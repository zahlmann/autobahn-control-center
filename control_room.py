"""
ASFINAG Webcam Control Room

Live 3x3 grid of highway webcam feeds with pagination.
Run: uv run control_room.py
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import re
import time
import threading
from pathlib import Path

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from openai import AsyncOpenAI
from pydantic import BaseModel

from analyze_feeds import analyze_cameras

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("control_room")

# --- ASFINAG public webcam service params ---
# These are public parameters from the ASFINAG webcam viewer page
# (https://www.asfinag.at/verkehr-sicherheit/webcams/). They are used to
# request a short-lived secToken which authorizes fetching camera frames.
SSID = "7926870e-4c5a-422b-ae3f-3675f744556d"
TOKEN = "IyMjIzUwOSM0MTUjYXRERSNpbmZvI2ZhbHNlI05vdFVzZWQjMzY2NDA4NzU3Nw"
PAGE_URL = "https://webcams2.asfinag.at/webcamviewer/SingleScreenServlet/"
IMAGE_BASE = "https://webcamsservice.asfinag.at/wcsservices/services/image"
CAMERAS_FILE = Path(__file__).parent / "asfinag_cameras.txt"
DATA_DIR = Path(__file__).parent / "data"
PER_PAGE = 6

# --- OpenAI client + pricing (USD per 1M tokens) ---
MODEL_PRICING = {
    "gpt-5.2":    {"input": 1.75, "output": 14.00},
    "gpt-5.1":    {"input": 1.25, "output": 10.00},
    "gpt-5":      {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1":    {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o":     {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3":         {"input": 2.00, "output": 8.00},
    "o4-mini":    {"input": 1.10, "output": 4.40},
}

VISION_MODEL = "gpt-4.1"

openai_client = AsyncOpenAI() if os.environ.get("OPENAI_API_KEY") else None


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

# --- Parse camera list ---
def parse_cameras(path: Path) -> list[dict]:
    cameras = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Camera IDs are indented by 2 spaces, not 4
        if re.match(r"^  [A-Z]", line) and not line.strip().startswith("==="):
            wcsid = line.strip()
            short_code = wcsid.split(",")[0]
            desc_lines = []
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                desc_lines.append(lines[i].strip())
                i += 1
            if not any(short_code.startswith(p) for p in ("HAK-", "BRENNERO-", "DARS-", "ITS-")):
                highway = ""
                direction = ""
                if len(desc_lines) >= 2:
                    highway = desc_lines[1].split(",")[0].strip()
                    m = re.search(r"Blickrichtung\s+(.+?)(?:\s*-\s*Km|\s*$)", desc_lines[1])
                    if m:
                        direction = m.group(1).strip()
                cameras.append({
                    "wcsid": wcsid,
                    "code": short_code,
                    "description": " | ".join(desc_lines),
                    "highway": highway,
                    "direction": direction,
                })
        else:
            i += 1
    return cameras

CAMERAS = parse_cameras(CAMERAS_FILE)

# --- Token cache (single shared secToken, tokens work for any camera) ---
_sec_token: str | None = None
_sec_token_time: float = 0
_token_lock = threading.Lock()
TOKEN_TTL = 120  # seconds before refreshing


def get_sec_token() -> str | None:
    global _sec_token, _sec_token_time
    now = time.time()
    with _token_lock:
        if _sec_token and now - _sec_token_time < TOKEN_TTL:
            return _sec_token

    # Use any camera to fetch a token (tokens are not camera-specific)
    params = {
        "user": "webcamstartseite",
        "wcsid": CAMERAS[0]["wcsid"],
        "ssid": SSID,
        "token": TOKEN,
    }
    try:
        resp = requests.get(PAGE_URL, params=params, timeout=10)
        resp.raise_for_status()
    except Exception:
        return None

    match = re.search(r'name="[^"]*secToken=([^&"]+)', resp.text)
    if not match:
        match = re.search(r'src="https://webcamsservice[^"]*secToken=([^&"]+)', resp.text)
    if not match:
        return None

    with _token_lock:
        _sec_token = match.group(1)
        _sec_token_time = now
        return _sec_token


def fetch_frame(wcsid: str) -> bytes | None:
    sec_token = get_sec_token()
    if not sec_token:
        return None
    params = {
        "modul": "ss",
        "secToken": sec_token,
        "dlink": "webcamstartseite",
        "ratio": "false",
        "wcsid": wcsid,
        "width": "509",
        "height": "415",
        "time": str(time.time()),
    }
    try:
        resp = requests.get(IMAGE_BASE, params=params, timeout=10)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            return resp.content
    except Exception:
        pass
    return None


# --- Filter state ---
_active_filter: dict | None = None
_filter_lock = threading.Lock()

# --- Vision prompt + schema (replaces query planner) ---

VISION_PROMPT = """\
You are filtering Austrian highway webcam images. Answer "yes" if the image matches the query, "no" otherwise.

Query: "{user_query}"

The camera name is prepended above (e.g. "A01, zwischen Wien-Auhof und Pressbaum, Blickrichtung Wien"). Use it:
- If the name contains "Rastplatz", "Raststätte", or "LKW Stellplatz", this is a parking/rest area camera — answer "no" for traffic or road condition queries.
- If the name contains "Parkplatz", same thing — not a road camera.

Image details:
- ASFINAG highway camera snapshot. Top overlay shows air temp, time, road temp. Bottom bar is the ASFINAG logo — ignore it.
- Tunnel cameras have yellow/orange tint from sodium lighting — this is normal.

Rules:
- For traffic queries (stau, traffic, vehicles): only match if the image shows a road with vehicles. Parked vehicles in a lot do not count.
- For weather queries (schnee, snow, rain, nebel, fog, ice): look at road surface, shoulders, and surroundings.
- For broken/offline camera queries: only match genuinely broken feeds (placeholder image with cartoon traffic cone, fully black frame, solid color). Dark nighttime roads with visible lanes or lights are NOT broken.
- If the image is too dark or unclear to confidently judge, answer "no".
"""

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["match"],
    "additionalProperties": False,
}


# --- Background frame fetcher ---

async def _background_fetcher():
    """Fetch all camera frames to disk every 20 seconds."""
    sem = asyncio.Semaphore(300)

    async def _fetch_one(wcsid: str):
        async with sem:
            data = await asyncio.to_thread(fetch_frame, wcsid)
            if data:
                (DATA_DIR / f"{wcsid}.jpg").write_bytes(data)

    while True:
        t0 = time.time()
        await asyncio.gather(
            *[_fetch_one(cam["wcsid"]) for cam in CAMERAS],
            return_exceptions=True,
        )
        elapsed = time.time() - t0
        n_files = sum(1 for _ in DATA_DIR.glob("*.jpg"))
        log.info("Background fetch: %d/%d frames in %.1fs", n_files, len(CAMERAS), elapsed)
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app):
    DATA_DIR.mkdir(exist_ok=True)
    task = asyncio.create_task(_background_fetcher())
    yield
    task.cancel()


# --- App ---
app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    prompt: str
    highway: str | None = None
    direction: str | None = None


def _pre_filter(cameras: list[dict], highway: str | None, direction: str | None) -> list[dict]:
    """Filter cameras by highway and/or direction."""
    result = cameras
    if highway:
        result = [c for c in result if c.get("highway") == highway]
    if direction:
        result = [c for c in result if c.get("direction") == direction]
    return result


@app.get("/api/filter-options")
def api_filter_options(highway: str | None = None, direction: str | None = None):
    """Return unique highways and directions for populating dropdowns."""
    highways = sorted({c["highway"] for c in CAMERAS if c.get("highway")})
    # Narrow directions to selected highway (if any), otherwise show all
    if highway:
        directions = sorted({c["direction"] for c in CAMERAS if c.get("direction") and c.get("highway") == highway})
    else:
        directions = sorted({c["direction"] for c in CAMERAS if c.get("direction")})
    return {"highways": highways, "directions": directions}


@app.get("/api/cameras")
def api_cameras(
    page: int = Query(1, ge=1),
    all: bool = Query(False),
    highway: str | None = None,
    direction: str | None = None,
):
    pre_filtered = _pre_filter(CAMERAS, highway, direction)

    if all:
        return {"cameras": pre_filtered}

    with _filter_lock:
        active = _active_filter

    if active:
        ids = active["camera_ids"]
        if active.get("inverted"):
            filtered = [c for c in pre_filtered if c["wcsid"] not in ids]
        else:
            filtered = [c for c in pre_filtered if c["wcsid"] in ids]
    else:
        filtered = pre_filtered

    total = len(filtered)
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = min(page, total_pages)
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    return {
        "cameras": filtered[start:end],
        "page": page,
        "total_pages": total_pages,
        "total": total,
    }


@app.post("/api/query")
async def api_query(req: QueryRequest):
    if not req.prompt.strip():
        return {"status": "error", "message": "Please provide a filter query."}

    if not openai_client:
        return {"status": "error", "message": "OPENAI_API_KEY not configured."}

    # Pre-filter cameras by highway/direction
    pre_filtered = _pre_filter(CAMERAS, req.highway, req.direction)
    wcsids = {c["wcsid"] for c in pre_filtered}

    log.info("Query received: %s (pre-filtered to %d cameras)", req.prompt, len(wcsids))

    t0 = time.time()

    vision_prompt = VISION_PROMPT.format(user_query=req.prompt)
    try:
        cam_descs = {c["wcsid"]: c["description"] for c in pre_filtered}
        results, vision_usage = await analyze_cameras(
            client=openai_client,
            frames_dir=DATA_DIR,
            prompt=vision_prompt,
            schema=VISION_SCHEMA,
            model=VISION_MODEL,
            wcsids=wcsids if (req.highway or req.direction) else None,
            camera_descriptions=cam_descs,
        )
    except Exception as e:
        log.error("Camera analysis failed: %s", e, exc_info=True)
        return {"status": "error", "message": f"Analysis failed: {e}"}

    elapsed = time.time() - t0

    vision_cost = calc_cost(VISION_MODEL, vision_usage["input_tokens"], vision_usage["output_tokens"])
    log.info(
        "Query cost: vision=%s %din/%dout=$%.4f in %.1fs",
        VISION_MODEL, vision_usage["input_tokens"], vision_usage["output_tokens"],
        vision_cost, elapsed,
    )

    camera_ids = [r["wcsid"] for r in results if r.get("match") == "yes"]

    description = f"Showing {len(camera_ids)} cameras matching '{req.prompt}'"
    with _filter_lock:
        global _active_filter
        _active_filter = {
            "camera_ids": set(camera_ids),
            "query": req.prompt,
            "description": description,
        }

    log.info("Filter applied: %d cameras matched for query: %s", len(camera_ids), req.prompt)

    return {
        "status": "success",
        "filter": {
            "camera_ids": camera_ids,
            "query": req.prompt,
            "description": description,
            "count": len(camera_ids),
        },
        "usage": {
            "model": VISION_MODEL,
            **vision_usage,
            "cost": round(vision_cost, 6),
            "elapsed_seconds": round(elapsed, 1),
        },
    }


@app.get("/api/filter")
def api_filter():
    with _filter_lock:
        if _active_filter:
            return {
                "active": True,
                "query": _active_filter["query"],
                "description": _active_filter["description"],
                "count": len(_active_filter["camera_ids"]),
                "inverted": _active_filter.get("inverted", False),
            }
    return {"active": False}


@app.post("/api/filter/invert")
def api_filter_invert():
    with _filter_lock:
        if not _active_filter:
            return {"status": "error", "message": "No active filter"}
        _active_filter["inverted"] = not _active_filter.get("inverted", False)
        inverted = _active_filter["inverted"]
        count = len(CAMERAS) - len(_active_filter["camera_ids"]) if inverted else len(_active_filter["camera_ids"])
        _active_filter["description"] = (
            f"Showing {count} cameras NOT matching '{_active_filter['query']}'"
            if inverted
            else f"Showing {count} cameras matching '{_active_filter['query']}'"
        )
        return {"status": "ok", "inverted": inverted, "description": _active_filter["description"]}


@app.delete("/api/filter")
def api_filter_delete():
    global _active_filter
    with _filter_lock:
        _active_filter = None
    return {"status": "ok"}


@app.get("/api/frame/{wcsid:path}")
def api_frame(wcsid: str):
    data = fetch_frame(wcsid)
    if data:
        return Response(content=data, media_type="image/jpeg")
    # Return a 1x1 transparent pixel as fallback
    return Response(
        content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82",
        media_type="image/png",
        status_code=502,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autobahn Control Center</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: #111118;
    border-bottom: 1px solid #222;
  }
  header h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #8af;
  }
  .status {
    font-size: 12px;
    color: #666;
  }
  .status .live {
    color: #4f4;
    font-weight: bold;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 4px;
    padding: 4px;
    flex: 1;
    min-height: 0;
  }
  .cell {
    position: relative;
    background: #111;
    border: 1px solid #222;
    overflow: hidden;
    min-height: 0;
  }
  .cell img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .cell .label {
    position: absolute;
    top: 6px;
    left: 6px;
    background: rgba(0,0,0,0.7);
    color: #8af;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 3px;
    letter-spacing: 0.5px;
  }
  .cell .desc {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.8));
    color: #aaa;
    font-size: 10px;
    padding: 16px 8px 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 20ch;
  }
  .cell:hover .desc {
    max-width: none;
    white-space: normal;
  }
  .cell.loading img {
    opacity: 0.3;
  }

  .pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 14px;
    background: #111118;
    border-top: 1px solid #222;
    flex-shrink: 0;
  }
  .pagination button {
    background: #1a1a2e;
    border: 1px solid #333;
    color: #8af;
    padding: 6px 18px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    border-radius: 4px;
  }
  .pagination button:hover:not(:disabled) {
    background: #252540;
  }
  .pagination button:disabled {
    opacity: 0.3;
    cursor: default;
  }
  .pagination .page-info {
    font-size: 13px;
    color: #666;
    min-width: 120px;
    text-align: center;
  }
  .refresh-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #333;
    display: inline-block;
    transition: background 0.3s;
  }
  .refresh-indicator.active {
    background: #4f4;
  }

  .filter-chip {
    display: none;
    align-items: center;
    gap: 6px;
    background: #1a3a1a;
    border: 1px solid #3a6;
    color: #6d6;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 12px;
    margin-left: 12px;
  }
  .filter-chip.visible { display: inline-flex; }
  .filter-chip .clear-btn {
    background: none;
    border: none;
    color: #6d6;
    cursor: pointer;
    font-size: 14px;
    padding: 0 2px;
    line-height: 1;
  }
  .filter-chip .clear-btn:hover { color: #f66; }
  .filter-chip .invert-btn {
    background: none;
    border: 1px solid #6d6;
    color: #6d6;
    cursor: pointer;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    line-height: 1;
    letter-spacing: 0.5px;
  }
  .filter-chip .invert-btn:hover { background: rgba(102,221,102,0.15); }
  .filter-chip.inverted { background: #3a1a1a; border-color: #a66; color: #fa6; }
  .filter-chip.inverted .clear-btn { color: #fa6; }
  .filter-chip.inverted .invert-btn { border-color: #fa6; color: #fa6; }

  .pre-filter-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 24px;
    background: #0e0e14;
    border-bottom: 1px solid #222;
    flex-shrink: 0;
  }
  .pre-filter-bar label {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .pre-filter-bar select {
    background: #1a1a2e;
    border: 1px solid #333;
    color: #e0e0e0;
    font-family: inherit;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    outline: none;
    min-width: 120px;
  }
  .pre-filter-bar select:focus { border-color: #8af; }
  .pre-filter-bar .cam-count {
    font-size: 11px;
    color: #555;
    margin-left: auto;
  }

  .query-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    background: #0e0e14;
    border-bottom: 1px solid #222;
    flex-shrink: 0;
  }
  .query-bar input {
    flex: 1;
    background: #1a1a2e;
    border: 1px solid #333;
    color: #e0e0e0;
    font-family: inherit;
    font-size: 13px;
    padding: 7px 12px;
    border-radius: 4px;
    outline: none;
  }
  .query-bar input:focus { border-color: #8af; }
  .query-bar input:disabled { opacity: 0.5; }
  .query-bar button {
    background: #1a2a4e;
    border: 1px solid #3a5a9a;
    color: #8af;
    padding: 7px 18px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    border-radius: 4px;
    white-space: nowrap;
  }
  .query-bar button:hover:not(:disabled) { background: #253a6a; }
  .query-bar button:disabled { opacity: 0.4; cursor: default; }

  .spinner {
    display: none;
    width: 18px;
    height: 18px;
    border: 2px solid #333;
    border-top-color: #8af;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .spinner.visible { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .toast {
    position: fixed;
    bottom: 70px;
    left: 50%;
    transform: translateX(-50%);
    background: #3a1a1a;
    border: 1px solid #a33;
    color: #f88;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 13px;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    z-index: 100;
  }
  .toast.visible { opacity: 1; pointer-events: auto; }
</style>
</head>
<body>
<header>
  <h1>Autobahn Control Center</h1>
  <div class="status">
    <span class="refresh-indicator" id="indicator"></span>
    <span class="live">LIVE</span>
    <span id="cam-count"></span>
    <span class="filter-chip" id="filter-chip">
      <span id="filter-text"></span>
      <button class="invert-btn" onclick="invertFilter()" title="Invert filter">INV</button>
      <button class="clear-btn" onclick="clearFilter()" title="Clear filter">&times;</button>
    </span>
  </div>
</header>

<div class="pre-filter-bar">
  <label>Highway</label>
  <select id="filter-highway" onchange="applyPreFilter()">
    <option value="">All</option>
  </select>
  <label>Direction</label>
  <select id="filter-direction" onchange="applyPreFilter()">
    <option value="">All</option>
  </select>
  <span class="cam-count" id="pre-filter-count"></span>
</div>

<div class="query-bar">
  <input type="text" id="query-input" placeholder="Filter cameras... (e.g. traffic jam, snow, broken camera)"
         onkeydown="if(event.key==='Enter')submitQuery()">
  <div class="spinner" id="query-spinner"></div>
  <button id="query-btn" onclick="submitQuery()">Analyze</button>
</div>

<div class="grid" id="grid"></div>

<div class="pagination">
  <button id="btn-first" onclick="goPage(1)">&laquo;</button>
  <button id="btn-prev" onclick="goPage(currentPage-1)">&lsaquo; Prev</button>
  <div class="page-info" id="page-info">-</div>
  <button id="btn-next" onclick="goPage(currentPage+1)">Next &rsaquo;</button>
  <button id="btn-last" onclick="goPage(totalPages)">&raquo;</button>
</div>

<div class="toast" id="toast"></div>

<script>
let currentPage = 1;
let totalPages = 1;
let cameras = [];
let refreshTimer = null;
const REFRESH_MS = 500;
let preFilterHighway = '';
let preFilterDirection = '';

async function loadFilterOptions() {
  const params = new URLSearchParams();
  if (preFilterHighway) params.set('highway', preFilterHighway);
  if (preFilterDirection) params.set('direction', preFilterDirection);
  const res = await fetch(`/api/filter-options?${params}`);
  const data = await res.json();

  document.getElementById('filter-highway').innerHTML = '<option value="">All</option>' +
    data.highways.map(h => `<option value="${h}"${h === preFilterHighway ? ' selected' : ''}>${h}</option>`).join('');
  document.getElementById('filter-direction').innerHTML = '<option value="">All</option>' +
    data.directions.map(d => `<option value="${d}"${d === preFilterDirection ? ' selected' : ''}>${d}</option>`).join('');
}

function applyPreFilter() {
  preFilterHighway = document.getElementById('filter-highway').value;
  preFilterDirection = document.getElementById('filter-direction').value;
  loadFilterOptions();
  loadPage(1);
}

async function loadPage(page) {
  const params = new URLSearchParams({page});
  if (preFilterHighway) params.set('highway', preFilterHighway);
  if (preFilterDirection) params.set('direction', preFilterDirection);
  const res = await fetch(`/api/cameras?${params}`);
  const data = await res.json();
  cameras = data.cameras;
  currentPage = data.page;
  totalPages = data.total_pages;

  document.getElementById('cam-count').textContent = `${data.total} cameras`;
  document.getElementById('pre-filter-count').textContent =
    (preFilterHighway || preFilterDirection) ? `${data.total} cameras` : '';
  document.getElementById('page-info').textContent = `Page ${currentPage} / ${totalPages}`;
  document.getElementById('btn-prev').disabled = currentPage <= 1;
  document.getElementById('btn-first').disabled = currentPage <= 1;
  document.getElementById('btn-next').disabled = currentPage >= totalPages;
  document.getElementById('btn-last').disabled = currentPage >= totalPages;

  renderGrid();
  refreshFrames();
}

function renderGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  cameras.forEach((cam, i) => {
    const cell = document.createElement('div');
    cell.className = 'cell loading';
    cell.id = `cell-${i}`;
    cell.innerHTML = `
      <img id="img-${i}" src="" alt="${cam.code}">
      <div class="label">${cam.code}</div>
      <div class="desc">${cam.description}</div>
    `;
    grid.appendChild(cell);
  });
}

async function refreshFrames() {
  if (refreshTimer) clearTimeout(refreshTimer);

  const indicator = document.getElementById('indicator');
  indicator.classList.add('active');

  const promises = cameras.map((cam, i) => {
    const url = `/api/frame/${encodeURIComponent(cam.wcsid)}?t=${Date.now()}`;
    return fetch(url).then(res => {
      if (res.ok) return res.blob();
      return null;
    }).then(blob => {
      if (blob) {
        const img = document.getElementById(`img-${i}`);
        const cell = document.getElementById(`cell-${i}`);
        if (img && cell) {
          const oldSrc = img.src;
          img.src = URL.createObjectURL(blob);
          if (oldSrc.startsWith('blob:')) URL.revokeObjectURL(oldSrc);
          cell.classList.remove('loading');
        }
      }
    }).catch(() => {});
  });

  await Promise.all(promises);

  setTimeout(() => indicator.classList.remove('active'), 300);
  refreshTimer = setTimeout(refreshFrames, REFRESH_MS);
}

function goPage(page) {
  if (page < 1 || page > totalPages) return;
  loadPage(page);
}

// Keyboard navigation (skip when input focused)
document.addEventListener('keydown', (e) => {
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') goPage(currentPage - 1);
  if (e.key === 'ArrowRight') goPage(currentPage + 1);
});

// --- Query / Filter ---
async function submitQuery() {
  const input = document.getElementById('query-input');
  const btn = document.getElementById('query-btn');
  const spinner = document.getElementById('query-spinner');
  const prompt = input.value.trim();
  if (!prompt) return;

  input.disabled = true;
  btn.disabled = true;
  spinner.classList.add('visible');

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, highway: preFilterHighway || undefined, direction: preFilterDirection || undefined}),
    });
    const data = await res.json();
    if (data.status === 'success') {
      showFilterChip(data.filter.description || `${data.filter.count} cameras`);
      input.value = '';
      loadPage(1);
    } else {
      showToast(data.message || 'Analysis failed.');
    }
  } catch (err) {
    showToast('Connection error. Try again.');
  } finally {
    input.disabled = false;
    btn.disabled = false;
    spinner.classList.remove('visible');
  }
}

async function clearFilter() {
  await fetch('/api/filter', {method: 'DELETE'});
  hideFilterChip();
  loadPage(1);
}

async function invertFilter() {
  const res = await fetch('/api/filter/invert', {method: 'POST'});
  const data = await res.json();
  if (data.status === 'ok') {
    showFilterChip(data.description, data.inverted);
    loadPage(1);
  }
}

function showFilterChip(text, inverted) {
  const chip = document.getElementById('filter-chip');
  document.getElementById('filter-text').textContent = text;
  chip.classList.add('visible');
  chip.classList.toggle('inverted', !!inverted);
}

function hideFilterChip() {
  const chip = document.getElementById('filter-chip');
  chip.classList.remove('visible', 'inverted');
}

let toastTimer = null;
function showToast(message) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.classList.add('visible');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('visible'), 5000);
}

async function checkFilter() {
  try {
    const res = await fetch('/api/filter');
    const data = await res.json();
    if (data.active) {
      showFilterChip(data.description || `${data.count} cameras`, data.inverted);
    }
  } catch {}
}

checkFilter();
loadFilterOptions();
loadPage(1);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    log.info("Loaded %d cameras (%d pages)", len(CAMERAS), (len(CAMERAS) + PER_PAGE - 1) // PER_PAGE)
    log.info("Starting at http://localhost:8050")
    uvicorn.run(app, host="0.0.0.0", port=8050)
