"""
ASFINAG camera feed analysis using OpenAI Vision API.

Reads cached camera frames from disk (populated by control_room.py background
fetcher) and sends each individually to GPT for categorization.

Run: uv run analyze_feeds.py --prompt "..." --schema '...'
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("analyze_feeds")


def load_frames(frames_dir: Path) -> dict[str, bytes]:
    """Load all cached camera frames from disk."""
    frames = {}
    for path in frames_dir.glob("*.jpg"):
        frames[path.stem] = path.read_bytes()
    return frames


async def analyze_single(
    client: AsyncOpenAI,
    wcsid: str,
    frame_data: bytes,
    prompt: str,
    schema: dict,
    model: str,
) -> dict:
    """Send a single camera frame to GPT for categorization."""
    b64 = base64.b64encode(frame_data).decode()
    response = await client.responses.create(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
        ]}],
        text={"format": {"type": "json_schema", "name": "camera_analysis", "strict": True, "schema": schema}},
        temperature=0.5,
    )
    result = json.loads(response.output_text)
    return {
        **result,
        "wcsid": wcsid,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


async def analyze_cameras(
    *,
    client: AsyncOpenAI,
    frames_dir: Path,
    prompt: str,
    schema: dict,
    model: str = "gpt-4.1",
    wcsids: set[str] | None = None,
    camera_descriptions: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """Analyze all cameras and return (results, usage).

    This is the main reusable entry point, called both from the CLI
    and from control_room.py in-process.
    """
    frames = load_frames(frames_dir)
    if wcsids is not None:
        frames = {k: v for k, v in frames.items() if k in wcsids}
    log.info("Loaded %d frames from %s", len(frames), frames_dir)

    if not frames:
        log.warning("No frames found in %s", frames_dir)
        return [], {"input_tokens": 0, "output_tokens": 0}

    t0 = time.time()
    sem = asyncio.Semaphore(100)

    async def process_one(wcsid: str, frame_data: bytes) -> dict | None:
        async with sem:
            try:
                cam_prompt = prompt
                if camera_descriptions and wcsid in camera_descriptions:
                    cam_prompt = f"Camera: {camera_descriptions[wcsid]}\n\n{prompt}"
                result = await analyze_single(client, wcsid, frame_data, cam_prompt, schema, model)
                return result
            except Exception as e:
                log.error("Camera %s: OpenAI API error: %s", wcsid, e, exc_info=True)
                return None

    results = await asyncio.gather(
        *[process_one(wcsid, data) for wcsid, data in frames.items()]
    )

    all_results = [r for r in results if r is not None]
    total_in = sum(r["input_tokens"] for r in all_results)
    total_out = sum(r["output_tokens"] for r in all_results)
    elapsed = time.time() - t0
    log.info("Done: %d/%d camera results in %.1fs (%d in / %d out tokens)",
             len(all_results), len(frames), elapsed, total_in, total_out)

    # Strip token counts from results before returning
    for r in all_results:
        del r["input_tokens"]
        del r["output_tokens"]

    return all_results, {"input_tokens": total_in, "output_tokens": total_out}


async def async_main(args):
    if not os.environ.get("OPENAI_API_KEY"):
        json.dump({"status": "error", "message": "OPENAI_API_KEY not set"}, sys.stdout)
        sys.exit(1)

    try:
        schema = json.loads(args.schema)
    except json.JSONDecodeError as e:
        json.dump({"status": "error", "message": f"Invalid schema JSON: {e}"}, sys.stdout)
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        json.dump({"status": "error", "message": f"Frames directory not found: {frames_dir}"}, sys.stdout)
        sys.exit(1)

    client = AsyncOpenAI()

    try:
        results, usage = await analyze_cameras(
            client=client,
            frames_dir=frames_dir,
            prompt=args.prompt,
            schema=schema,
        )
    except Exception as e:
        json.dump({"status": "error", "message": str(e)}, sys.stdout)
        sys.exit(1)

    json.dump({"cameras": results, "usage": usage}, sys.stdout)


def main():
    parser = argparse.ArgumentParser(description="Analyze ASFINAG camera feeds with OpenAI Vision")
    parser.add_argument("--frames-dir", default="data", help="Directory containing cached frame JPEGs")
    parser.add_argument("--prompt", required=True, help="Vision categorization prompt")
    parser.add_argument("--schema", required=True, help="JSON schema string for structured output")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
