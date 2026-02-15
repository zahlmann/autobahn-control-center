"""
ASFINAG Webcam Continuous Fetcher

Fetches webcam snapshots from ASFINAG's highway webcam service.
The official viewer limits sessions to ~20 seconds; this script
bypasses that by directly polling the image endpoint.

Usage:
    uv run --with requests --with opencv-python-headless asfinag_webcam.py [--interval 0.5] [--output ./frames] [--duration 60] [--wcsid CAMERA_ID]
"""

import argparse
import glob
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import requests

# Default camera: A02 Süd - Knoten Guntramsdorf
DEFAULT_WCSID = "A023_010,535_0_00113785"
DEFAULT_SSID = "7926870e-4c5a-422b-ae3f-3675f744556d"
DEFAULT_TOKEN = "IyMjIzUwOSM0MTUjYXRERSNpbmZvI2ZhbHNlI05vdFVzZWQjMzY2NDA4NzU3Nw"

PAGE_URL = "https://webcams2.asfinag.at/webcamviewer/SingleScreenServlet/"
IMAGE_BASE = "https://webcamsservice.asfinag.at/wcsservices/services/image"


def get_sec_token(wcsid: str, ssid: str, token: str) -> str | None:
    """Fetch the viewer page and extract a fresh secToken."""
    params = {
        "user": "webcamstartseite",
        "wcsid": wcsid,
        "ssid": ssid,
        "token": token,
    }
    resp = requests.get(PAGE_URL, params=params, timeout=10)
    resp.raise_for_status()

    # Extract the secToken from the img name attribute (used for polling)
    match = re.search(r'name="[^"]*secToken=([^&"]+)', resp.text)
    if match:
        return match.group(1)

    # Fallback: try src attribute
    match = re.search(r'src="https://webcamsservice[^"]*secToken=([^&"]+)', resp.text)
    if match:
        return match.group(1)

    return None


def fetch_frame(sec_token: str, wcsid: str) -> bytes | None:
    """Fetch a single JPEG frame from the webcam."""
    params = {
        "modul": "ss",
        "secToken": sec_token,
        "dlink": "webcamstartseite",
        "ratio": "false",
        "wcsid": wcsid,
        "width": "509",
        "height": "415",
        "time": str(time.time()),  # cache buster
    }
    resp = requests.get(IMAGE_BASE, params=params, timeout=10)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
        return resp.content
    return None


def main():
    parser = argparse.ArgumentParser(description="Continuously fetch ASFINAG webcam images")
    parser.add_argument("--interval", type=float, default=5, help="Seconds between fetches (default: 5)")
    parser.add_argument("--output", type=str, default="./asfinag_frames", help="Output directory for frames")
    parser.add_argument("--duration", type=int, default=0, help="Total seconds to run (0 = indefinite)")
    parser.add_argument("--wcsid", type=str, default=DEFAULT_WCSID, help="Camera ID")
    parser.add_argument("--ssid", type=str, default=DEFAULT_SSID, help="Session ID")
    parser.add_argument("--token", type=str, default=DEFAULT_TOKEN, help="Auth token")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Camera: {args.wcsid}")
    print(f"Interval: {args.interval}s")
    print(f"Output: {args.output}")
    if args.duration > 0:
        print(f"Duration: {args.duration}s")
    print()

    # Get initial secToken
    print("Fetching initial secToken...", end=" ", flush=True)
    sec_token = get_sec_token(args.wcsid, args.ssid, args.token)
    if not sec_token:
        print("FAILED - could not extract secToken from page")
        return
    print(f"OK ({sec_token[:16]}...)")

    start_time = time.time()
    frame_count = 0
    error_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\nDuration reached ({args.duration}s). Stopping.")
                break

            # Fetch frame
            frame_data = fetch_frame(sec_token, args.wcsid)
            if frame_data:
                frame_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}_{frame_count:05d}.jpg"
                filepath = os.path.join(args.output, filename)
                with open(filepath, "wb") as f:
                    f.write(frame_data)
                elapsed_min = int(elapsed) // 60
                elapsed_sec = int(elapsed) % 60
                print(f"[{frame_count:5d}] {filename} ({len(frame_data):,} bytes) [+{elapsed_min}m{elapsed_sec:02d}s]")
            else:
                error_count += 1
                elapsed_min = int(elapsed) // 60
                elapsed_sec = int(elapsed) % 60
                print(f"[ERROR] Failed to fetch frame (errors: {error_count}) [+{elapsed_min}m{elapsed_sec:02d}s]")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print()

    print(f"\nCaptured {frame_count} frames in {args.output}/")

    if frame_count < 2:
        print("Not enough frames for a video.")
        return

    # Build video from captured frames
    frames_pattern = os.path.join(args.output, "frame_*.jpg")
    frame_files = sorted(glob.glob(frames_pattern))
    if not frame_files:
        print("No frame files found.")
        return

    # Read first frame to get dimensions
    first = cv2.imread(frame_files[0])
    h, w = first.shape[:2]

    # Use actual capture rate as video fps (capped at 30)
    actual_elapsed = time.time() - start_time
    fps = min(frame_count / max(actual_elapsed, 1), 30)
    fps = max(fps, 1)

    video_path = os.path.join(args.output, "webcam.mp4")
    print(f"Creating video: {video_path} ({frame_count} frames, {fps:.1f} fps, {w}x{h})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for f in frame_files:
        img = cv2.imread(f)
        if img is not None:
            writer.write(img)
    writer.release()

    file_size = os.path.getsize(video_path)
    duration_s = frame_count / fps
    print(f"Done: {file_size / 1024 / 1024:.1f} MB, ~{duration_s:.1f}s playback")


if __name__ == "__main__":
    main()
