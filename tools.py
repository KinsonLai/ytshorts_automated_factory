import os
import re
import random
import asyncio
import subprocess
import time
import json
import hashlib
import unicodedata
from typing import Any

from duckduckgo_search import DDGS
import requests
import edge_tts
from openai import OpenAI
import PIL.Image

if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
    ImageClip, ColorClip, CompositeAudioClip,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings

_agent_settings: dict[str, Any] = {}
"""
Module-level settings injected by ``agent.configure_agent()``.  This is the
single source of truth for API keys, paths, model choices, and enrichment
flags.  All tool functions read from it directly so callers don't need to
thread a settings dict through every function signature.
"""


# ---- SRT parsing ------------------------------------------------------------

def parse_srt_time(time_str: str) -> float:
    """Convert an SRT timestamp like ``HH:MM:SS,mmm`` into seconds."""
    h, m, s = time_str.split(":")
    s, ms = s.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt_segments(srt_file: str) -> list[tuple[float, float, str]]:
    """Parse an SRT file into a list of ``(start_s, end_s, text)`` tuples."""
    segments: list[tuple[float, float, str]] = []
    with open(srt_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = (
        r"(\d+)\n"
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.*?)(?=\n\n|\Z)"
    )
    for m in re.findall(pattern, content, re.DOTALL):
        start = parse_srt_time(m[1])
        end = parse_srt_time(m[2])
        text = m[3].strip().replace("\n", " ")
        segments.append((start, end, text))
    return segments


# ---- LLM tool schemas -------------------------------------------------------

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_trends",
            "description": "Search the web for the latest viral trends, news, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'latest AI news June 2026'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_video_specification",
            "description": "Submit the final script and metadata for the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Viral title under 60 characters ending in #shorts",
                    },
                    "script": {
                        "type": "string",
                        "description": "Exact spoken text, 60-80 words.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description with hashtags.",
                    },
                    "hook_text": {
                        "type": "string",
                        "description": "2-4 word hook for the on-screen overlay.",
                    },
                    "image_query": {
                        "type": "string",
                        "description": "3-4 visual queries separated by '|' matching script sections.",
                    },
                    "enrichment": {
                        "type": "object",
                        "description": "Optional engagement overlays.",
                        "properties": {
                            "countdown_enabled": {
                                "type": "boolean",
                                "description": "Add countdown timer at end.",
                            },
                            "quiz_question": {
                                "type": "string",
                                "description": "Quiz question for engagement overlay.",
                            },
                            "quiz_answer": {
                                "type": "string",
                                "description": "Correct answer shown after a pause.",
                            },
                            "progress_bar": {
                                "type": "boolean",
                                "description": "Add progress bar at bottom.",
                            },
                            "lower_third": {
                                "type": "string",
                                "description": "Lower-third text for channel branding.",
                            },
                        },
                    },
                },
                "required": [
                    "title", "script", "description", "hook_text", "image_query",
                ],
            },
        },
    },
]


# ---- Image acquisition (Pixabay -> Pexels -> DuckDuckGo fallback) -----------

PEXELS_API_BASE = "https://api.pexels.com/v1"


def _validate_image_bytes(data: bytes) -> bool:
    """Quick header sniff -- reject payloads that aren't real images."""
    if len(data) < 2048:
        return False
    header = data[:4]
    return (
        header.startswith(b"\xff\xd8")          # JPEG
        or header.startswith(b"\x89PNG")         # PNG
        or (header.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP")
    )


def _validate_image_response(response: requests.Response) -> bool:
    """Check that an HTTP response actually contains usable image bytes."""
    if response.status_code != 200:
        return False
    content_type = response.headers.get("Content-Type", "").lower()
    if any(tag in content_type for tag in ("text", "html", "json")):
        return False
    return _validate_image_bytes(response.content)


def _download_with_retry(
    url: str,
    timeout: int = 20,
    max_retries: int = 3,
    headers: dict[str, str] | None = None,
) -> requests.Response | None:
    """GET *url* with exponential back-off for rate-limit and server errors."""
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503, 502):
                time.sleep((attempt + 1) * 3)
                continue
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
        except Exception:
            pass
        break
    return None


def _try_pixabay_image(pixabay_key: str, query: str) -> bytes | None:
    """Search Pixabay photos for *query* and return the first valid image."""
    if not pixabay_key:
        return None
    try:
        resp = requests.get(
            "https://pixabay.com/api/",
            params={
                "key": pixabay_key, "q": query, "image_type": "photo",
                "per_page": 5, "orientation": "vertical", "safesearch": "true",
            },
            timeout=15,
        )
        data = resp.json()
        if not isinstance(data, dict):
            return None
        for hit in data.get("hits", []):
            url = hit.get("largeImageURL") or hit.get("webformatURL")
            if not url:
                continue
            img_res = _download_with_retry(url, timeout=20)
            if img_res and _validate_image_response(img_res):
                print(f"[Tool: Image] Pixabay hit: {url[:80]}...")
                return img_res.content
    except Exception as e:
        print(f"[Tool: Image] Pixabay error: {str(e)[:120]}")
    return None


def _try_pexels_image(pexels_key: str, query: str) -> bytes | None:
    """Search Pexels for *query* and return the first valid image."""
    if not pexels_key:
        return None
    try:
        resp = requests.get(
            f"{PEXELS_API_BASE}/search",
            headers={"Authorization": pexels_key},
            params={"query": query, "per_page": 5, "orientation": "portrait"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        for photo in data.get("photos", []):
            url = photo.get("src", {}).get("original") or photo.get("src", {}).get("large")
            if not url:
                continue
            img_res = _download_with_retry(url, timeout=20)
            if img_res and _validate_image_response(img_res):
                print(f"[Tool: Image] Pexels hit: {url[:80]}...")
                return img_res.content
    except Exception as e:
        print(f"[Tool: Image] Pexels error: {str(e)[:120]}")
    return None


def _try_duckduckgo_image(query: str) -> bytes | None:
    """Last-resort image source -- DuckDuckGo image search."""
    try:
        ddgs = DDGS()
        results = ddgs.images(query, max_results=8)
        for result in results:
            url = result.get("image")
            if not url:
                continue
            img_res = _download_with_retry(url, timeout=15)
            if img_res and _validate_image_response(img_res):
                print("[Tool: Image] DDG fallback hit")
                return img_res.content
    except Exception as e:
        print(f"[Tool: Image] DDG error: {str(e)[:120]}")
    return None


def _hash_image_data(img_bytes: bytes) -> str:
    return hashlib.md5(img_bytes).hexdigest()


def _download_unique_image(
    query: str,
    output_prefix: str,
    existing_hashes: set[str],
    existing_paths: list[str],
) -> bool:
    """Download one image for *query*, skipping content already collected.

    Sources are tried in order of reliability/quality: Pixabay, then Pexels,
    then DuckDuckGo.  A content-hash check prevents near-duplicate images.
    """
    pixabay_key: str = _agent_settings.get("pixabay_api_key", "")
    pexels_key: str = _agent_settings.get("pexels_api_key", "")

    datasources: list = []
    if pixabay_key:
        datasources.append(lambda: _try_pixabay_image(pixabay_key, query))
    if pexels_key:
        datasources.append(lambda: _try_pexels_image(pexels_key, query))
    datasources.append(lambda: _try_duckduckgo_image(query))

    for source_fn in datasources:
        img_data = source_fn()
        if img_data is None:
            continue
        content_hash = _hash_image_data(img_data)
        if content_hash in existing_hashes:
            print(f"[Tool: Image] Duplicate skipped for '{query}'")
            continue
        existing_hashes.add(content_hash)
        path = f"{output_prefix}_{len(existing_paths)}.jpg"
        with open(path, "wb") as f:
            f.write(img_data)
        existing_paths.append(path)
        return True

    return False


def download_images(
    query: str, max_results: int = 4, output_prefix: str = "downloaded_image"
) -> list[str]:
    """Download one unique image per pipe-separated sub-query.

    Duplicates are detected via content hashing so visually identical images
    from different sources or queries don't end up in the same video.
    """
    queries = [q.strip() for q in query.split("|") if q.strip()]
    if not queries:
        queries = [query.strip()]

    downloaded_paths: list[str] = []
    seen_hashes: set[str] = set()

    for i, q in enumerate(queries):
        print(f"[Tool: Image] Processing '{q}' ({i + 1}/{len(queries)})...")
        succeeded = _download_unique_image(q, output_prefix, seen_hashes, downloaded_paths)
        if not succeeded:
            print(f"[Tool: Image] All sources exhausted for '{q}'.")

    print(f"[Tool: Image] Downloaded {len(downloaded_paths)}/{len(queries)} unique images")
    return downloaded_paths


# ---- Responsive subtitle rendering ------------------------------------------

def _sanitize_text(text: str) -> str:
    """Normalise text for subtitle rendering.

    Strips Unicode control characters, zero-width spaces, byte-order marks,
    and replaces curly quotes / em-dashes with ASCII equivalents.
    """
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate({
        0x200B: None, 0x200C: None, 0x200D: None, 0x200E: None, 0x200F: None,
        0xFEFF: None,
    })
    text = "".join(c for c in text if c == "\n" or c == " " or ord(c) >= 32)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u2026", "...")
    return " ".join(text.split())


def _calc_responsive_font(
    text: str, max_width: int, max_height: int | None = None, base_ratio: float = 0.08
) -> int:
    """Pick a font size that keeps the subtitle readable without overflow."""
    words = len(text.split())
    chars = len(text)
    longest_word = max((len(w) for w in text.split()), default=1)

    if words <= 2:
        size = int(max_width * 0.10)
    elif words <= 3:
        size = int(max_width * 0.08)
    elif words <= 5:
        size = int(max_width * 0.065)
    elif words <= 8:
        size = int(max_width * 0.052)
    elif words <= 12:
        size = int(max_width * 0.042)
    else:
        size = int(max_width * 0.035)

    if longest_word > 10:
        size = min(size, int(max_width * 0.045))
    if chars > 80:
        size = min(size, int(max_width * 0.038))

    return max(18, min(size, 72))


def _wrap_text_lines(
    text: str, fontsize: int, max_width: int, max_lines: int = 2
) -> list[str]:
    """Break *text* into visually balanced subtitle lines.

    Prefers 2 lines for readability; falls back to 3 only for long text.
    """
    words = text.split()
    estimated_char_width = fontsize * 0.55
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        if len(test_line) * estimated_char_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # single word longer than a line -- character-level break
                for i in range(len(word), 0, -1):
                    if i * estimated_char_width <= max_width:
                        lines.append(word[:i])
                        current_line = word[i:]
                        break
                else:
                    lines.append(word)
                    current_line = ""
    if current_line:
        lines.append(current_line)

    # force a single long line onto 2 lines for readability
    if len(lines) == 1 and len(lines[0].split()) >= 4:
        words = lines[0].split()
        mid = len(words) // 2
        candidate1 = " ".join(words[:mid])
        candidate2 = " ".join(words[mid:])
        if (
            len(candidate1) * estimated_char_width <= max_width
            and len(candidate2) * estimated_char_width <= max_width
        ):
            lines = [candidate1, candidate2]

    # enforce max_lines by roughly splitting word count
    if len(lines) > max_lines:
        all_words = text.split()
        total = len(all_words)
        per_line = max(1, (total + max_lines - 1) // max_lines)
        lines = [
            " ".join(all_words[i : i + per_line])
            for i in range(0, total, per_line)
        ][:max_lines]

    return lines[:max_lines]


def generate_responsive_subtitle_clip(
    txt: str, target_w: int, target_h: int | None = None
) -> TextClip:
    """Build a MoviePy ``TextClip`` whose font size and line breaks auto-scale."""
    txt = _sanitize_text(txt)
    max_text_width = target_w * 0.90
    fontsize = _calc_responsive_font(txt, target_w)
    lines = _wrap_text_lines(txt, fontsize, max_text_width)
    display_text = "\n".join(lines)

    if len(lines) > 4:
        fontsize = max(int(target_w * 0.03), 18)
        lines = _wrap_text_lines(txt, fontsize, max_text_width)
        display_text = "\n".join(lines)

    color_options = _agent_settings.get("subtitle_color", "yellow")

    return TextClip(
        display_text,
        font="Arial-Bold",
        fontsize=fontsize,
        color=color_options,
        stroke_width=0,
        method="caption",
        size=(target_w * 0.92, None),
        align="center",
    )


# ---- Background music (Pixabay audio) ---------------------------------------

def download_background_music(
    query: str = "upbeat background", output_file: str = "bg_music.mp3"
) -> str | None:
    """Grab a royalty-free music track from Pixabay and save it locally."""
    pixabay_key: str = _agent_settings.get("pixabay_api_key", "")
    if not pixabay_key:
        print("[Tool: Music] No Pixabay key, skipping background music.")
        return None
    try:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params={
                "key": pixabay_key, "q": query, "video_type": "music",
                "per_page": 3,
            },
            timeout=15,
        ).json()
        hits = resp.get("hits", [])
        for hit in hits:
            for size_key in ("large", "medium", "small"):
                music_url = hit.get("videos", {}).get(size_key, {}).get("url")
                if not music_url:
                    continue
                music_res = requests.get(music_url, timeout=30)
                if music_res.status_code == 200 and len(music_res.content) > 10000:
                    with open(output_file, "wb") as f:
                        f.write(music_res.content)
                    print(f"[Tool: Music] Downloaded: {output_file}")
                    return output_file
        print("[Tool: Music] No music tracks found.")
    except Exception as e:
        print(f"[Tool: Music] Failed: {e}")
    return None


# ---- Engagement overlays (timer, quiz, progress bar, etc.) -----------------

def create_countdown_timer(
    duration: float,
    target_w: int,
    position: tuple[str, float] = ("center", 0.85),
    start_offset: float = 0,
) -> CompositeVideoClip | None:
    """Render a ticking countdown overlay (e.g. ``3, 2, 1``) in the top-right."""
    try:
        frame_w = int(target_w * 0.18)
        frame_h = int(frame_w * 0.55)
        bg = ColorClip(size=(frame_w, frame_h), color=(0, 0, 0, 160))

        clips = []
        for t in range(int(duration) + 1):
            remaining = max(0, int(duration) - t)
            mins, secs = remaining // 60, remaining % 60
            time_str = f"{mins}:{secs:02d}" if mins > 0 else str(secs)

            txt = TextClip(
                time_str, font="Arial-Bold", fontsize=int(frame_h * 0.6),
                color="white", stroke_color="black", stroke_width=2,
            )
            clips.append(txt.set_position("center").set_duration(1).set_start(t))

        timer = CompositeVideoClip([bg, *clips], size=(frame_w, frame_h))
        timer = timer.set_position(position, relative=True)
        if start_offset > 0:
            timer = timer.set_start(start_offset)
        print(f"[Tool: Enrichment] Countdown timer created ({int(duration)}s)")
        return timer
    except Exception as e:
        print(f"[Tool: Enrichment] Countdown timer failed: {e}")
        return None


def create_progress_bar(
    duration: float,
    target_w: int,
    color: str = "#8b5cf6",
    position: tuple[str, float] = ("center", 0.97),
) -> CompositeVideoClip | None:
    """Draw a thin progress bar that fills left-to-right across the bottom."""
    try:
        bar_h = int(target_w * 0.012)
        bar_w = int(target_w * 0.88)
        bg = ColorClip(size=(bar_w, bar_h), color=(0, 0, 0, 100)).set_duration(duration)

        clips = [bg]
        for pct in range(0, 101, 2):
            t = (pct / 100.0) * duration
            curr_w = int(bar_w * pct / 100)
            if curr_w > 0:
                seg = ColorClip(size=(curr_w, bar_h), color=(139, 92, 246))
                seg = seg.set_start(t).set_duration(duration - t + 0.1)
                clips.append(seg)

        bar = CompositeVideoClip(clips, size=(bar_w, bar_h))
        bar = bar.set_position(position, relative=True)
        print("[Tool: Enrichment] Progress bar created")
        return bar
    except Exception as e:
        print(f"[Tool: Enrichment] Progress bar failed: {e}")
        return None


def create_lower_third(
    text: str,
    duration: float,
    target_w: int,
    position: tuple[str, float] = ("center", 0.88),
) -> TextClip | None:
    """Render a lower-third text banner for channel branding."""
    try:
        fontsize = int(target_w * 0.04)
        txt = TextClip(
            text, font="Arial", fontsize=fontsize,
            color="white", stroke_color="black", stroke_width=2,
            method="caption", size=(target_w * 0.8, None),
        )
        txt = txt.set_position(position, relative=True).set_duration(duration)
        print(f"[Tool: Enrichment] Lower third added: {text}")
        return txt
    except Exception as e:
        print(f"[Tool: Enrichment] Lower third failed: {e}")
        return None


def create_quiz_overlay(
    question: str,
    answer: str,
    duration: float,
    target_w: int,
    reveal_at: float | None = None,
) -> CompositeVideoClip | None:
    """Show a question overlay, then cross-fade to the answer.

    If *reveal_at* isn't set, the answer appears about a third of the way in.
    """
    if reveal_at is None:
        reveal_at = duration * 0.35
    try:
        fontsize_q = int(target_w * 0.042)
        fontsize_a = int(target_w * 0.038)

        q_clip = TextClip(
            f"Q: {question}", font="Arial-Bold", fontsize=fontsize_q,
            color="#fbbf24", stroke_color="black", stroke_width=2,
            method="caption", size=(target_w * 0.85, None),
        )
        q_clip = q_clip.set_position(("center", 0.55), relative=True)
        q_clip = q_clip.set_start(0.5).set_duration(reveal_at)

        answer_prefix: str = _agent_settings.get("quiz_prefix", "Answer: ")
        answer_text = f"{answer_prefix}{answer}" if answer_prefix else answer

        a_clip = TextClip(
            answer_text, font="Arial-Bold", fontsize=fontsize_a,
            color="#4ade80", stroke_color="black", stroke_width=2,
            method="caption", size=(target_w * 0.85, None),
        )
        a_clip = a_clip.set_position(("center", 0.55), relative=True)
        a_clip = a_clip.set_start(reveal_at).set_duration(duration - reveal_at)
        a_clip = a_clip.crossfadein(0.4)

        print(f"[Tool: Enrichment] Quiz overlay: '{question}'")
        return CompositeVideoClip([q_clip, a_clip])
    except Exception as e:
        print(f"[Tool: Enrichment] Quiz overlay failed: {e}")
        return None


def create_watermark(
    text: str, duration: float, target_w: int, opacity: float = 0.45
) -> TextClip | None:
    """Render a subtle, semi-transparent watermark at the top of the frame."""
    try:
        fontsize = int(target_w * 0.028)
        txt = TextClip(
            text, font="Arial", fontsize=fontsize,
            color="white", stroke_color="black", stroke_width=1,
        )
        txt = txt.set_opacity(opacity)
        txt = txt.set_position(("center", 0.03), relative=True).set_duration(duration)
        print(f"[Tool: Enrichment] Watermark: {text}")
        return txt
    except Exception as e:
        print(f"[Tool: Enrichment] Watermark failed: {e}")
        return None


# ---- FFmpeg-based video editor utilities ------------------------------------

def get_video_dimensions(file_path: str) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` of *file_path* via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0", file_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)
    except Exception:
        return None, None


def trim_video(
    input_path: str, output_path: str, start_time: float, end_time: float
) -> bool:
    """Losslessly trim a video segment using ffmpeg stream copy."""
    try:
        duration = end_time - start_time
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", input_path,
            "-t", str(duration), "-c", "copy", output_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"[Tool: Editor] Trim failed: {e}")
        return False


def extract_clip(
    input_path: str, output_path: str, start_time: float, duration: float
) -> bool:
    """Extract a chunk of video by start time and duration (stream copy)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", input_path,
            "-t", str(duration), "-c", "copy", output_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"[Tool: Editor] Extract clip failed: {e}")
        return False


def add_effect_overlay(
    input_path: str,
    output_path: str,
    effect_type: str = "blur",
    start_time: float = 0,
    duration: float | None = None,
) -> bool:
    """Apply a video filter (blur, grayscale, vignette) to a segment."""
    try:
        if duration is None:
            duration = get_video_duration(input_path)
        filters = {
            "blur": "boxblur=10:2",
            "grayscale": "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3",
            "vignette": "vignette=PI/4",
        }
        if effect_type not in filters:
            return False
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", input_path,
            "-t", str(duration), "-vf", filters[effect_type],
            "-c:a", "copy", output_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"[Tool: Editor] Effect failed: {e}")
        return False


# ---- TTS and web search implementations -------------------------------------

def execute_search_trends(query: str) -> str:
    """Run a DuckDuckGo text search for recent trends/news."""
    max_results: int = _agent_settings.get("max_search_results", 4)
    time_limit: str = _agent_settings.get("search_time_limit", "w")
    print(f"[Tool: Search] Searching for recent news: '{query}'")
    try:
        results = DDGS().text(query, max_results=max_results, timelimit=time_limit)
        if not results:
            return "No results found. Try a different query or broader trend."
        return "\n".join([f"- {res['title']}: {res['body']}" for res in results])
    except Exception as e:
        return f"Search failed: {str(e)}"


async def generate_audio(
    text: str, output_file: str = "voice.mp3", output_srt: str = "voice.srt"
) -> None:
    """Generate TTS audio via Microsoft Edge TTS with word-level subtitle data."""
    print("[Tool: Audio] Generating TTS and subtitles...")
    voice: str = _agent_settings.get("tts_voice", "en-US-ChristopherNeural")
    communicate = edge_tts.Communicate(text, voice)
    submaker = edge_tts.SubMaker()
    had_word_boundary = False

    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                had_word_boundary = True
                submaker.feed(chunk)
            elif chunk["type"] == "SentenceBoundary" and not had_word_boundary:
                submaker.feed(chunk)

    srt_content = submaker.get_srt()
    if not srt_content.strip():
        print("[Tool: Audio] Warning: empty SRT, using placeholder subtitles.")
        srt_content = "1\n00:00:00,000 --> 00:00:03,000\n \n"

    with open(output_srt, "w", encoding="utf-8") as file:
        file.write(srt_content)
    print("[Tool: Audio] TTS and subtitles generated.")


# ---- Full video compositing pipeline ----------------------------------------

def create_video(
    audio_file: str = "voice.mp3",
    bg_video: str = "assets/background_video.mp4",
    output_file: str | None = None,
    hook_text: str = "Wait for it...",
    srt_file: str = "voice.srt",
    image_query: str = "parkour or extreme sports",
    bg_music: str | None = None,
    enrichment: dict[str, Any] | None = None,
) -> str:
    """Composite the final 9:16 Short video from all generated assets.

    The pipeline:
    1. Extract 2-4 random chunks from *bg_video*, cross-fading between them.
    2. Crop to 9:16 aspect ratio.
    3. Layer hook text, synced subtitles, images matched to subtitle segments,
       and any requested enrichment overlays (quiz, countdown, etc.).
    4. Mix in background music at low volume.
    5. Encode with libx264 at 8 Mbps.
    """
    if output_file is None:
        output_file = _agent_settings.get("output_file", "final_short.mp4")
    if enrichment is None:
        enrichment = _agent_settings.get("enrichment", {})

    print("[Tool: Video] Starting video processing...")

    temp_files: list[str] = []
    audio = None
    full_bg = None
    final_video = None
    image_paths: list[str] = []

    try:
        from moviepy.editor import concatenate_videoclips

        audio = AudioFileClip(audio_file)
        total_duration = audio.duration

        # split total duration randomly into 2-4 segments for visual variety
        num_clips = random.randint(2, 4)
        clip_durations: list[float] = []
        remaining = total_duration
        for j in range(num_clips):
            if j == num_clips - 1:
                clip_durations.append(remaining)
            else:
                portion = max(3.0, remaining * random.uniform(0.25, 0.45))
                clip_durations.append(portion)
                remaining -= portion

        bg_segments: list[VideoFileClip] = []
        for j, seg_dur in enumerate(clip_durations):
            chunk_file = f"temp_bg_{j}.mp4"
            temp_files.append(chunk_file)
            chunk = extract_random_chunk(
                bg_video, chunk_file, duration=max(65, int(seg_dur) + 5)
            )
            if not chunk or not os.path.exists(chunk_file):
                raise RuntimeError(f"Failed to extract chunk {j} from {bg_video}")
            clip = VideoFileClip(chunk_file).subclip(0, seg_dur)
            bg_segments.append(clip)

        for j in range(1, len(bg_segments)):
            bg_segments[j] = bg_segments[j].crossfadein(0.4)

        full_bg = concatenate_videoclips(bg_segments, method="compose", padding=-0.4)

        # 9:16 crop (center-crop from the background)
        w, h = full_bg.size
        target_w = int(h * 9 / 16)
        x_center = w / 2
        full_bg = full_bg.crop(
            x1=x_center - target_w / 2, y1=0,
            x2=x_center + target_w / 2, y2=h,
        )
        full_bg = full_bg.subclip(0, total_duration)

        # hook text overlay (appears for 1.8s at the start)
        hook_fontsize = min(
            _calc_responsive_font(hook_text, target_w, base_ratio=0.06), 32
        )
        txt_clip = TextClip(
            hook_text, fontsize=hook_fontsize, color="white", font="Arial-Bold",
            stroke_width=0, size=(target_w * 0.88, None),
            method="caption", align="center",
        )
        txt_clip = txt_clip.set_position(("center", 0.04), relative=True).set_duration(1.8)

        # images synced to subtitle segments
        image_clips: list[ImageClip] = []
        image_paths = download_images(image_query, max_results=4)
        if image_paths and os.path.exists(srt_file):
            try:
                srt_segments = parse_srt_segments(srt_file)
                num_images = len(image_paths)
                num_segments = len(srt_segments)
                if num_segments >= num_images and num_segments > 0:
                    step = num_segments / num_images
                    segment_indices = [int(i * step) for i in range(num_images)]
                else:
                    segment_indices = list(range(min(num_segments, num_images)))
                for i, seg_idx in enumerate(segment_indices):
                    if i >= len(image_paths):
                        break
                    seg_start, seg_end, _ = srt_segments[seg_idx]
                    seg_duration = max(seg_end - seg_start, 1.5)
                    clip = ImageClip(image_paths[i]).set_duration(seg_duration)
                    clip = clip.resize(width=target_w * 0.55)
                    clip = clip.set_position(("center", 0.18), relative=True)
                    clip = clip.set_start(seg_start)
                    image_clips.append(clip)
                print(
                    f"[Tool: Video] Placed {len(image_clips)} images at "
                    f"subtitle segments: {segment_indices}"
                )
            except Exception as e:
                print(f"[Tool: Video] Failed to add images: {e}")

        # subtitles
        try:
            def make_subtitle(txt):
                return generate_responsive_subtitle_clip(txt, target_w)
            subtitles = SubtitlesClip(srt_file, make_subtitle)
            if subtitles.subtitles:
                subtitles = subtitles.set_position(("center", "center"))
            else:
                subtitles = (
                    ColorClip(size=(target_w, h), color=(0, 0, 0, 0))
                    .set_duration(audio.duration)
                )
        except Exception as e:
            print(f"[Tool: Video] Subtitles unavailable ({e}), proceeding without.")
            subtitles = (
                ColorClip(size=(target_w, h), color=(0, 0, 0, 0))
                .set_duration(audio.duration)
            )

        clips_to_composite = [full_bg, txt_clip]
        if image_clips:
            clips_to_composite.extend(image_clips)
        clips_to_composite.append(subtitles)

        if enrichment.get("progress_bar"):
            bar = create_progress_bar(audio.duration, target_w)
            if bar:
                clips_to_composite.append(bar)

        if enrichment.get("countdown_enabled"):
            countdown_start = max(0, audio.duration - 5)
            timer = create_countdown_timer(5, target_w, start_offset=countdown_start)
            if timer:
                clips_to_composite.append(timer)

        if enrichment.get("quiz_question") and enrichment.get("quiz_answer"):
            quiz = create_quiz_overlay(
                enrichment["quiz_question"],
                enrichment["quiz_answer"],
                audio.duration,
                target_w,
            )
            if quiz:
                clips_to_composite.append(quiz)

        if enrichment.get("lower_third"):
            lt = create_lower_third(
                enrichment["lower_third"], audio.duration, target_w
            )
            if lt:
                clips_to_composite.append(lt)

        watermark_text: str = _agent_settings.get("watermark_text", "")
        if watermark_text:
            wm = create_watermark(watermark_text, audio.duration, target_w)
            if wm:
                clips_to_composite.append(wm)

        final_video = CompositeVideoClip(clips_to_composite)

        # mix background music into the audio track
        if bg_music and os.path.exists(bg_music):
            try:
                music = AudioFileClip(bg_music).volumex(0.12)
                if music.duration < audio.duration:
                    music = music.loop(duration=audio.duration)
                else:
                    music = music.subclip(0, audio.duration)
                final_video = final_video.set_audio(
                    CompositeAudioClip([audio, music])
                )
                print("[Tool: Video] Background music mixed in.")
            except Exception as e:
                print(f"[Tool: Video] Music mixing failed ({e}), using TTS only.")
                final_video = final_video.set_audio(audio)
        else:
            final_video = final_video.set_audio(audio)

        final_video.write_videofile(
            output_file,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k",
            preset="medium",
            logger=None,
        )
        print("[Tool: Video] Render complete!")

    finally:
        # close everything to release file handles, then clean up temp files
        for clip_obj in (audio, full_bg, final_video):
            if clip_obj is not None:
                try:
                    clip_obj.close()
                except Exception:
                    pass
        for tf in temp_files:
            try:
                if tf and os.path.exists(tf):
                    os.remove(tf)
            except Exception:
                pass
        for p in image_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    return output_file


# ---- Background video extraction helpers ------------------------------------

def get_video_duration(file_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path,
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"ffprobe failed for {file_path}: "
            f"{result.stderr or 'empty output'}"
        )
    return float(result.stdout.strip())


def extract_random_chunk(
    input_video: str, output_chunk: str = "temp_bg.mp4", duration: float = 60
) -> str | None:
    """Extract a random *duration*-second chunk from *input_video* via ffmpeg."""
    if not os.path.exists(input_video):
        print(f"[Tool: Video] ERROR: Input video not found: {input_video}")
        return None
    total_length = get_video_duration(input_video)
    max_start = max(0, total_length - duration)
    if max_start <= 0:
        print(
            f"[Tool: Video] ERROR: Video too short ({total_length:.1f}s), "
            f"need at least {duration}s"
        )
        return None
    start_time = random.uniform(0, max_start)
    print(
        f"[Tool: Video] Extracting random {duration}s chunk at {start_time:.2f}s..."
    )
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-i", input_video,
        "-t", str(duration), "-c", "copy", output_chunk,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0 or not os.path.exists(output_chunk):
        print(
            f"[Tool: Video] ERROR: ffmpeg chunk extraction failed "
            f"(code {proc.returncode})"
        )
        return None
    return output_chunk


# ---- Video preview snapshot -------------------------------------------------

def get_video_snapshot(
    video_path: str, time_sec: float = 0, output_jpg: str = "preview.jpg"
) -> bool:
    """Extract a single JPEG frame from *video_path* at *time_sec* via ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(time_sec), "-i", video_path,
            "-vframes", "1", "-q:v", "2", output_jpg,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_jpg)
    except Exception as e:
        print(f"[Tool: Preview] Snapshot failed: {e}")
        return False
