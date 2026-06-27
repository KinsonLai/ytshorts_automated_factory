# Auto Video Creator

An end-to-end pipeline that autonomously researches trending topics, writes
scripts, generates voiceovers, sources royalty-free images, composites a 9:16
short-form video, and uploads it to YouTube -- all driven by a large language
model through OpenRouter.

Built as a desktop application with a dark-themed CustomTkinter GUI, this
project demonstrates working knowledge of LLM tool-calling, video compositing
with MoviePy, OAuth-based API integration, and multithreaded desktop UI design.

---

## Features

- **AI Research** -- the agent searches the web via DuckDuckGo for trending
  topics from the last 24 hours to 30 days, picking subjects with high viral
  potential.
- **Script Generation** -- an LLM (DeepSeek, GPT-4o, or Claude) writes a 60-80
  word script, viral title, hook text, description with hashtags, and image
  queries. A detailed system prompt (`skills.md`) guides creative tone and
  platform-specific optimisation.
- **Text-to-Speech** -- Microsoft Edge TTS generates natural speech with
  word-level timestamps for precise subtitle synchronisation.
- **Stock Image Sourcing** -- a 3-tier fallback chain (Pixabay -> Pexels ->
  DuckDuckGo) downloads unique images with content-hash deduplication.
- **Video Compositing** -- MoviePy assembles random background chunks with
  cross-fades, 9:16 crop, hook overlay, responsive subtitles, images matched to
  subtitle segments, and optional enrichment overlays.
- **Enrichment Overlays** -- countdown timer, quiz question/answer reveal,
  progress bar, lower-third banner, and semi-transparent watermark.
- **Background Music** -- optional royalty-free music from Pixabay, mixed at
  low volume under the voiceover.
- **YouTube Upload** -- Google OAuth 2.0 with multi-account support, automatic
  token refresh, and session tracking.
- **In-App Video Editor** -- trim and apply ffmpeg filters (blur, grayscale,
  vignette) without leaving the application.
- **Crash Resilience** -- AI-generated scripts are checkpointed to disk so a
  crash or stop doesn't waste API tokens on re-generation.
- **Ideas Archive** -- every video specification is logged with timestamp,
  script, and metadata for review.
- **Real-time Console** -- a colour-coded log view captures every `print()` from
  the pipeline, running in a background thread so the UI stays responsive.

---

## Architecture

```
config.json        skills.md         client_secret*.json
    |                  |                    |
    v                  v                    v
+---------------------------------------------------+
|                     agent.py                       |
|  - loads config, skills, and memory                |
|  - runs the LLM tool-calling loop                  |
|  - orchestrates TTS, render, and upload            |
+---------------------------------------------------+
    |          |          |            |
    v          v          v            v
+---------+ +--------+ +----------+ +-----------+
| tools.py| |gui.py  | |uploader.py| |edge-tts   |
|         | |        | |           | |ffmpeg     |
| - search| | tabs:  | | - OAuth   | |MoviePy    |
| - images| | Dashbd | | - multi-  | |Pixabay    |
| - TTS   | | Gen    | |   account | |Pexels     |
| - video | | Prev   | | - API     | |DuckDuckGo |
| - edit  | | Editor | |   upload  | |OpenRouter |
+---------+ | Ideas  | +-----------+ |Google API |
            | Accts  |               +-----------+
            | Settings|
            +--------+
```

The agent module is the orchestrator. It reads the `skills.md` system prompt,
wires up the LLM with function-calling tool schemas, and delegates every
concrete operation (search, image download, TTS, video compositing, upload)
to the tools and uploader modules. The GUI wraps this pipeline with a
thread-safe progress bar, live console, and tabbed settings panels.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Desktop GUI | CustomTkinter, Pillow |
| AI / LLM | OpenRouter API (OpenAI SDK), models: DeepSeek, GPT-4o, Claude 3.5 |
| TTS | Microsoft Edge TTS (`edge-tts`) |
| Video | MoviePy, ImageIO, ffmpeg / ffprobe |
| Images | Pixabay API, Pexels API, DuckDuckGo Images (`duckduckgo_search`) |
| Search | DuckDuckGo Text Search |
| YouTube | Google API Python Client, google-auth-oauthlib |
| HTTP | requests |

---

## Setup

### Prerequisites

- **Python 3.10+** (developed on 3.13)
- **ffmpeg** and **ffprobe** on your PATH
- **ImageMagick** (Windows only -- MoviePy uses it for text rendering; on
  Linux/macOS this is handled differently)
- **API keys**: [OpenRouter](https://openrouter.ai/keys),
  [Pixabay](https://pixabay.com/api/docs/),
  [Pexels](https://www.pexels.com/api/)
- **Google OAuth client secret**: Create a project in the
  [Google Cloud Console](https://console.cloud.google.com/), enable the
  YouTube Data API v3, and download the OAuth client secret JSON.

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/auto-video-creator.git
cd auto-video-creator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy `config.example.json` to `config.json`** and fill in your API keys:

   ```json
   {
       "openrouter_api_key": "sk-or-v1-...",
       "pixabay_api_key": "...",
       "pexels_api_key": "...",
       "ai_model": "deepseek/deepseek-v4-flash",
       "bg_video": "assets/background_video.mp4",
       "output_file": "outputs/final_short.mp4",
       "tts_voice": "en-US-ChristopherNeural",
       "imagemagick_path": "C:\\Program Files (x86)\\ImageMagick...\\magick.exe"
   }
   ```

   All fields are configurable through the GUI's Settings tab as well.

2. **Copy `client_secret.example.json` to `client_secret.json`** with your
   Google OAuth credentials for YouTube upload. For multiple YouTube accounts,
   create `client_secret_AccountName.json` files.

3. **Place a background video** in `assets/`. This is used as the visual
   backdrop for all generated Shorts. A 90+ second compilation of fast-paced
   footage (parkour, city timelapse, etc.) works well.

### Running

**Desktop GUI:**
```bash
python gui.py
```

**Headless / CLI:**
```bash
python agent.py
```

The headless mode reads `config.json` and runs one generation cycle end-to-end.

---

## Project Structure

```
auto-video-creator/
|-- agent.py              # AI orchestrator and main pipeline
|-- tools.py              # Video, audio, image, search, and enrichment logic
|-- uploader.py           # YouTube OAuth and upload management
|-- gui.py                # CustomTkinter desktop application
|-- skills.md             # System prompt that guides the AI's creative output
|-- requirements.txt      # Python dependencies
|-- config.example.json   # Configuration template (copy to config.json)
|-- client_secret.example.json  # Google OAuth template
|-- LICENSE               # MIT License
|-- .gitignore
|-- assets/               # Background video and static assets (gitignored)
|-- outputs/              # Generated videos (gitignored)
```

---

## Pipeline Flow

1. **Memory Check** -- if a previous run left a valid script on disk, skip the
   LLM round-trip to save tokens and time.
2. **AI Generation** -- the LLM calls `search_trends` (DuckDuckGo) to find a
   viral topic, then calls `submit_video_specification` with title, script,
   hook text, description, and image queries.
3. **TTS & Subtitles** -- Microsoft Edge TTS generates the voiceover with
   per-word boundary timestamps, producing both `.mp3` and `.srt` files.
4. **Background Music** -- an optional royalty-free track is downloaded from
   Pixabay's music library.
5. **Image Download** -- up to 4 unique images are sourced with content-hash
   deduplication across the 3-tier fallback chain.
6. **Video Composite** -- MoviePy slices 2-4 random chunks from the background
   video, cross-fades between them, crops to 9:16, and layers hook text,
   word-synced subtitles, images at subtitle segment boundaries, and any
   requested enrichment overlays. Background music is mixed at 12% volume.
7. **YouTube Upload** -- the final `.mp4` is uploaded as a public Short via the
   YouTube Data API v3.
8. **Cleanup** -- the memory checkpoint is deleted, temp files are removed, and
   the script is archived to the ideas history.

---

## Key Design Decisions

- **Global settings via module variable**: `tools._agent_settings` is injected
  once by `agent.configure_agent()` and read by every tool function. This keeps
  function signatures clean but means the tools module depends on the agent
  module calling `configure_agent` first.
- **Print-based logging with stdout capture**: The GUI captures `stdout` in the
  worker thread so every `print()` from the pipeline appears live in the
  colour-coded console. This is simple and effective for a desktop app but
  would be swapped for the `logging` module in a headless server deployment.
- **Stream-copy trimming**: The editor uses ffmpeg's `-c copy` for lossless
  trim operations. Effect overlays require re-encoding only the affected
  segment.
- **Token economy**: The memory checkpoint system means a crash during video
  compositing doesn't require re-running the (expensive) LLM generation step.

---

## Future Work

Some ideas I'm considering for the next iteration:

- Support for TikTok and Instagram Reels upload APIs
- Scheduled generation (run every N hours in a cron-like loop)
- Video templates with configurable layout and animation presets
- A/B testing different hooks and thumbnails
- Analytics dashboard pulling view/engagement data from YouTube
- Web-based interface instead of a native desktop app
- Direct background video generation via text-to-video models

---

## License

MIT -- see [LICENSE](LICENSE) for details.
