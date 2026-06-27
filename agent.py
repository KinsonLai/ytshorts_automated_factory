import json
import asyncio
import os
from datetime import datetime
from typing import Any

from openai import OpenAI

from tools import AGENT_TOOLS, execute_search_trends, generate_audio, create_video
import tools as tools_module
from uploader import upload_to_youtube

AGENT_SETTINGS: dict[str, Any] = {}

MEMORY_FILE = "memory.json"
IDEAS_HISTORY_FILE = "ideas_history.json"


def configure_agent(settings: dict[str, Any] | None = None) -> None:
    """Wire the global agent settings into both the agent and tools modules.

    This is called once at startup (from both the GUI and the standalone entry
    point) so every part of the pipeline sees the same API keys, paths, model
    choice, and enrichment flags without passing them around manually.
    """
    global AGENT_SETTINGS
    AGENT_SETTINGS = settings or {}
    tools_module._agent_settings = AGENT_SETTINGS

    from moviepy.config import change_settings
    imagick_path = AGENT_SETTINGS.get("imagemagick_path")
    if imagick_path and os.path.exists(imagick_path):
        change_settings({"IMAGEMAGICK_BINARY": imagick_path})


def _get_api_key() -> str:
    return AGENT_SETTINGS.get("openrouter_api_key", "")


def _get_model() -> str:
    return AGENT_SETTINGS.get("ai_model", "deepseek/deepseek-v4-flash")


def _get_bg_video() -> str:
    return AGENT_SETTINGS.get("bg_video", "assets/background_video.mp4")


def _refresh_client() -> None:
    global client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_get_api_key(),
    )


client = None  # initialised lazily by configure_agent -> _refresh_client


def read_skills() -> str:
    """Read the markdown system prompt that guides the AI's creative behaviour."""
    if not os.path.exists("skills.md"):
        print("[Agent] Warning: skills.md not found, using default instructions.")
        return "You are a viral video content creator specializing in YouTube Shorts."
    with open("skills.md", "r", encoding="utf-8") as file:
        return file.read()


def load_memory() -> dict[str, Any] | None:
    """Restore a partially-completed video specification from disk.

    If the app crashed or was stopped mid-generation, the AI's work is still
    stored here so we can skip the expensive LLM round-trip and pick up
    directly at the rendering stage.
    """
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_memory(data: dict[str, Any]) -> None:
    """Persist the AI's video specification so a crash doesn't waste tokens."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def clear_memory() -> None:
    """Remove the checkpoint file after a successful upload."""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)


def load_ideas_history() -> list[dict[str, Any]]:
    """Return every video idea the agent has ever generated."""
    if os.path.exists(IDEAS_HISTORY_FILE):
        with open(IDEAS_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_idea_to_history(video_spec: dict[str, Any]) -> None:
    """Append a completed video spec to the ideas archive.

    The archive is displayed in the GUI's Ideas tab for review and reuse.
    """
    ideas = load_ideas_history()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "title": video_spec.get("title", ""),
        "script": video_spec.get("script", ""),
        "hook_text": video_spec.get("hook_text", ""),
        "image_query": video_spec.get("image_query", ""),
        "description": video_spec.get("description", ""),
        "enrichment": video_spec.get("enrichment", {}),
    }
    ideas.append(entry)
    with open(IDEAS_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(ideas, f, indent=4)


def run_agent() -> None:
    """Orchestrate a full generation cycle: research -> script -> render -> upload.

    The flow has three stages:

    1. Check memory -- if a previous run left a complete spec on disk we skip
       the AI round-trip and go straight to rendering.
    2. AI generation -- the LLM is given the skills.md system prompt plus live
       tool schemas. It calls ``search_trends`` to scope a topic, then calls
       ``submit_video_specification`` to commit title, script, hook text,
       description, and image queries.
    3. Execution -- TTS, background music, final video composite, and an
       optional YouTube upload.
    """
    _refresh_client()
    print("\n=== STARTING AUTONOMOUS AGENT ===")

    video_spec = load_memory()

    if video_spec:
        print("[Memory] Recovered previous script from memory, skipping AI generation.")
    else:
        print("[Agent] No memory found, brainstorming new video...")
        today_date = datetime.now().strftime("%A, %B %d, %Y")

        skills_content = read_skills()
        custom_instructions = AGENT_SETTINGS.get("custom_instructions", "")
        if custom_instructions:
            skills_content += f"\n\nCUSTOM INSTRUCTIONS:\n{custom_instructions}"

        dynamic_system_prompt = (
            skills_content
            + f"\n\nCRITICAL CONTEXT:\n- Today's exact date is {today_date}.\n"
            "- You MUST include the current month/year in your search queries."
        )

        model = _get_model()
        messages = [
            {"role": "system", "content": dynamic_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Today is {today_date}. Create today's viral YouTube Short. "
                    "Call the search tool to find a trend from the last 24-48 hours, "
                    "then submit the final video specification."
                ),
            },
        ]

        video_spec = None
        while True:
            print("[Agent] Thinking...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            messages.append(message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if func_name == "search_trends":
                        search_result = execute_search_trends(args["query"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": search_result,
                        })
                    elif func_name == "submit_video_specification":
                        video_spec = args
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": "Accepted.",
                        })
                        break
            else:
                break

            if video_spec:
                break

        if video_spec:
            enrichment = video_spec.get("enrichment", {})
            save_memory(video_spec)
            save_idea_to_history(video_spec)
            print("[Memory] Script saved to memory.json and ideas_history.json.")

    if not video_spec:
        print("[Error] Agent failed to finalize video.")
        return

    print("\n--- FINAL VIDEO SPECIFICATION ---")
    print(f"Title: {video_spec['title']}")
    print(f"Hook: {video_spec['hook_text']}")
    print(f"Script: {video_spec['script']}")
    print(f"Image Query: {video_spec.get('image_query', 'N/A')}")
    print("---------------------------------")

    title: str = video_spec["title"]
    script: str = video_spec["script"]
    hook_text: str = video_spec["hook_text"]
    description: str = video_spec["description"]
    image_query: str = video_spec.get("image_query", "parkour or extreme sports")
    enrichment: dict[str, Any] = video_spec.get("enrichment", {})

    asyncio.run(generate_audio(script, "voice.mp3", "voice.srt"))

    from tools import download_background_music
    bg_music = download_background_music(query="background music", output_file="bg_music.mp3")

    create_video(
        audio_file="voice.mp3",
        bg_video=_get_bg_video(),
        output_file="final_short.mp4",
        hook_text=hook_text,
        srt_file="voice.srt",
        image_query=image_query,
        bg_music=bg_music,
        enrichment=enrichment,
    )

    if AGENT_SETTINGS.get("auto_upload", True):
        upload_to_youtube("final_short.mp4", title, description)

    clear_memory()
    print("=== AGENT WORKFLOW COMPLETE ===\n")


if __name__ == "__main__":
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                configure_agent(json.load(f))
        run_agent()
    except Exception:
        import traceback
        traceback.print_exc()
