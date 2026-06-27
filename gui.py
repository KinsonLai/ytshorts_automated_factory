"""
Desktop GUI for the Auto Video Creator pipeline.

Built with CustomTkinter, this provides a dark-themed tabbed interface for
configuring and running the full AI-powered video generation flow:
  - Dashboard   -- quick stats, one-click generation, recent output
  - Generate    -- model selection, video source, enrichment toggles, live log
  - Preview     -- frame snapshots of generated or imported videos
  - Editor      -- trim and apply effects via ffmpeg
  - Ideas       -- browse the history of every generated script
  - Accounts    -- manage multiple YouTube OAuth accounts
  - Settings    -- API keys, file paths, TTS voice, search config

The generation worker runs in a background thread and captures stdout/stderr so
every print() from the agent, tools, and uploader modules appears live in the
console tabs.
"""
import os
import sys
import json
import io
import queue
import threading
import traceback
import subprocess
from datetime import datetime
from tkinter import filedialog, messagebox, Text, END
from typing import Any

import customtkinter as ctk
from PIL import Image as PILImage
import PIL.Image

if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "config.json")

DEFAULT_SETTINGS: dict[str, Any] = {
    "openrouter_api_key": "",
    "pixabay_api_key": "",
    "pexels_api_key": "",
    "ai_model": "deepseek/deepseek-v4-flash",
    "bg_video": "assets/background_video.mp4",
    "output_file": "outputs/final_short.mp4",
    "tts_voice": "en-US-ChristopherNeural",
    "auto_upload": True,
    "max_search_results": 4,
    "search_time_limit": "w",
    "imagemagick_path": r"C:\Program Files (x86)\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
    "custom_instructions": "",
    "selected_account": "Default",
    "subtitle_color": "yellow",
    "watermark_text": "",
    "quiz_prefix": "Answer: ",
    "preview_auto_refresh": False,
    "editor_temp_dir": "editor_temp",
}

MODEL_CHOICES = [
    "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro",
    "openai/gpt-4o", "anthropic/claude-3.5-sonnet",
]

VOICE_CHOICES = [
    "en-US-ChristopherNeural", "en-US-EricNeural", "en-US-GuyNeural",
    "en-US-JennyNeural", "en-US-AriaNeural",
    "en-UK-RyanNeural", "en-UK-SoniaNeural",
]

SUBTITLE_COLORS = ["yellow", "white", "#4ade80", "#fbbf24", "#60a5fa", "#f472b6"]

TIME_LIMIT_MAP = {"24 hours": "d", "7 days": "w", "30 days": "m"}
TIME_LIMIT_REVERSE = {"d": "24 hours", "w": "7 days", "m": "30 days"}


def load_settings() -> dict[str, Any]:
    """Read config.json, back-filling any missing keys from DEFAULT_SETTINGS."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                data.setdefault(k, v)
            return data
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(data: dict[str, Any]) -> None:
    """Write the settings dict to config.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception:
        pass


class ColoredConsole(Text):
    """A read-only Tk text widget that colour-codes log lines by prefix.

    Prefixes like ``[Agent]``, ``[Tool:``, ``[Success]``, and ``[Error]``
    are mapped to different colours so the console is scannable at a glance.
    """

    TAG_COLORS = {
        "agent": "#8b5cf6", "tool": "#10b981", "success": "#22c55e",
        "error": "#ef4444", "memory": "#f59e0b", "youtube": "#f43f5e",
        "banner": "#ec4899", "url": "#60a5fa", "gui": "#10b981",
        "warn": "#f59e0b", "normal": "#d1d5db",
    }

    def __init__(self, parent, **kwargs):
        bg = kwargs.pop("bg", "#0b0c12")
        fg = kwargs.pop("fg", "#d1d5db")
        font = kwargs.pop("font", ("Consolas", 11))
        kwargs.setdefault("wrap", "word")
        kwargs.setdefault("borderwidth", 1)
        kwargs.setdefault("relief", "solid")
        kwargs.setdefault("padx", 8)
        kwargs.setdefault("pady", 6)
        super().__init__(parent, bg=bg, fg=fg, font=font, **kwargs)
        self.configure(
            insertbackground=fg,
            selectbackground="#374151",
            selectforeground="#f1f5f9",
        )
        for tag, color in self.TAG_COLORS.items():
            self.tag_configure(tag, foreground=color)
        self.tag_configure("bold", font=("Consolas", 11, "bold"))

    def append_line(self, line: str) -> None:
        lower = line.lower()
        if line.startswith("==="):
            self.insert(END, line + "\n", ("banner", "bold"))
        elif line.startswith(("[Error]", "[ERROR]")) or "traceback" in lower:
            self.insert(END, line + "\n", ("error", "bold"))
        elif line.startswith("[Agent]"):
            self.insert(END, line + "\n", ("agent", "bold"))
        elif line.startswith(("[Tool:", "[GUI]")):
            self.insert(END, line + "\n", ("tool", "bold"))
        elif line.startswith("[Memory]"):
            self.insert(END, line + "\n", ("memory", "bold"))
        elif line.startswith("[Success]"):
            self.insert(END, line + "\n", ("success", "bold"))
        elif "[YouTube]" in line:
            self.insert(END, line + "\n", ("youtube", "bold"))
        elif "http" in lower:
            self.insert(END, line + "\n", "url")
        elif line.startswith("[Warn]"):
            self.insert(END, line + "\n", "warn")
        else:
            self.insert(END, line + "\n", "normal")
        self.see(END)


class App(ctk.CTk):
    """Main application window hosting the tabbed interface."""

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.worker_thread: threading.Thread | None = None
        self.is_running = False
        self._stop_event = threading.Event()
        self.log_queue: queue.Queue = queue.Queue()
        self._log_entries: list[str] = []
        self.preview_image = None
        self.preview_after_id = None
        self.editor_clips: list = []

        self.title("Auto Video Creator")
        self.geometry("1280x860")
        self.minsize(1050, 700)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_header()
        self._build_tabview()
        self._build_status_bar()
        self._poll_queues()
        self._after(2000, self._refresh_preview_on_startup)

    # -- layout helpers ---------------------------------------------------

    def _section_frame(self, parent, title: str) -> ctk.CTkFrame:
        f = ctk.CTkFrame(
            parent, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        f.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            f, text=title, font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f1f5f9", anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 2))
        return f

    def _stat_card(self, parent, value: str, label: str) -> ctk.CTkFrame:
        c = ctk.CTkFrame(
            parent, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        c.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            c, text=value, font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#f1f5f9", anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        ctk.CTkLabel(
            c, text=label, font=ctk.CTkFont(size=11),
            text_color="#6b7280", anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))
        return c

    def _labeled_combo(
        self, parent, label: str, values: list[str], default: str, attr: str, row: int,
    ) -> None:
        ctk.CTkLabel(
            parent, text=label, text_color="#d1d5db",
            font=ctk.CTkFont(size=12), anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        cb = ctk.CTkComboBox(
            parent, values=values, width=280, fg_color="#1a1b26",
            border_color="#2d2f3d", button_color="#2d2f3d",
            button_hover_color="#3d3f4d", dropdown_fg_color="#1a1b26",
            dropdown_hover_color="#2d2f3d", dropdown_text_color="#e2e8f0",
            corner_radius=6,
        )
        cb.set(default)
        cb.grid(row=row, column=1, columnspan=2, sticky="w", pady=3)
        setattr(self, f"_{attr}_cb", cb)

    def _labeled_entry(
        self, parent, label: str, default: str, attr: str, row: int, browse: bool,
    ) -> None:
        ctk.CTkLabel(
            parent, text=label, text_color="#d1d5db",
            font=ctk.CTkFont(size=12), anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        entry = ctk.CTkEntry(
            parent, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        setattr(self, f"_{attr}_entry", entry)
        if browse:
            ctk.CTkButton(
                parent, text="Browse", width=60, height=28, corner_radius=5,
                fg_color="#1e2030", hover_color="#2a2c3d",
                border_color="#2d2f3d", border_width=1,
                font=ctk.CTkFont(size=10),
                command=lambda e=entry: self._browse(
                    e, [("Videos", "*.mp4 *.mov *.avi *.mkv")]
                ),
            ).grid(row=row, column=2, padx=(4, 0), pady=3)

    def _spath(
        self, parent, row: int, label: str, attr: str, default: str,
        filetypes: list[tuple[str, str]],
    ) -> None:
        ctk.CTkLabel(
            parent, text=label, text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=row, column=0, sticky="w", padx=(14, 6), pady=3)
        e = ctk.CTkEntry(
            parent, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        e.insert(0, default)
        e.grid(row=row, column=1, sticky="ew", pady=3)
        setattr(self, f"_{attr}", e)
        ctk.CTkButton(
            parent, text="Browse", width=60, height=28, corner_radius=5,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=10),
            command=lambda ent=e, ft=filetypes: self._browse(ent, ft),
        ).grid(row=row, column=2, padx=(4, 14), pady=3)

    def _browse(self, entry, filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    # -- header and status bar --------------------------------------------

    def _build_header(self) -> None:
        h = ctk.CTkFrame(self, height=50, fg_color="#0d0e15", corner_radius=0)
        h.grid(row=0, column=0, sticky="ew")
        h.grid_propagate(False)
        h.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            h, text="Auto Video Creator",
            font=ctk.CTkFont(size=17, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, padx=(18, 0), pady=10, sticky="w")
        self.status_dot = ctk.CTkLabel(
            h, text="", width=10, height=10, fg_color="#6b7280", corner_radius=5,
        )
        self.status_dot.grid(row=0, column=1, padx=(0, 6), pady=10, sticky="e")
        self.status_label = ctk.CTkLabel(
            h, text="Ready", font=ctk.CTkFont(size=12), text_color="#9ca3af",
        )
        self.status_label.grid(row=0, column=2, padx=(0, 16), pady=10, sticky="e")

    def _build_status_bar(self) -> None:
        bar = ctk.CTkFrame(self, height=26, fg_color="#0d0e15", corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        self.status_bar_text = ctk.CTkLabel(
            bar, text="", font=ctk.CTkFont(size=10), text_color="#6b7280",
        )
        self.status_bar_text.pack(side="left", padx=10)

    # -- tab container ----------------------------------------------------

    def _build_tabview(self) -> None:
        self.tabview = ctk.CTkTabview(
            self, fg_color="#0d0e15",
            segmented_button_fg_color="#14151e",
            segmented_button_selected_color="#2d2a3d",
            segmented_button_selected_hover_color="#3d3a4d",
            segmented_button_unselected_color="#14151e",
            segmented_button_unselected_hover_color="#1f2030",
            text_color="#6b7280",
        )
        self.tabview.grid(row=1, column=0, sticky="nsew")
        for name in ("Dashboard", "Generate", "Preview", "Editor",
                      "Ideas", "Accounts", "Settings"):
            self.tabview.add(name)
        self._build_dashboard()
        self._build_generate()
        self._build_preview()
        self._build_editor()
        self._build_ideas()
        self._build_accounts()
        self._build_settings()
        self.tabview.set("Dashboard")

    # -- Dashboard tab ----------------------------------------------------

    def _build_dashboard(self) -> None:
        t = self.tabview.tab("Dashboard")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(2, weight=1)

        cards = ctk.CTkFrame(t, fg_color="transparent")
        cards.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        cards.grid_columnconfigure((0, 1, 2), weight=1)
        self.card_total = self._stat_card(cards, "0", "Total Generated")
        self.card_total.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=4)
        self.card_today = self._stat_card(cards, "0", "Generated Today")
        self.card_today.grid(row=0, column=1, sticky="ew", padx=(2, 4), pady=4)
        self.card_auth = self._stat_card(cards, "Check", "Auth Status")
        self.card_auth.grid(row=0, column=2, sticky="ew", padx=(4, 0), pady=4)

        actions = self._section_frame(t, "Quick Actions")
        actions.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 0))
        act_row = ctk.CTkFrame(actions, fg_color="transparent")
        act_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 10))
        ctk.CTkButton(
            act_row, text="Generate New Video",
            font=ctk.CTkFont(weight="bold"), fg_color="#8b5cf6",
            hover_color="#7c3aed", corner_radius=8, height=36,
            command=self._start,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            act_row, text="Open Output", corner_radius=8, height=36,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            command=lambda: self._open_output_file(),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            act_row, text="Settings", corner_radius=8, height=36,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            command=lambda: self.tabview.set("Settings"),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            act_row, text="Preview Last", corner_radius=8, height=36,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            command=lambda: self._load_for_preview("final_short.mp4"),
        ).pack(side="left", padx=3)

        rec = self._section_frame(t, "Recent Output")
        rec.grid(row=2, column=0, sticky="nsew", padx=16, pady=(10, 14))
        rec.grid_columnconfigure(0, weight=1)
        rec.grid_rowconfigure(1, weight=1)
        self.dash_console = ColoredConsole(rec, height=10)
        self.dash_console.grid(
            row=1, column=0, sticky="nsew", padx=14, pady=(0, 10)
        )
        self.dash_console.insert(
            END, "[GUI] Ready. Click 'Generate New Video' to start.\n", "gui"
        )
        self.dash_console.configure(state="disabled")

    def _open_output_file(self) -> None:
        out = self.settings.get("output_file", "outputs/final_short.mp4")
        path = os.path.join(BASE_DIR, out)
        if os.path.exists(path):
            if os.name == "nt":
                os.startfile(BASE_DIR)
            else:
                subprocess.run(["open", BASE_DIR])
        else:
            messagebox.showinfo("Info", f"Output file not found:\n{path}")

    # -- Generate tab -----------------------------------------------------

    def _build_generate(self) -> None:
        t = self.tabview.tab("Generate")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(4, weight=1)

        ctrl = self._section_frame(t, "Generation Controls")
        ctrl.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        ctrl.grid_columnconfigure(1, weight=1)
        inner = ctk.CTkFrame(ctrl, fg_color="transparent")
        inner.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))
        r = 0
        self._labeled_combo(
            inner, "AI Model:", MODEL_CHOICES,
            self.settings.get("ai_model", MODEL_CHOICES[0]), "model", r,
        )
        r += 1
        self._labeled_entry(
            inner, "BG Video:", self.settings.get("bg_video", ""), "bg", r, True,
        )
        r += 1

        up_row = ctk.CTkFrame(inner, fg_color="transparent")
        up_row.grid(row=r, column=1, columnspan=2, sticky="ew", pady=4)
        r += 1
        self.upload_var = ctk.BooleanVar(value=self.settings.get("auto_upload", True))
        ctk.CTkSwitch(
            up_row, text="Auto Upload to YouTube", variable=self.upload_var,
            font=ctk.CTkFont(size=12), text_color="#d1d5db",
            progress_color="#8b5cf6", button_color="#8b5cf6",
            button_hover_color="#7c3aed",
        ).pack(side="left")

        self._labeled_entry(
            inner, "Output:", self.settings.get("output_file", "outputs/final_short.mp4"),
            "output", r, False,
        )
        r += 1

        enrich = self._section_frame(t, "Enrichment Features")
        enrich.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 0))
        enrich.grid_columnconfigure(1, weight=1)
        e_inner = ctk.CTkFrame(enrich, fg_color="transparent")
        e_inner.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 8))
        er = 0
        ctk.CTkLabel(
            e_inner, text="Watermark:", text_color="#d1d5db",
            font=ctk.CTkFont(size=12), anchor="w",
        ).grid(row=er, column=0, sticky="w", padx=(0, 8), pady=2)
        self.enrich_watermark = ctk.CTkEntry(
            e_inner, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=200,
        )
        self.enrich_watermark.insert(0, self.settings.get("watermark_text", ""))
        self.enrich_watermark.grid(row=er, column=1, sticky="w", pady=2)
        er += 1

        ctk.CTkLabel(
            e_inner, text="Subtitle Color:", text_color="#d1d5db",
            font=ctk.CTkFont(size=12), anchor="w",
        ).grid(row=er, column=0, sticky="w", padx=(0, 8), pady=2)
        self.enrich_sub_color = ctk.CTkComboBox(
            e_inner, values=SUBTITLE_COLORS, width=140,
            fg_color="#1a1b26", border_color="#2d2f3d", button_color="#2d2f3d",
            dropdown_fg_color="#1a1b26", dropdown_hover_color="#2d2f3d",
            dropdown_text_color="#e2e8f0", corner_radius=6,
        )
        self.enrich_sub_color.set(self.settings.get("subtitle_color", "yellow"))
        self.enrich_sub_color.grid(row=er, column=1, sticky="w", pady=2)
        er += 1

        ctk.CTkLabel(
            e_inner, text="Quiz Prefix:", text_color="#d1d5db",
            font=ctk.CTkFont(size=12), anchor="w",
        ).grid(row=er, column=0, sticky="w", padx=(0, 8), pady=2)
        self.enrich_quiz_prefix = ctk.CTkEntry(
            e_inner, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=200,
        )
        self.enrich_quiz_prefix.insert(0, self.settings.get("quiz_prefix", "Answer: "))
        self.enrich_quiz_prefix.grid(row=er, column=1, sticky="w", pady=2)
        er += 1

        btns = ctk.CTkFrame(inner, fg_color="transparent")
        btns.grid(row=r, column=1, columnspan=2, sticky="w", pady=(6, 0))
        r += 1
        self.gen_btn = ctk.CTkButton(
            btns, text="Start Generation", font=ctk.CTkFont(weight="bold"),
            fg_color="#8b5cf6", hover_color="#7c3aed", corner_radius=8,
            height=40, width=165, command=self._start,
        )
        self.gen_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ctk.CTkButton(
            btns, text="Stop", font=ctk.CTkFont(weight="bold"),
            fg_color="#2a1a1a", hover_color="#3a2020",
            border_color="#ef4444", border_width=1,
            text_color="#ef4444", text_color_disabled="#374151",
            corner_radius=8, height=40, width=85, state="disabled",
            command=self._stop,
        )
        self.stop_btn.pack(side="left")

        ci = self._section_frame(t, "Custom Instructions")
        ci.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 0))
        ci.grid_columnconfigure(0, weight=1)
        ci.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(
            ci, text="Injected into the AI agent's system prompt to guide behaviour.",
            font=ctk.CTkFont(size=11), text_color="#6b7280",
            anchor="w", wraplength=700,
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 4))
        self.ci_textbox = ctk.CTkTextbox(
            ci, fg_color="#1a1b26", border_color="#2d2f3d", border_width=1,
            corner_radius=6, height=70,
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#e2e8f0", wrap="word",
        )
        self.ci_textbox.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 8))
        self.ci_textbox.insert("1.0", self.settings.get("custom_instructions", ""))

        self.progress_bar = ctk.CTkProgressBar(
            t, fg_color="#1a1b26", progress_color="#8b5cf6",
            border_color="#1e2030", border_width=1, corner_radius=6, height=12,
        )
        self.progress_bar.grid(row=3, column=0, sticky="ew", padx=16, pady=(6, 0))
        self.progress_bar.set(0)

        con = self._section_frame(t, "Live Console Output")
        con.grid(row=4, column=0, sticky="nsew", padx=16, pady=(6, 14))
        con.grid_columnconfigure(0, weight=1)
        con.grid_rowconfigure(1, weight=1)
        ch = ctk.CTkFrame(con, fg_color="transparent")
        ch.grid(row=0, column=0, sticky="ew", padx=14, pady=(6, 0))
        ctk.CTkButton(
            ch, text="Clear", width=55, height=24, corner_radius=5,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=10),
            command=lambda: self._clear_console(self.gen_console),
        ).pack(side="right")
        self.gen_console = ColoredConsole(con)
        self.gen_console.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.gen_console.configure(state="disabled")

    # -- Preview tab ------------------------------------------------------

    def _build_preview(self) -> None:
        t = self.tabview.tab("Preview")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(0, weight=1)
        main = ctk.CTkFrame(t, fg_color="transparent")
        main.grid(row=0, column=0, sticky="nsew", padx=16, pady=(14, 14))
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        img_frame = self._section_frame(main, "Video Preview")
        img_frame.grid(row=0, column=0, sticky="nsew")
        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_rowconfigure(1, weight=1)

        self.preview_label = ctk.CTkLabel(
            img_frame,
            text="No video loaded.\nClick 'Browse' to select a video.",
            fg_color="#0b0c12", corner_radius=8,
            font=ctk.CTkFont(size=14), text_color="#6b7280",
            anchor="center", height=320,
        )
        self.preview_label.grid(
            row=1, column=0, sticky="nsew", padx=14, pady=(0, 10)
        )

        ctrl_bar = ctk.CTkFrame(img_frame, fg_color="transparent")
        ctrl_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.preview_path_entry = ctk.CTkEntry(
            ctrl_bar, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=30, placeholder_text="Video file path...",
        )
        self.preview_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(
            ctrl_bar, text="Browse", width=70, height=30, corner_radius=6,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=11),
            command=self._browse_preview_video,
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            ctrl_bar, text="Load Preview", width=100, height=30, corner_radius=6,
            fg_color="#8b5cf6", hover_color="#7c3aed",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._load_preview,
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            ctrl_bar, text="Play", width=60, height=30, corner_radius=6,
            fg_color="#10b981", hover_color="#059669",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._play_video,
        ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(
            ctrl_bar, text="Snapshot", width=70, height=30, corner_radius=6,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=11),
            command=self._capture_snapshot,
        ).pack(side="left")

        nav_bar = ctk.CTkFrame(img_frame, fg_color="transparent")
        nav_bar.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))
        ctk.CTkLabel(
            nav_bar, text="Snapshot time (s):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 6))
        self.preview_time_entry = ctk.CTkEntry(
            nav_bar, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=70,
        )
        self.preview_time_entry.insert(0, "0.0")
        self.preview_time_entry.pack(side="left", padx=(0, 8))
        self.preview_auto_var = ctk.BooleanVar(
            value=self.settings.get("preview_auto_refresh", False)
        )
        ctk.CTkSwitch(
            nav_bar, text="Auto-refresh after generation",
            variable=self.preview_auto_var,
            font=ctk.CTkFont(size=11), text_color="#d1d5db",
            progress_color="#8b5cf6", button_color="#8b5cf6",
        ).pack(side="left")

    def _browse_preview_video(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv")]
        )
        if path:
            self.preview_path_entry.delete(0, "end")
            self.preview_path_entry.insert(0, path)
            self._load_for_preview(path)

    def _load_for_preview(self, path: str | None = None) -> None:
        if not path:
            path = self.preview_path_entry.get().strip()
        if not path or not os.path.exists(
            os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
        ):
            path = (
                os.path.join(BASE_DIR, path)
                if not os.path.isabs(path)
                else path
            )
        if not os.path.exists(path):
            self._set_status("error", "Video not found")
            return
        self.preview_path_entry.delete(0, "end")
        self.preview_path_entry.insert(0, path)
        try:
            from tools import get_video_duration, get_video_dimensions
            dur = get_video_duration(path)
            w, h = get_video_dimensions(path)
            res_str = f"{w}x{h}" if w and h else "unknown"
            self._set_status("running", f"Loaded: {dur:.1f}s ({res_str})")
        except Exception:
            self._set_status("running", "Loaded")
        self._capture_snapshot()

    def _load_preview(self) -> None:
        self._load_for_preview()

    def _capture_snapshot(self) -> None:
        path = self.preview_path_entry.get().strip()
        if not path or not os.path.exists(
            os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
        ):
            path = (
                os.path.join(BASE_DIR, path)
                if not os.path.isabs(path)
                else path
            )
        if not os.path.exists(path):
            self.preview_label.configure(text="Video file not found.", image=None)
            return
        try:
            t_str = self.preview_time_entry.get().strip() or "0"
            t = float(t_str)
        except ValueError:
            t = 0
        snapshot_path = os.path.join(BASE_DIR, "preview_snapshot.jpg")
        try:
            from tools import get_video_snapshot
            ok = get_video_snapshot(path, t, snapshot_path)
            if ok and os.path.exists(snapshot_path):
                img = ctk.CTkImage(
                    light_image=PILImage.open(snapshot_path),
                    dark_image=PILImage.open(snapshot_path),
                    size=(420, 700),
                )
                self.preview_label.configure(text="", image=img)
            else:
                self.preview_label.configure(
                    text=f"No snapshot at {t}s", image=None
                )
        except Exception as e:
            self.preview_label.configure(text=f"Preview error: {e}", image=None)

    def _play_video(self) -> None:
        path = self.preview_path_entry.get().strip()
        if not path:
            return
        full_path = (
            os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
        )
        if not os.path.exists(full_path):
            messagebox.showerror("Error", f"Video not found:\n{full_path}")
            return
        if os.name == "nt":
            try:
                os.startfile(full_path)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot play video: {e}")
        else:
            subprocess.Popen(["xdg-open", full_path])

    def _refresh_preview_on_startup(self) -> None:
        out = self.settings.get("output_file", "outputs/final_short.mp4")
        path = os.path.join(BASE_DIR, out)
        if os.path.exists(path):
            self.preview_path_entry.delete(0, "end")
            self.preview_path_entry.insert(0, out)

    # -- Editor tab -------------------------------------------------------

    def _build_editor(self) -> None:
        t = self.tabview.tab("Editor")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(2, weight=1)

        hdr = ctk.CTkFrame(t, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        ctk.CTkLabel(
            hdr, text="In-App Video Editor",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#f1f5f9",
        ).pack(side="left")

        load_frame = ctk.CTkFrame(
            t, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        load_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        load_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            load_frame, text="Source Video:", text_color="#d1d5db",
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 2))
        self.editor_source_entry = ctk.CTkEntry(
            load_frame, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        self.editor_source_entry.insert(
            0, self.settings.get("output_file", "outputs/final_short.mp4")
        )
        self.editor_source_entry.grid(
            row=0, column=1, sticky="ew", padx=6, pady=(10, 2)
        )
        ctk.CTkButton(
            load_frame, text="Browse", width=60, height=28, corner_radius=5,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=10),
            command=self._editor_browse,
        ).grid(row=0, column=2, padx=(0, 14), pady=(10, 2))
        ctk.CTkLabel(
            load_frame, text="Output File:", text_color="#d1d5db",
            font=ctk.CTkFont(size=12),
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(2, 10))
        self.editor_output_entry = ctk.CTkEntry(
            load_frame, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        self.editor_output_entry.insert(0, "edited_output.mp4")
        self.editor_output_entry.grid(
            row=1, column=1, sticky="ew", padx=6, pady=(2, 10)
        )

        edit_frame = ctk.CTkFrame(
            t, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        edit_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 14))
        edit_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            edit_frame, text="Editing Controls",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 4))

        # trim section
        trim_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        trim_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8)
        )
        ctk.CTkLabel(
            trim_frame, text="Trim:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(
            trim_frame, text="Start (s):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, sticky="w", padx=(0, 6))
        self.editor_trim_start = ctk.CTkEntry(
            trim_frame, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=80,
        )
        self.editor_trim_start.insert(0, "0")
        self.editor_trim_start.grid(row=1, column=1, sticky="w", padx=(0, 14))
        ctk.CTkLabel(
            trim_frame, text="End (s):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=2, sticky="w", padx=(0, 6))
        self.editor_trim_end = ctk.CTkEntry(
            trim_frame, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=80,
        )
        self.editor_trim_end.insert(0, "60")
        self.editor_trim_end.grid(row=1, column=3, sticky="w")
        ctk.CTkButton(
            trim_frame, text="Apply Trim", width=90, height=30, corner_radius=6,
            fg_color="#8b5cf6", hover_color="#7c3aed",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._editor_trim,
        ).grid(row=1, column=4, padx=(16, 0))

        # effects section
        effect_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        effect_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8)
        )
        ctk.CTkLabel(
            effect_frame, text="Effects:",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.editor_effect_cb = ctk.CTkComboBox(
            effect_frame, values=["blur", "grayscale", "vignette"], width=120,
            fg_color="#1a1b26", border_color="#2d2f3d", button_color="#2d2f3d",
            dropdown_fg_color="#1a1b26", dropdown_hover_color="#2d2f3d",
            dropdown_text_color="#e2e8f0", corner_radius=6,
        )
        self.editor_effect_cb.set("blur")
        self.editor_effect_cb.grid(row=1, column=0, sticky="w")
        ctk.CTkLabel(
            effect_frame, text="Start (s):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=1, sticky="w", padx=(14, 6))
        self.editor_effect_start = ctk.CTkEntry(
            effect_frame, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=70,
        )
        self.editor_effect_start.insert(0, "0")
        self.editor_effect_start.grid(row=1, column=2, sticky="w", padx=(0, 14))
        ctk.CTkLabel(
            effect_frame, text="Dur (s):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=3, sticky="w", padx=(0, 6))
        self.editor_effect_dur = ctk.CTkEntry(
            effect_frame, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=26, width=70,
        )
        self.editor_effect_dur.insert(0, "5")
        self.editor_effect_dur.grid(row=1, column=4, sticky="w")
        ctk.CTkButton(
            effect_frame, text="Apply Effect", width=90, height=30,
            corner_radius=6, fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=11),
            command=self._editor_effect,
        ).grid(row=1, column=5, padx=(16, 0))

        info_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        info_frame.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 8)
        )
        self.editor_info = ctk.CTkLabel(
            info_frame, text="Load a video to begin editing.",
            font=ctk.CTkFont(size=11), text_color="#6b7280",
            anchor="w", wraplength=600,
        )
        self.editor_info.pack(side="left")
        ctk.CTkButton(
            info_frame, text="Show Info", width=80, height=28, corner_radius=5,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=10),
            command=self._editor_show_info,
        ).pack(side="right")

    def _editor_browse(self) -> None:
        """Open a file picker to select the source video for editing."""
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv")]
        )
        if path:
            self.editor_source_entry.delete(0, "end")
            self.editor_source_entry.insert(0, path)
            self._editor_show_info()

    def _editor_show_info(self) -> None:
        """Display duration, resolution, and file size for the source video."""
        src = self._editor_source_path()
        if not os.path.exists(src):
            self.editor_info.configure(text="Source file not found.")
            return
        try:
            from tools import get_video_duration, get_video_dimensions
            dur = get_video_duration(src)
            w, h = get_video_dimensions(src)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            self.editor_info.configure(
                text=f"Duration: {dur:.1f}s | Resolution: {w}x{h} "
                     f"| Size: {size_mb:.1f}MB"
            )
        except Exception as e:
            self.editor_info.configure(text=f"Error reading video: {e}")

    def _editor_source_path(self) -> str:
        src = self.editor_source_entry.get().strip()
        return os.path.join(BASE_DIR, src) if not os.path.isabs(src) else src

    def _editor_output_path(self) -> str:
        out = self.editor_output_entry.get().strip()
        return os.path.join(BASE_DIR, out) if not os.path.isabs(out) else out

    def _editor_trim(self) -> None:
        """Trim the source video to the specified start/end times via ffmpeg."""
        src = self._editor_source_path()
        out = self._editor_output_path()
        if not os.path.exists(src):
            messagebox.showerror("Error", "Source video not found.")
            return
        try:
            start = float(self.editor_trim_start.get() or 0)
            end = float(self.editor_trim_end.get() or 60)
        except ValueError:
            messagebox.showerror("Error", "Invalid trim times.")
            return
        from tools import trim_video
        ok = trim_video(src, out, start, end)
        if ok:
            self._set_status("success", "Trim complete")
            messagebox.showinfo("Success", f"Trimmed video saved to:\n{out}")
        else:
            self._set_status("error", "Trim failed")
            messagebox.showerror("Error", "Failed to trim video.")

    def _editor_effect(self) -> None:
        """Apply a video filter (blur, grayscale, vignette) via ffmpeg."""
        src = self._editor_source_path()
        out = self._editor_output_path()
        if not os.path.exists(src):
            messagebox.showerror("Error", "Source video not found.")
            return
        effect = self.editor_effect_cb.get()
        try:
            start = float(self.editor_effect_start.get() or 0)
            dur = (
                float(self.editor_effect_dur.get() or 5)
                if self.editor_effect_dur.get().strip()
                else None
            )
        except ValueError:
            messagebox.showerror("Error", "Invalid effect parameters.")
            return
        from tools import add_effect_overlay
        ok = add_effect_overlay(src, out, effect, start, dur)
        if ok:
            self._set_status("success", f"{effect} applied")
            messagebox.showinfo(
                "Success", f"Effect '{effect}' applied.\nSaved to:\n{out}"
            )
        else:
            self._set_status("error", "Effect failed")
            messagebox.showerror("Error", f"Failed to apply {effect}.")

    # -- Ideas tab --------------------------------------------------------

    def _build_ideas(self) -> None:
        t = self.tabview.tab("Ideas")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(1, weight=1)
        hdr = ctk.CTkFrame(t, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            hdr, text="Ideas History",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#f1f5f9",
        ).pack(side="left")
        ctk.CTkButton(
            hdr, text="Refresh", width=70, height=28, corner_radius=6,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=11), command=self._refresh_ideas,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            hdr, text="Clear All", width=70, height=28, corner_radius=6,
            fg_color="#2a1a1a", hover_color="#3a2020",
            border_color="#3b1e1e", border_width=1,
            font=ctk.CTkFont(size=11), command=self._clear_ideas,
        ).pack(side="right")
        self.ideas_scroll = ctk.CTkScrollableFrame(t, fg_color="transparent")
        self.ideas_scroll.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self.ideas_scroll.grid_columnconfigure(0, weight=1)

    def _refresh_ideas(self) -> None:
        """Populate the Ideas tab with cards for every generated video spec."""
        for w in self.ideas_scroll.winfo_children():
            w.destroy()
        try:
            from agent import load_ideas_history
            ideas = load_ideas_history()
        except Exception:
            ideas = []
        if not ideas:
            ctk.CTkLabel(
                self.ideas_scroll, text="No ideas generated yet.",
                text_color="#6b7280", font=ctk.CTkFont(size=13),
            ).pack(pady=40)
            return
        for idea in reversed(ideas):
            card = ctk.CTkFrame(
                self.ideas_scroll, fg_color="#14151e", border_color="#1e2030",
                border_width=1, corner_radius=10,
            )
            card.pack(fill="x", pady=3)
            title_row = ctk.CTkFrame(card, fg_color="transparent")
            title_row.pack(fill="x", padx=12, pady=(8, 2))
            ctk.CTkLabel(
                title_row, text=idea.get("title", "Untitled"),
                font=ctk.CTkFont(size=13, weight="bold"), text_color="#f1f5f9",
            ).pack(side="left")
            ts = idea.get("timestamp", "")[:19].replace("T", " ")
            ctk.CTkLabel(
                title_row, text=ts, font=ctk.CTkFont(size=10),
                text_color="#6b7280",
            ).pack(side="right")
            hook = idea.get("hook_text", "")
            if hook:
                ctk.CTkLabel(
                    card, text=f"Hook: {hook}", font=ctk.CTkFont(size=11),
                    text_color="#a78bfa", anchor="w",
                ).pack(anchor="w", padx=12)
            img = idea.get("image_query", "")
            if img:
                ctk.CTkLabel(
                    card, text=f"Image: {img}", font=ctk.CTkFont(size=10),
                    text_color="#6b7280", anchor="w",
                ).pack(anchor="w", padx=12)
            enrich = idea.get("enrichment", {})
            if enrich.get("quiz_question"):
                ctk.CTkLabel(
                    card, text=f"Quiz: {enrich['quiz_question']}",
                    font=ctk.CTkFont(size=10), text_color="#fbbf24", anchor="w",
                ).pack(anchor="w", padx=12)
            sp = idea.get("script", "")[:150]
            if len(idea.get("script", "")) > 150:
                sp += "..."
            ctk.CTkLabel(
                card, text=sp, font=ctk.CTkFont(size=11),
                text_color="#9ca3af", anchor="w", wraplength=800,
            ).pack(anchor="w", padx=12, pady=(2, 8))
        self._update_dashboard_stats(ideas)

    def _update_dashboard_stats(self, ideas: list[dict[str, Any]]) -> None:
        total = len(ideas)
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(
            1 for i in ideas if i.get("timestamp", "").startswith(today)
        )
        self.card_total.configure(text=str(total))
        self.card_today.configure(text=str(today_count))
        try:
            from uploader import get_auth_status
            auth = get_auth_status(
                self.settings.get("selected_account", "Default")
            )
            status_text = "Authenticated" if auth.get("is_authenticated") else "Not Auth"
            self.card_auth.configure(text=status_text)
        except Exception:
            self.card_auth.configure(text="Unknown")

    def _clear_ideas(self) -> None:
        if not messagebox.askyesno("Clear Ideas", "Delete all idea history?"):
            return
        try:
            os.remove(os.path.join(BASE_DIR, "ideas_history.json"))
        except OSError:
            pass
        self._refresh_ideas()

    # -- Accounts tab -----------------------------------------------------

    def _build_accounts(self) -> None:
        t = self.tabview.tab("Accounts")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(3, weight=1)
        hdr = ctk.CTkFrame(t, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        ctk.CTkLabel(
            hdr, text="YouTube Accounts",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#f1f5f9",
        ).pack(side="left")
        ctk.CTkButton(
            hdr, text="Add Account", width=100, height=30, corner_radius=6,
            fg_color="#8b5cf6", hover_color="#7c3aed",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._add_account,
        ).pack(side="right")
        ctk.CTkButton(
            hdr, text="Refresh", width=70, height=30, corner_radius=6,
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            font=ctk.CTkFont(size=12), command=self._refresh_accounts,
        ).pack(side="right", padx=(4, 0))

        sel_row = ctk.CTkFrame(t, fg_color="transparent")
        sel_row.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
        ctk.CTkLabel(
            sel_row, text="Active Upload Account:", text_color="#d1d5db",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=(0, 6))
        self.account_cb = ctk.CTkComboBox(
            sel_row, values=["Default"], width=200,
            fg_color="#1a1b26", border_color="#2d2f3d",
            button_color="#2d2f3d", button_hover_color="#3d3f4d",
            dropdown_fg_color="#1a1b26", dropdown_hover_color="#2d2f3d",
            dropdown_text_color="#e2e8f0", corner_radius=6,
            command=self._on_account_select,
        )
        self.account_cb.pack(side="left")
        self.account_cb.set(self.settings.get("selected_account", "Default"))
        ctk.CTkButton(
            sel_row, text="Re-Auth", width=70, height=28, corner_radius=6,
            fg_color="#2a1a1a", hover_color="#3a2020",
            border_color="#3b1e1e", border_width=1,
            font=ctk.CTkFont(size=10),
            command=self._reauthenticate,
        ).pack(side="left", padx=(10, 0))

        auth_frame = self._section_frame(t, "Auth Status")
        auth_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 4))
        self.auth_status_label = ctk.CTkLabel(
            auth_frame, text="Loading...", font=ctk.CTkFont(size=12),
            text_color="#6b7280", anchor="w",
        )
        self.auth_status_label.grid(
            row=1, column=0, sticky="w", padx=14, pady=(2, 10)
        )

        self.accounts_scroll = ctk.CTkScrollableFrame(t, fg_color="transparent")
        self.accounts_scroll.grid(
            row=3, column=0, sticky="nsew", padx=16, pady=(0, 14)
        )
        self.accounts_scroll.grid_columnconfigure(0, weight=1)
        self._refresh_accounts()

    def _refresh_accounts(self) -> None:
        """Rebuild the account list from ``client_secret_*.json`` files."""
        for w in self.accounts_scroll.winfo_children():
            w.destroy()
        try:
            from uploader import list_available_accounts, get_all_auth_statuses
            accounts = list_available_accounts()
            auth_data = get_all_auth_statuses()
        except Exception:
            accounts = ["Default"]
            auth_data = {"accounts": [], "total_authenticated": 0}

        current = self.account_cb.get()
        self.account_cb.configure(values=accounts)
        if current in accounts:
            self.account_cb.set(current)
        elif "Default" in accounts:
            self.account_cb.set("Default")

        auth_map = {a["account"]: a for a in auth_data.get("accounts", [])}

        for acc in accounts:
            a_info = auth_map.get(acc, {})
            row = ctk.CTkFrame(
                self.accounts_scroll, fg_color="#14151e",
                border_color="#1e2030", border_width=1, corner_radius=8,
            )
            row.pack(fill="x", pady=2)
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)
            ctk.CTkLabel(
                inner, text=acc, font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#f1f5f9",
            ).pack(side="left")
            if a_info.get("is_authenticated"):
                ctk.CTkLabel(
                    inner, text="(auth)", font=ctk.CTkFont(size=10),
                    text_color="#10b981",
                ).pack(side="left", padx=6)
            else:
                ctk.CTkLabel(
                    inner, text="(not auth)", font=ctk.CTkFont(size=10),
                    text_color="#ef4444",
                ).pack(side="left", padx=6)
            exp = a_info.get("expires_in_minutes")
            if exp is not None and exp > 0:
                ctk.CTkLabel(
                    inner, text=f"expires in {exp}m",
                    font=ctk.CTkFont(size=9), text_color="#6b7280",
                ).pack(side="left", padx=6)
            ctk.CTkButton(
                inner, text="Set Active", width=80, height=26, corner_radius=5,
                fg_color="#1e2030", hover_color="#2a2c3d",
                border_color="#2d2f3d", border_width=1,
                font=ctk.CTkFont(size=10),
                command=lambda a=acc: self._set_active(a),
            ).pack(side="right")
            if acc != "Default":
                ctk.CTkButton(
                    inner, text="Remove", width=65, height=26, corner_radius=5,
                    fg_color="#2a1a1a", hover_color="#3a2020",
                    border_color="#3b1e1e", border_width=1,
                    text_color="#ef4444", font=ctk.CTkFont(size=10),
                    command=lambda a=acc: self._remove_account(a),
                ).pack(side="right", padx=4)

        total_auth = auth_data.get("total_authenticated", 0)
        self.auth_status_label.configure(
            text=(
                f"{total_auth}/{len(accounts)} accounts authenticated | "
                f"Token auto-refresh enabled"
            )
        )

    def _set_active(self, name: str) -> None:
        self.account_cb.set(name)
        self.settings["selected_account"] = name
        save_settings(self.settings)

    def _on_account_select(self, choice: str) -> None:
        self.settings["selected_account"] = choice
        save_settings(self.settings)

    def _reauthenticate(self) -> None:
        acc = self.account_cb.get()
        if not messagebox.askyesno(
            "Re-Authenticate",
            f"Force re-authentication for '{acc}'?\nExisting token will be cleared.",
        ):
            return
        try:
            from uploader import force_reauthenticate
            ok = force_reauthenticate(acc)
            if ok:
                self._set_status("success", "Re-authenticated")
                messagebox.showinfo("Success", f"Re-authenticated '{acc}'")
            else:
                messagebox.showerror("Error", "Re-authentication failed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self._refresh_accounts()

    def _add_account(self) -> None:
        path = filedialog.askopenfilename(
            title="Select client_secret JSON file",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        from tkinter import simpledialog
        name = simpledialog.askstring(
            "Account Name", "Enter a short name for this account:", parent=self
        )
        if not name or not name.strip():
            return
        name = name.strip().replace(" ", "_")
        try:
            from uploader import add_account
            if add_account(path, name):
                self._refresh_accounts()
            else:
                messagebox.showerror("Error", "Failed to copy client_secret file.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _remove_account(self, name: str) -> None:
        if not messagebox.askyesno(
            "Remove Account",
            f"Remove account '{name}'?\nDeletes client_secret_{name}.json",
        ):
            return
        try:
            from uploader import remove_account
            if remove_account(name):
                if self.account_cb.get() == name:
                    self.account_cb.set("Default")
                    self.settings["selected_account"] = "Default"
                    save_settings(self.settings)
                self._refresh_accounts()
            else:
                messagebox.showerror("Error", "Failed to remove account.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -- Settings tab -----------------------------------------------------

    def _build_settings(self) -> None:
        t = self.tabview.tab("Settings")
        t.grid_columnconfigure(0, weight=1)
        t.grid_rowconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(
            t, fg_color="transparent", scrollbar_fg_color="#14151e"
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=16, pady=(14, 8))
        scroll.grid_columnconfigure(0, weight=1)

        s_api = ctk.CTkFrame(
            scroll, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        s_api.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        s_api.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            s_api, text="API Configuration",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 4))

        ctk.CTkLabel(
            s_api, text="OpenRouter Key:", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, sticky="w", padx=(14, 6), pady=3)
        self.s_api_key = ctk.CTkEntry(
            s_api, show="*", fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        self.s_api_key.insert(0, self.settings.get("openrouter_api_key", ""))
        self.s_api_key.grid(row=1, column=1, sticky="ew", pady=3)
        self.show_key_var = ctk.BooleanVar()
        ctk.CTkCheckBox(
            s_api, text="Show", variable=self.show_key_var,
            font=ctk.CTkFont(size=10), text_color="#9ca3af",
            border_color="#4b5563", checkmark_color="#8b5cf6",
            command=lambda: self.s_api_key.configure(
                show="" if self.show_key_var.get() else "*"
            ),
        ).grid(row=1, column=2, padx=(4, 14), pady=3)

        ctk.CTkLabel(
            s_api, text="Pixabay Key (free):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=2, column=0, sticky="w", padx=(14, 6), pady=3)
        self.s_pixabay_key = ctk.CTkEntry(
            s_api, show="*", fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        self.s_pixabay_key.insert(0, self.settings.get("pixabay_api_key", ""))
        self.s_pixabay_key.grid(row=2, column=1, sticky="ew", pady=3)

        ctk.CTkLabel(
            s_api, text="Pexels Key (free):", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, sticky="w", padx=(14, 6), pady=3)
        self.s_pexels_key = ctk.CTkEntry(
            s_api, show="*", fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        self.s_pexels_key.insert(0, self.settings.get("pexels_api_key", ""))
        self.s_pexels_key.grid(row=3, column=1, sticky="ew", pady=3)

        ctk.CTkLabel(
            s_api, text="TTS Voice:", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=4, column=0, sticky="w", padx=(14, 6), pady=(3, 10))
        self.s_voice = ctk.CTkComboBox(
            s_api, values=VOICE_CHOICES, width=220, fg_color="#1a1b26",
            border_color="#2d2f3d", button_color="#2d2f3d",
            button_hover_color="#3d3f4d", dropdown_fg_color="#1a1b26",
            dropdown_hover_color="#2d2f3d", dropdown_text_color="#e2e8f0",
            corner_radius=6,
        )
        self.s_voice.set(self.settings.get("tts_voice", VOICE_CHOICES[0]))
        self.s_voice.grid(row=4, column=1, columnspan=2, sticky="w", pady=(3, 10))

        s_paths = ctk.CTkFrame(
            scroll, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        s_paths.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        s_paths.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            s_paths, text="File Paths",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 4))
        self._spath(
            s_paths, 1, "ImageMagick:", "s_im",
            self.settings.get("imagemagick_path", ""),
            [("Exe", "*.exe")],
        )
        self._spath(
            s_paths, 2, "BG Video:", "s_bg",
            self.settings.get("bg_video", ""),
            [("Videos", "*.mp4 *.mov *.avi")],
        )
        ctk.CTkLabel(
            s_paths, text="Output File:", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, sticky="w", padx=(14, 6), pady=(3, 10))
        self.s_output = ctk.CTkEntry(
            s_paths, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28,
        )
        self.s_output.insert(
            0, self.settings.get("output_file", "outputs/final_short.mp4")
        )
        self.s_output.grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(0, 14), pady=(3, 10)
        )

        s_search = ctk.CTkFrame(
            scroll, fg_color="#14151e", border_color="#1e2030",
            border_width=1, corner_radius=10,
        )
        s_search.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        s_search.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            s_search, text="Search Configuration",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#f1f5f9",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 4))
        ctk.CTkLabel(
            s_search, text="Max Results:", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=1, column=0, sticky="w", padx=(14, 6), pady=3)
        self.s_results = ctk.CTkEntry(
            s_search, fg_color="#1a1b26", border_color="#2d2f3d",
            corner_radius=6, height=28, width=80,
        )
        self.s_results.insert(0, str(self.settings.get("max_search_results", 4)))
        self.s_results.grid(row=1, column=1, sticky="w", pady=3)
        ctk.CTkLabel(
            s_search, text="Time Limit:", text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
        ).grid(row=2, column=0, sticky="w", padx=(14, 6), pady=(3, 10))
        tl_vals = list(TIME_LIMIT_MAP.keys())
        self.s_time = ctk.CTkComboBox(
            s_search, values=tl_vals, width=150, fg_color="#1a1b26",
            border_color="#2d2f3d", button_color="#2d2f3d",
            button_hover_color="#3d3f4d", dropdown_fg_color="#1a1b26",
            dropdown_hover_color="#2d2f3d", dropdown_text_color="#e2e8f0",
            corner_radius=6,
        )
        displayed = TIME_LIMIT_REVERSE.get(
            self.settings.get("search_time_limit", "w"), "7 days"
        )
        self.s_time.set(displayed)
        self.s_time.grid(row=2, column=1, sticky="w", pady=(3, 10))

        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ctk.CTkButton(
            btn_row, text="Save All Settings",
            font=ctk.CTkFont(weight="bold"), fg_color="#8b5cf6",
            hover_color="#7c3aed", corner_radius=8, height=36, width=150,
            command=self._save_all_settings,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Reset to Defaults",
            fg_color="#1e2030", hover_color="#2a2c3d",
            border_color="#2d2f3d", border_width=1,
            corner_radius=8, height=36, width=150,
            command=self._reset_all,
        ).pack(side="left")

    def _save_all_settings(self) -> None:
        """Collect values from all Settings widgets and persist to config.json."""
        s = self.settings
        s["openrouter_api_key"] = self.s_api_key.get()
        s["pixabay_api_key"] = self.s_pixabay_key.get()
        s["pexels_api_key"] = self.s_pexels_key.get()
        s["tts_voice"] = self.s_voice.get()
        s["imagemagick_path"] = getattr(self, "_s_im").get()
        s["bg_video"] = getattr(self, "_s_bg").get()
        s["output_file"] = self.s_output.get()
        try:
            s["max_search_results"] = int(self.s_results.get())
        except Exception:
            s["max_search_results"] = 4
        s["search_time_limit"] = TIME_LIMIT_MAP.get(self.s_time.get(), "w")
        s["custom_instructions"] = self.ci_textbox.get("1.0", "end-1c")
        s["ai_model"] = (
            getattr(self, "_model_cb").get()
            if hasattr(self, "_model_cb")
            else s.get("ai_model", MODEL_CHOICES[0])
        )
        s["auto_upload"] = self.upload_var.get()
        s.setdefault("selected_account", "Default")
        s["subtitle_color"] = (
            self.enrich_sub_color.get()
            if hasattr(self, "enrich_sub_color")
            else "yellow"
        )
        s["watermark_text"] = (
            self.enrich_watermark.get()
            if hasattr(self, "enrich_watermark")
            else ""
        )
        s["quiz_prefix"] = (
            self.enrich_quiz_prefix.get()
            if hasattr(self, "enrich_quiz_prefix")
            else "Answer: "
        )
        s["preview_auto_refresh"] = (
            self.preview_auto_var.get()
            if hasattr(self, "preview_auto_var")
            else False
        )
        save_settings(s)
        messagebox.showinfo("Settings", "All settings saved successfully.")

    def _reset_all(self) -> None:
        """Restore every setting to its factory default and update the UI."""
        if not messagebox.askyesno("Reset", "Reset ALL settings to defaults?"):
            return
        self.settings = dict(DEFAULT_SETTINGS)
        self.s_api_key.delete(0, "end")
        self.s_api_key.insert(0, DEFAULT_SETTINGS["openrouter_api_key"])
        self.s_pixabay_key.delete(0, "end")
        self.s_pixabay_key.insert(0, "")
        self.s_pexels_key.delete(0, "end")
        self.s_pexels_key.insert(0, "")
        self.s_voice.set(DEFAULT_SETTINGS["tts_voice"])
        getattr(self, "_s_im").delete(0, "end")
        getattr(self, "_s_im").insert(0, DEFAULT_SETTINGS["imagemagick_path"])
        getattr(self, "_s_bg").delete(0, "end")
        getattr(self, "_s_bg").insert(0, DEFAULT_SETTINGS["bg_video"])
        self.s_output.delete(0, "end")
        self.s_output.insert(0, DEFAULT_SETTINGS["output_file"])
        self.s_results.delete(0, "end")
        self.s_results.insert(0, str(DEFAULT_SETTINGS["max_search_results"]))
        displayed = TIME_LIMIT_REVERSE.get(DEFAULT_SETTINGS["search_time_limit"], "7 days")
        self.s_time.set(displayed)
        self.ci_textbox.delete("1.0", "end")
        if hasattr(self, "_model_cb"):
            self._model_cb.set(DEFAULT_SETTINGS["ai_model"])
        if hasattr(self, "_bg_entry"):
            self._bg_entry.delete(0, "end")
            self._bg_entry.insert(0, DEFAULT_SETTINGS["bg_video"])
        if hasattr(self, "_output_entry"):
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, DEFAULT_SETTINGS["output_file"])
        self.upload_var.set(True)
        save_settings(self.settings)

    # -- generation worker -------------------------------------------------

    def _start(self) -> None:
        """Begin a generation run in a background thread.

        The worker captures stdout/stderr so all print() output from the agent,
        tools, and uploader modules is forwarded to the GUI console in real time.
        """
        if self.is_running:
            return
        self._log_entries = []
        self._clear_console(self.gen_console)
        self._clear_console(self.dash_console)
        self.progress_bar.set(0)
        self.is_running = True
        self._stop_event.clear()
        self.gen_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_status("running", "Generating video...")
        self._emit_log("=== Generation Started ===", "banner")

        frozen_settings = dict(self.settings)
        frozen_settings["custom_instructions"] = self.ci_textbox.get("1.0", "end-1c")
        frozen_settings["ai_model"] = (
            getattr(self, "_model_cb").get()
            if hasattr(self, "_model_cb")
            else self.settings.get("ai_model", MODEL_CHOICES[0])
        )
        frozen_settings["bg_video"] = (
            getattr(self, "_bg_entry").get()
            if hasattr(self, "_bg_entry")
            else self.settings.get("bg_video", "")
        )
        frozen_settings["output_file"] = (
            getattr(self, "_output_entry").get()
            if hasattr(self, "_output_entry")
            else self.settings.get("output_file", "outputs/final_short.mp4")
        )
        frozen_settings["auto_upload"] = self.upload_var.get()
        frozen_settings["subtitle_color"] = (
            self.enrich_sub_color.get()
            if hasattr(self, "enrich_sub_color")
            else "yellow"
        )
        frozen_settings["watermark_text"] = (
            self.enrich_watermark.get()
            if hasattr(self, "enrich_watermark")
            else ""
        )
        frozen_settings["quiz_prefix"] = (
            self.enrich_quiz_prefix.get()
            if hasattr(self, "enrich_quiz_prefix")
            else "Answer: "
        )
        self.worker_thread = threading.Thread(
            target=self._run_worker, args=(frozen_settings,), daemon=True,
        )
        self.worker_thread.start()

    def _stop(self) -> None:
        """Signal the background worker to abort at the next checkpoint."""
        if not self.is_running:
            return
        self._stop_event.set()
        self._set_status("idle", "Stopping...")
        self._emit_log("=== Stop Requested ===", "banner")

    def _run_worker(self, frozen_settings: dict[str, Any]) -> None:
        """Execute the full agent pipeline in a background thread.

        stdout/stderr are replaced with a queue-based writer so every log line
        is forwarded to the GUI's console and progress bar.
        """
        old_out, old_err = sys.stdout, sys.stderr

        class _QueueWriter(io.TextIOBase):
            def __init__(self, q):
                self.q = q
                self.buf = ""

            def write(self, s):
                self.buf += s
                if "\n" in self.buf:
                    for line in self.buf.split("\n")[:-1]:
                        if line.strip():
                            self.q.put(("log", line.strip()))
                    self.buf = self.buf.split("\n")[-1]
                return len(s)

            def flush(self):
                if self.buf.strip():
                    self.q.put(("log", self.buf.strip()))
                    self.buf = ""

        sys.stdout = _QueueWriter(self.log_queue)
        sys.stderr = _QueueWriter(self.log_queue)

        try:
            if self._stop_event.is_set():
                return
            self.log_queue.put(("log", "[GUI] Configuring agent..."))
            self.log_queue.put(("progress", 3))
            from agent import configure_agent
            configure_agent(frozen_settings)
            if self._stop_event.is_set():
                return
            from agent import run_agent
            self.log_queue.put(("progress", 10))
            run_agent()
            if self._stop_event.is_set():
                self.log_queue.put(("log", "[GUI] Generation stopped by user."))
                self.log_queue.put(("done", (False, "Manually stopped")))
                return
            self.log_queue.put(("progress", 100))
            self.log_queue.put(
                ("done", (True, "Video generation completed successfully"))
            )
        except Exception as e:
            self.log_queue.put(("log", f"[ERROR] {e}"))
            for line in traceback.format_exc().split("\n"):
                if line.strip():
                    self.log_queue.put(("log", line.strip()))
            self.log_queue.put(("done", (False, str(e))))
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    # -- event loop callbacks ----------------------------------------------

    def _poll_queues(self) -> None:
        """Drain the log queue every ~80ms and update the GUI accordingly."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg[0] == "log":
                    self._append_console(msg[1])
                elif msg[0] == "progress":
                    self.progress_bar.set(msg[1] / 100.0)
                    if msg[1] >= 100:
                        self._set_status("success", "Done")
                elif msg[0] == "done":
                    self._finish(msg[1][0], msg[1][1])
        except queue.Empty:
            pass
        self.after(80, self._poll_queues)

    def _emit_log(self, text: str, level: str = "normal") -> None:
        self._append_console(text)
        self._log_entries.append(text)

    def _append_console(self, text: str) -> None:
        """Push a line into both the Generate and Dashboard console widgets."""
        for console in (self.gen_console, self.dash_console):
            try:
                console.configure(state="normal")
                console.append_line(text)
                console.configure(state="disabled")
            except Exception:
                pass

    def _clear_console(self, console) -> None:
        try:
            console.configure(state="normal")
            console.delete("1.0", END)
            console.configure(state="disabled")
        except Exception:
            pass

    def _finish(self, success: bool, msg: str) -> None:
        """Restore button state after a generation run finishes."""
        self.is_running = False
        self.gen_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if success:
            self.progress_bar.set(1.0)
            self._set_status("success", "Complete")
            self._emit_log("=== Generation Completed ===", "banner")
            if self.preview_auto_var.get():
                self._after(
                    1000, lambda: self._load_for_preview("final_short.mp4")
                )
            messagebox.showinfo("Success", "Video generated successfully!")
        else:
            self.progress_bar.set(0)
            self._set_status("error", "Error")
            self._emit_log("=== Generation Failed ===", "banner")
            messagebox.showerror("Error", f"Generation failed:\n{msg}")

    def _set_status(self, status: str, text: str) -> None:
        """Update the header status dot colour and label text."""
        colors = {
            "idle": "#6b7280", "running": "#3b82f6",
            "success": "#10b981", "error": "#ef4444",
        }
        try:
            self.status_dot.configure(fg_color=colors.get(status, "#6b7280"))
            self.status_label.configure(text=text)
        except Exception:
            pass

    def _on_close(self) -> None:
        self.is_running = False
        self.destroy()

    def _after(self, ms: int, callback) -> None:
        self.after(ms, callback)


def main() -> None:
    """Launch the desktop application."""
    App().mainloop()


if __name__ == "__main__":
    main()
