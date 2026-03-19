from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk
import math
import os
import subprocess
import tempfile
import threading
import tkinter as tk

from .pipeline import (
    CropSettings,
    NoSpeechDetectedError,
    ProcessingOptions,
    ProcessingResult,
    UserCancelledError,
    build_output_path,
    check_runtime,
    probe_video_metadata,
    process_video,
)
from .presets import PRESETS, build_options


NATURAL_SETTINGS = PRESETS["natural"].settings


@dataclass(frozen=True)
class QueueEvent:
    kind: str
    payload: object


class CropSelectionDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        input_path: Path,
        source_width: int,
        source_height: int,
        source_duration: float,
        initial_crop: tuple[int, int, int, int] | None,
    ) -> None:
        super().__init__(parent)
        self.title("Choose Crop")
        self.configure(bg="#f5f1e8")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: tuple[int, int, int, int] | None = None
        self._source_width = source_width
        self._source_height = source_height
        self._temp_dir = tempfile.TemporaryDirectory(prefix="speech_cutter_crop_")
        preview_path = self._extract_preview_frame(
            input_path=input_path,
            source_duration=source_duration,
            output_dir=Path(self._temp_dir.name),
        )
        self._image = tk.PhotoImage(file=str(preview_path))
        self._preview_width = self._image.width()
        self._preview_height = self._image.height()

        self._start_x = 0
        self._start_y = 0
        self._current_rect: tuple[int, int, int, int] | None = None
        self._rect_id: int | None = None
        self._overlay_id: int | None = None

        self.summary_var = tk.StringVar(value="Drag a rectangle on the preview.")

        self._build_layout()
        self._bind_events()

        if initial_crop:
            self._current_rect = self._source_crop_to_preview(initial_crop)
            self._redraw_selection()

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus_set()

    def destroy(self) -> None:
        try:
            self._temp_dir.cleanup()
        except Exception:
            pass
        super().destroy()

    def _build_layout(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Drag a box over the part of the frame you want to zoom into.",
            wraplength=860,
        ).pack(anchor="w")

        self.canvas = tk.Canvas(
            frame,
            width=self._preview_width,
            height=self._preview_height,
            highlightthickness=1,
            highlightbackground="#c9baa7",
            bg="#000000",
            cursor="crosshair",
        )
        self.canvas.pack(pady=(12, 10))
        self.canvas.create_image(0, 0, image=self._image, anchor="nw")
        self._overlay_id = self.canvas.create_rectangle(
            0,
            0,
            self._preview_width,
            self._preview_height,
            outline="",
            fill="",
        )

        ttk.Label(frame, textvariable=self.summary_var, wraplength=860).pack(anchor="w")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))

        ttk.Button(buttons, text="Clear Selection", command=self._clear_selection).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(buttons, text="Use This Crop", command=self._apply).pack(side="right", padx=(0, 8))

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event: tk.Event[tk.Canvas]) -> None:
        self._start_x, self._start_y = self._clamp_preview_point(event.x, event.y)
        self._current_rect = (self._start_x, self._start_y, self._start_x, self._start_y)
        self._redraw_selection()

    def _on_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._current_rect is None:
            return
        end_x, end_y = self._clamp_preview_point(event.x, event.y)
        self._current_rect = (self._start_x, self._start_y, end_x, end_y)
        self._redraw_selection()

    def _on_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._current_rect is None:
            return
        end_x, end_y = self._clamp_preview_point(event.x, event.y)
        self._current_rect = (self._start_x, self._start_y, end_x, end_y)
        self._redraw_selection()

    def _redraw_selection(self) -> None:
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)

        if not self._current_rect:
            self.summary_var.set("No crop selected.")
            return

        x0, y0, x1, y1 = self._normalize_preview_rect(self._current_rect)
        self._rect_id = self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#ffb347",
            width=2,
        )

        crop = self._preview_to_source_crop((x0, y0, x1, y1))
        if crop is None:
            self.summary_var.set("Selection is too small. Drag a larger box.")
            return

        x, y, width, height = crop
        self.summary_var.set(f"Crop selected: x={x}, y={y}, width={width}, height={height}")

    def _clear_selection(self) -> None:
        self._current_rect = None
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self.summary_var.set("No crop selected.")

    def _apply(self) -> None:
        if not self._current_rect:
            messagebox.showerror("Missing crop", "Draw a crop box first.", parent=self)
            return

        crop = self._preview_to_source_crop(self._normalize_preview_rect(self._current_rect))
        if crop is None:
            messagebox.showerror("Invalid crop", "The crop box is too small.", parent=self)
            return

        self.result = crop
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()

    def _source_crop_to_preview(self, crop: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = crop
        scale_x = self._preview_width / self._source_width
        scale_y = self._preview_height / self._source_height
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round((x + width) * scale_x)),
            int(round((y + height) * scale_y)),
        )

    def _preview_to_source_crop(
        self,
        rect: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = rect
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None

        scale_x = self._source_width / self._preview_width
        scale_y = self._source_height / self._preview_height

        left = max(0, int(math.floor(x0 * scale_x)))
        top = max(0, int(math.floor(y0 * scale_y)))
        right = min(self._source_width, int(math.ceil(x1 * scale_x)))
        bottom = min(self._source_height, int(math.ceil(y1 * scale_y)))

        width = right - left
        height = bottom - top
        if width < 2 or height < 2:
            return None

        if width % 2 == 1 and width > 2:
            width -= 1
        if height % 2 == 1 and height > 2:
            height -= 1

        if left + width > self._source_width:
            left = max(0, self._source_width - width)
        if top + height > self._source_height:
            top = max(0, self._source_height - height)

        return left, top, width, height

    def _normalize_preview_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        return left, top, right, bottom

    def _clamp_preview_point(self, x: int, y: int) -> tuple[int, int]:
        return max(0, min(x, self._preview_width)), max(0, min(y, self._preview_height))

    def _extract_preview_frame(
        self,
        *,
        input_path: Path,
        source_duration: float,
        output_dir: Path,
    ) -> Path:
        preview_path = output_dir / "preview.png"
        preview_time = min(max(source_duration * 0.25, 0.0), max(source_duration - 0.1, 0.0))
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{preview_time:.3f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:540:force_original_aspect_ratio=decrease",
            str(preview_path),
        ]
        subprocess.run(command, check=True, **_subprocess_kwargs())
        return preview_path


class SpeechCutterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Speech Cutter")
        self.geometry("940x760")
        self.minsize(880, 680)
        self.configure(bg="#f5f1e8")

        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._events: Queue[QueueEvent] = Queue()
        self._last_result: ProcessingResult | None = None
        self._crop_widgets: list[tk.Widget] = []
        self._source_path: Path | None = None
        self._source_metadata: dict[str, float | bool | None] = {}
        self._crop_rect: tuple[int, int, int, int] | None = None

        self.input_var = tk.StringVar()
        self.source_info_var = tk.StringVar(value="Pick a video to load its source size.")
        self.crop_summary_var = tk.StringVar(value="No crop selected.")
        self.status_var = tk.StringVar(value="Pick a video and press Start.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.padding_seconds_var = tk.DoubleVar(value=NATURAL_SETTINGS.speech_pad_ms / 1000)
        self.padding_value_var = tk.StringVar()

        self.crop_enabled_var = tk.BooleanVar(value=False)
        self.crop_every_var = tk.DoubleVar(value=3.0)
        self.crop_every_value_var = tk.StringVar()
        self.crop_min_seconds_var = tk.DoubleVar(value=1.2)
        self.crop_min_seconds_value_var = tk.StringVar()

        self.padding_seconds_var.trace_add("write", self._refresh_setting_labels)
        self.crop_every_var.trace_add("write", self._refresh_setting_labels)
        self.crop_min_seconds_var.trace_add("write", self._refresh_setting_labels)

        self._configure_styles()
        self._refresh_setting_labels()
        self._build_layout()
        self._update_crop_state()
        self.after(100, self._drain_queue)

        issues = check_runtime()
        if issues:
            messagebox.showwarning(
                "Missing dependencies",
                "Some requirements are missing:\n\n" + "\n".join(f"- {issue}" for issue in issues),
            )

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Root.TFrame", background="#f5f1e8")
        style.configure("Card.TFrame", background="#fffaf3")
        style.configure("CardInner.TFrame", background="#fffaf3")
        style.configure("Title.TLabel", background="#f5f1e8", foreground="#1f2b34", font=("Segoe UI Semibold", 24))
        style.configure("Subtitle.TLabel", background="#f5f1e8", foreground="#5b655d", font=("Segoe UI", 11))
        style.configure("Section.TLabel", background="#fffaf3", foreground="#27343b", font=("Segoe UI Semibold", 11))
        style.configure("Body.TLabel", background="#fffaf3", foreground="#46535a")
        style.configure("Status.TLabel", background="#fffaf3", foreground="#263238", font=("Segoe UI Semibold", 10))
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10))
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#e8ded0", background="#b45f3c", bordercolor="#e8ded0")
        style.configure(
            "Accent.Horizontal.TScale",
            background="#fffaf3",
            troughcolor="#d8def8",
            bordercolor="#d8def8",
            lightcolor="#5a74ff",
            darkcolor="#5a74ff",
        )

    def _refresh_setting_labels(self, *_args: object) -> None:
        self.padding_value_var.set(f"{float(self.padding_seconds_var.get()):.2f}s")
        self.crop_every_value_var.set(str(max(1, int(round(float(self.crop_every_var.get()))))))
        self.crop_min_seconds_value_var.set(f"{float(self.crop_min_seconds_var.get()):.1f}s")

    def _on_padding_scale(self, value: str) -> None:
        snapped = min(3.0, max(0.0, round(float(value) / 0.05) * 0.05))
        snapped = round(snapped, 2)
        if abs(float(self.padding_seconds_var.get()) - snapped) > 1e-9:
            self.padding_seconds_var.set(snapped)

    def _on_crop_every_scale(self, value: str) -> None:
        snapped = max(1, min(12, int(round(float(value)))))
        if max(1, int(round(float(self.crop_every_var.get())))) != snapped:
            self.crop_every_var.set(float(snapped))

    def _on_crop_min_scale(self, value: str) -> None:
        snapped = min(10.0, max(0.0, round(float(value) / 0.1) * 0.1))
        snapped = round(snapped, 1)
        if abs(float(self.crop_min_seconds_var.get()) - snapped) > 1e-9:
            self.crop_min_seconds_var.set(snapped)

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        hero = ttk.Frame(root, style="Root.TFrame")
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(hero, text="Speech Cutter", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="Load a video, keep only the spoken parts, and save the MP4 next to the original.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        files_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        files_card.grid(row=1, column=0, sticky="ew")
        files_card.columnconfigure(1, weight=1)

        ttk.Label(files_card, text="Source video", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(files_card, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Button(files_card, text="Browse...", command=self._browse_input).grid(row=0, column=2)
        ttk.Label(files_card, textvariable=self.source_info_var, style="Body.TLabel", wraplength=780).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(10, 0),
        )

        trim_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        trim_card.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        trim_card.columnconfigure(0, weight=1)

        trim_header = ttk.Frame(trim_card, style="CardInner.TFrame")
        trim_header.grid(row=0, column=0, sticky="ew")
        trim_header.columnconfigure(0, weight=1)
        ttk.Label(trim_header, text="Keep Around Speech (seconds)", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(trim_header, textvariable=self.padding_value_var, style="Body.TLabel").grid(row=0, column=1, sticky="e")

        self.padding_scale = ttk.Scale(
            trim_card,
            from_=0.0,
            to=3.0,
            variable=self.padding_seconds_var,
            command=self._on_padding_scale,
            style="Accent.Horizontal.TScale",
        )
        self.padding_scale.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            trim_card,
            text="Natural trimming is always used. This setting keeps a little extra time before and after speech so cuts feel less abrupt.",
            style="Body.TLabel",
            wraplength=780,
        ).grid(row=2, column=0, sticky="w", pady=(12, 0))

        crop_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        crop_card.grid(row=3, column=0, sticky="ew", pady=14)
        crop_card.columnconfigure(2, weight=1)

        ttk.Checkbutton(
            crop_card,
            text="Enable crop / zoom on selected segments",
            variable=self.crop_enabled_var,
            command=self._update_crop_state,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        self.pick_crop_button = ttk.Button(crop_card, text="Choose Crop Visually...", command=self._choose_crop_visually)
        self.pick_crop_button.grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.clear_crop_button = ttk.Button(crop_card, text="Clear Crop", command=self._clear_crop_selection)
        self.clear_crop_button.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(12, 0))
        ttk.Label(crop_card, textvariable=self.crop_summary_var, style="Body.TLabel", wraplength=520).grid(
            row=1,
            column=2,
            sticky="w",
            padx=(16, 0),
            pady=(12, 0),
        )

        crop_every_header = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_every_header.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        crop_every_header.columnconfigure(0, weight=1)
        ttk.Label(crop_every_header, text="Apply Crop Every Nth Segment", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(crop_every_header, textvariable=self.crop_every_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.crop_every_scale = ttk.Scale(
            crop_card,
            from_=1,
            to=12,
            variable=self.crop_every_var,
            command=self._on_crop_every_scale,
            style="Accent.Horizontal.TScale",
        )
        self.crop_every_scale.grid(row=3, column=0, columnspan=3, sticky="ew")

        crop_min_header = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_min_header.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        crop_min_header.columnconfigure(0, weight=1)
        ttk.Label(crop_min_header, text="Skip Crop Below (seconds)", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(crop_min_header, textvariable=self.crop_min_seconds_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.crop_min_scale = ttk.Scale(
            crop_card,
            from_=0.0,
            to=10.0,
            variable=self.crop_min_seconds_var,
            command=self._on_crop_min_scale,
            style="Accent.Horizontal.TScale",
        )
        self.crop_min_scale.grid(row=5, column=0, columnspan=3, sticky="ew")

        ttk.Label(
            crop_card,
            text=(
                "The crop box is picked on a frame preview and uses the original video size under the hood. "
                "Cropped segments are scaled back up, so this behaves like a zoom. Short segments can be skipped to avoid fast jump cuts."
            ),
            style="Body.TLabel",
            wraplength=780,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self._crop_widgets = [self.pick_crop_button, self.clear_crop_button, self.crop_every_scale, self.crop_min_scale]

        activity_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        activity_card.grid(row=4, column=0, sticky="nsew")
        activity_card.columnconfigure(0, weight=1)
        activity_card.rowconfigure(4, weight=1)

        actions = ttk.Frame(activity_card, style="CardInner.TFrame")
        actions.grid(row=0, column=0, sticky="ew")
        actions.columnconfigure(3, weight=1)

        self.start_button = ttk.Button(actions, text="Start", style="Primary.TButton", command=self._start_processing)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel_processing, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.open_folder_button = ttk.Button(actions, text="Open Output Folder", command=self._open_output_folder, state="disabled")
        self.open_folder_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

        ttk.Label(activity_card, text="Progress", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 6))
        ttk.Progressbar(
            activity_card,
            style="Accent.Horizontal.TProgressbar",
            variable=self.progress_var,
            maximum=100,
        ).grid(row=2, column=0, sticky="ew")

        ttk.Label(activity_card, textvariable=self.status_var, style="Status.TLabel").grid(
            row=3,
            column=0,
            sticky="nw",
            pady=(12, 6),
        )

        self.log_box = scrolledtext.ScrolledText(
            activity_card,
            wrap="word",
            height=14,
            relief="flat",
            bg="#fffdf8",
            fg="#243238",
            font=("Cascadia Mono", 10),
            padx=10,
            pady=10,
        )
        self.log_box.grid(row=4, column=0, sticky="nsew", pady=(4, 0))
        self.log_box.insert("end", "Ready.\n")
        self.log_box.configure(state="disabled")

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.flv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)
            self._load_source_metadata(Path(path))

    def _load_source_metadata(self, input_path: Path) -> None:
        try:
            metadata = probe_video_metadata(input_path)
        except Exception as exc:  # noqa: BLE001
            self._source_info_reset("Could not read the source video size.")
            self._append_log(f"Could not inspect source video: {exc}")
            return

        self._source_path = input_path
        self._source_metadata = metadata
        self._clear_crop_selection(update_message=False)

        width = int(metadata.get("video_width") or 0)
        height = int(metadata.get("video_height") or 0)
        if width > 0 and height > 0:
            self.source_info_var.set(
                f"Source size: {width} x {height}. Crop is chosen visually from a frame preview."
            )
        else:
            self.source_info_var.set("Source size could not be read from this file.")

    def _source_info_reset(self, message: str) -> None:
        self._source_path = None
        self._source_metadata = {}
        self.source_info_var.set(message)

    def _update_crop_state(self) -> None:
        state = "normal" if self.crop_enabled_var.get() else "disabled"
        for widget in self._crop_widgets:
            widget.configure(state=state)

    def _choose_crop_visually(self) -> None:
        input_path = self._get_input_path_or_warn()
        if input_path is None:
            return

        if self._source_path != input_path:
            self._load_source_metadata(input_path)

        width = int(self._source_metadata.get("video_width") or 0)
        height = int(self._source_metadata.get("video_height") or 0)
        duration = float(self._source_metadata.get("duration") or 0.0)
        if width <= 0 or height <= 0:
            messagebox.showerror("Missing video size", "Could not read the source video size for crop selection.")
            return

        try:
            dialog = CropSelectionDialog(
                self,
                input_path=input_path,
                source_width=width,
                source_height=height,
                source_duration=duration,
                initial_crop=self._crop_rect,
            )
            self.wait_window(dialog)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not open crop picker", str(exc))
            return

        if dialog.result:
            self._crop_rect = dialog.result
            self.crop_enabled_var.set(True)
            self._update_crop_state()
            x, y, crop_width, crop_height = dialog.result
            self.crop_summary_var.set(
                f"Crop selected: x={x}, y={y}, width={crop_width}, height={crop_height}"
            )

    def _clear_crop_selection(self, update_message: bool = True) -> None:
        self._crop_rect = None
        if update_message:
            self.crop_summary_var.set("No crop selected.")
        else:
            self.crop_summary_var.set("No crop selected.")

    def _start_processing(self) -> None:
        input_path = self._get_input_path_or_warn()
        if input_path is None:
            return

        if self._source_path != input_path:
            self._load_source_metadata(input_path)

        try:
            padding_seconds = max(0.0, float(self.padding_seconds_var.get()))
            crop_every = max(1, int(round(float(self.crop_every_var.get()))))
            crop_min_seconds = max(0.0, float(self.crop_min_seconds_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid settings", "One of the trim or crop settings is not valid.")
            return

        if self.crop_enabled_var.get() and not self._crop_rect:
            messagebox.showerror("Missing crop", "Enable crop is on, but no crop box has been chosen yet.")
            return

        crop_settings = CropSettings(
            enabled=bool(self.crop_enabled_var.get() and self._crop_rect),
            x=self._crop_rect[0] if self._crop_rect else 0,
            y=self._crop_rect[1] if self._crop_rect else 0,
            width=self._crop_rect[2] if self._crop_rect else 0,
            height=self._crop_rect[3] if self._crop_rect else 0,
            every_n_segments=crop_every,
            min_segment_seconds=crop_min_seconds,
        )

        output_path = build_output_path(input_path)
        if output_path.exists():
            overwrite = messagebox.askyesno(
                "Replace existing file?",
                f"This output already exists:\n\n{output_path}\n\nDo you want to replace it?",
            )
            if not overwrite:
                return

        self._last_result = None
        self._cancel_event.clear()
        self._set_running(True)
        self._append_log(f"Starting: {input_path.name}")
        self.status_var.set("Preparing...")
        self.progress_var.set(0.0)

        options = build_options(
            "natural",
            padding_ms=int(round(padding_seconds * 1000)),
            merge_gap_ms=NATURAL_SETTINGS.merge_gap_ms,
            crop=crop_settings,
        )

        self._worker = threading.Thread(
            target=self._worker_main,
            args=(input_path, output_path, options),
            daemon=True,
        )
        self._worker.start()

    def _get_input_path_or_warn(self) -> Path | None:
        input_text = self.input_var.get().strip()
        if not input_text:
            messagebox.showerror("Missing video", "Choose a source video first.")
            return None

        input_path = Path(input_text)
        if not input_path.exists():
            messagebox.showerror("Missing file", "The selected source video does not exist.")
            return None
        return input_path

    def _worker_main(
        self,
        input_path: Path,
        output_path: Path,
        options: ProcessingOptions,
    ) -> None:
        try:
            result = process_video(
                input_path,
                output_path,
                options=options,
                progress_callback=lambda value, status: self._events.put(
                    QueueEvent("progress", (value, status))
                ),
                log_callback=lambda message: self._events.put(QueueEvent("log", message)),
                cancel_callback=self._cancel_event.is_set,
            )
        except UserCancelledError:
            self._events.put(QueueEvent("cancelled", None))
        except NoSpeechDetectedError as exc:
            self._events.put(QueueEvent("error", str(exc)))
        except Exception as exc:  # noqa: BLE001
            self._events.put(QueueEvent("error", str(exc)))
        else:
            self._events.put(QueueEvent("done", result))

    def _cancel_processing(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self.status_var.set("Cancelling...")
            self._append_log("Cancellation requested.")

    def _drain_queue(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                self._handle_event(event)
        except Empty:
            pass
        finally:
            self.after(100, self._drain_queue)

    def _handle_event(self, event: QueueEvent) -> None:
        if event.kind == "progress":
            value, status = event.payload  # type: ignore[misc]
            self.progress_var.set(float(value) * 100)
            self.status_var.set(str(status))
            return

        if event.kind == "log":
            self._append_log(str(event.payload))
            return

        if event.kind == "done":
            result = event.payload  # type: ignore[assignment]
            self._last_result = result
            self._set_running(False)
            self.progress_var.set(100)
            summary = (
                f"Finished. Kept {result.kept_duration:.1f}s of speech and removed "
                f"{result.removed_duration:.1f}s."
            )
            self.status_var.set(summary)
            self._append_log(summary)
            self.open_folder_button.configure(state="normal")
            messagebox.showinfo("Done", f"Speech-only video saved to:\n\n{result.output_path}")
            return

        if event.kind == "cancelled":
            self._set_running(False)
            self.progress_var.set(0)
            self.status_var.set("Cancelled.")
            self._append_log("Processing cancelled.")
            return

        if event.kind == "error":
            self._set_running(False)
            self.progress_var.set(0)
            self.status_var.set("Stopped because of an error.")
            self._append_log(f"Error: {event.payload}")
            messagebox.showerror("Could not finish", str(event.payload))

    def _set_running(self, is_running: bool) -> None:
        self.start_button.configure(state="disabled" if is_running else "normal")
        self.cancel_button.configure(state="normal" if is_running else "disabled")
        if is_running:
            self.open_folder_button.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message.rstrip() + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _open_output_folder(self) -> None:
        if not self._last_result:
            return
        folder = self._last_result.output_path.parent
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Output folder", str(folder))


def _subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def launch_gui() -> None:
    app = SpeechCutterApp()
    app.mainloop()
