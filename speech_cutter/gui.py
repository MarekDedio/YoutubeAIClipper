from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import threading
import tkinter as tk

from .pipeline import (
    NoSpeechDetectedError,
    ProcessingOptions,
    ProcessingResult,
    UserCancelledError,
    check_runtime,
    process_video,
)
from .presets import PRESETS, build_options


@dataclass(frozen=True)
class QueueEvent:
    kind: str
    payload: object


class SpeechCutterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Speech Cutter")
        self.geometry("880x620")
        self.minsize(820, 560)
        self.configure(bg="#f5f1e8")

        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._events: Queue[QueueEvent] = Queue()
        self._last_result: ProcessingResult | None = None
        self._last_auto_output = ""

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Pick a video and press Start.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.preset_var = tk.StringVar(value="balanced")
        self.preset_description_var = tk.StringVar()
        self.padding_var = tk.IntVar(value=PRESETS["balanced"].settings.speech_pad_ms)
        self.merge_gap_var = tk.IntVar(value=PRESETS["balanced"].settings.merge_gap_ms)

        self._configure_styles()
        self._build_layout()
        self._apply_preset("balanced")
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

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        hero = ttk.Frame(root, style="Root.TFrame")
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(hero, text="Speech Cutter", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="Load a video, keep only the spoken parts, and export a fast-start MP4.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        files_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        files_card.grid(row=1, column=0, sticky="ew")
        files_card.columnconfigure(1, weight=1)

        ttk.Label(files_card, text="Source video", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(files_card, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Button(files_card, text="Browse...", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(files_card, text="Output file", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(files_card, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=10, pady=(10, 0))
        ttk.Button(files_card, text="Save as...", command=self._browse_output).grid(row=1, column=2, pady=(10, 0))

        settings_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        settings_card.grid(row=2, column=0, sticky="ew", pady=14)
        settings_card.columnconfigure(1, weight=1)
        settings_card.columnconfigure(3, weight=1)
        settings_card.columnconfigure(5, weight=1)

        ttk.Label(settings_card, text="Trim style", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        preset_box = ttk.Combobox(
            settings_card,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
        )
        preset_box.grid(row=0, column=1, sticky="ew", padx=(10, 20))
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_changed)

        ttk.Label(settings_card, text="Speech padding (ms)", style="Section.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(settings_card, from_=20, to=600, increment=10, textvariable=self.padding_var, width=8).grid(
            row=0,
            column=3,
            sticky="w",
            padx=(10, 20),
        )

        ttk.Label(settings_card, text="Merge pauses under (ms)", style="Section.TLabel").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(settings_card, from_=0, to=1500, increment=25, textvariable=self.merge_gap_var, width=8).grid(
            row=0,
            column=5,
            sticky="w",
            padx=(10, 0),
        )

        ttk.Label(settings_card, textvariable=self.preset_description_var, style="Body.TLabel", wraplength=780).grid(
            row=1,
            column=0,
            columnspan=6,
            sticky="w",
            pady=(12, 0),
        )

        activity_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        activity_card.grid(row=3, column=0, sticky="nsew")
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
        self.progress = ttk.Progressbar(
            activity_card,
            style="Accent.Horizontal.TProgressbar",
            variable=self.progress_var,
            maximum=100,
        )
        self.progress.grid(row=2, column=0, sticky="ew")

        ttk.Label(activity_card, textvariable=self.status_var, style="Status.TLabel").grid(row=3, column=0, sticky="nw", pady=(12, 6))

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
            self._update_auto_output(path)

    def _browse_output(self) -> None:
        initial_file = Path(self.output_var.get()).name if self.output_var.get() else "speech_only.mp4"
        path = filedialog.asksaveasfilename(
            title="Save trimmed video as",
            defaultextension=".mp4",
            initialfile=initial_file,
            filetypes=[("MP4 video", "*.mp4")],
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_var.set(path)

    def _on_preset_changed(self, _event: object | None = None) -> None:
        self._apply_preset(self.preset_var.get())

    def _apply_preset(self, preset_key: str) -> None:
        preset = PRESETS[preset_key]
        self.preset_description_var.set(preset.description)
        self.padding_var.set(preset.settings.speech_pad_ms)
        self.merge_gap_var.set(preset.settings.merge_gap_ms)

    def _update_auto_output(self, input_path: str) -> None:
        source = Path(input_path)
        auto_output = str(source.with_name(f"{source.stem}_speech_only.mp4"))
        if not self.output_var.get() or self.output_var.get() == self._last_auto_output:
            self.output_var.set(auto_output)
        self._last_auto_output = auto_output

    def _start_processing(self) -> None:
        input_text = self.input_var.get().strip()
        output_text = self.output_var.get().strip()

        if not input_text:
            messagebox.showerror("Missing video", "Choose a source video first.")
            return
        if not output_text:
            messagebox.showerror("Missing output", "Choose where the speech-only MP4 should be saved.")
            return

        input_path = Path(input_text)
        output_path = Path(output_text)
        if not input_path.exists():
            messagebox.showerror("Missing file", "The selected source video does not exist.")
            return

        self._last_result = None
        self._cancel_event.clear()
        self._set_running(True)
        self._append_log(f"Starting: {input_path.name}")
        self.status_var.set("Preparing...")
        self.progress_var.set(0.0)

        options = build_options(
            self.preset_var.get(),
            padding_ms=self.padding_var.get(),
            merge_gap_ms=self.merge_gap_var.get(),
        )

        self._worker = threading.Thread(
            target=self._worker_main,
            args=(input_path, output_path, options),
            daemon=True,
        )
        self._worker.start()

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


def launch_gui() -> None:
    app = SpeechCutterApp()
    app.mainloop()
