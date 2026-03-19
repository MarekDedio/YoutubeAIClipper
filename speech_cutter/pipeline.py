from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import json
import os
import shutil
import subprocess
import tempfile
import time

from .vad import (
    DetectionCancelledError,
    SileroOnnxVad,
    SpeechSegment,
    VadSettings,
    detect_speech_in_wav,
)


MODEL_PATH = Path(__file__).resolve().parent / "assets" / "silero_vad_16k_op15.onnx"


class ProcessingError(RuntimeError):
    """Base error for the video cutting pipeline."""


class NoSpeechDetectedError(ProcessingError):
    """Raised when no speech was detected in the input video."""


class UserCancelledError(ProcessingError):
    """Raised when the user cancels processing."""


@dataclass(frozen=True)
class ProcessingOptions:
    vad: VadSettings = field(default_factory=VadSettings)
    video_crf: int = 18
    video_preset: str = "veryfast"


@dataclass(frozen=True)
class ProcessingResult:
    input_path: Path
    output_path: Path
    input_duration: float
    kept_duration: float
    segments: list[SpeechSegment]

    @property
    def removed_duration(self) -> float:
        return max(self.input_duration - self.kept_duration, 0.0)


ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]


def check_runtime() -> list[str]:
    issues: list[str] = []
    if shutil.which("ffmpeg") is None:
        issues.append("`ffmpeg` was not found in PATH.")
    if shutil.which("ffprobe") is None:
        issues.append("`ffprobe` was not found in PATH.")
    if not MODEL_PATH.exists():
        issues.append(f"Speech model is missing: {MODEL_PATH}")
    return issues


def process_video(
    input_path: Path,
    output_path: Path,
    options: ProcessingOptions | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ProcessingResult:
    options = options or ProcessingOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)

    _ensure_not_cancelled(cancel_callback)
    _emit_progress(progress_callback, 0.01, "Checking the source video...")
    _log(log_callback, "Reading video metadata.")
    metadata = _probe_video(input_path)

    if not metadata["has_video"]:
        raise ProcessingError("The selected file does not contain a video stream.")
    if not metadata["has_audio"]:
        raise ProcessingError("The selected file does not contain an audio stream to analyze.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="speech_cutter_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        wav_path = temp_dir_path / "audio.wav"
        filter_script_path = temp_dir_path / "speech_concat.ffscript"

        _log(log_callback, "Extracting mono 16 kHz audio for speech detection.")
        _run_ffmpeg_with_progress(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-progress",
                "pipe:1",
                "-nostats",
                str(wav_path),
            ],
            total_duration=max(metadata["duration"], 0.001),
            progress_start=0.02,
            progress_end=0.20,
            status="Extracting audio...",
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

        _log(log_callback, "Running speech detection with the bundled Silero VAD model.")
        detector = SileroOnnxVad(MODEL_PATH)
        try:
            segments = detect_speech_in_wav(
                wav_path,
                detector,
                options.vad,
                progress_callback=lambda ratio: _emit_progress(
                    progress_callback,
                    0.20 + ratio * 0.40,
                    "Detecting spoken sections...",
                ),
                cancel_callback=cancel_callback,
            )
        except DetectionCancelledError as exc:
            raise UserCancelledError(str(exc)) from exc

        if not segments:
            raise NoSpeechDetectedError("No human speech was detected in that video.")

        kept_duration = sum(segment.duration for segment in segments)
        _log(
            log_callback,
            f"Detected {len(segments)} speech segment(s). Keeping {kept_duration:.1f}s out of {metadata['duration']:.1f}s.",
        )

        filter_script_path.write_text(_build_filter_script(segments), encoding="utf-8")
        _log(log_callback, "Rendering the final speech-only MP4.")
        _run_ffmpeg_with_progress(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(input_path),
                "-filter_complex_script",
                str(filter_script_path),
                "-map",
                "[outv]",
                "-map",
                "[outa]",
                "-c:v",
                "libx264",
                "-preset",
                options.video_preset,
                "-crf",
                str(options.video_crf),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                str(output_path),
            ],
            total_duration=max(kept_duration, 0.001),
            progress_start=0.60,
            progress_end=1.00,
            status="Rendering output video...",
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    _emit_progress(progress_callback, 1.0, "Finished.")
    _log(log_callback, f"Saved output to {output_path}")
    return ProcessingResult(
        input_path=input_path,
        output_path=output_path,
        input_duration=metadata["duration"],
        kept_duration=kept_duration,
        segments=segments,
    )


def _probe_video(input_path: Path) -> dict[str, float | bool]:
    completed = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})

    duration = _first_float(
        format_info.get("duration"),
        *(stream.get("duration") for stream in streams),
    )

    return {
        "duration": duration,
        "has_video": any(stream.get("codec_type") == "video" for stream in streams),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def _build_filter_script(segments: list[SpeechSegment]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments):
        start = f"{segment.start:.6f}"
        end = f"{segment.end:.6f}"
        lines.append(f"[0:v:0]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]")
        lines.append(f"[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")

    concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(segments)))
    lines.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    return ";\n".join(lines) + "\n"


def _run_ffmpeg_with_progress(
    command: list[str],
    *,
    total_duration: float,
    progress_start: float,
    progress_end: float,
    status: str,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_startup_kwargs(),
    )

    try:
        saw_end = False
        while True:
            _ensure_not_cancelled(cancel_callback, process)

            assert process.stdout is not None
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            line = line.strip()
            if line.startswith("out_time="):
                current = _parse_ffmpeg_timecode(line.partition("=")[2])
                if current is not None:
                    ratio = min(current / total_duration, 1.0)
                    progress = progress_start + ratio * (progress_end - progress_start)
                    _emit_progress(progress_callback, progress, status)
            elif line == "progress=end":
                _emit_progress(progress_callback, progress_end, status)
                saw_end = True
                break

        stdout_tail, stderr_output = process.communicate(timeout=30)
        return_code = process.returncode
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    if return_code != 0:
        if cancel_callback and cancel_callback():
            raise UserCancelledError("Processing was cancelled.")
        message = stderr_output.strip() or "ffmpeg exited with an unknown error."
        raise ProcessingError(message)

    if not saw_end:
        _emit_progress(progress_callback, progress_end, status)


def _run_command(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        **_startup_kwargs(),
    )


def _ensure_not_cancelled(
    cancel_callback: CancelCallback | None,
    process: subprocess.Popen[str] | None = None,
) -> None:
    if cancel_callback and cancel_callback():
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        raise UserCancelledError("Processing was cancelled.")


def _emit_progress(callback: ProgressCallback | None, value: float, status: str) -> None:
    if callback:
        callback(max(0.0, min(value, 1.0)), status)


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _parse_ffmpeg_timecode(value: str) -> float | None:
    parts = value.split(":")
    if len(parts) != 3:
        return None

    hours, minutes, seconds = parts
    try:
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _first_float(*values: object) -> float:
    for value in values:
        if value in (None, "N/A", ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _startup_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs
