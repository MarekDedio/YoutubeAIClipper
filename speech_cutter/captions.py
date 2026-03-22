from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import os
import re
import subprocess


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]

DEFAULT_MODEL_NAME = "small"
_MODEL_CACHE: dict[tuple[str, str], object] = {}

_PROFANITY_PREFIXES = (
    "asshole",
    "bastard",
    "bitch",
    "bullshit",
    "cock",
    "cunt",
    "dick",
    "fuck",
    "motherfuck",
    "nigger",
    "nigga",
    "pussy",
    "shit",
    "slut",
    "whore",
    "chuj",
    "cipa",
    "jeb",
    "kurw",
    "pierdol",
    "pizd",
    "skurw",
    "spierdal",
    "wkurw",
    "zajeb",
)


@dataclass(frozen=True)
class CaptionSettings:
    enabled: bool = False
    profanity_filter: bool = False
    model_name: str = DEFAULT_MODEL_NAME
    max_words_per_caption: int = 4
    max_caption_seconds: float = 2.2
    max_word_gap_seconds: float = 0.45


@dataclass(frozen=True)
class CaptionWord:
    start: float
    end: float
    text: str
    display_text: str
    is_profanity: bool


@dataclass(frozen=True)
class CaptionArtifactResult:
    subtitle_path: Path | None
    profanity_intervals: list[tuple[float, float]]
    caption_event_count: int
    censored_word_count: int
    transcribed_word_count: int


def needs_transcription(settings: CaptionSettings) -> bool:
    return bool(settings.enabled or settings.profanity_filter)


def create_caption_artifacts(
    input_path: Path,
    output_dir: Path,
    *,
    video_width: int,
    video_height: int,
    total_duration: float,
    settings: CaptionSettings,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> CaptionArtifactResult:
    if not needs_transcription(settings):
        return CaptionArtifactResult(
            subtitle_path=None,
            profanity_intervals=[],
            caption_event_count=0,
            censored_word_count=0,
            transcribed_word_count=0,
        )

    transcription_input = _prepare_transcription_audio(
        input_path,
        output_dir=output_dir,
        total_duration=max(total_duration, 0.001),
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    words = _transcribe_words(
        transcription_input,
        total_duration=max(total_duration, 0.001),
        settings=settings,
        progress_callback=progress_callback,
        log_callback=log_callback,
        cancel_callback=cancel_callback,
    )

    subtitle_path: Path | None = None
    caption_event_count = 0
    if settings.enabled and words:
        subtitle_path = output_dir / "captions.ass"
        caption_event_count = _write_ass_subtitles(
            subtitle_path,
            words,
            video_width=video_width,
            video_height=video_height,
            settings=settings,
        )

    profanity_intervals = _collect_profanity_intervals(words) if settings.profanity_filter else []
    censored_word_count = sum(1 for word in words if word.is_profanity) if settings.profanity_filter else 0
    return CaptionArtifactResult(
        subtitle_path=subtitle_path,
        profanity_intervals=profanity_intervals,
        caption_event_count=caption_event_count,
        censored_word_count=censored_word_count,
        transcribed_word_count=len(words),
    )


def _transcribe_words(
    input_path: Path,
    *,
    total_duration: float,
    settings: CaptionSettings,
    progress_callback: ProgressCallback | None,
    log_callback: LogCallback | None,
    cancel_callback: CancelCallback | None,
) -> list[CaptionWord]:
    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    model = _load_model(settings.model_name, log_callback)
    if log_callback:
        log_callback("Transcribing the kept speech for captions and profanity timing.")

    segments, _info = model.transcribe(
        str(input_path),
        beam_size=4,
        best_of=4,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    words: list[CaptionWord] = []
    segment_list = list(segments)
    for segment in segment_list:
        if cancel_callback and cancel_callback():
            raise RuntimeError("Processing was cancelled.")

        for token in _extract_segment_words(segment):
            start = max(0.0, min(float(token.start), total_duration))
            end = max(start + 0.02, min(float(token.end), total_duration))
            text = token.text.strip()
            if not text:
                continue

            is_profanity = _is_profanity(text)
            display_text = _mask_profanity(text) if is_profanity and settings.profanity_filter else text
            words.append(
                CaptionWord(
                    start=start,
                    end=end,
                    text=text,
                    display_text=display_text,
                    is_profanity=is_profanity,
                )
            )

        if progress_callback:
            segment_end = max(0.0, min(float(getattr(segment, "end", 0.0) or 0.0), total_duration))
            progress_callback(segment_end / total_duration, "Transcribing speech for captions...")

    if progress_callback:
        progress_callback(1.0, "Transcribing speech for captions...")
    return [word for word in words if word.end > word.start]


def _prepare_transcription_audio(
    input_path: Path,
    *,
    output_dir: Path,
    total_duration: float,
    progress_callback: ProgressCallback | None,
    log_callback: LogCallback | None,
) -> Path:
    wav_path = output_dir / "captions_audio.wav"
    if log_callback:
        log_callback("Extracting clean mono audio for more stable subtitle timing.")
    if progress_callback:
        progress_callback(0.03, "Preparing audio for captions...")

    command = [
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
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True, **_subprocess_kwargs())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Could not prepare audio for subtitles.") from exc

    if progress_callback:
        progress_callback(min(0.10, max(0.03, 2.0 / max(total_duration, 0.001))), "Preparing audio for captions...")
    return wav_path


def _extract_segment_words(segment: object) -> list[CaptionWord]:
    raw_words = getattr(segment, "words", None) or []
    extracted: list[CaptionWord] = []
    for word in raw_words:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        text = getattr(word, "word", None) or getattr(word, "text", None) or ""
        if start is None or end is None or not str(text).strip():
            continue
        extracted.append(
            CaptionWord(
                start=float(start),
                end=float(end),
                text=str(text).strip(),
                display_text=str(text).strip(),
                is_profanity=False,
            )
        )

    if extracted:
        return extracted

    segment_text = str(getattr(segment, "text", "") or "").strip()
    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
    segment_end = float(getattr(segment, "end", segment_start) or segment_start)
    tokens = [token for token in segment_text.split() if token.strip()]
    if not tokens:
        return []

    duration = max(segment_end - segment_start, 0.01)
    slice_duration = duration / len(tokens)
    fallback: list[CaptionWord] = []
    for index, token in enumerate(tokens):
        start = segment_start + slice_duration * index
        end = segment_start + slice_duration * (index + 1)
        fallback.append(
            CaptionWord(
                start=start,
                end=end,
                text=token,
                display_text=token,
                is_profanity=False,
            )
        )
    return fallback


def _write_ass_subtitles(
    subtitle_path: Path,
    words: list[CaptionWord],
    *,
    video_width: int,
    video_height: int,
    settings: CaptionSettings,
) -> int:
    events = _build_caption_events(words, settings)
    subtitle_path.write_text(
        _build_ass_document(events, video_width=video_width, video_height=video_height),
        encoding="utf-8",
    )
    return len(events)


def _build_caption_events(words: list[CaptionWord], settings: CaptionSettings) -> list[tuple[float, float, str]]:
    groups: list[list[CaptionWord]] = []
    current: list[CaptionWord] = []
    for word in words:
        if not current:
            current = [word]
            continue

        previous = current[-1]
        group_duration = word.end - current[0].start
        punctuation_break = previous.text.rstrip().endswith((".", "!", "?", ";", ":"))
        gap = max(0.0, word.start - previous.end)
        should_split = (
            len(current) >= max(settings.max_words_per_caption, 1)
            or group_duration > max(settings.max_caption_seconds, 0.1)
            or gap > max(settings.max_word_gap_seconds, 0.05)
            or punctuation_break
        )
        if should_split:
            groups.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        groups.append(current)

    events: list[tuple[float, float, str]] = []
    for group_index, group in enumerate(groups):
        next_group_start = groups[group_index + 1][0].start if group_index + 1 < len(groups) else None
        for index, word in enumerate(group):
            start = word.start
            if index < len(group) - 1:
                end = max(word.end, group[index + 1].start)
            else:
                end = max(word.end, group[-1].end + 0.08)
                if next_group_start is not None:
                    end = min(end, max(word.end, next_group_start - 0.001))
            if end <= start:
                end = start + 0.01
            events.append((start, end, _format_event_text(group, index)))
    return _normalize_caption_events(events)


def _build_ass_document(
    events: list[tuple[float, float, str]],
    *,
    video_width: int,
    video_height: int,
) -> str:
    font_size = max(28, int(round(video_height * 0.055)))
    active_font_size = max(font_size + 8, int(round(font_size * 1.12)))
    outline = max(3, int(round(font_size * 0.10)))
    margin_h = max(42, int(round(video_width * 0.06)))
    margin_v = max(84, int(round(video_height * 0.12)))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        (
            "Style: TikTok,Arial,"
            f"{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline},0,2,{margin_h},{margin_h},{margin_v},1"
        ),
        (
            "Style: TikTokActive,Arial,"
            f"{active_font_size},&H0000FFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline},0,2,{margin_h},{margin_h},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for start, end, text in events:
        lines.append(
            f"Dialogue: 0,{_format_ass_time_start(start)},{_format_ass_time_end(end)},TikTok,,0,0,0,,{{\\an2\\blur0.7}}{text}"
        )
    return "\n".join(lines) + "\n"


def _format_event_text(words: list[CaptionWord], active_index: int) -> str:
    rendered_words: list[str] = []
    visible_lengths: list[int] = []
    for index, word in enumerate(words):
        display = _escape_ass(word.display_text.upper())
        visible_lengths.append(len(word.display_text))
        if index == active_index:
            rendered_words.append(r"{\rTikTokActive}" + display + r"{\rTikTok}")
        else:
            rendered_words.append(display)

    if len(rendered_words) >= 4 or sum(visible_lengths) > 18:
        midpoint = max(1, len(rendered_words) // 2)
        return " ".join(rendered_words[:midpoint]) + r"\N" + " ".join(rendered_words[midpoint:])
    return " ".join(rendered_words)


def _collect_profanity_intervals(words: list[CaptionWord]) -> list[tuple[float, float]]:
    intervals = [
        (max(0.0, word.start - 0.04), max(word.end + 0.04, word.start + 0.06))
        for word in words
        if word.is_profanity
    ]
    if not intervals:
        return []

    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 0.03:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _is_profanity(text: str) -> bool:
    core = re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)
    if not core:
        return False
    return any(core == prefix or core.startswith(prefix) for prefix in _PROFANITY_PREFIXES)


def _mask_profanity(text: str) -> str:
    match = re.match(r"^([^\w]*)([\w']+)([^\w]*)$", text, flags=re.UNICODE)
    if not match:
        return "***"
    prefix, core, suffix = match.groups()
    if len(core) <= 1:
        masked = "*"
    elif len(core) == 2:
        masked = core[0] + "*"
    else:
        masked = core[0] + "*" * (len(core) - 1)
    return prefix + masked + suffix


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _normalize_caption_events(events: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    if not events:
        return []

    normalized: list[tuple[float, float, str]] = []
    previous_end = -1.0
    min_duration = 0.01

    for start, end, text in events:
        safe_start = max(start, previous_end if previous_end >= 0 else start)
        safe_end = max(end, safe_start + min_duration)
        normalized.append((safe_start, safe_end, text))
        previous_end = safe_end

    return normalized


def _format_ass_time_start(seconds: float) -> str:
    total_centiseconds = max(0, int(math.ceil(seconds * 100)))
    return _format_ass_centiseconds(total_centiseconds)


def _format_ass_time_end(seconds: float) -> str:
    total_centiseconds = max(0, int(math.floor(seconds * 100)))
    return _format_ass_centiseconds(total_centiseconds)


def _format_ass_centiseconds(total_centiseconds: int) -> str:
    centiseconds = total_centiseconds % 100
    total_seconds = total_centiseconds // 100
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02d}:{seconds_part:02d}.{centiseconds:02d}"


def _load_model(model_name: str, log_callback: LogCallback | None) -> object:
    cache_key = ("cpu", model_name)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "Subtitles and profanity filtering need the `faster-whisper` package. "
            "Install the updated requirements first."
        ) from exc

    if log_callback:
        log_callback(f"Loading speech-to-text model '{model_name}'. The first run may download it.")

    download_root = Path.home() / ".cache" / "speech_cutter" / "whisper"
    download_root.mkdir(parents=True, exist_ok=True)

    cpu_threads = max((os.cpu_count() or 4) - 1, 1)
    try:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(download_root),
            cpu_threads=cpu_threads,
        )
    except ValueError:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="float32",
            download_root=str(download_root),
            cpu_threads=cpu_threads,
        )
    _MODEL_CACHE[cache_key] = model
    return model


def _subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs
