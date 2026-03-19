from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import wave

import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 16_000
CHUNK_SIZE = 512
CONTEXT_SIZE = 64


class DetectionCancelledError(RuntimeError):
    """Raised when the user cancels detection."""


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VadSettings:
    threshold: float = 0.50
    min_speech_ms: int = 250
    min_silence_ms: int = 150
    speech_pad_ms: int = 120
    merge_gap_ms: int = 350


class SileroOnnxVad:
    """Thin ONNX wrapper around the official Silero 16 kHz VAD model."""

    def __init__(self, model_path: Path) -> None:
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)

    def predict(self, chunk: np.ndarray) -> float:
        if chunk.shape != (CHUNK_SIZE,):
            raise ValueError(f"Expected a mono chunk of {CHUNK_SIZE} samples, got {chunk.shape!r}")

        model_input = np.concatenate((self._context, chunk.reshape(1, -1)), axis=1)
        output, state = self._session.run(
            None,
            {
                "input": model_input.astype(np.float32, copy=False),
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        self._state = state.astype(np.float32, copy=False)
        self._context = model_input[:, -CONTEXT_SIZE:].astype(np.float32, copy=False)
        return float(output[0, 0])


def detect_speech_in_wav(
    wav_path: Path,
    detector: SileroOnnxVad,
    settings: VadSettings,
    *,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> list[SpeechSegment]:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()

        if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected 16 kHz mono PCM WAV, got {channels} channel(s), "
                f"{sample_width * 8}-bit, {sample_rate} Hz."
            )

        detector.reset_states()
        raw_segments = _collect_segments(
            wav_file,
            total_frames,
            detector,
            settings,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    merged = _merge_close_segments(raw_segments, int(SAMPLE_RATE * settings.merge_gap_ms / 1000))
    padded = _pad_segments(merged, total_frames, int(SAMPLE_RATE * settings.speech_pad_ms / 1000))

    return [SpeechSegment(start / SAMPLE_RATE, end / SAMPLE_RATE) for start, end in padded]


def _collect_segments(
    wav_file: wave.Wave_read,
    total_frames: int,
    detector: SileroOnnxVad,
    settings: VadSettings,
    *,
    progress_callback: Callable[[float], None] | None,
    cancel_callback: Callable[[], bool] | None,
) -> list[tuple[int, int]]:
    min_speech_samples = int(SAMPLE_RATE * settings.min_speech_ms / 1000)
    min_silence_samples = int(SAMPLE_RATE * settings.min_silence_ms / 1000)
    neg_threshold = max(settings.threshold - 0.15, 0.01)

    raw_segments: list[tuple[int, int]] = []
    triggered = False
    current_start = 0
    temp_end: int | None = None
    processed = 0

    while processed < total_frames:
        if cancel_callback and cancel_callback():
            raise DetectionCancelledError("Speech detection was cancelled.")

        frames_to_read = min(CHUNK_SIZE, total_frames - processed)
        frame_bytes = wav_file.readframes(frames_to_read)
        chunk = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0

        if chunk.size < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - chunk.size))

        probability = detector.predict(chunk)
        current_sample = processed
        processed += frames_to_read

        if probability >= settings.threshold:
            if not triggered:
                triggered = True
                current_start = current_sample
            temp_end = None
        elif triggered:
            if temp_end is None:
                temp_end = current_sample

            if current_sample - temp_end >= min_silence_samples or probability < neg_threshold:
                if current_sample - temp_end < min_silence_samples and processed < total_frames:
                    continue
                if temp_end - current_start >= min_speech_samples:
                    raw_segments.append((current_start, temp_end))
                triggered = False
                temp_end = None

        if progress_callback:
            progress_callback(min(processed / max(total_frames, 1), 1.0))

    if triggered and total_frames - current_start >= min_speech_samples:
        raw_segments.append((current_start, total_frames))

    return raw_segments


def _merge_close_segments(
    segments: list[tuple[int, int]],
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged: list[tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_samples:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _pad_segments(
    segments: list[tuple[int, int]],
    total_frames: int,
    pad_samples: int,
) -> list[tuple[int, int]]:
    padded: list[tuple[int, int]] = []
    for start, end in segments:
        padded_start = max(0, start - pad_samples)
        padded_end = min(total_frames, end + pad_samples)

        if padded and padded_start <= padded[-1][1]:
            prev_start, prev_end = padded[-1]
            padded[-1] = (prev_start, max(prev_end, padded_end))
        else:
            padded.append((padded_start, padded_end))

    return padded
