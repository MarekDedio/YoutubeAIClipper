from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

from .pipeline import ProcessingOptions
from .vad import VadSettings


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    settings: VadSettings


PRESETS: "OrderedDict[str, Preset]" = OrderedDict(
    [
        (
            "natural",
            Preset(
                key="natural",
                label="Keep Natural Pauses",
                description="Leaves a little more breathing room around spoken parts.",
                settings=VadSettings(
                    threshold=0.45,
                    min_speech_ms=220,
                    min_silence_ms=220,
                    speech_pad_ms=180,
                    merge_gap_ms=500,
                ),
            ),
        ),
        (
            "balanced",
            Preset(
                key="balanced",
                label="Balanced",
                description="Good default for most talking-head videos, podcasts, and lessons.",
                settings=VadSettings(
                    threshold=0.50,
                    min_speech_ms=250,
                    min_silence_ms=160,
                    speech_pad_ms=120,
                    merge_gap_ms=350,
                ),
            ),
        ),
        (
            "aggressive",
            Preset(
                key="aggressive",
                label="Trim Harder",
                description="Cuts tighter and removes short pauses more aggressively.",
                settings=VadSettings(
                    threshold=0.58,
                    min_speech_ms=260,
                    min_silence_ms=120,
                    speech_pad_ms=70,
                    merge_gap_ms=180,
                ),
            ),
        ),
    ]
)


def build_options(
    preset_key: str,
    *,
    padding_ms: int | None = None,
    merge_gap_ms: int | None = None,
) -> ProcessingOptions:
    preset = PRESETS[preset_key]
    settings = preset.settings

    if padding_ms is not None:
        settings = replace(settings, speech_pad_ms=int(padding_ms))
    if merge_gap_ms is not None:
        settings = replace(settings, merge_gap_ms=int(merge_gap_ms))

    return ProcessingOptions(vad=settings)
