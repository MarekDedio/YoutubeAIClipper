from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speech_cutter.captions import CaptionSettings
from speech_cutter.gui import launch_gui
from speech_cutter.pipeline import build_output_path, check_runtime, process_video
from speech_cutter.presets import PRESETS, build_options


def _make_cli_progress_callback():
    last_percent = -1
    last_status = ""

    def callback(value: float, status: str) -> None:
        nonlocal last_percent, last_status
        percent = int(value * 100)
        if percent != last_percent or status != last_status:
            print(f"{value * 100:5.1f}%  {status}", flush=True)
            last_percent = percent
            last_status = status

    return callback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep only spoken parts from a video and export the result as MP4."
    )
    parser.add_argument("input", nargs="?", help="Source video file.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Optional legacy argument. Output is now always saved next to the source video.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="natural",
        help="Speech trimming preset to use in CLI mode.",
    )
    parser.add_argument("--padding-seconds", type=float, help="Extra padding to keep around each speech segment.")
    parser.add_argument("--padding-ms", type=int, help="Legacy option for padding in milliseconds.")
    parser.add_argument("--merge-gap-ms", type=int, help="Merge nearby speech chunks when the pause is shorter than this.")
    parser.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Burn in TikTok-style subtitles on the output video.",
    )
    parser.add_argument(
        "--profanity-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically mute profane words using the subtitle timing pass.",
    )
    parser.add_argument(
        "--caption-model",
        default=CaptionSettings().model_name,
        help="Whisper model name to use for subtitles and profanity timing.",
    )
    args = parser.parse_args(argv)

    issues = check_runtime()
    if issues:
        for issue in issues:
            print(f"Error: {issue}", file=sys.stderr)
        return 1

    if args.input:
        output_path = build_output_path(Path(args.input))
        if args.output:
            print(
                f"Output path is automatic in this version. Saving to {output_path}",
                flush=True,
            )
        options = build_options(
            args.preset,
            padding_ms=(
                int(round(args.padding_seconds * 1000))
                if args.padding_seconds is not None
                else args.padding_ms
            ),
            merge_gap_ms=args.merge_gap_ms,
            captions=CaptionSettings(
                enabled=bool(args.captions),
                profanity_filter=bool(args.profanity_filter),
                model_name=str(args.caption_model),
            ),
        )
        result = process_video(
            Path(args.input),
            output_path,
            options=options,
            progress_callback=_make_cli_progress_callback(),
            log_callback=lambda message: print(message, flush=True),
        )
        print(
            f"Saved {result.output_path} with {len(result.segments)} speech segment(s). "
            f"Kept {result.kept_duration:.1f}s / {result.input_duration:.1f}s."
        )
        return 0

    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
