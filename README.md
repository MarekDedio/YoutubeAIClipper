# Speech Cutter

Speech Cutter is a small desktop app that loads a video, finds the parts with human speech, and exports a new MP4 that keeps only those spoken sections.

## What it does

- Accepts common video formats like `mp4`, `webm`, `mkv`, `mov`, and more.
- Detects spoken audio with the bundled Silero VAD ONNX model.
- Removes non-speech stretches and joins the kept parts into a single MP4.
- Runs in a simple Windows-friendly Tkinter GUI and also supports a CLI mode.

## Requirements

- Python 3.11 or newer
- `ffmpeg` and `ffprobe` available in `PATH`

This machine already has `ffmpeg`, so the main extra step is installing Python packages from `requirements.txt`.

## Run it

### Option 1: double-click launcher

Run [run.bat](C:\Users\Marel\Desktop\CODEX CLIP\run.bat).

The launcher will:

1. Create `.venv` if it does not exist.
2. Install the required Python packages.
3. Start the GUI.

### Option 2: terminal

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

## CLI example

```powershell
.\.venv\Scripts\python app.py input.mp4 output.mp4 --preset balanced
```

Available presets:

- `natural`
- `balanced`
- `aggressive`

## Notes

- The exported file is always MP4 for wide compatibility.
- Very noisy videos can still produce occasional false positives or false negatives.
- The first run may take a little longer while dependencies install.

## Third-party component

This app bundles the official Silero VAD 16 kHz ONNX model under the MIT license.

- Repo: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- License copy: [third_party_licenses/Silero-VAD-LICENSE.txt](C:\Users\Marel\Desktop\CODEX CLIP\third_party_licenses\Silero-VAD-LICENSE.txt)

