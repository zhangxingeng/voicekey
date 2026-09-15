# voicekey

Hotkey-triggered local speech-to-text. Press a key, speak, get text you can paste.
Runs entirely on your machine — nothing is uploaded.

> **Status: early.** The scaffold, tooling, and packaging are in place. The
> inference engine is next. See [`STATE.md`](STATE.md) for the full design and
> every decision made so far.

## What it does

Press the hotkey → a small popup appears and recording starts immediately →
stop → the transcription appears, ready to copy.

That is the whole product. No prompt library, no history, no settings sprawl.

## How it works

Whisper `large-v3-turbo` driven directly through ONNX Runtime at the graph level —
log-mel spectrogram, encoder, then a KV-cached autoregressive decode loop. There is
no wrapper library between the app and the model, which is deliberate: it is about
250 lines and it is the interesting part.

GPU is used when available and falls back to CPU automatically. The execution
provider is probed at runtime, not assumed at install time:

| Platform | Accelerator | Weights |
|---|---|---|
| Linux / Windows + NVIDIA | CUDA | fp16 |
| macOS (Apple Silicon) | CoreML | fp16 |
| anything else | CPU | int8 |

## Install

Download the artifact for your platform from Releases and run it — no Python, no
pip, no system packages.

On first run it downloads the speech model (~1.0 GB) into your platform's data
directory. If an NVIDIA GPU is detected it offers to fetch the CUDA runtime too.

## Development

Requires [uv](https://docs.astral.sh/uv/). It manages the Python interpreter and
Tk, so there is nothing else to install — except PortAudio on Linux:

```sh
sudo apt install libportaudio2      # Linux only, build/dev requirement
uv sync --dev
uv run voicekey
```

Checks:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Frozen build (what CI produces):

```sh
uv run pyinstaller --noconfirm --clean --name voicekey --windowed \
  --collect-binaries sounddevice src/voicekey/__main__.py
```

## Layout

```
src/voicekey/
  backend.py   execution-provider probe → providers + quantization
  paths.py     per-OS model/data locations
  ui.py        the popup (tkinter, behind a 4-method interface)
  __main__.py  entry point
salvage/       distilled notes from the previous Tauri/Rust app
STATE.md       design decisions, open questions, machine facts
```

## License

MIT
