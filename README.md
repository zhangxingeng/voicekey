# voicekey

Local dictation. Press a key, speak, get text you can fix and paste. Everything
runs on your machine — nothing is uploaded.

> **Status: in progress.** The engine works and transcribes real speech. The
> hotkey, level meter and continuous recording are being built now. See
> [`STATE.md`](STATE.md) for the design and every decision made so far.

## What it does

Launch it, then press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd> to start
recording and again to stop. The text appears in an editable box — fix whatever
Whisper got wrong, then copy it.

Recording never waits. You can stop one burst and immediately start another
while the first is still decoding; the results append in order.

That is the whole product. No prompt library, no settings sprawl.

## How it works

Whisper `large-v3-turbo` driven directly through ONNX Runtime at the graph
level — log-mel spectrogram, encoder, then a KV-cached autoregressive decode
loop. There is no wrapper library between the app and the model, which is
deliberate: it is about 250 lines and it is the interesting part.

GPU is used when available and falls back to CPU automatically. The execution
provider is probed at runtime and then **confirmed against the live session**,
because ONNX Runtime degrades to CPU silently.

One int8 model serves every backend. An fp16 build was planned until it was
measured: int8 on CUDA is 2.15x faster than int8 on CPU, so the second model
bought nothing and doubled the download.

| Platform | Accelerator |
|---|---|
| Linux / Windows + NVIDIA | CUDA |
| macOS (Apple Silicon) | CoreML |
| anything else | CPU |

## Install

Download the artifact for your platform from Releases and run it — no Python, no
pip, no system packages.

On first run it downloads the speech model (~1.0 GB) into your platform's data
directory, verifying it before use.

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
  mel.py       log-mel spectrogram, Whisper-exact
  engine.py    encoder + KV-cached decode loop
  backend.py   execution-provider probe and live confirmation
  cuda.py      NVIDIA library preload (ORT does not find CUDA 13 on its own)
  audio.py     microphone capture, loopback filtering
  vad.py       silence gate — Whisper invents text for silence
  meter.py     microphone level for the visual cue
  models.py    first-run model download and verification
  session.py   burst queue: recording and transcription run independently
  display.py   the one type the session and the UI agree on
  paths.py     per-OS model/data locations
  ui.py        the window (tkinter)
  __main__.py  entry point
salvage/       distilled notes from the previous Tauri/Rust app
STATE.md       design decisions, measured findings, open questions
```

## License

MIT
