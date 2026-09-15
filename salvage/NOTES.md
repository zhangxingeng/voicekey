# Salvage notes — everything worth keeping from prompt-compose v0.3.3

The old app is being scrapped. This directory is the distilled, stack-independent
knowledge: what we learned about Whisper, audio capture, and model handling, so a
rewrite in *any* stack doesn't relearn it the hard way.

Full original source is recoverable at git tag `pre-rewrite-v0.3.3`.

## Contents

- `download-whisper-model.sh` — standalone model fetch+verify+extract. Works
  today, no build step, no app.
- `reference/engine.rs` — the sherpa-onnx Whisper recognizer wrapper (~100 lines).
- `reference/audio.rs` — cpal capture → mono downmix → 16kHz resample (~150 lines).
- `reference/session-state.rs` — start/stop session lifecycle on a background
  thread, with the decode happening once on stop.

These three files are the *entire* useful core. Everything else in the old repo
(4,900 lines of prompt-library UI, semantic match, SQLite embedding cache,
self-updater, project tabs, variable grammar) is not being carried forward.

## Hard-won facts

**Model choice: Whisper large-v3-turbo int8, not SenseVoice-Small.**
SenseVoice is 3x smaller and much faster but mangled English technical jargon
beyond usability ("GitHub" → "GET UP", "Kubernetes" → "CORNATTIE ENGINES").
Whisper turbo handled the same clips correctly in both English and Mandarin.
This was tested on real voice, not benchmarks. Don't re-litigate it without
re-running that comparison.

**Whisper decodes in 30-second windows.** sherpa-onnx's Whisper decoder silently
truncates any single `decode_stream` call at 30s of audio. This is a runtime
limit, not a model limit — every real Whisper deployment chunks internally. Split
into sequential, non-overlapping 30s windows and join the text. Cost stays linear.
A word landing exactly on a boundary can come out mangled; that's the standard trade.

**Do not do live partial transcription by redecoding the buffer.** An earlier
version redecoded the whole growing buffer every 800ms to show interim text. Cost
is quadratic in utterance length — a 10-minute utterance burns hours of CPU, and
individual redecodes take tens of seconds long before that. No accurate
genuine-streaming (persistent-state, linear-cost) bilingual model was found. One
decode per utterance on release is the right architecture.

**Audio capture shape.** cpal gives whatever native rate/format/channel count the
device offers. You must: handle both F32 and I16 sample formats, average channels
to mono, and resample to exactly 16kHz f32 — Whisper accepts nothing else.
`sherpa_onnx::LinearResampler` does the resample, so no extra crate is needed if
you're already linking sherpa-onnx.

**Language flag.** Whisper's auto-detect is triggered by an *empty string*, not
the word "auto". `language = ""` for auto, `"en"`, `"zh"` otherwise.

**Model artifacts.** Three files, ~1.0GB total:
`turbo-encoder.int8.onnx` (675MB), `turbo-decoder.int8.onnx` (361MB),
`turbo-tokens.txt` (817KB). Already downloaded on this machine at
`~/.prompt-compose/models/sherpa-onnx-whisper-turbo/` — reuse them, don't refetch.
Verify by presence, never by re-hashing 1GB on every check.

**Download discipline worth keeping.** Pinned URL + hardcoded sha256, verified
*before* extracting anything, and atomic write (temp sibling + rename) so a crash
can't leave a truncated-but-plausible model on disk. `download-whisper-model.sh`
preserves this.

## GPU: the open problem

Machine has an RTX 3090 Ti (24GB). The old app ran CPU-only —
`config.model_config.provider = "cpu"`, `num_threads = 2`.

**The `sherpa-onnx` Rust crate (1.13.4) has no CUDA path.** Its only features are
`static` and `shared`; `sherpa-onnx-sys`'s build.rs downloads a prebuilt CPU
native library at build time. Setting `provider = "cuda"` against that lib will
not give you GPU. Getting GPU through this crate means building sherpa-onnx from
source with CUDA enabled and linking it yourself.

Alternatives to weigh when picking the stack:
- `whisper.cpp` — has CUDA and Vulkan backends, ships GGML models, `whisper-rs`
  is a maintained Rust binding with a `cuda` feature. Runtime provider fallback
  to CPU is a build-flag question, not free.
- `faster-whisper` (CTranslate2, Python) — mature CUDA support, `device="cuda"`
  with clean `device="cpu"` fallback at runtime, one line. Easiest GPU story by
  far; cost is a Python runtime in the loop.
- ONNX Runtime CUDA EP directly — provider list `["CUDAExecutionProvider",
  "CPUExecutionProvider"]` gives genuine automatic fallback, but means dropping
  sherpa-onnx's convenience wrapper and driving Whisper's encoder/decoder loop
  ourselves. Most work.

GPU-vs-CPU on turbo-int8 is roughly the difference between "text appears while
you're still letting go of the key" and "a couple of seconds of waiting" for a
short utterance. For a hotkey dictation tool that gap is the whole product feel.

## Platform constraint that decides the stack

Desktop is **Ubuntu on Wayland**. A global hotkey that works while another app
has focus is the hard part, not the transcription:

- Wayland has no global-hotkey protocol for ordinary clients. X11-era grabbing
  (and anything built on it, including Tauri's `global-shortcut` plugin) does not
  work under a Wayland session.
- Workable routes: (a) register a GNOME custom keybinding that runs a tiny CLI
  which signals a long-running daemon — supported, survives updates, no root;
  (b) read `/dev/input` via evdev — needs the user in the `input` group, sees all
  keystrokes, heavier trust; (c) the XDG global-shortcuts portal — correct in
  principle, GNOME support is not something to assume without testing.

(a) is the safe default. This should be settled before writing app code.
