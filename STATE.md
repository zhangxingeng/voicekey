# Project state — read this first

External memory. If context was compacted, this file is the picture.

Last updated: 2026-09-18

---

## What we're building

A dictation **app** (not a service). You launch it; it stays open. Press
Ctrl+Shift+D to start recording, press again to stop. Transcribed text lands in
an editable box you can fix and copy.

- **Recording is instant and never blocks.** It does not wait for the model, the
  GPU, or a previous transcription. Recording is snappy; transcription can take
  as long as it needs.
- **Continuous bursts.** Stop one burst and immediately start another while the
  first is still decoding. Results append in submission order.
- **Editable transcript.** You fix Whisper's mistakes in place before copying.
- **Visual level meter** so a dead or muted mic is visible while you speak.
- GPU-accelerated Whisper, CPU fallback, one-button install.
- Explicitly NOT wanted: prompt library, snippets, projects, variables, semantic
  match, self-updater — the old app's entire feature set.

### Open product question

**Is Ctrl+Shift+D global or in-window?** Unresolved, and it decides whether
`hotkey.py` + `ipc.py` exist at all.

- *In-window*: a Tk binding. Zero extra modules. You must focus the window first.
- *Global*: GNOME custom keybinding → tiny CLI → unix socket → running app.
  ~120 lines. Works from anywhere, which is the point of dictation.

Dropping the *daemon* does not force in-window — a GNOME keybinding plus a
socket is still an app you launch.

---

## Stack — SETTLED

**Python, single language, all the way.** No Rust-in-Python, no Node/Bun, no
split-language architecture.

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.14 via uv | uv's **managed** builds bundle the interpreter *and* Tk 9.0 |
| Inference | `onnxruntime` / `onnxruntime-gpu`, driven at the bare-graph level | learning goal + real runtime CPU fallback |
| Numerics | `numpy` | mel spectrogram, tensor glue |
| Audio | `sounddevice` | industry-standard PortAudio, official PyInstaller hook |
| UI | `tkinter` (Tk 9.0) | zero install, bundled by uv, ~335ms cold start |
| Model dir | `platformdirs` | correct per-OS location |
| Packaging | PyInstaller via GitHub Actions matrix | AppImage / .app / .exe, zero user deps |

### Why not the alternatives (don't re-litigate without new evidence)

- **Rust-in-Python (PyO3):** no CPU-bound Python hot loop exists — the hot loop
  is ORT (C++). Would *create* per-platform wheel pain rather than solve it.
- **Node/Bun:** no numpy, worse audio capture, Electron for UI.
- **Whole thing in Rust:** cleanest single binary, but the app is noise against a
  1–3GB model download, and it costs REPL-driven tensor debugging.
- **HTML/JS UI:** there is no "raw HTML" on desktop — you either borrow the OS
  webview (Linux = WebKitGTK + PyGObject + typelibs; this machine has two
  incompatible versions installed) or ship Chromium (100MB+). We already deleted
  a Tauri app that did the former.
- **Flet / NiceGUI / Reflex:** NiceGUI and Reflex are web servers with a browser
  in the critical path. Flet bundles the Flutter engine (~60MB) and runs the UI
  as a second process.
- **Dear PyGui:** genuinely lighter and faster, but has no native text widget —
  selection and clipboard would be hand-rolled, and an editable text box is
  precisely what this app is.

---

## Key design rules

**1. The engine is a pure function.** numpy in, text out. No tkinter, no asyncio,
no IPC anywhere in it. A future Android port becomes a rewrite of the shell, not
the brain.

**2. Provider is a runtime probe, never an install-time assumption** — confirmed
against a *live session*, because ORT degrades to CPU silently. See
`backend.resolve()` / `backend.confirm()` and finding #2 below.

**3. Models are never packaged.** ~1GB, downloaded on first run into a
`platformdirs` location, sha256-verified before extraction, temp-then-move so an
interrupted run cannot leave a half-model.

**4. Recording never waits on anything.** Audio capture opens in ~20ms; a Tk
window maps in ~50–100ms; the model loads in seconds. Capture starts first and
bursts queue up even before the model exists.

**5. Inject the expensive dependency.** Every module takes its slow collaborator
(the model, the network fetcher, the subprocess runner) as an argument with a
real default. This is why the modules can be built and tested in parallel,
without a GPU, a network, or a display.

**6. Cost is linear in audio length — no cutting needed.** Whisper's encoder is
O(n²) in sequence length, but its input is *fixed* at 30s / 1500 frames; longer
audio is chunked at a 30s stride (`engine.py`) and the KV cache makes decode
linear in tokens. A 5-minute recording costs exactly 10x a 30-second one.
Quadratic cost only appears with **live partials** (re-decoding a growing
buffer), which is why we don't do them.

*Known flaw, deliberately deferred:* the 30s stride can cut a word in half. The
cheap fix is moving the boundary to the quietest 100ms nearby, not
overlap-and-stitch (which needs the timestamp tokens we disabled). Most
dictation bursts never reach 30s.

---

## The bare-ONNX interface (verified against the real model files)

```
encoder:  mel[n_audio,128,T]  →  cross_k[4,·,T,1280], cross_v[4,·,T,1280]
decoder:  tokens, in_self_k_cache[4,·,448,1280], in_self_v_cache,
          cross_k, cross_v, offset
       →  logits[·,·,51866], out_self_k_cache, out_self_v_cache
```

Why this export is unusually good for bare interfacing:

- **KV cache is explicitly exposed** (`in_/out_self_k_cache` + `offset`) → the
  decode loop is linear-cost, one token per step.
- **Encoder emits cross-attention K/V directly** → it runs once per 30s window.
- **All Whisper constants live in encoder metadata**: `sot=50258`, `eot=50257`,
  `transcribe=50360`, `no_timestamps=50364`, `n_mels=128`, `n_text_ctx=448`,
  `n_vocab=51866`, `sot_sequence=50258,50259,50360`, full language table.
  Read them off the file; don't hardcode.
- **Token table is base64 raw bytes** (`IQ==  0` → `!`). Detokenize = b64decode,
  concat, utf-8 decode. No GPT-2 byte-BPE dance.
- Turbo shape: **32 encoder layers, only 4 decoder layers** — the looped part is
  tiny.
- **Auto-detect language is the empty string**, not `"auto"`.

---

## Measured findings (don't re-derive these)

Benchmarked on the JFK sample (11.0s, 16kHz mono), RTX 3090 Ti:

| backend | decode | realtime factor |
|---|---|---|
| CPU int8 | 8.73s | 1.26x |
| **CUDA int8** | **4.06s** | **2.71x** |

**1. int8-on-CUDA is fine — the fp16 plan was wrong.** The prediction was that
ORT's CUDA EP would handle quantized ops so badly that int8-on-CUDA would lose to
CPU. Measured, it is **2.15x faster**. So there is **one int8 model for every
backend**, which halves the download and deletes an axis of configuration.

**2. ONNX Runtime falls back to CPU silently.** `get_available_providers()`
reports what ORT was *built* with, not what it can load. With CUDA libraries
missing it still listed `CUDAExecutionProvider`, then quietly bound CPU — right
answers, 2x slower, only a stderr warning. **Read `session.get_providers()` after
construction**, never the pre-flight probe.

**3. `onnxruntime-gpu` does not bundle the CUDA runtime.** It needs cuBLAS,
cuDNN 9, cuRAND, cuFFT, cuSPARSE, cuSOLVER — ~2GB. For CUDA 13 these live on
**`https://pypi.nvidia.com`** (the PyPI names are redirect stubs), and CUDA 13
moved them to a shared **`nvidia/cu13/lib`** that ORT 1.30 does not search.
`cuda.py` dlopens them with `RTLD_GLOBAL` before session creation.

**4. Whisper hallucinates on silence.** Digital silence decodes to `"you"`, room
tone to `"."`. The model's `<|nospeech|>` head would be the principled fix, but
**this export does not produce one** — `P(<|nospeech|>)` is `0.000000` for
silence and speech alike, at every position. So the gate is audio-side
(`vad.py`), using a noise floor estimated from the recording itself rather than
an absolute RMS threshold, which would encode *this* microphone's gain:

| | rms | crest |
|---|---|---|
| room tone | 0.0086 | 1.1 (flat) |
| speech | 0.1421 | 5.5 (peaky) |

**5. The loopback filter matched nothing.** One device, two spellings:
PulseAudio's *description* is `"Monitor of <sink>"` (what `pactl` shows), but
PortAudio reports the PipeWire *node name*, where it is a `.monitor` **suffix**.
Both forms are matched now, with tests using the real observed strings.

**6. uv bundles Tk only in *managed* builds.** A system interpreter satisfies
`requires-python` equally well and may have no Tk — Homebrew's `python@3.14`
splits it into a separate formula, which broke macOS CI. Pinned via
`python-preference = "only-managed"`, guarded by `tests/test_packaging.py`.

---

## Wayland reality (GNOME 50.1, verified on this machine)

| Want | Status |
|---|---|
| Global hotkey | ✅ GNOME custom keybinding via `gsettings`. No root, no extension. |
| Always-on-top | ✅ `root.attributes("-topmost", True)`. |
| Position window at a screen corner | ❌ Wayland clients cannot position themselves. Needs layer-shell, which GNOME doesn't offer apps and Tk can't speak. |
| Type into the focused app | ❌ `/dev/uinput` is root-only; `wtype` needs a protocol GNOME doesn't implement. Only route is the RemoteDesktop portal (consent dialog, unproven). **Deferred — clipboard only.** |

---

## Architecture

One frozen dataclass carries the coordination: `Session` produces `Display`, the
UI consumes it and knows nothing else — not audio, not numpy, not the model.

The transcript is deliberately **not** in `Display`. The box is editable, so the
UI owns the text and the session only hands it fragments to append. Otherwise
every burst finishing mid-edit would have to reconcile against what the user had
just typed.

```
src/voicekey/
  mel.py       log-mel spectrogram, Whisper-exact   DONE, 10 tests
  engine.py    encoder + KV-cached decode loop      DONE, verified on real speech
  cuda.py      NVIDIA library preload               DONE, 4 tests
  backend.py   provider probe + confirm             DONE, 5 tests
  paths.py     per-OS model locations               DONE, 3 tests
  vad.py       silence gate (adaptive noise floor)  DONE, 9 tests
  audio.py     mic capture + loopback filter        DONE, 10 tests
  display.py   the Display/State contract           DONE (frozen seam)
  meter.py     mic level → 0..1, dB-scaled          wave 1
  models.py    first-run download + verify          wave 1
  session.py   burst queue, ordering, status        wave 1
  ui.py        Tk window, meter, editable box       wave 2
  __main__.py  wiring                               rewrite after wave 2
```

The mel and vad tests pin *invariants* rather than output text, because both
fail silently: a wrong front-end hallucinates fluently, and a wrong gate either
drops real speech or lets invented text through.

---

## Build/packaging notes

- Build machine (and CI) needs `libportaudio2` — 78KB apt package. PyInstaller
  collects the system `.so` into the bundle, so **end users need nothing**.
- **No in-app `apt install`.** Needs a polkit password dialog, is Debian-only,
  breaks on immutable distros and containers, and contradicts the zero-install
  goal. The freeze already solves it.
- Build the AppImage on an **older base** (Ubuntu 22.04 / manylinux) — bundling a
  system `.so` makes glibc compatibility a real concern.
- macOS gets **CoreML EP inside the standard `onnxruntime` wheel** — free.
- `onnxruntime-gpu` is Linux/Windows x86_64 only → optional extra.
- **`build.yml` has never been executed.** Freezing is unproven.

---

## Machine facts

- Ubuntu, Wayland, GNOME Shell 50.1.
- NVIDIA RTX 3090 Ti 24GB, driver 595.91.07, CUDA 13.2, no system CUDA toolkit.
- Python 3.14.6 via uv (Tk 9.0 bundled).
- Available: `uv`, `wl-copy`, `xclip`, `gsettings`, `dbus-send`, `notify-send`.
- Absent: `xdotool`, `ydotool`, `wtype`. `/dev/uinput` is root-only.
- Mic is likely `USB Audio Device Mono`. Most enumerated capture devices are
  loopback monitors (system output, not mics) — see finding #5.
- Model on disk at `~/.prompt-compose/models/sherpa-onnx-whisper-turbo/` (int8,
  ~1.0GB), left by the old app. `__main__.py` falls back to it.

---

## Open questions

1. **Global vs in-window hotkey** — see top. Blocks wave 2.
2. **Is fp16 faster than int8 on CUDA?** Untested, needs a separate ~1.6GB
   download. Only worth it if 2.71x realtime proves too slow in practice.
3. **RemoteDesktop portal for cursor injection** — unproven, deferred.
4. **Full loop with human speech into the mic.** Capture is proven, the engine is
   proven on a speech file, and the GUI has been used successfully — but no
   automated end-to-end test exists.
