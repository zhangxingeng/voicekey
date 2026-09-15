# Project state — read this first

External memory for the rewrite. Updated as decisions land. If context was
compacted, this file plus `salvage/NOTES.md` is the whole picture.

Last updated: 2026-09-15

---

## What we're building

A background service. Press a hotkey → a small popup appears and recording starts
**immediately** (no warm-up). Stop → popup shows it's transcribing → transcribed
text appears for the user to copy and paste.

That is the entire product.

- Runs in the background as a service; hotkey-triggered.
- Recording starts the instant the popup shows.
- Small popup, clear visual cues per state (recording / transcribing / done).
- GPU-accelerated Whisper, CPU fallback.
- One-button install, no user-facing dependency installs.
- Explicitly NOT wanted: prompt library, snippets, projects, variables, semantic
  match, self-updater — the old app's entire feature set.

---

## Stack — SETTLED

**Python, single language, all the way.** No Rust-in-Python, no Node/Bun, no
split-language architecture.

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.14 via uv | uv bundles the interpreter *and* Tk 9.0 — no system Python |
| Inference | `onnxruntime` / `onnxruntime-gpu`, driven at the bare-graph level | learning goal + real runtime CPU fallback |
| Numerics | `numpy` | mel spectrogram, tensor glue |
| Audio | `sounddevice` | industry-standard PortAudio; more widely used than miniaudio |
| UI | `tkinter` (Tk 9.0) | zero install, bundled by uv, genuinely cross-platform, ~40ms startup |
| Model dir | `platformdirs` | correct per-OS location |
| Packaging | PyInstaller via GitHub Actions matrix | AppImage / .app / .exe, zero user deps |

### Why not the alternatives (don't re-litigate without new evidence)

- **Rust-in-Python (PyO3):** no CPU-bound Python hot loop exists to optimize — the
  hot loop is ORT (C++). Python overhead is <1% of decode time. Would *create*
  per-platform wheel-building pain rather than solve it.
- **Node/Bun:** no numpy (the mel spectrogram is the part we want to learn), much
  worse audio capture story, Electron for UI. Loses on the two axes that matter.
- **Split languages (Python engine + other shell):** ships two runtimes plus an IPC
  bridge and process-lifecycle management. Strictly more complex for a text box.
- **Whole thing in Rust:** cleanest single-binary install (~70MB vs ~200MB), and
  honestly acknowledged as such — but the app binary is noise against a 1–3GB
  model+CUDA download, and it costs the REPL-driven tensor debugging that is the
  whole point of going bare-metal.
- **miniaudio instead of sounddevice:** miniaudio needs no system lib at all, but
  once frozen, PyInstaller bundles PortAudio anyway, so sounddevice's only
  disadvantage disappears. sounddevice is more widely used and has an official hook.

---

## Key design rules

**1. The engine is a pure function.** numpy in, text out. No tkinter, no asyncio,
no daemon imports anywhere in it. This is what makes a future Kotlin/Android port a
rewrite of the shell rather than the brain — and it costs nothing now.

**2. Provider is a runtime probe, never an install-time assumption.** Quantization
follows from it.

```python
def pick_backend():
    avail = ort.get_available_providers()
    for ep in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"):
        if ep in avail:
            return [ep, "CPUExecutionProvider"], "fp16"
    return ["CPUExecutionProvider"], "int8"
```

**3. Models are never packaged.** 1–2GB, downloaded on first run into a
`platformdirs` location, sha256-verified before extraction. GPU runtime rides the
same first-run flow:

```
launch → "Downloading speech model (1.0 GB)" → detects NVIDIA GPU
       → "Enable GPU acceleration? (+1.6 GB)" → ready
```

Ship a modest CPU-only artifact that works for everyone; GPU is an in-app opt-in.
Better one-button UX than a 3GB installer.

**4. Start audio capture BEFORE mapping the popup window.** Capture open is ~20ms,
GTK/Tk window map is ~50–100ms. This is what makes "no delay" true.

**5. The daemon is mandatory, not a nicety.** CUDA EP session init takes 5–15s
cold. "Recording starts immediately" is only possible with the model warm in VRAM.

---

## The bare-ONNX interface (verified against the real model files)

```
encoder:  mel[n_audio,128,T]  →  cross_k[4,·,T,1280], cross_v[4,·,T,1280]
decoder:  tokens, in_self_k_cache[4,·,448,1280], in_self_v_cache,
          cross_k, cross_v, offset
       →  logits[·,·,51866], out_self_k_cache, out_self_v_cache
```

Why this export is unusually good for bare interfacing:

- **KV cache is explicitly exposed** (`in_/out_self_k_cache` + `offset`) → decode
  loop is linear-cost, one token per step. This was the biggest risk; it's resolved.
- **Encoder emits cross-attention K/V directly**, not raw features → encoder runs
  once per 30s window, decoder loop is a clean function of (tokens, caches).
- **All Whisper constants live in encoder metadata**: `sot=50258`, `eot=50257`,
  `transcribe=50360`, `no_timestamps=50364`, `n_mels=128`, `n_text_ctx=448`,
  `n_vocab=51866`, `sot_sequence=50258,50259,50360`, full language-token table.
  Read them off the file; don't hardcode.
- **Token table is base64 raw bytes** (`IQ==  0` → `!`). Detokenize = b64decode,
  concat, utf-8 decode. No GPT-2 byte-BPE dance. File has ids 0..50256; specials
  are ≥50257 and get filtered.
- Turbo shape: **32 encoder layers, only 4 decoder layers** — the looped part is tiny.

### Work remaining to implement it (~150–250 lines)

| Piece | Lines | Notes |
|---|---|---|
| Log-mel (hann 400, hop 160, 128 mels, log10, clamp, normalize) | ~50 | only real DSP; numpy STFT + mel filterbank |
| Encoder call | ~5 | trivial |
| Greedy decode loop w/ KV cache + offset | ~50 | the part worth learning |
| Prompt construction | ~20 | read from metadata |
| Detokenize | ~10 | base64 |
| 30s chunking | ~15 | see salvage/NOTES.md |

---

## Open questions

1. **Does int8-on-CUDA actually work?** ORT's CUDA EP handles `MatMulInteger` /
   `QuantizeLinear` poorly and may silently fall back to CPU per-node, making it
   *slower* than pure CPU. This is why the fp16/int8 split exists. **Measure it.**
2. **Does `onnxruntime-gpu` bundle CUDA/cuDNN or expect a system install?** It
   resolved as a single package with no `nvidia-*` deps, which is ambiguous.
   Determines whether GPU install is one command or "install CUDA toolkit first."
3. **Wayland global hotkey.** Tauri-style global grabs don't work. Leading
   candidate: GNOME custom keybinding running a tiny CLI that signals the daemon
   over a unix socket. Verify on this machine before building it.
4. **miniaudio has no PyInstaller hook; sounddevice does.** Moot given the choice
   above, but noted.

---

## Build/packaging notes

- Build machine (and CI image) needs `libportaudio2` — 78KB apt package. PyInstaller
  collects the system `.so` into the bundle, so **end users need nothing**.
  Confirmed by reading `hook-sounddevice.py`.
- **No in-app `apt install`.** Rejected: needs pkexec/polkit password dialog, is
  Debian-only, breaks on immutable distros/containers/sandboxes, fails for non-admin
  users, and contradicts the zero-install goal. The freeze already solves it.
- Build AppImage on an **older base** (Ubuntu 22.04 / manylinux) — bundling a system
  `.so` makes glibc compatibility across distros a real concern.
- macOS gets **CoreML EP inside the standard `onnxruntime` wheel** — Apple Silicon
  acceleration costs zero packaging effort.
- `onnxruntime-gpu` is Linux/Windows x86_64 only → belongs in an optional extra.

---

## Planned minimal scope (first milestone)

Nothing but the model working. No daemon, no hotkey, no systemd, no clipboard.

```
engine.py    mel → encoder → decode loop → text   (pure; numpy + ort only)
audio.py     sounddevice capture → 16kHz mono f32
models.py    download + sha256 verify + extract   (port of salvage script to Python)
app.py       tkinter box: [● recording] / [… transcribing] / [text + Copy]
```

Run from terminal, button to start/stop. Once the model provably works, the daemon,
hotkey, and freeze are mechanical additions.

---

## Already-settled model facts (from the old app, don't re-litigate)

- **Whisper large-v3-turbo, not SenseVoice-Small.** SenseVoice mangled English
  technical jargon ("GitHub" → "GET UP", "Kubernetes" → "CORNATTIE ENGINES").
  Tested on real voice.
- **One decode per utterance, no live partials.** Redecoding a growing buffer is
  quadratic-cost. Cut deliberately.
- **30-second chunking is mandatory** for utterances over 30s.
- **Whisper auto-detect language = empty string**, not `"auto"`.
- **Models already on disk** at `~/.prompt-compose/models/sherpa-onnx-whisper-turbo/`
  (int8, ~1.0GB). Reuse for CPU testing; fp16 still needs fetching for GPU.

---

## Machine facts

- Ubuntu, Wayland, GNOME.
- NVIDIA RTX 3090 Ti 24GB, driver 595.91.07, CUDA 13.2, no system CUDA toolkit.
- Python 3.14.6 via uv (Tk 9.0 bundled); system Python 3.14.4 (Tk 8.6, GTK4 OK).
- `uv`, `wl-copy`, `notify-send`, `zenity`, `pw-cat`, `pactl` available.
- Verified working: `onnxruntime` 1.30.0, `numpy` 2.3.5, `miniaudio` 1.71,
  `tkinter`/Tk 9.0. `onnxruntime-gpu` 1.30.0 resolves for cp314.
- Mic is likely `USB Audio Device Mono`. NOTE: most enumerated capture devices are
  `Monitor of ...` PulseAudio **loopback** sources (system output, not mics) —
  filter or deprioritize them in any device picker, or the default could silently
  record the speakers.
- Old app data in `~/.prompt-compose/` (1.3GB — two models + a dead embedding cache).

---

## Repo state

- Old Tauri+SvelteKit app still present, **not yet deleted**. Recoverable at git tag
  `pre-rewrite-v0.3.3`.
- `salvage/` holds the keepers: `download-whisper-model.sh` (working, verified),
  `reference/{engine,audio,session-state}.rs`, and `NOTES.md`.
- Deletion of the old app proposed but **not yet approved** — awaiting go-ahead.
