# Salvage — what survived from the old Tauri/Rust app

The old app (4,900 lines of prompt library, semantic match, SQLite embedding
cache, self-updater, project tabs, variable grammar) is gone. Full source is
recoverable at git tag `pre-rewrite-v0.3.3`.

Almost everything worth keeping has since been ported to Python and folded into
[`STATE.md`](../STATE.md), which is the live document. The Rust reference files
that lived here were deleted once `engine.py`, `audio.py` and `session.py`
superseded them.

## What is still here

`download-whisper-model.sh` — the original fetch + verify + extract, kept only
until `models.py` has been proven against the real 540MB download. It is the
last independent copy of the pinned URL and sha256. **Delete it once `models.py`
has fetched the real model successfully at least once.**

## Model artifacts

Three files, ~1.0GB extracted:

| file | size |
|---|---|
| `turbo-encoder.int8.onnx` | 675MB |
| `turbo-decoder.int8.onnx` | 361MB |
| `turbo-tokens.txt` | 817KB |

Check for presence, never by re-hashing 1GB on every launch.
