#!/usr/bin/env bash
# Download + verify + extract the Whisper large-v3-turbo ONNX model used for
# dictation. Stack-independent: plain shell, no Rust/Node/Python needed.
#
# This replaces the ~180 lines of Rust in the old src-tauri/src/dictate/model.rs.
# Whatever stack we land on, the model is just three files on disk; fetching
# them does not need to live inside the app.
#
#   usage: ./download-whisper-model.sh [dest-dir]
#   default dest: ${VOICE_MODEL_DIR:-$HOME/.local/share/voicekey/models}

set -euo pipefail

DEST="${1:-${VOICE_MODEL_DIR:-$HOME/.local/share/voicekey/models}}/sherpa-onnx-whisper-turbo"

# Pinned k2-fsa release asset. int8-quantised large-v3-turbo, ~540MB archive,
# ~1.0GB extracted. Bilingual EN/ZH and accurate on technical English —
# SenseVoice-Small (3x smaller, 5x faster) was tried first and rejected:
# it rendered "GitHub" as "GET UP" and "Kubernetes" as "CORNATTIE ENGINES".
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2"
SHA256="b11acbbcd660b44a8e0df33724feb5aaa709cf65668f2823d59f656312544f22"
PREFIX="sherpa-onnx-whisper-turbo"

# The only three members we need. The archive also carries a test_wavs/ dir.
MEMBERS=(turbo-encoder.int8.onnx turbo-decoder.int8.onnx turbo-tokens.txt)

have_all=1
for m in "${MEMBERS[@]}"; do [ -f "$DEST/$m" ] || have_all=0; done
if [ "$have_all" = 1 ]; then
  echo "already present: $DEST"
  exit 0
fi

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "downloading ~540MB -> $TMP/model.tar.bz2"
curl -fL --progress-bar -o "$TMP/model.tar.bz2" "$URL"

echo "verifying sha256"
echo "$SHA256  $TMP/model.tar.bz2" | sha256sum -c - || {
  echo "CHECKSUM MISMATCH — discarding" >&2
  exit 1
}

echo "extracting"
tar -xjf "$TMP/model.tar.bz2" -C "$TMP" \
  "${MEMBERS[@]/#/$PREFIX/}"
for m in "${MEMBERS[@]}"; do
  mv "$TMP/$PREFIX/$m" "$DEST/$m"
done

echo "ready: $DEST"
ls -lh "$DEST"
