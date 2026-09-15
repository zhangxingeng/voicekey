"""Whisper large-v3-turbo driven directly through ONNX Runtime.

There is no wrapper library between this and the model. The exported graphs
make that practical:

    encoder:  mel[1,128,3000] -> cross_k[4,1,1500,1280], cross_v[...]
    decoder:  tokens, self_k_cache, self_v_cache, cross_k, cross_v, offset
           -> logits[1,n,51866], self_k_cache', self_v_cache'

Two properties are what make a hand-written loop reasonable rather than
masochistic:

* **The KV cache is explicit.** `in_/out_n_layer_self_*_cache` plus `offset`
  means each step decodes exactly one new token against the accumulated state,
  so generation is linear in output length. Without this you would re-run the
  full decoder per token and pay O(n^2).
* **The encoder emits cross-attention K/V directly**, not raw audio features.
  It runs once per 30s window; the decode loop is then a pure function of
  (tokens, caches).

Every Whisper constant -- token ids, layer counts, the language table -- is
read from the encoder's ONNX metadata rather than hardcoded, so this stays
correct if the model is re-exported.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from voicekey import backend as backend_mod
from voicekey import cuda
from voicekey import mel as mel_mod
from voicekey.backend import Backend


def _load_tokens(path: Path) -> dict[int, bytes]:
    """`<base64> <id>` per line; the payload is raw UTF-8 bytes.

    Notably *not* GPT-2 byte-level BPE in its escaped form -- sherpa's export
    base64s the real bytes, so detokenizing is decode-and-concatenate.
    """
    table: dict[int, bytes] = {}
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        payload, _, ident = line.rpartition(b" ")
        try:
            table[int(ident)] = base64.b64decode(payload)
        except Exception:
            # One degenerate row exists near the special-token boundary; it is
            # never emitted, so an empty mapping is the right outcome.
            continue
    return table


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str


class Whisper:
    """Loaded model. Construct once and keep it -- session init is the slow part."""

    def __init__(self, model_dir: Path, backend: Backend, *, num_threads: int = 0) -> None:
        if backend.accelerated:
            cuda.preload()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if num_threads:
            opts.intra_op_num_threads = num_threads

        suffix = ".int8" if backend.quantization == "int8" else ""
        self.encoder = ort.InferenceSession(
            str(model_dir / f"turbo-encoder{suffix}.onnx"), opts, providers=backend.providers
        )
        self.decoder = ort.InferenceSession(
            str(model_dir / f"turbo-decoder{suffix}.onnx"), opts, providers=backend.providers
        )
        # What the session *actually* bound, which is not necessarily what was
        # requested: ORT falls back to CPU silently when a provider's native
        # library fails to load. This is the value the UI should show.
        self.backend = backend_mod.confirm(self.encoder.get_providers())

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.sot = int(meta["sot"])
        self.eot = int(meta["eot"])
        self.transcribe_tok = int(meta["transcribe"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.no_speech = int(meta["no_speech"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_state = int(meta["n_text_state"])
        self.n_mels = int(meta["n_mels"])

        lang_tokens = [int(t) for t in meta["all_language_tokens"].split(",")]
        lang_codes = meta["all_language_codes"].split(",")
        self.lang_token = dict(zip(lang_codes, lang_tokens, strict=True))
        self.token_lang = dict(zip(lang_tokens, lang_codes, strict=True))

        self._tokens = _load_tokens(model_dir / "turbo-tokens.txt")
        self._filters = mel_mod.mel_filterbank(self.n_mels)

    # -- internals ---------------------------------------------------------

    def _empty_cache(self) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.n_text_layer, 1, self.n_text_ctx, self.n_text_state)
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def _step(
        self,
        tokens: list[int],
        k: np.ndarray,
        v: np.ndarray,
        cross_k: np.ndarray,
        cross_v: np.ndarray,
        offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, k_out, v_out = self.decoder.run(
            None,
            {
                "tokens": np.array([tokens], dtype=np.int64),
                "in_n_layer_self_k_cache": k,
                "in_n_layer_self_v_cache": v,
                "n_layer_cross_k": cross_k,
                "n_layer_cross_v": cross_v,
                "offset": np.array([offset], dtype=np.int64),
            },
        )
        return logits, k_out, v_out

    def _detect_language(self, cross_k: np.ndarray, cross_v: np.ndarray) -> int:
        """One decoder step from a bare <|startoftranscript|>, argmax over the
        language tokens only. Caches are thrown away afterwards."""
        k, v = self._empty_cache()
        logits, _, _ = self._step([self.sot], k, v, cross_k, cross_v, 0)
        row = logits[0, -1]
        candidates = np.array(list(self.token_lang.keys()))
        return int(candidates[np.argmax(row[candidates])])

    def _decode_window(self, samples: np.ndarray, language: str | None) -> tuple[list[int], str]:
        audio = mel_mod.pad_or_trim(samples)
        features = mel_mod.log_mel(audio, self.n_mels, self._filters)[None, :, :]

        cross_k, cross_v = self.encoder.run(None, {"mel": features})

        if language is None:
            lang_tok = self._detect_language(cross_k, cross_v)
        else:
            lang_tok = self.lang_token[language]

        prompt = [self.sot, lang_tok, self.transcribe_tok, self.no_timestamps]
        k, v = self._empty_cache()
        logits, k, v = self._step(prompt, k, v, cross_k, cross_v, 0)
        offset = len(prompt)

        out: list[int] = []
        while offset < self.n_text_ctx:
            row = logits[0, -1].copy()
            # Never emit a special token mid-stream; <|endoftext|> is handled
            # by the break below, so suppressing the rest keeps stray control
            # tokens out of the text.
            row[self.eot + 1 :] = -np.inf
            row[self.no_speech] = -np.inf

            nxt = int(np.argmax(row))
            if nxt == self.eot:
                break
            out.append(nxt)
            logits, k, v = self._step([nxt], k, v, cross_k, cross_v, offset)
            offset += 1

        return out, self.token_lang[lang_tok]

    def detokenize(self, ids: list[int]) -> str:
        return b"".join(self._tokens.get(i, b"") for i in ids).decode("utf-8", errors="replace")

    # -- public ------------------------------------------------------------

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Transcription:
        """16kHz mono f32 of any length -> text.

        Audio longer than 30s is split into sequential, non-overlapping windows
        and decoded once each. This is how every real Whisper deployment does
        long-form: the decoder silently truncates a single call at 30s, and a
        sliding-window replay would make cost quadratic. A word landing exactly
        on a boundary can come out mangled; that is the standard trade.
        """
        samples = np.asarray(samples, dtype=np.float32)
        parts: list[str] = []
        detected = language or "en"

        for start in range(0, max(len(samples), 1), mel_mod.N_SAMPLES):
            window = samples[start : start + mel_mod.N_SAMPLES]
            if len(window) == 0:
                break
            ids, detected = self._decode_window(window, language)
            text = self.detokenize(ids).strip()
            if text:
                parts.append(text)

        return Transcription(text=" ".join(parts), language=detected)
