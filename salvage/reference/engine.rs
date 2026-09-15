//! Wraps `sherpa_onnx::OfflineRecognizer` loaded with the downloaded Whisper
//! large-v3-turbo model.
//!
//! Dictation is one-shot: capture starts on Space-down, and the whole
//! buffer is decoded exactly once on Space-up (`dictate::state`). There is no
//! live "partial" feed anymore — an earlier version redecoded the entire
//! growing buffer every 800ms for interim results, but that cost is
//! quadratic in utterance length (a 10-minute utterance would burn on the
//! order of three hours of cumulative CPU time, and individual redecodes
//! would themselves take tens of seconds well before the 10-minute mark) and
//! is not how genuine streaming ASR works. No accurate genuine-streaming
//! (persistent-state, linear-cost) bilingual model was found to replace it
//! with, so live partials were cut entirely rather than shipped as a patch on
//! top of that cost — one accurate decode per utterance instead.
//!
//! sherpa-onnx's Whisper decoder hard-caps a single `decode_stream` call at
//! 30 seconds of audio (it silently discards anything beyond that — this is
//! the ONNX runtime's own limit, not a Whisper model limit: OpenAI's own CLI
//! and other Whisper runtimes handle longer audio by chunking internally).
//! `decode_long` does the same: split into sequential, non-overlapping
//! 30-second windows, decode each once, join the text. This is the standard
//! way every real Whisper deployment handles long-form audio — not a
//! sliding-window replay — so total cost stays linear in utterance length. A
//! word that happens to land exactly on a chunk boundary may come out
//! slightly mangled; every Whisper-based long-form transcription tool makes
//! this same trade.

use std::path::Path;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig};

/// sherpa-onnx's Whisper decoder silently truncates a single call to this
/// many seconds of audio — `decode_long` chunks around that limit.
const MAX_CHUNK_SECONDS: usize = 30;
const SAMPLE_RATE: usize = 16_000;

/// `OfflineRecognizer` is already `Send + Sync` (the sherpa-onnx crate treats
/// the underlying C library as thread-safe for single-object use), so `Engine`
/// inherits both automatically — no unsafe impl needed here.
pub struct Engine {
    recognizer: OfflineRecognizer,
}

impl Engine {
    /// Load the recognizer from the on-disk encoder/decoder/tokens files.
    /// `language` is the app's `"auto" | "en" | "zh"` — Whisper's own
    /// auto-detection is triggered by an empty string, not the word "auto".
    pub fn load(
        encoder_path: &Path,
        decoder_path: &Path,
        tokens_path: &Path,
        language: &str,
    ) -> Result<Self, String> {
        let whisper_language = if language == "auto" { "" } else { language };

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(encoder_path.to_string_lossy().into_owned()),
            decoder: Some(decoder_path.to_string_lossy().into_owned()),
            language: Some(whisper_language.to_string()),
            task: Some("transcribe".to_string()),
            ..Default::default()
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().into_owned());
        config.model_config.provider = Some("cpu".to_string());
        config.model_config.num_threads = 2;

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or("failed to create the Whisper recognizer — check the model files")?;
        Ok(Self { recognizer })
    }

    /// Decode up to `MAX_CHUNK_SECONDS` of audio (16kHz mono f32) as one
    /// utterance and return the transcribed text. Longer input is silently
    /// truncated by sherpa-onnx itself — callers with potentially longer
    /// audio should use `decode_long` instead.
    fn decode(&self, samples: &[f32]) -> Result<String, String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE as i32, samples);
        self.recognizer.decode(&stream);
        Ok(stream.get_result().map(|r| r.text).unwrap_or_default())
    }

    /// Decode audio of any length by splitting it into sequential,
    /// non-overlapping `MAX_CHUNK_SECONDS` windows and joining the text —
    /// see the module docs for why this (not a sliding-window replay) is the
    /// standard, linear-cost way to run Whisper over long-form audio.
    pub fn decode_long(&self, samples: &[f32]) -> Result<String, String> {
        let chunk_len = MAX_CHUNK_SECONDS * SAMPLE_RATE;
        let mut parts = Vec::new();
        for chunk in samples.chunks(chunk_len) {
            let text = self.decode(chunk)?;
            let text = text.trim();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
        Ok(parts.join(" "))
    }
}
