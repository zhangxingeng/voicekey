//! Tauri commands for dictation — `list_audio_devices` / `start_dictation` /
//! `stop_dictation`, all async, `Result<T, String>`, snake_case, the same
//! conventions as `prompts::state`.
//!
//! Unlike the Prompt Library's semantic match (`prompts/embed.rs`), which
//! downloads and degrades silently because it is never a prerequisite for
//! anything, dictation is an explicit user action: the user clicked the mic
//! and is waiting on it. So every failure here — no mic permission, no input
//! devices, a failed model download — is a returned `Err` the frontend turns
//! into a toast, never a silent degradation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use super::audio::AudioDevice;
use super::{audio, engine, model};

pub struct DictateState {
    inner: Arc<DictateInner>,
}

/// How often the session loop wakes up just to check `stop_requested` — no
/// decoding happens on this cadence anymore, so this is a cheap idle poll,
/// not a cost driver.
const STOP_POLL_INTERVAL_MS: u64 = 100;

struct DictateInner {
    /// Set for the lifetime of one capture+decode session — guards against a
    /// second `start_dictation` racing the first.
    running: AtomicBool,
    /// Polled by the session loop every `STOP_POLL_INTERVAL_MS`; set by
    /// `stop_dictation` to end the session and trigger the one decode.
    stop_requested: AtomicBool,
    /// Set for the lifetime of one `download_dictate_model` call — guards
    /// against a second click racing the first, same discipline as `running`.
    downloading: AtomicBool,
}

impl DictateState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DictateInner {
                running: AtomicBool::new(false),
                stop_requested: AtomicBool::new(false),
                downloading: AtomicBool::new(false),
            }),
        }
    }
}

impl Default for DictateState {
    fn default() -> Self {
        Self::new()
    }
}

fn root() -> Result<PathBuf, String> {
    crate::datadir::data_root()
}

#[derive(Serialize, Clone)]
struct TextPayload {
    text: String,
}

#[derive(Serialize, Clone)]
struct ProgressPayload {
    fraction: f32,
}

/// The exact error `start_dictation` returns when the model hasn't been
/// downloaded yet. The frontend matches on this string to show its "download
/// the model in Settings first" message instead of a generic failure toast —
/// keep the two in sync (`src/lib/dictate.svelte.ts`).
pub const MODEL_NOT_DOWNLOADED: &str = "MODEL_NOT_DOWNLOADED";

#[tauri::command]
pub async fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    tauri::async_runtime::spawn_blocking(audio::list_input_devices).await.map_err(|e| e.to_string())?
}

/// Is the Whisper large-v3-turbo model already on disk? Settings uses this to
/// decide whether to show "Download" or "Ready", and the compose box uses it
/// (via the frontend's cached copy) to decide whether Space is allowed to
/// start a session at all.
#[tauri::command]
pub async fn dictate_model_status() -> Result<bool, String> {
    tauri::async_runtime::spawn_blocking(|| Ok(model::artifacts_present(&root()?)))
        .await
        .map_err(|e| e.to_string())?
}

/// Download the Whisper large-v3-turbo model, reporting progress via
/// `dictate:model-progress` (`{ fraction: 0.0..=1.0 }`). Explicitly
/// user-triggered from Settings — `start_dictation` never downloads on its
/// own, so holding Space never looks like it's doing nothing while ~540MB
/// fetches silently in the background.
#[tauri::command]
pub async fn download_dictate_model(
    app: AppHandle,
    state: tauri::State<'_, DictateState>,
) -> Result<(), String> {
    let inner = state.inner.clone();
    if inner.downloading.swap(true, Ordering::SeqCst) {
        return Err("the model is already downloading".to_string());
    }
    let result = tauri::async_runtime::spawn_blocking(move || {
        let root = root()?;
        model::download_artifacts(&root, |fraction| {
            let _ = app.emit("dictate:model-progress", ProgressPayload { fraction });
        })
    })
    .await
    .map_err(|e| e.to_string())?;
    inner.downloading.store(false, Ordering::SeqCst);
    result
}

/// Start one capture+decode session. Requires the model to already be on
/// disk — downloading it is Settings' job now, not this command's, since the
/// user just took an explicit action (held Space) and would otherwise wonder
/// why nothing happens for as long as a ~540MB download takes. Blocks only
/// long enough to load the recognizer and open the input device — everything
/// that can fail synchronously — then hands the running session off to a
/// dedicated background thread and returns. `stop_dictation` ends it.
///
/// `Engine` and `audio::Capture` are both `Send` (the sherpa-onnx recognizer
/// and cpal's `Stream` are each `Send + Sync` on every backend), so setup can
/// run on one blocking-pool thread and, once it succeeds, hand both off to a
/// second thread that owns the session — no channel needed to bridge them.
#[tauri::command]
pub async fn start_dictation(
    app: AppHandle,
    state: tauri::State<'_, DictateState>,
    device_id: Option<String>,
    language: String,
) -> Result<(), String> {
    let inner = state.inner.clone();
    if inner.running.swap(true, Ordering::SeqCst) {
        return Err("dictation is already running".to_string());
    }
    inner.stop_requested.store(false, Ordering::SeqCst);

    let setup: Result<(engine::Engine, audio::Capture), String> =
        tauri::async_runtime::spawn_blocking(move || {
            let root = root()?;
            if !model::artifacts_present(&root) {
                return Err(MODEL_NOT_DOWNLOADED.to_string());
            }
            let engine = engine::Engine::load(
                &model::encoder_path(&root),
                &model::decoder_path(&root),
                &model::tokens_path(&root),
                &language,
            )?;
            let capture = audio::start_capture(device_id.as_deref())?;
            Ok((engine, capture))
        })
        .await
        .map_err(|e| e.to_string())?;

    match setup {
        Ok((engine, capture)) => {
            let session_inner = inner.clone();
            // Fire-and-forget: this JoinHandle is intentionally dropped
            // unawaited. The session runs until `stop_dictation` sets
            // `stop_requested`; awaiting it here would block the command
            // until the user stops dictating.
            tauri::async_runtime::spawn_blocking(move || run_session(app, session_inner, engine, capture));
            Ok(())
        }
        Err(e) => {
            inner.running.store(false, Ordering::SeqCst);
            Err(e)
        }
    }
}

/// Signal the running session to stop capturing and start (the one) decode.
/// Resolves immediately — the decode itself happens on the session's
/// background thread and its result arrives later via `dictate:final`, with
/// `dictate:done` always following once decoding finishes (empty result or
/// not) so the frontend knows to stop showing a "transcribing…" state.
#[tauri::command]
pub async fn stop_dictation(state: tauri::State<'_, DictateState>) -> Result<(), String> {
    state.inner.stop_requested.store(true, Ordering::SeqCst);
    Ok(())
}

/// One-shot capture+decode: wait for `stop_requested` (checked every
/// `STOP_POLL_INTERVAL_MS` — a cheap idle poll, no decoding happens on this
/// cadence), then decode the entire captured buffer exactly once and emit it
/// as `dictate:final`. `dictate:done` always fires after, whether or not
/// there was any text, so the frontend can clear its "transcribing" state.
fn run_session(app: AppHandle, inner: Arc<DictateInner>, engine: engine::Engine, capture: audio::Capture) {
    loop {
        std::thread::sleep(Duration::from_millis(STOP_POLL_INTERVAL_MS));
        if inner.stop_requested.load(Ordering::SeqCst) {
            break;
        }
    }

    let samples = capture.buffer.lock().ok().map(|buf| buf.clone()).unwrap_or_default();
    capture.stop();

    if !samples.is_empty() {
        match engine.decode_long(&samples) {
            Ok(text) if !text.trim().is_empty() => {
                let _ = app.emit("dictate:final", TextPayload { text });
            }
            Ok(_) => {}
            Err(e) => eprintln!("[dictate] decode failed: {e}"),
        }
    }
    let _ = app.emit("dictate:done", ());

    inner.running.store(false, Ordering::SeqCst);
}
