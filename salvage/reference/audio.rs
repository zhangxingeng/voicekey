//! Microphone enumeration and live capture: `cpal` reads whatever native rate
//! the device offers, downmixes to mono, and `sherpa_onnx::LinearResampler`
//! (already a dependency of `engine.rs` — no separate resampling crate is
//! needed) converts it to the 16kHz mono f32 Whisper expects. The result
//! lands in a shared buffer that the session (`dictate::state`) reads once,
//! in full, when Space is released — capture just appends for as long as
//! Space is held.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use sherpa_onnx::LinearResampler;

/// One selectable input device. `id` round-trips through `cpal::DeviceId`'s
/// `Display`/`FromStr` — it is what the frontend sends back in
/// `start_dictation`'s `device_id`.
#[derive(Debug, Clone, Serialize)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
}

pub fn list_input_devices() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let devices = host.input_devices().map_err(|e| e.to_string())?;
    Ok(devices
        .filter_map(|d| {
            let id = d.id().ok()?.to_string();
            let name = d.description().map(|desc| desc.name().to_string()).unwrap_or_else(|_| d.to_string());
            Some(AudioDevice { id, name })
        })
        .collect())
}

fn resolve_device(host: &cpal::Host, device_id: Option<&str>) -> Result<cpal::Device, String> {
    match device_id {
        Some(id) => {
            let parsed: cpal::DeviceId = id.parse().map_err(|e: cpal::Error| e.to_string())?;
            host.device_by_id(&parsed).ok_or_else(|| format!("input device not found: {id}"))
        }
        None => host.default_input_device().ok_or_else(|| "no input device available".to_string()),
    }
}

/// The 16kHz mono f32 buffer capture appends to and the recognition loop reads
/// (and later truncates, once a stretch has been committed).
pub type SampleBuffer = Arc<Mutex<Vec<f32>>>;

/// A running capture. Holding this alive keeps the `cpal::Stream` playing;
/// dropping it (`stop`) tears the stream down. `cpal::Stream` is `Send + Sync`
/// on every backend (cpal 0.18 asserts this at the definition site of each
/// platform's `Stream`), so a `Capture` can be built on one thread and handed
/// off to the dedicated session thread that runs the decode loop.
pub struct Capture {
    stream: cpal::Stream,
    pub buffer: SampleBuffer,
}

impl Capture {
    pub fn stop(self) {
        drop(self.stream); // Stream's Drop tears down the platform stream.
    }
}

/// Downmix one interleaved frame block to mono by averaging channels.
fn downmix(data: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    let channels = channels as usize;
    data.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect()
}

/// Start capturing from `device_id` (or the system default when `None`).
/// Every callback: downmix to mono, resample to 16kHz, append to `buffer`.
pub fn start_capture(device_id: Option<&str>) -> Result<Capture, String> {
    let host = cpal::default_host();
    let device = resolve_device(&host, device_id)?;
    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let channels = config.channels();
    let native_rate = config.sample_rate();
    let sample_format = config.sample_format();

    let buffer: SampleBuffer = Arc::new(Mutex::new(Vec::new()));
    let buffer_cb = buffer.clone();
    let resampler = LinearResampler::create(native_rate as i32, 16_000)
        .ok_or("could not create resampler for this device's sample rate")?;

    let err_fn = |e: cpal::Error| eprintln!("[dictate] audio stream error: {e}");

    let push = move |mono: Vec<f32>| {
        let resampled = resampler.resample(&mono, false);
        if resampled.is_empty() {
            return;
        }
        if let Ok(mut buf) = buffer_cb.lock() {
            buf.extend_from_slice(&resampled);
        }
    };

    let stream_config: cpal::StreamConfig = config.into();
    let stream = match sample_format {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| push(downmix(data, channels)),
                err_fn,
                None,
            )
            .map_err(|e| e.to_string())?,
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let floats: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    push(downmix(&floats, channels))
                },
                err_fn,
                None,
            )
            .map_err(|e| e.to_string())?,
        other => return Err(format!("unsupported input sample format: {other}")),
    };

    stream.play().map_err(|e| e.to_string())?;
    Ok(Capture { stream, buffer })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_interleaved_channels() {
        // Stereo: (L, R) pairs average to one mono sample per frame.
        let stereo = vec![1.0, 3.0, 0.0, 0.0, -1.0, 1.0];
        assert_eq!(downmix(&stereo, 2), vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn downmix_is_a_no_op_for_mono() {
        let mono = vec![0.1, 0.2, 0.3];
        assert_eq!(downmix(&mono, 1), mono);
    }
}
