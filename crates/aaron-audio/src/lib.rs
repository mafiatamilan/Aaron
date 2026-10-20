//! Microphone capture, voice activity detection and audio sources.
//!
//! This crate owns everything from "open the device" to "here is a trimmed
//! mono 16 kHz buffer". It never interprets what was said; that is
//! `aaron-asr`'s job.
//!
//! Real capture is behind the `capture` feature. Without it — in CI, in a
//! container, or on a `--text-only` install — the crate still provides the
//! synthetic and file-backed sources, so the rest of AARON builds and tests
//! without ALSA.

#[cfg(feature = "capture")]
pub mod capture;
pub mod source;
pub mod vad;

pub use source::{
    measure_level, record_utterance, CaptureSpec, FileSource, LevelReport, NullSource,
    RecordedUtterance, SyntheticSource,
};
pub use vad::{crop, VadOutcome, VoiceActivityDetector};

#[cfg(feature = "capture")]
pub use capture::{
    capture_available, capture_host_name, default_device_name, input_devices, pipewire_is_running,
    InputDeviceInfo,
};

/// One entry describing an available capture device.
pub fn describe_devices() -> Vec<String> {
    #[cfg(feature = "capture")]
    {
        capture::input_devices()
            .into_iter()
            .map(|d| {
                format!(
                    "{}{} — {} Hz",
                    d.name,
                    if d.supports_input { "" } else { " (no input)" },
                    d.default_input_sample_rate
                        .map(|r| r.to_string())
                        .unwrap_or_else(|| "unknown rate".into())
                )
            })
            .collect()
    }
    #[cfg(not(feature = "capture"))]
    {
        vec!["audio capture was not compiled in (build with --features capture)".into()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_description_never_panics() {
        let devices = describe_devices();
        assert!(!devices.is_empty());
    }
}
