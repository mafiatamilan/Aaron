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
pub use capture::{capture_available, default_device_name, input_devices, InputDeviceInfo};

/// Which host the capture goes through, or `"none"` when capture was not
/// compiled in.
///
/// `aaron doctor` has to be able to answer this in a text-only build, so the
/// answer has to exist whether or not the capture backend does. Reporting
/// `"alsa"` when there is no ALSA would be a lie, so the honest answer is
/// `"none"`.
#[cfg(feature = "capture")]
pub fn capture_host_name() -> &'static str {
    capture::capture_host_name()
}

/// See the [`capture_host_name`] documentation.
#[cfg(not(feature = "capture"))]
pub fn capture_host_name() -> &'static str {
    "none"
}

/// Whether a PipeWire server is running for this user.
///
/// This checks for the server's socket in the runtime directory, so it needs
/// no audio library at all — which is why it is available in a text-only
/// build. The doctor uses it to explain the ALSA-playback-endpoint trap.
pub fn pipewire_is_running() -> bool {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        return std::path::Path::new(&dir).join("pipewire-0").exists();
    }
    false
}

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
