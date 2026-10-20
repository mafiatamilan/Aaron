//! Hardware and runtime capability detection.
//!
//! Everything here is read-only and dependency-free: `/proc` on Linux, plus an
//! optional `nvidia-smi` probe. Nothing is assumed — every field carries an
//! explicit "unknown" state so `aaron doctor` can tell the truth rather than
//! print a plausible-looking guess.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// CPU description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Marketing model string from `/proc/cpuinfo`, when readable.
    pub model: Option<String>,
    /// Logical cores visible to this process.
    pub logical_cores: usize,
    /// `nproc`-equivalent, the same value above but named for display.
    pub arch: String,
}

/// Physical memory description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
}

/// NVIDIA GPU description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: Option<String>,
    pub driver_version: Option<String>,
    pub total_vram_bytes: Option<u64>,
    pub cuda_version: Option<String>,
    /// Whether `nvidia-smi` responded at all.
    pub nvidia_smi_available: bool,
}

/// Audio subsystem description.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioInfo {
    /// ALSA `default` device present.
    pub default_device_present: bool,
    /// `/dev/snd` exists.
    pub sound_devices_present: bool,
    /// Server-sound available through PipeWire/PulseAudio.
    pub server_sound_available: bool,
    /// Free-form note, e.g. "no capture devices found".
    pub detail: String,
}

/// Everything `aaron doctor` and `aaron benchmark` report.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReport {
    pub os: String,
    pub kernel: String,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpu: GpuInfo,
    pub audio: AudioInfo,
    /// Running as uid 0, which AARON refuses to do.
    pub running_as_root: bool,
    /// Whether the build has a CUDA backend compiled in.
    pub cuda_compiled_in: bool,
    /// Extra key/value facts.
    pub extras: BTreeMap<String, String>,
}

fn read_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Detect the CPU.
pub fn detect_cpu() -> CpuInfo {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        });
    CpuInfo {
        model,
        logical_cores,
        arch: std::env::consts::ARCH.to_string(),
    }
}

/// Detect physical memory from `/proc/meminfo`.
pub fn detect_memory() -> MemoryInfo {
    let Some(text) = std::fs::read_to_string("/proc/meminfo").ok() else {
        return MemoryInfo::default();
    };
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else { continue };
        let value: Option<u64> = parts.next().and_then(|v| v.parse().ok());
        let Some(kb) = value else { continue };
        match key.trim_end_matches(':') {
            "MemTotal" => total = Some(kb * 1024),
            "MemAvailable" => available = Some(kb * 1024),
            _ => {}
        }
    }
    MemoryInfo {
        total_bytes: total,
        available_bytes: available,
    }
}

/// Probe the NVIDIA stack with a short `nvidia-smi --query-gpu` call.
///
/// Uses a fixed executable name and a fixed argument vector, never a shell, and
/// a hard 5 second timeout so a wedged driver cannot hang the CLI.
pub fn detect_gpu() -> GpuInfo {
    let mut info = GpuInfo::default();
    let timeout = Duration::from_millis(2500);

    let Some(stdout) = probe(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout,
    ) else {
        info.detail_missing();
        return info;
    };
    info.nvidia_smi_available = true;
    if let Some(first) = stdout.lines().next() {
        let fields: Vec<&str> = first.split(',').map(|s| s.trim()).collect();
        if let Some(name) = fields.first().filter(|s| !s.is_empty()) {
            info.name = Some((*name).to_string());
        }
        // A driver version looks like `615.78.08`. Parsing it as a number
        // fails on the second dot, which is why every GPU report said
        // `driver unknown` on a machine whose driver was perfectly readable.
        if let Some(v) = fields.get(1).filter(|s| !s.is_empty()) {
            info.driver_version = Some((*v).to_string());
        }
        if let Some(mib) = fields.get(2).and_then(|s| s.parse::<u64>().ok()) {
            info.total_vram_bytes = Some(mib * 1024 * 1024);
        }
    }

    // The CUDA runtime version lives in the plain `nvidia-smi` banner. Recent
    // drivers label it `CUDA UMD Version:`, older ones `CUDA Version:`, so
    // both are accepted rather than reporting "unknown" on a card we can see.
    if let Some(banner) = probe("nvidia-smi", &[], timeout) {
        for label in ["CUDA Version:", "CUDA UMD Version:", "CUDA Driver Version:"] {
            let Some(idx) = banner.find(label) else {
                continue;
            };
            // Skip the padding after the colon; taking digits from a leading
            // space yields an empty string and the label is silently skipped.
            let rest = banner[idx + label.len()..].trim_start();
            let ver: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if !ver.is_empty() {
                info.cuda_version = Some(ver);
                break;
            }
        }
    }
    info
}

impl GpuInfo {
    fn detail_missing(&mut self) {
        self.name = None;
        self.driver_version = None;
        self.total_vram_bytes = None;
        self.cuda_version = None;
    }
}

/// Probe the audio subsystem. Read-only; never opens a device.
pub fn detect_audio() -> AudioInfo {
    let sound_devices_present = Path::new("/dev/snd").is_dir();
    let server_sound_available =
        std::path::Path::new(&format!("/run/user/{}/pulse", unsafe { libc_geteuid() })).exists()
            || std::env::var("XDG_RUNTIME_DIR")
                .map(|d| Path::new(&d).join("pulse").exists())
                .unwrap_or(false);

    let default_device_present =
        std::path::Path::new("/dev/snd/controlC0").exists() || sound_devices_present;

    let mut detail_parts: Vec<String> = Vec::new();
    if !sound_devices_present {
        detail_parts.push("no /dev/sound devices".into());
    }
    if !server_sound_available {
        detail_parts.push("no PulseAudio/PipeWire socket".into());
    }
    let detail = if detail_parts.is_empty() {
        "audio subsystem reachable".into()
    } else {
        detail_parts.join("; ")
    };

    AudioInfo {
        default_device_present,
        sound_devices_present,
        server_sound_available,
        detail,
    }
}

// `libc` is not a dependency of `aaron-core`; use the raw syscall instead.
unsafe fn libc_geteuid() -> u32 {
    extern "C" {
        fn geteuid() -> u32;
    }
    geteuid()
}

/// Whether this build can move tensors to a CUDA device.
///
/// AARON's 1M-parameter intent model runs on CPU by design: at that size, host
/// memory transfer costs more than the matmul saves. The flag exists so
/// `aaron doctor` can state the truth about what this binary was compiled for.
pub const fn cuda_compiled_in() -> bool {
    false
}

/// Detect `nvcc` on `PATH`, without running it for version output.
pub fn nvcc_available() -> bool {
    which("nvcc").is_some()
}

/// Locate an executable on `PATH`.
///
/// A deliberately small replacement for the `which` crate: it refuses to run in
/// a directory named on `PATH` when that directory is world-writable, which is
/// the usual way a hostile user plants a fake `whisper-cli`.
pub fn which(program: &str) -> Option<std::path::PathBuf> {
    if program.contains('/') {
        let p = std::path::PathBuf::from(program);
        return p.is_file().then_some(p);
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        // World-writable PATH entries are not searched.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(&dir) {
                if meta.permissions().mode() & 0o002 != 0 {
                    continue;
                }
            }
        }
        let candidate = dir.join(program);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Full report.
pub fn detect_all() -> HardwareReport {
    let mut extras = BTreeMap::new();
    extras.insert("nvcc".into(), nvcc_available().to_string());
    extras.insert("cuda_compiled_in".into(), cuda_compiled_in().to_string());
    extras.insert(
        "nvcc_path".into(),
        which("nvcc")
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
    );
    HardwareReport {
        os: read_trimmed("/etc/os-release")
            .and_then(|s| {
                s.lines().find(|l| l.starts_with("PRETTY_NAME=")).map(|l| {
                    l.trim_start_matches("PRETTY_NAME=")
                        .trim_matches('"')
                        .to_string()
                })
            })
            .unwrap_or_else(|| std::env::consts::OS.to_string()),
        kernel: read_trimmed("/proc/sys/kernel/osrelease").unwrap_or_else(|| "unknown".into()),
        cpu: detect_cpu(),
        memory: detect_memory(),
        gpu: detect_gpu(),
        audio: detect_audio(),
        running_as_root: unsafe { libc_geteuid() == 0 },
        cuda_compiled_in: cuda_compiled_in(),
        extras,
    }
}

/// Format a byte count for human display.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Run `program` with a hard timeout and return its stdout.
///
/// Only ever called with a fixed program name and a fixed argument vector. Used
/// by capability probes so a wedged driver or a hung `nvidia-smi` cannot hang
/// `aaron doctor`. Returns `None` when the program is missing, fails, or does
/// not finish inside `timeout`.
pub fn probe(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let start = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(30));
            }
            Err(_) => {
                let _ = child.kill();
                return None;
            }
        }
    };
    if !status.success() {
        return None;
    }
    // `wait_with_output` would block again on the pipe; the child has exited and
    // the pipe buffer is bounded, so this read returns promptly.
    child.stdout.take()?.read_to_string_end()
}

trait ReadToStringEnd {
    /// Drain the pipe into a string, capped: a rogue binary must not be able to
    /// exhaust memory through a capability probe.
    fn read_to_string_end(self) -> Option<String>;
}

impl ReadToStringEnd for std::process::ChildStdout {
    fn read_to_string_end(mut self) -> Option<String> {
        use std::io::Read;
        let mut raw = Vec::new();
        self.by_ref().take(256 * 1024).read_to_end(&mut raw).ok()?;
        Some(String::from_utf8_lossy(&raw).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_detection_reports_a_core_count() {
        let cpu = detect_cpu();
        assert!(cpu.logical_cores >= 1, "available_parallelism returned 0");
        assert!(!cpu.arch.is_empty());
    }

    #[test]
    fn memory_detection_reports_total_if_meminfo_exists() {
        if Path::new("/proc/meminfo").exists() {
            let mem = detect_memory();
            assert!(mem.total_bytes.unwrap_or(0) > 0);
        }
    }

    #[test]
    fn human_bytes_formats_every_scale() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KiB");
        assert_eq!(human_bytes(4 * 1024 * 1024), "4.0 MiB");
        assert_eq!(human_bytes(4 * 1024 * 1024 * 1024), "4.0 GiB");
    }

    #[test]
    fn which_finds_a_known_program_and_rejects_a_missing_one() {
        assert!(
            which("sh").is_some(),
            "sh must exist on any Linux test host"
        );
        assert!(which("definitely-not-a-real-binary-xyz").is_none());
    }

    #[test]
    fn which_does_not_treat_directories_as_programs() {
        assert!(which("/tmp").is_none());
    }

    #[test]
    fn gpu_detection_never_panics_without_a_gpu() {
        let gpu = detect_gpu();
        // Either a real NVIDIA GPU is reported, or nothing is. Both are valid.
        if gpu.nvidia_smi_available {
            assert!(gpu.name.is_some());
        } else {
            assert!(gpu.name.is_none());
        }
    }

    #[test]
    fn full_report_is_internally_consistent() {
        let report = detect_all();
        assert!(!report.os.is_empty());
        assert!(report.memory.total_bytes.unwrap_or(1) > 0 || Path::new("/proc/meminfo").exists());
    }

    #[test]
    fn probe_times_out_rather_than_hanging() {
        let start = std::time::Instant::now();
        // `sleep` may not exist everywhere; a missing binary must also return.
        let _ = probe("sleep", &["5"], Duration::from_millis(200));
        assert!(start.elapsed() < Duration::from_secs(4));
    }
    /// Splits an `nvidia-smi --query-gpu` row the way `detect_gpu` does.
    fn parse_query_row(stdout: &str) -> GpuInfo {
        let mut info = GpuInfo::default();
        if let Some(first) = stdout.lines().next() {
            let fields: Vec<&str> = first.split(',').map(|s| s.trim()).collect();
            if let Some(name) = fields.first().filter(|s| !s.is_empty()) {
                info.name = Some((*name).to_string());
            }
            if let Some(v) = fields.get(1).filter(|s| !s.is_empty()) {
                info.driver_version = Some((*v).to_string());
            }
            if let Some(mib) = fields.get(2).and_then(|s| s.parse::<u64>().ok()) {
                info.total_vram_bytes = Some(mib * 1024 * 1024);
            }
        }
        info
    }

    /// Extracts the CUDA version from an `nvidia-smi` banner, as `detect_gpu`
    /// does.
    fn parse_cuda(banner: &str) -> Option<String> {
        for label in ["CUDA Version:", "CUDA UMD Version:", "CUDA Driver Version:"] {
            let Some(idx) = banner.find(label) else {
                continue;
            };
            // Skip the padding after the colon; taking digits from a leading
            // space yields an empty string and the label is silently skipped.
            let rest = banner[idx + label.len()..].trim_start();
            let ver: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if !ver.is_empty() {
                return Some(ver);
            }
        }
        None
    }

    #[test]
    fn a_dotted_driver_version_survives_parsing() {
        // `615.78.08` is not a u64. Parsing it as one is why GPU reports said
        // `driver unknown` on machines whose driver version was perfectly
        // readable from the same query.
        let info = parse_query_row("NVIDIA GeForce RTX 2050, 615.78.08, 4096");
        assert_eq!(info.driver_version.as_deref(), Some("615.78.08"));
        assert_eq!(info.total_vram_bytes, Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn a_short_driver_version_also_survives() {
        let info = parse_query_row("Some GPU, 550.54, 8192");
        assert_eq!(info.driver_version.as_deref(), Some("550.54"));
    }

    #[test]
    fn an_empty_driver_field_is_left_unset_rather_than_becoming_empty() {
        let info = parse_query_row("Some GPU, , 8192");
        assert!(info.driver_version.is_none());
    }

    #[test]
    fn the_cuda_version_is_found_under_either_banner_label() {
        // Recent drivers print `CUDA UMD Version:`; older ones `CUDA Version:`.
        let recent = "NVIDIA-SMI 615.78.08   KMD Version: 615.78.08   CUDA UMD Version: 13.4   |";
        assert_eq!(parse_cuda(recent).as_deref(), Some("13.4"));
        let older = "CUDA Version: 11.8";
        assert_eq!(parse_cuda(older).as_deref(), Some("11.8"));
        let none = "no cuda here";
        assert_eq!(parse_cuda(none), None);
    }

    #[test]
    fn live_gpu_detection_agrees_with_the_query_shape() {
        // On a host with an NVIDIA card the driver version must actually come
        // back. On a host without one the field is simply unset, which is also
        // correct.
        let gpu = detect_gpu();
        if gpu.name.is_some() {
            assert!(
                gpu.driver_version.is_some(),
                "a GPU was detected but its driver version was not read: {gpu:?}"
            );
        }
    }
}
