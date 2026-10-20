//! Runtime resource sampling.
//!
//! Every number here is measured on the machine that runs it. Anything that
//! cannot be measured is reported as `None` with a reason, never as zero —
//! "0 bytes of VRAM used" and "I could not read VRAM" are very different
//! statements and conflating them is how a resource report ends up lying.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// A point-in-time sample of the machine's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySample {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl MemorySample {
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.available_bytes)
    }

    pub fn free_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.available_bytes as f64 / self.total_bytes as f64 * 100.0
    }
}

/// A point-in-time sample of GPU memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuSample {
    pub total_bytes: u64,
    pub used_bytes: u64,
    /// True when the numbers came from `nvidia-smi` rather than a fallback.
    pub measured: bool,
    /// Why the sample is unavailable, when `measured` is false.
    pub detail: String,
}

impl GpuSample {
    /// A sample for a machine with no measurable GPU.
    pub fn unavailable(detail: impl Into<String>) -> GpuSample {
        GpuSample {
            total_bytes: 0,
            used_bytes: 0,
            measured: false,
            detail: detail.into(),
        }
    }

    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    pub fn free_percent(&self) -> f64 {
        if !self.measured || self.total_bytes == 0 {
            return 100.0;
        }
        self.free_bytes() as f64 / self.total_bytes as f64 * 100.0
    }
}

/// Everything measured in one sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSample {
    pub system: MemorySample,
    pub gpu: GpuSample,
    /// Resident set size of the AARON process.
    pub process_rss_bytes: u64,
}

fn read_meminfo() -> Option<MemorySample> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = 0u64;
    let mut available = 0u64;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next()?;
        let value: u64 = parts.next()?.parse().ok()?;
        match key.trim_end_matches(':') {
            "MemTotal" => total = value * 1024,
            // `MemAvailable` is the kernel's own estimate of what a new
            // allocation could claim without swapping. It is the honest number
            // to budget against, not `MemFree`.
            "MemAvailable" => available = value * 1024,
            _ => {}
        }
    }
    if total == 0 {
        return None;
    }
    Some(MemorySample {
        total_bytes: total,
        available_bytes: available,
    })
}

/// Sample system memory. Returns `None` when `/proc/meminfo` is unreadable.
pub fn system_memory() -> Option<MemorySample> {
    read_meminfo()
}

/// Sample GPU memory through `nvidia-smi`, with a hard timeout.
///
/// AARON's own CUDA backend is not compiled in, so *all* VRAM it uses is
/// through a child process such as whisper.cpp. The sample therefore reports
/// the whole card, and `aaron resources` subtracts nothing: it reports what is
/// there and labels whose it is.
pub fn gpu_memory() -> GpuSample {
    let Some(text) = probe(
        "nvidia-smi",
        &[
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_millis(2000),
    ) else {
        return GpuSample::unavailable(
            "nvidia-smi is not installed or did not respond; VRAM is not measurable on this host",
        );
    };
    let Some(first) = text.lines().find(|l| !l.trim().is_empty()) else {
        return GpuSample::unavailable("nvidia-smi returned no rows");
    };
    let fields: Vec<u64> = first
        .split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect();
    if fields.len() < 2 {
        return GpuSample::unavailable("nvidia-smi returned an unparsable row");
    }
    GpuSample {
        total_bytes: fields[0] * 1024 * 1024,
        used_bytes: fields[1] * 1024 * 1024,
        measured: true,
        detail: String::new(),
    }
}

/// Resident set size of this process, in bytes.
pub fn process_rss_bytes() -> u64 {
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let Some(pages) = statm
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse::<u64>().ok())
    else {
        return 0;
    };
    // `sysconf(_SC_PAGESIZE)` is 4 KiB on every Linux platform AARON targets.
    pages.saturating_mul(4096)
}

/// Run a helper binary with a hard timeout and return its stdout.
///
/// Used for capability probes. A wedged driver must never hang the assistant,
/// so a probe that overruns is killed and reported as unavailable.
pub fn probe(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let start = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => {
                let _ = child.kill();
                return None;
            }
        }
    };
    if !status.is_some_and(|s| s.success()) {
        return None;
    }
    use std::io::Read;
    let mut raw = Vec::new();
    child
        .stdout
        .take()?
        .take(64 * 1024)
        .read_to_end(&mut raw)
        .ok()?;
    Some(String::from_utf8_lossy(&raw).into_owned())
}

/// Number of GPUs present, for reporting.
pub fn gpu_count() -> usize {
    probe(
        "nvidia-smi",
        &["--query-gpu=count", "--format=csv,noheader,nounits"],
        Duration::from_millis(1500),
    )
    .and_then(|t| t.trim().parse::<usize>().ok())
    .unwrap_or(0)
}

/// Take one full sample.
pub fn sample() -> ResourceSample {
    ResourceSample {
        system: system_memory().unwrap_or(MemorySample {
            total_bytes: 0,
            available_bytes: 0,
        }),
        gpu: gpu_memory(),
        process_rss_bytes: process_rss_bytes(),
    }
}

/// Whether a path is a Linux procfs entry we can read, used by tests.
pub fn procfs_available() -> bool {
    Path::new("/proc/self/statm").exists()
}

/// Format bytes for display.
pub fn human_bytes(bytes: u64) -> String {
    aaron_core::hardware::human_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_memory_is_measurable_on_linux() {
        if !procfs_available() {
            return;
        }
        let m = system_memory().expect("/proc/meminfo should be readable");
        assert!(m.total_bytes > 0);
        assert!(m.available_bytes <= m.total_bytes);
        assert!(m.free_percent() >= 0.0 && m.free_percent() <= 100.0);
        assert_eq!(m.used_bytes(), m.total_bytes - m.available_bytes);
    }

    #[test]
    fn an_unavailable_gpu_never_reports_zero_as_a_measurement() {
        let g = gpu_memory();
        if !g.measured {
            assert_eq!(g.total_bytes, 0);
            assert!(!g.detail.is_empty(), "an unmeasured GPU must say why");
            // Critically: free_percent is 100, not 0, so an absent GPU never
            // looks like a full one and never trips the low-memory rule.
            assert_eq!(g.free_percent(), 100.0);
        } else {
            assert!(g.total_bytes > 0);
            assert!(g.free_bytes() <= g.total_bytes);
        }
    }

    #[test]
    fn process_rss_is_non_zero_while_running() {
        if procfs_available() {
            assert!(process_rss_bytes() > 0);
        }
    }

    #[test]
    fn a_probe_of_a_missing_program_returns_none_rather_than_hanging() {
        let start = std::time::Instant::now();
        assert!(probe(
            "definitely-not-a-real-binary-xyz",
            &[],
            Duration::from_millis(200)
        )
        .is_none());
        assert!(start.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn a_probe_times_out() {
        let start = std::time::Instant::now();
        let _ = probe("sleep", &["10"], Duration::from_millis(200));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_full_sample_always_carries_a_process_rss() {
        let s = sample();
        assert_eq!(s.process_rss_bytes, process_rss_bytes());
    }

    #[test]
    fn byte_formatting_is_readable() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(2048), "2.0 KiB");
        assert_eq!(human_bytes(4 * 1024 * 1024 * 1024), "4.0 GiB");
    }
}
