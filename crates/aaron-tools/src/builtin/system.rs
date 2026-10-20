//! `system_status`: report what the machine looks like right now.
//!
//! Read-only. No configuration is read from the filesystem, no file is opened
//! for writing, and every probe is bounded by a timeout.

use aaron_core::error::Result;
use aaron_core::hardware::{detect_all, human_bytes};
use serde_json::json;

use crate::registry::{Tool, ToolContext, ToolOutput};
use crate::spec::{ToolArgs, ToolSpec};
use aaron_security::{PermissionCategory, RiskLevel};

pub const SPEC: ToolSpec = ToolSpec {
    id: "system_status",
    description: "Report CPU, memory, GPU and audio status",
    category: PermissionCategory::SystemRead,
    risk: RiskLevel::Low,
    timeout_secs: 8,
    max_output_bytes: 8192,
    requires_confirmation: false,
    args: &[],
};

#[derive(Debug)]
pub struct SystemStatusTool;

impl Tool for SystemStatusTool {
    fn spec(&self) -> &'static ToolSpec {
        &SPEC
    }

    fn execute(&self, _ctx: &ToolContext, _args: &ToolArgs) -> Result<ToolOutput> {
        let r = detect_all();

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("System: {} (kernel {})", r.os, r.kernel));
        lines.push(format!(
            "CPU: {} — {} logical cores, {}",
            r.cpu
                .model
                .clone()
                .unwrap_or_else(|| "unknown model".into()),
            r.cpu.logical_cores,
            r.cpu.arch
        ));
        lines.push(match (r.memory.total_bytes, r.memory.available_bytes) {
            (Some(t), Some(a)) => format!(
                "Memory: {} total, {} available",
                human_bytes(t),
                human_bytes(a)
            ),
            _ => "Memory: unknown (/proc/meminfo unavailable)".into(),
        });
        lines.push(match (&r.gpu.name, r.gpu.total_vram_bytes) {
            (Some(name), Some(vram)) => format!(
                "GPU: {name}, {} VRAM, driver {}, CUDA runtime {}",
                human_bytes(vram),
                r.gpu
                    .driver_version
                    .clone()
                    .unwrap_or_else(|| "unknown".into()),
                r.gpu
                    .cuda_version
                    .clone()
                    .unwrap_or_else(|| "unknown".into())
            ),
            (Some(name), None) => format!("GPU: {name} (VRAM unknown)"),
            _ => "GPU: no NVIDIA GPU detected".into(),
        });
        lines.push(format!("Audio: {}", r.audio.detail));
        lines.push(format!(
            "Inference: {} (CUDA backend {})",
            if r.cuda_compiled_in { "cuda" } else { "cpu" },
            if r.cuda_compiled_in {
                "compiled in"
            } else {
                "not compiled in"
            }
        ));
        lines.push(format!(
            "Privilege: {}",
            if r.running_as_root {
                "running as root (AARON refuses to execute tools)"
            } else {
                "normal user"
            }
        ));

        let data = json!({
            "os": r.os,
            "kernel": r.kernel,
            "cpu": r.cpu,
            "memory": r.memory,
            "gpu": r.gpu,
            "audio": r.audio,
            "cuda_compiled_in": r.cuda_compiled_in,
            "running_as_root": r.running_as_root,
        });
        Ok(ToolOutput::text(lines.join("\n"))
            .with_data(data)
            .capped(SPEC.max_output_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_a_truthful_status() {
        let out = SystemStatusTool
            .execute(&ToolContext::for_tests("/tmp"), &ToolArgs::default())
            .unwrap();
        let text = out.text.to_lowercase();
        assert!(text.contains("cpu"));
        assert!(text.contains("memory"));
        assert!(text.contains("gpu"));
        assert!(text.contains("audio"));
        assert!(out.data.is_some());
    }

    #[test]
    fn never_claims_a_gpu_it_did_not_see() {
        let out = SystemStatusTool
            .execute(&ToolContext::for_tests("/tmp"), &ToolArgs::default())
            .unwrap();
        let gpu = &out.data.unwrap()["gpu"];
        if gpu["name"].is_null() {
            assert!(out.text.contains("no NVIDIA GPU detected"));
        }
    }

    #[test]
    fn output_stays_within_the_declared_cap() {
        // Compare against the *shipped* policy limit rather than a literal.
        // Comparing `SPEC.max_output_bytes` to a constant is a compile-time
        // comparison, which the compiler resolves and clip rightly rejects: it
        // tests nothing. Reading the limit from the shipped policy file makes
        // this a statement about the shipped product instead.
        let shipped = include_str!("../../../../config/permissions.toml");
        let declared = aaron_security::PolicyFile::from_toml(shipped)
            .expect("the shipped permissions file must parse")
            .limits
            .max_output_bytes;
        assert!(
            SPEC.max_output_bytes <= declared,
            "the tool declares {} bytes but the shipped global cap is {declared}; \
             a per-tool cap can only lower the limit, so this is a configuration bug",
            SPEC.max_output_bytes
        );
        let out = SystemStatusTool
            .execute(&ToolContext::for_tests("/tmp"), &ToolArgs::default())
            .unwrap();
        assert!(out.text.chars().count() <= SPEC.max_output_bytes);
    }
}
