//! Versioned checkpoint format with integrity verification.
//!
//! A checkpoint is one file:
//!
//! ```text
//! [u32 LE]  manifest length in bytes
//! [JSON    ] manifest (UTF-8)
//! [f32 LE  ] concatenated tensor data, in manifest order
//! ```
//!
//! The manifest records, for every tensor, its name, shape, byte offset, length
//! and SHA-256, plus a SHA-256 of the whole data block and the architecture
//! fingerprint. Loading therefore answers three questions before a single
//! weight is used:
//!
//! 1. Is this file the architecture AARON expects?
//! 2. Does it contain exactly the tensors the architecture declares?
//! 3. Is every byte the one that was written?
//!
//! A corrupted or foreign checkpoint is a clear error, never a silent
//! misclassification.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use aaron_core::error::{Error, Result};

use crate::model::{
    arch_id, expected_parameter_count, parameter_specs, Heads, ParamSpec, CLASSES, DIM, HIDDEN,
    VOCAB,
};

/// Current container format version.
pub const CHECKPOINT_FORMAT: u32 = 1;

/// One tensor as recorded in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    /// Byte offset from the start of the data block.
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
    /// SHA-256 of this tensor's bytes.
    pub sha256: String,
}

/// Checkpoint manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    /// Architecture fingerprint; loading refuses a mismatch.
    pub arch: String,
    /// Total trainable parameters recorded at save time.
    pub param_count: usize,
    pub vocab: usize,
    pub dim: usize,
    pub hidden: usize,
    pub classes: usize,
    pub seq_len: usize,
    /// Does this checkpoint carry the readiness head?
    ///
    /// The head's presence changes the tensor set, the parameter count and
    /// the architecture fingerprint, so it has to be recorded: a loader that
    /// guessed would either fail to find the tensors or, worse, build a model
    /// with the wrong shape and load weights into it.
    ///
    /// `false` for a checkpoint written before release 0.5.
    #[serde(default)]
    pub readiness_head: bool,
    pub created_at_unix: u64,
    /// Dataset version this checkpoint was trained from, if known.
    #[serde(default)]
    pub dataset_version: String,
    /// Training seed, so a run can be reproduced.
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub epochs: u32,
    /// Training metrics recorded at save time.
    #[serde(default)]
    pub metrics: serde_json::Value,
    /// SHA-256 of the whole data block.
    pub data_sha256: String,
    pub tensors: Vec<TensorEntry>,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Metadata attached when saving a checkpoint.
#[derive(Debug, Clone, Default)]
pub struct CheckpointInfo {
    pub dataset_version: String,
    pub seed: u64,
    pub epochs: u32,
    pub metrics: serde_json::Value,
}

impl CheckpointInfo {
    /// A manifest-only view, used by `aaron model info` without loading weights.
    pub fn metrics_or_null(&self) -> serde_json::Value {
        if self.metrics.is_null() {
            serde_json::Value::Null
        } else {
            self.metrics.clone()
        }
    }
}

/// Serialise a set of tensors into the on-disk container format.
pub fn encode(tensors: &HashMap<String, Tensor>, info: &CheckpointInfo) -> Result<Vec<u8>> {
    let heads = if tensors.contains_key("fc_ready.weight") {
        Heads::WithReadiness
    } else {
        Heads::IntentOnly
    };
    let specs = parameter_specs(heads);
    let mut entries = Vec::with_capacity(specs.len());
    let mut data: Vec<u8> = Vec::new();
    let mut param_count = 0usize;

    for spec in &specs {
        let tensor = tensors
            .get(&spec.name)
            .ok_or_else(|| Error::Model(format!("checkpoint is missing tensor `{}`", spec.name)))?;
        let actual: Vec<usize> = tensor.shape().dims().to_vec();
        if actual != spec.shape {
            return Err(Error::Model(format!(
                "tensor `{}` has shape {actual:?} but the architecture declares {:?}",
                spec.name, spec.shape
            )));
        }
        let values: Vec<f32> = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::Model(format!("tensor `{}` is not f32: {e}", spec.name)))?;

        let offset = data.len() as u64;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let sha = sha256_hex(&bytes);
        param_count += values.len();
        entries.push(TensorEntry {
            name: spec.name.clone(),
            shape: spec.shape.clone(),
            offset,
            len: bytes.len() as u64,
            sha256: sha,
        });
        data.extend_from_slice(&bytes);
    }

    let manifest = Manifest {
        format: CHECKPOINT_FORMAT,
        arch: arch_id(heads),
        param_count,
        vocab: VOCAB,
        dim: DIM,
        hidden: HIDDEN,
        classes: CLASSES,
        seq_len: crate::model::SEQ_LEN,
        readiness_head: heads.has_readiness(),
        created_at_unix: now_unix(),
        dataset_version: info.dataset_version.clone(),
        seed: info.seed,
        epochs: info.epochs,
        metrics: info.metrics_or_null(),
        data_sha256: sha256_hex(&data),
        tensors: entries,
    };

    let json = serde_json::to_vec(&manifest).map_err(|e| Error::Model(e.to_string()))?;
    let mut out = Vec::with_capacity(4 + json.len() + data.len());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&data);
    Ok(out)
}

/// Parse and fully verify a checkpoint container.
pub fn decode(bytes: &[u8]) -> Result<(Manifest, HashMap<String, Tensor>)> {
    if bytes.len() < 4 {
        return Err(Error::Model("checkpoint file is truncated".into()));
    }
    let json_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if bytes.len() < 4 + json_len {
        return Err(Error::Model("checkpoint manifest is truncated".into()));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes[4..4 + json_len])
        .map_err(|e| Error::Model(format!("checkpoint manifest is not valid JSON: {e}")))?;
    let data = &bytes[4 + json_len..];

    verify_manifest(&manifest, data.len() as u64)?;

    let device = Device::Cpu;
    let mut tensors = HashMap::new();
    for entry in &manifest.tensors {
        let start = entry.offset as usize;
        let end = start + entry.len as usize;
        if end > data.len() {
            return Err(Error::Model(format!(
                "tensor `{}` claims bytes {start}..{end} but the data block is {} bytes",
                entry.name,
                data.len()
            )));
        }
        let slice = &data[start..end];
        if sha256_hex(slice) != entry.sha256 {
            return Err(Error::Model(format!(
                "tensor `{}` failed its SHA-256 check; the file is corrupt",
                entry.name
            )));
        }
        let count = slice.len() / 4;
        let values: Vec<f32> = slice
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        debug_assert_eq!(values.len(), count);
        let tensor = Tensor::new(values.as_slice(), &device)
            .and_then(|t| t.reshape(entry.shape.clone()))
            .map_err(|e| Error::Model(format!("tensor `{}` is unreadable: {e}", entry.name)))?;
        tensors.insert(entry.name.clone(), tensor);
    }
    Ok((manifest, tensors))
}

/// The heads a manifest describes.
pub fn manifest_heads(m: &Manifest) -> Heads {
    if m.readiness_head {
        Heads::WithReadiness
    } else {
        Heads::IntentOnly
    }
}

fn verify_manifest(m: &Manifest, data_len: u64) -> Result<()> {
    if m.format != CHECKPOINT_FORMAT {
        return Err(Error::Model(format!(
            "checkpoint format {} is not supported (this build reads {CHECKPOINT_FORMAT})",
            m.format
        )));
    }
    let heads = manifest_heads(m);
    let expected_arch = arch_id(heads);
    if m.arch != expected_arch {
        return Err(Error::Model(format!(
            "checkpoint architecture is `{}` but this build expects `{expected_arch}`",
            m.arch
        )));
    }
    let expected_count = expected_parameter_count(heads);
    if m.param_count != expected_count {
        return Err(Error::Model(format!(
            "checkpoint declares {} parameters but the architecture has {expected_count}",
            m.param_count
        )));
    }
    // The declared tensor set must be exactly the architecture's tensor set.
    let specs: Vec<ParamSpec> = parameter_specs(heads);
    if m.tensors.len() != specs.len() {
        return Err(Error::Model(format!(
            "checkpoint has {} tensors but the architecture declares {}",
            m.tensors.len(),
            specs.len()
        )));
    }
    for spec in &specs {
        let Some(entry) = m.tensors.iter().find(|e| e.name == spec.name) else {
            return Err(Error::Model(format!(
                "checkpoint is missing tensor `{}`",
                spec.name
            )));
        };
        if entry.shape != spec.shape {
            return Err(Error::Model(format!(
                "tensor `{}` has shape {:?}, expected {:?}",
                spec.name, entry.shape, spec.shape
            )));
        }
    }
    let total: u64 = m.tensors.iter().map(|e| e.len).sum();
    if total != data_len {
        return Err(Error::Model(format!(
            "checkpoint manifest describes {total} bytes of tensors but the data block holds {data_len}"
        )));
    }
    Ok(())
}

/// Read and verify a checkpoint file without materialising tensors.
///
/// Used by `aaron model info` so a 4 MB file does not have to be fully decoded
/// just to print its metadata.
pub fn read_manifest(path: impl AsRef<Path>) -> Result<Manifest> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| Error::io(path.as_ref(), e))?;
    if bytes.len() < 4 {
        return Err(Error::Model("checkpoint file is truncated".into()));
    }
    let json_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if bytes.len() < 4 + json_len {
        return Err(Error::Model("checkpoint manifest is truncated".into()));
    }
    let manifest: Manifest = serde_json::from_slice(&bytes[4..4 + json_len])
        .map_err(|e| Error::Model(format!("checkpoint manifest is not valid JSON: {e}")))?;
    verify_manifest(&manifest, (bytes.len() - 4 - json_len) as u64)?;
    Ok(manifest)
}

/// Read and verify a checkpoint file, returning tensors ready for inference.
pub fn load(path: impl AsRef<Path>) -> Result<(Manifest, HashMap<String, Tensor>)> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    decode(&bytes).map_err(|e| match e {
        Error::Model(msg) => Error::Model(format!("{}: {}", path.display(), msg)),
        other => other,
    })
}

/// Write a checkpoint, creating parent directories.
pub fn save(
    path: impl AsRef<Path>,
    tensors: &HashMap<String, Tensor>,
    info: &CheckpointInfo,
) -> Result<Manifest> {
    let path = path.as_ref();
    let bytes = encode(tensors, info)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    // Write to a temporary sibling and rename, so an interrupted save never
    // leaves a half-written checkpoint where a valid one used to be.
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| Error::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| Error::io(path, e))?;
    let (manifest, _) = decode(&bytes)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{initial_tensors, IntentModel};

    fn tensors() -> HashMap<String, Tensor> {
        initial_tensors(11, crate::model::Heads::IntentOnly).unwrap()
    }

    #[test]
    fn round_trips_a_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.ckpt");
        let info = CheckpointInfo {
            dataset_version: "v1".into(),
            seed: 11,
            epochs: 3,
            metrics: serde_json::json!({"accuracy": 0.9}),
        };
        let manifest = save(&path, &tensors(), &info).unwrap();
        assert_eq!(manifest.param_count, 1_000_000);
        assert_eq!(manifest.dataset_version, "v1");

        let (loaded_manifest, loaded) = load(&path).unwrap();
        assert_eq!(loaded_manifest, manifest);
        let original = tensors();
        for (name, t) in &original {
            let a: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = loaded[name].flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(a, b, "tensor `{name}` changed across a save/load cycle");
        }
    }

    #[test]
    fn file_size_matches_the_parameter_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.ckpt");
        save(&path, &tensors(), &CheckpointInfo::default()).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        // 4 bytes of manifest + 1,000,000 f32 + a small JSON manifest.
        assert!(meta.len() > 4_000_000);
        assert!(meta.len() < 4_200_000);
    }

    #[test]
    fn rejects_a_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.ckpt");
        save(&path, &tensors(), &CheckpointInfo::default()).unwrap();
        let bytes = std::fs::read(&path).unwrap();

        assert!(decode(&bytes[..10]).is_err());
        assert!(decode(&bytes[..bytes.len() / 2]).is_err());
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn rejects_a_single_flipped_weight_byte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.ckpt");
        save(&path, &tensors(), &CheckpointInfo::default()).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let err = decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::Model(_)));
        assert!(err.to_string().contains("SHA-256") || err.to_string().contains("sha256"));
    }

    #[test]
    fn rejects_a_foreign_architecture() {
        let bytes = encode(&tensors(), &CheckpointInfo::default()).unwrap();
        // Rewrite the manifest with a different architecture fingerprint.
        let json_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut manifest: Manifest = serde_json::from_slice(&bytes[4..4 + json_len]).unwrap();
        manifest.arch = "some-other-model-v9".into();
        let new_json = serde_json::to_vec(&manifest).unwrap();
        let mut forged = Vec::new();
        forged.extend_from_slice(&(new_json.len() as u32).to_le_bytes());
        forged.extend_from_slice(&new_json);
        forged.extend_from_slice(&bytes[4 + json_len..]);

        let err = decode(&forged).unwrap_err();
        assert!(err.to_string().contains("architecture"));
    }

    #[test]
    fn rejects_a_wrong_parameter_count() {
        let bytes = encode(&tensors(), &CheckpointInfo::default()).unwrap();
        let json_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut manifest: Manifest = serde_json::from_slice(&bytes[4..4 + json_len]).unwrap();
        manifest.param_count = 999_999;
        let new_json = serde_json::to_vec(&manifest).unwrap();
        let mut forged = Vec::new();
        forged.extend_from_slice(&(new_json.len() as u32).to_le_bytes());
        forged.extend_from_slice(&new_json);
        forged.extend_from_slice(&bytes[4 + json_len..]);
        assert!(decode(&forged)
            .unwrap_err()
            .to_string()
            .contains("parameters"));
    }

    #[test]
    fn rejects_a_manifest_missing_a_tensor() {
        let bytes = encode(&tensors(), &CheckpointInfo::default()).unwrap();
        let json_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut manifest: Manifest = serde_json::from_slice(&bytes[4..4 + json_len]).unwrap();
        manifest.tensors.pop();
        let new_json = serde_json::to_vec(&manifest).unwrap();
        let mut forged = Vec::new();
        forged.extend_from_slice(&(new_json.len() as u32).to_le_bytes());
        forged.extend_from_slice(&new_json);
        forged.extend_from_slice(&bytes[4 + json_len..]);
        assert!(decode(&forged).is_err());
    }

    #[test]
    fn read_manifest_validates_without_loading_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.ckpt");
        save(&path, &tensors(), &CheckpointInfo::default()).unwrap();
        let m = read_manifest(&path).unwrap();
        assert_eq!(m.arch, arch_id(crate::model::Heads::IntentOnly));
        assert_eq!(m.tensors.len(), 9);
    }

    #[test]
    fn read_manifest_on_garbage_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.ckpt");
        std::fs::write(&path, b"this is not a checkpoint").unwrap();
        assert!(read_manifest(&path).is_err());
        assert!(read_manifest(dir.path().join("missing.ckpt")).is_err());
    }

    #[test]
    fn loaded_tensors_rebuild_the_same_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.ckpt");
        save(&path, &tensors(), &CheckpointInfo::default()).unwrap();
        let (_, loaded) = load(&path).unwrap();
        let model =
            IntentModel::from_tensors(loaded, &Device::Cpu, crate::model::Heads::IntentOnly)
                .unwrap();

        let batch =
            crate::model::FeatureBatch::from_texts(&["check the system status".to_string()]);
        let logits = model.forward(&batch).unwrap();
        assert_eq!(logits.dims(), &[1, CLASSES]);

        let direct =
            IntentModel::initialised(11, &Device::Cpu, crate::model::Heads::IntentOnly).unwrap();
        let a: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = direct
            .forward(&batch)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn missing_tensor_is_reported_at_save_time() {
        let mut t = tensors();
        t.remove("fc2.bias");
        let err = encode(&t, &CheckpointInfo::default()).unwrap_err();
        assert!(err.to_string().contains("fc2.bias"));
    }
}
