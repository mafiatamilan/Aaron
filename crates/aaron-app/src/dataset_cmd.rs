//! `aaron dataset` — inspect the training data.
//!
//! The data quality pipeline lives in `aaron-brain::dataquality`. This
//! module is its command-line surface, and it exists so the checks can be run
//! as a gate (in CI, or before a training run) rather than only from inside a
//! test.

use std::path::Path;

use aaron_core::error::{Error, Result};

use crate::context::AppContext;

/// `aaron dataset check`
///
/// Validates the dataset and reports every defect: duplicate ids, duplicate
/// and near-duplicate text, cross-split leakage, thin coverage and split
/// imbalance.
///
/// Exits non-zero when a *blocking* problem is found, so it can be used as a
/// gate. Non-blocking findings — near-duplicates, thin coverage — are
/// reported and do not fail the check: they are quality signals, not
/// correctness errors.
pub fn check(ctx: &AppContext, dataset_path: Option<&Path>, json: bool) -> Result<i32> {
    let path = dataset_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| ctx.project_root.join(&ctx.config.learning.dataset_path));
    let dataset = aaron_brain::dataset::Dataset::load(&path)?;
    let report = aaron_brain::dataquality::analyse(&dataset);

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| Error::Other(e.to_string()))?
        );
    } else {
        println!("dataset     {}", report.version);
        println!("examples    {}", report.total);
        println!("sha256      {}", report.content_sha256);
        println!("splits      {:?}", report.split_counts);
        println!("provenance  {:?}", report.provenance);
        println!();
        for (intent, counts) in &report.per_intent {
            let flags = if report.intents_under_measured.contains(intent) {
                "  <- under-measured"
            } else if report.intents_under_trained.contains(intent) {
                "  <- under-trained"
            } else {
                ""
            };
            println!(
                "  {:<20} train {:>4}  validation {:>4}  test {:>4}{}",
                intent, counts.train, counts.validation, counts.test, flags
            );
        }
        println!();
        if report.findings.is_empty() {
            println!("no problems found.");
        } else {
            for finding in &report.findings {
                println!(
                    "  [{}] {:<24} x{}",
                    if finding.blocking { "BLOCK" } else { "warn " },
                    finding.kind.as_str(),
                    finding.count
                );
                println!("      {}", finding.detail);
            }
        }
    }
    Ok(if report.usable_for_training { 0 } else { 1 })
}

/// `aaron dataset report`
///
/// The same thing, but always as text. It exists so the report can be
/// requested by name without remembering that `check` has a `--json` flag.
pub fn report(ctx: &AppContext, dataset_path: Option<&Path>) -> Result<i32> {
    check(ctx, dataset_path, false)
}
