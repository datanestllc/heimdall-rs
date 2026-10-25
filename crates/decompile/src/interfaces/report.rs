use heimdall_vm::ext::exec::{ExecStats, TruncationReason, VMTrace};

use crate::interfaces::Effect;

/// How far the body of one function was analyzed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BodyStatus {
    /// Every path was explored, or stopped only by a loop heuristic.
    Done,
    /// A deterministic budget ran out or a path failed to execute; see the dropped paths.
    Partial,
    /// The wall-clock emergency stop fired.
    Timeout,
    /// The function could not be analyzed.
    Failed,
}

impl BodyStatus {
    /// The status that the dropped paths of one symbolic execution imply.
    pub fn from_stats(stats: &ExecStats) -> Self {
        if stats.truncated(TruncationReason::WallClock) > 0 {
            return Self::Timeout;
        }
        let partial = [
            TruncationReason::StepBudget,
            TruncationReason::BranchBudget,
            TruncationReason::StateBudget,
            TruncationReason::StepError,
        ];
        if partial.iter().any(|reason| stats.truncated(*reason) > 0) {
            return Self::Partial;
        }
        Self::Done
    }
}

/// What the decompiler did with one function: status, budgets used and paths dropped.
#[derive(Clone, Debug)]
pub struct FunctionReport {
    /// The 4-byte selector as 8 lowercase hex characters, or `fallback`.
    pub selector: String,
    /// The 0-based program counter where exploration of the body started.
    pub entry_pc: Option<u128>,
    /// How far the body was analyzed.
    pub status: BodyStatus,
    /// Why the function failed, when `status` is `Failed`.
    pub failure: Option<String>,
    /// Steps, branches, saved state bytes and dropped paths by reason.
    pub stats: ExecStats,
    /// Effectful opcodes of the analyzed paths, sorted by PC; kept when the analysis fails part
    /// way.
    pub effects: Vec<Effect>,
    /// 0-based PCs of every analyzed instruction, sorted.
    pub executed_pcs: Vec<u128>,
}

impl FunctionReport {
    /// The report for a function whose body was symbolically executed into `trace`.
    pub(crate) fn explored(selector: &str, trace: &VMTrace, stats: ExecStats) -> Self {
        Self {
            selector: selector.to_string(),
            entry_pc: Some(trace.instruction.saturating_sub(1)),
            status: BodyStatus::from_stats(&stats),
            failure: None,
            stats,
            effects: Vec::new(),
            executed_pcs: Vec::new(),
        }
    }

    /// The report for a function that could not be analyzed.
    pub(crate) fn failed(selector: &str, failure: String) -> Self {
        Self {
            selector: selector.to_string(),
            entry_pc: None,
            status: BodyStatus::Failed,
            failure: Some(failure),
            stats: ExecStats::default(),
            effects: Vec::new(),
            executed_pcs: Vec::new(),
        }
    }
}
