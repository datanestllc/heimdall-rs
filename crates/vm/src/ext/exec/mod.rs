mod jump_frame;
mod util;

use crate::{
    core::{
        stack::Stack,
        vm::{State, VM},
    },
    ext::exec::{
        jump_frame::JumpFrame,
        util::{
            historical_diffs_approximately_equal, jump_condition_appears_recursive,
            jump_condition_contains_mutated_memory_access,
            jump_condition_contains_mutated_storage_access,
            jump_stack_depth_less_than_max_stack_depth, stack_contains_too_many_items,
            stack_contains_too_many_of_the_same_item, stack_diff, stack_item_source_depth_too_deep,
            stack_position_shows_pattern,
        },
    },
};
use eyre::Result;
use hashbrown::HashMap;
use heimdall_common::utils::strings::decode_hex;
use std::{collections::BTreeMap, time::Instant};
use tracing::{trace, warn};

/// Why a path of the symbolic execution tree was cut short.
///
/// Every cut is recorded as a [`Truncation`] marker on the [`VMTrace`] node where it happened, so a
/// dropped path stays visible to the analyzer and to callers instead of disappearing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TruncationReason {
    /// The VM step budget of the function (or the remainder of the job budget) ran out.
    StepBudget,
    /// The branch budget ran out: the side of a JUMPI that the VM did not take was not explored.
    BranchBudget,
    /// The budget of saved VM state bytes ran out (every step keeps a [`State`] snapshot).
    StateBudget,
    /// The VM could not execute an instruction on this path (stack underflow, bad jump, ...).
    StepError,
    /// A loop-detection heuristic stopped the path at a JUMPI that was already explored.
    LoopHeuristic,
    /// The wall-clock deadline fired. This is an emergency stop and is not deterministic.
    WallClock,
}

impl TruncationReason {
    /// A stable snake_case name for the reason.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::StepBudget => "step_budget",
            Self::BranchBudget => "branch_budget",
            Self::StateBudget => "state_budget",
            Self::StepError => "step_error",
            Self::LoopHeuristic => "loop_heuristic",
            Self::WallClock => "wall_clock",
        }
    }
}

impl std::fmt::Display for TruncationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Marks a [`VMTrace`] node whose path was cut short.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Truncation {
    /// Why the path was cut.
    pub reason: TruncationReason,
    /// The 0-based program counter of the instruction that was not executed (for a loop heuristic:
    /// the JUMPI whose children were not explored).
    pub pc: u128,
}

/// Limits for one symbolic execution.
///
/// The step, branch and state budgets are counted in deterministic units, so the explored tree
/// does not depend on machine load. The deadline is only an emergency stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecBudget {
    /// Maximum VM steps across all paths (including the steps to reach the entry point).
    pub max_steps: u64,
    /// Maximum JUMPI forks.
    pub max_branches: u32,
    /// Maximum approximate bytes of saved VM states (see [`approximate_state_bytes`]).
    pub max_state_bytes: u64,
    /// Wall-clock emergency stop.
    pub deadline: Option<Instant>,
}

impl ExecBudget {
    /// No deterministic limit and no deadline.
    pub fn unlimited() -> Self {
        Self {
            max_steps: u64::MAX,
            max_branches: u32::MAX,
            max_state_bytes: u64::MAX,
            deadline: None,
        }
    }

    /// Only a wall-clock deadline (the behaviour of the `timeout` based entry points).
    pub fn with_deadline(deadline: Instant) -> Self {
        Self { deadline: Some(deadline), ..Self::unlimited() }
    }
}

impl Default for ExecBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// What one symbolic execution consumed and dropped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecStats {
    /// VM steps executed.
    pub steps: u64,
    /// JUMPI forks explored.
    pub branches: u32,
    /// Approximate bytes of saved VM states.
    pub state_bytes: u64,
    /// Paths cut short, by reason.
    pub truncations: BTreeMap<TruncationReason, u32>,
}

impl ExecStats {
    /// Number of paths cut short for the given reason.
    pub fn truncated(&self, reason: TruncationReason) -> u32 {
        self.truncations.get(&reason).copied().unwrap_or(0)
    }
}

/// Approximate heap bytes of one saved [`State`]: memory, stack frames, storage and logs.
pub fn approximate_state_bytes(state: &State) -> u64 {
    let memory = state.memory.memory.len() as u64;
    let stack = state.stack.size() as u64 * 96;
    let storage = (state.storage.storage.len() + state.storage.transient.len()) as u64 * 64;
    let events = state
        .events
        .iter()
        .map(|log| log.topics.len() as u64 * 32 + log.data.len() as u64 + 48)
        .sum::<u64>();
    256 + memory + stack + storage + events
}

/// Budget and counters shared by every path of one symbolic execution.
struct ExecContext {
    budget: ExecBudget,
    stats: ExecStats,
}

impl ExecContext {
    fn new(budget: ExecBudget) -> Self {
        Self { budget, stats: ExecStats::default() }
    }

    /// The first exhausted budget, if any.
    fn exhausted(&self) -> Option<TruncationReason> {
        if self.budget.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Some(TruncationReason::WallClock);
        }
        if self.stats.steps >= self.budget.max_steps {
            return Some(TruncationReason::StepBudget);
        }
        if self.stats.state_bytes >= self.budget.max_state_bytes {
            return Some(TruncationReason::StateBudget);
        }
        None
    }

    /// Mark `trace` as cut short at `pc` and count it.
    fn truncate(&mut self, mut trace: VMTrace, reason: TruncationReason, pc: u128) -> VMTrace {
        *self.stats.truncations.entry(reason).or_insert(0) += 1;
        trace.truncated = Some(Truncation { reason, pc });
        trace
    }
}

/// Represents a trace of virtual machine execution including operations and child calls
///
/// VMTrace is used to track the operations performed during VM execution, including
/// any nested calls that occur during execution (stored in the `children` field).
#[derive(Clone, Debug, Default)]
pub struct VMTrace {
    /// The instruction pointer at the start of this trace
    pub instruction: u128,

    /// The amount of gas used by this execution trace
    pub gas_used: u128,

    /// The sequence of VM states recorded during execution
    pub operations: Vec<State>,

    /// Child traces resulting from internal calls (CALL, DELEGATECALL, etc.)
    pub children: Vec<VMTrace>,

    /// Set when this path was cut short; `operations` holds what ran before the cut.
    pub truncated: Option<Truncation>,
}

impl VM {
    /// Run symbolic execution on a given function selector within a contract
    pub fn symbolic_exec_selector(
        &mut self,
        selector: &str,
        entry_point: u128,
        timeout: Instant,
    ) -> Result<(VMTrace, u32)> {
        let (trace, stats) = self.symbolic_exec_selector_with_budget(
            selector,
            entry_point,
            ExecBudget::with_deadline(timeout),
        )?;
        Ok((trace, stats.branches))
    }

    /// Run symbolic execution on a given function selector within a contract, under `budget`.
    ///
    /// The steps taken to reach `entry_point` count against the step budget. A path that runs
    /// out of budget is kept as a [`Truncation`] marker, never dropped.
    pub fn symbolic_exec_selector_with_budget(
        &mut self,
        selector: &str,
        entry_point: u128,
        budget: ExecBudget,
    ) -> Result<(VMTrace, ExecStats)> {
        self.calldata = decode_hex(selector)?;
        let mut ctx = ExecContext::new(budget);

        // step through the bytecode until we reach the entry point
        while self.bytecode.len() >= self.instruction as usize && (self.instruction <= entry_point)
        {
            if let Some(reason) = ctx.exhausted() {
                let pc = self.instruction.saturating_sub(1);
                let root = VMTrace { instruction: self.instruction, ..Default::default() };
                return Ok((ctx.truncate(root, reason, pc), ctx.stats));
            }
            ctx.stats.steps += 1;
            match self.step() {
                Ok(_) => {}
                Err(e) => {
                    warn!("failed to reach entry point for selector 0x{}: {:?}", selector, e);
                    return Err(e);
                }
            }

            // this shouldn't be necessary, but it's safer to have it
            if self.exitcode != 255 || !self.returndata.is_empty() {
                break;
            }
        }

        trace!("beginning symbolic execution for selector 0x{}", selector);

        // the VM is at the function entry point, begin tracing
        let trace = self.recursive_map(&mut ctx, &mut HashMap::new());
        if trace.operations.is_empty() && trace.children.is_empty() {
            warn!("symbolic execution returned no valid traces for selector 0x{}", selector);
        }
        Ok((trace, ctx.stats))
    }

    /// Performs symbolic execution on the entire contract to map out control flow
    ///
    /// This method executes the VM symbolically, starting from the beginning of the bytecode,
    /// to build a comprehensive map of all possible execution paths within the contract.
    /// It tracks branching and records operation states throughout execution.
    ///
    /// # Arguments
    /// * `timeout` - An Instant representing when execution should time out
    ///
    /// # Returns
    /// * A Result containing a tuple with:
    ///   - The execution trace (VMTrace)
    ///   - The number of branches encountered during execution
    pub fn symbolic_exec(&mut self, timeout: Instant) -> Result<(VMTrace, u32)> {
        let (trace, stats) = self.symbolic_exec_with_budget(ExecBudget::with_deadline(timeout))?;
        Ok((trace, stats.branches))
    }

    /// Performs symbolic execution on the entire contract under `budget`; see
    /// [`VM::symbolic_exec`].
    pub fn symbolic_exec_with_budget(
        &mut self,
        budget: ExecBudget,
    ) -> Result<(VMTrace, ExecStats)> {
        trace!("beginning contract-wide symbolic execution");

        let mut ctx = ExecContext::new(budget);
        let trace = self.recursive_map(&mut ctx, &mut HashMap::new());
        if trace.operations.is_empty() && trace.children.is_empty() {
            warn!("symbolic execution returned no valid traces");
        }
        Ok((trace, ctx.stats))
    }

    fn recursive_map(
        &mut self,
        ctx: &mut ExecContext,
        handled_jumps: &mut HashMap<JumpFrame, Vec<Stack>>,
    ) -> VMTrace {
        let vm = self;

        // create a new VMTrace object
        // this will essentially be a tree of executions, with each branch being a different path
        // that symbolic execution discovered
        let mut vm_trace = VMTrace {
            instruction: vm.instruction,
            gas_used: 0,
            operations: Vec::new(),
            children: Vec::new(),
            truncated: None,
        };

        // step through the bytecode until we find a JUMPI instruction
        while vm.bytecode.len() >= vm.instruction as usize {
            let pc = vm.instruction.saturating_sub(1);

            // a path that runs out of budget is kept, with a marker, instead of being dropped
            if let Some(reason) = ctx.exhausted() {
                return ctx.truncate(vm_trace, reason, pc);
            }

            // execute the next instruction. if the instruction fails, mark this path
            ctx.stats.steps += 1;
            let state = match vm.step() {
                Ok(state) => state,
                Err(e) => {
                    warn!("executing branch failed during step: {:?}", e);
                    return ctx.truncate(vm_trace, TruncationReason::StepError, pc);
                }
            };
            ctx.stats.state_bytes += approximate_state_bytes(&state);
            let last_instruction = state.last_instruction.clone();

            // update vm_trace
            vm_trace.operations.push(state);
            vm_trace.gas_used = vm.gas_used;

            // if we encounter a JUMP(I), create children taking both paths and break
            if last_instruction.opcode == 0x57 {
                trace!(
                    "found branch due to JUMP{} instruction at {}",
                    if last_instruction.opcode == 0x57 { "I" } else { "" },
                    last_instruction.instruction
                );

                let jump_condition: Option<String> =
                    last_instruction.input_operations.get(1).map(|op| op.solidify());
                let jump_taken =
                    last_instruction.inputs.get(1).map(|op| !op.is_zero()).unwrap_or(true);
                let jumpi_pc = last_instruction.instruction.saturating_sub(1);

                // build hashable jump frame
                let jump_frame = JumpFrame::new(
                    last_instruction.instruction,
                    last_instruction.inputs[0],
                    vm.stack.size(),
                    jump_taken,
                );

                // if the stack contains too many items, it's probably a loop
                if stack_contains_too_many_items(&vm.stack) {
                    return ctx.truncate(vm_trace, TruncationReason::LoopHeuristic, jumpi_pc);
                }

                // if the stack has over 16 items of the same source, it's probably a loop
                if stack_contains_too_many_of_the_same_item(&vm.stack) {
                    return ctx.truncate(vm_trace, TruncationReason::LoopHeuristic, jumpi_pc);
                }

                // if any item on the stack has a depth > 16, it's probably a loop (because of stack
                // too deep)
                if stack_item_source_depth_too_deep(&vm.stack) {
                    return ctx.truncate(vm_trace, TruncationReason::LoopHeuristic, jumpi_pc);
                }

                // if the jump stack depth is less than the max stack depth of all previous matching
                // jumps, it's probably a loop
                if jump_stack_depth_less_than_max_stack_depth(&jump_frame, handled_jumps) {
                    return ctx.truncate(vm_trace, TruncationReason::LoopHeuristic, jumpi_pc);
                }

                // perform heuristic checks on historical stacks
                match handled_jumps.get_mut(&jump_frame) {
                    Some(historical_stacks) => {
                        // for every stack that we have encountered for this jump, perform some
                        // heuristic checks to determine if this might be a loop
                        if historical_stacks.iter().any(|hist_stack| {
                            if let Some(jump_condition) = &jump_condition {

                                // check if any historical stack is the same as the current stack
                                if hist_stack == &vm.stack {
                                    trace!(
                                        "jump matches loop-detection heuristic: 'jump_path_already_handled'"
                                    );
                                    return true
                                }

                                // calculate the difference of the current stack and the historical stack
                                let stack_diff = stack_diff(&vm.stack, hist_stack);
                                if stack_diff.is_empty() {
                                    // the stack_diff is empty (the stacks are the same), so we've
                                    // already handled this path
                                    trace!(
                                        "jump matches loop-detection heuristic: 'stack_diff_is_empty'"
                                    );
                                    return true
                                }

                                trace!("stack diff: [{}]", stack_diff.iter().map(|frame| format!("{}", frame.value)).collect::<Vec<String>>().join(", "));

                                // check if the jump condition appears to be recursive
                                if jump_condition_appears_recursive(&stack_diff, jump_condition) {
                                    return true
                                }

                                // check for mutated memory accesses in the jump condition
                                if jump_condition_contains_mutated_memory_access(
                                    &stack_diff,
                                    jump_condition,
                                ) {
                                    return true
                                }

                                // check for mutated memory accesses in the jump condition
                                if jump_condition_contains_mutated_storage_access(
                                    &stack_diff,
                                    jump_condition,
                                ) {
                                    return true
                                }

                            }
                            false
                        }) {
                            trace!("jump terminated.");
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                            return ctx.truncate(vm_trace, TruncationReason::LoopHeuristic, jumpi_pc);
                        }

                        // check if any stack position shows a consistent pattern
                        // (increasing/decreasing/alternating)
                        if stack_position_shows_pattern(&vm.stack, historical_stacks) {
                            trace!("jump terminated.");
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                            return ctx.truncate(
                                vm_trace,
                                TruncationReason::LoopHeuristic,
                                jumpi_pc,
                            );
                        }

                        if historical_diffs_approximately_equal(&vm.stack, historical_stacks) {
                            trace!("jump terminated.");
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                            return ctx.truncate(
                                vm_trace,
                                TruncationReason::LoopHeuristic,
                                jumpi_pc,
                            );
                        }

                        {
                            trace!(
                                "adding historical stack {} to jump frame {:?}",
                                &format!("{:#016x?}", vm.stack.hash()),
                                jump_frame
                            );
                            trace!(
                                " - jump condition: {:?}\n        - stack: {}\n        - historical stacks: {}",
                                jump_condition,
                                vm.stack,
                                historical_stacks.iter().map(|stack| format!("{stack}")).collect::<Vec<String>>().join("\n            - ")
                            );

                            // this key exists, but the stack is different, so the jump is new
                            historical_stacks.push(vm.stack.clone());
                        }
                    }
                    None => {
                        // this key doesnt exist, so the jump is new
                        trace!("added new jump frame: {:?}", jump_frame);
                        handled_jumps.insert(jump_frame, vec![vm.stack.clone()]);
                    }
                }

                if last_instruction.opcode == 0x56 {
                    continue;
                }
                // the destination of the JUMPI and the fall-through, as 1-based instructions
                let destination = u128::try_from(last_instruction.inputs[0])
                    .unwrap_or(u128::MAX)
                    .saturating_add(1);
                let fallthrough = last_instruction.instruction + 1;

                // out of branches: follow the side the VM took, and keep the other side as a marker
                // so the JUMPI keeps both children and its condition
                if ctx.stats.branches >= ctx.budget.max_branches {
                    let other = VMTrace {
                        instruction: if jump_taken { fallthrough } else { destination },
                        ..Default::default()
                    };
                    let other = ctx.truncate(other, TruncationReason::BranchBudget, jumpi_pc);
                    vm_trace.children.push(other);
                    vm_trace.children.push(vm.recursive_map(ctx, handled_jumps));
                    break;
                }


                // we didnt break out, so now we crate branching paths to cover all possibilities
                ctx.stats.branches += 1;
                trace!(
                    "creating branching paths at instructions {} (JUMPDEST) and {} (CONTINUE)",
                    last_instruction.inputs[0],
                    last_instruction.instruction + 1
                );

                // we need to create a trace for the path that wasn't taken.
                let mut trace_vm = vm.clone();
                trace_vm.instruction = if jump_taken { fallthrough } else { destination };
                vm_trace.children.push(trace_vm.recursive_map(ctx, handled_jumps));

                // push the current path onto the stack
                vm_trace.children.push(vm.recursive_map(ctx, handled_jumps));
                break;
            }

            // when the vm exits, this path is complete
            if vm.exitcode != 255 || !vm.returndata.is_empty() {
                break;
            }
        }

        vm_trace
    }
}

#[cfg(test)]
mod tests {
    use alloy::primitives::Address;
    use heimdall_common::utils::strings::decode_hex;

    use super::*;
    use crate::core::opcodes::{JUMPI, SSTORE};

    fn vm(code: &str) -> VM {
        VM::new(
            &decode_hex(code).expect("bad hex"),
            &[],
            Address::default(),
            Address::default(),
            Address::default(),
            0,
            u128::MAX,
        )
    }

    fn truncated_nodes(trace: &VMTrace) -> Vec<Truncation> {
        let mut out = trace.truncated.into_iter().collect::<Vec<_>>();
        for child in &trace.children {
            out.extend(truncated_nodes(child));
        }
        out
    }

    #[test]
    fn step_budget_cuts_an_endless_path_with_a_marker() {
        // 0: JUMPDEST  1: PUSH1 0  3: JUMP  -- jumps back to 0 forever
        let budget = ExecBudget { max_steps: 100, ..ExecBudget::unlimited() };
        let (trace, stats) = vm("5b600056").symbolic_exec_with_budget(budget).unwrap();
        assert_eq!(stats.steps, 100);
        assert_eq!(trace.operations.len(), 100);
        assert_eq!(trace.truncated.map(|t| t.reason), Some(TruncationReason::StepBudget));
        assert_eq!(stats.truncated(TruncationReason::StepBudget), 1);

        // the same budget gives the same tree, whatever the machine load
        let (again, _) = vm("5b600056").symbolic_exec_with_budget(budget).unwrap();
        assert_eq!(again.operations.len(), trace.operations.len());
        assert_eq!(again.truncated, trace.truncated);
    }

    #[test]
    fn step_error_keeps_the_failed_child_and_the_jumpi() {
        // 0: PUSH1 0  2: CALLDATALOAD  3: PUSH1 0x0a  5: JUMPI
        // 6: POP (stack underflow on the fall-through)  7..9: STOP  10: JUMPDEST  11: STOP
        let (trace, stats) = vm("600035600a575000000000005b00")
            .symbolic_exec_with_budget(ExecBudget::unlimited())
            .unwrap();
        assert_eq!(trace.operations.last().unwrap().last_instruction.opcode, JUMPI);
        assert_eq!(trace.children.len(), 2, "both children of the JUMPI are kept");
        let failed = trace.children.iter().find(|c| c.truncated.is_some()).unwrap();
        assert_eq!(
            failed.truncated,
            Some(Truncation { reason: TruncationReason::StepError, pc: 6 })
        );
        assert_eq!(failed.instruction, 7);
        assert_eq!(stats.truncated(TruncationReason::StepError), 1);
    }

    #[test]
    fn branch_budget_follows_the_vm_and_marks_the_other_side() {
        // as above, but the fall-through is a clean STOP
        let budget = ExecBudget { max_branches: 0, ..ExecBudget::unlimited() };
        let (trace, stats) =
            vm("600035600a570000000000005b00").symbolic_exec_with_budget(budget).unwrap();
        assert_eq!(stats.branches, 0);
        assert_eq!(trace.children.len(), 2);
        // calldata is empty, so the VM falls through; the jump destination is the marked side
        let other = &trace.children[0];
        assert_eq!(other.instruction, 11);
        assert_eq!(
            other.truncated,
            Some(Truncation { reason: TruncationReason::BranchBudget, pc: 5 })
        );
        assert!(trace.children[1].truncated.is_none());
    }

    #[test]
    fn loop_heuristic_keeps_the_segment_before_the_cut() {
        // 0: JUMPDEST  1: PUSH1 1  3: PUSH1 0  5: SSTORE  6: PUSH1 0  8: CALLDATALOAD
        // 9: PUSH1 0  11: JUMPI (back to 0)  12: STOP
        let (trace, stats) = vm("5b600160005560003560005700")
            .symbolic_exec_with_budget(ExecBudget::unlimited())
            .unwrap();
        let cuts = truncated_nodes(&trace);
        assert_eq!(cuts, vec![Truncation { reason: TruncationReason::LoopHeuristic, pc: 11 }]);
        assert_eq!(stats.truncated(TruncationReason::LoopHeuristic), 1);
        let looped = trace.children.iter().find(|c| c.truncated.is_some()).unwrap();
        assert!(
            looped.operations.iter().any(|s| s.last_instruction.opcode == SSTORE),
            "the SSTORE executed before the cut stays in the trace"
        );
    }
}
