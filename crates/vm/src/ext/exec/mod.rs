mod jump_frame;
mod util;

use crate::{
    core::{
        opcodes::{
            WrappedInput, WrappedOpcode, ADD, AND, CALLDATALOAD, CALLDATASIZE, DIV, EQ, GT, ISZERO,
            JUMPDEST, LT, MOD, MUL, NOT, OR, PUSH0, PUSH1, PUSH32, SGT, SHL, SHR, SLT, SUB, XOR,
        },
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
use crate::ext::selectors::MAX_DISPATCH_STEPS;
use alloy::primitives::U256;
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
    /// Loop detection by calling context. Off: upstream heuristics, which compare a JUMPI with
    /// its earlier visits on any path by stack shape, so two paths through shared code (an
    /// internal function called from two places) cut each other as "loops" and the second
    /// caller's continuation is lost. On: a JUMPI is explored once per calling context, i.e. per
    /// set of return addresses on the stack (with the same stack depth and direction taken); a
    /// later visit in the same context is cut, on any path. A loop body is then explored once,
    /// and a helper once per call site.
    pub context_loops: bool,
}

impl ExecBudget {
    /// No deterministic limit and no deadline.
    pub fn unlimited() -> Self {
        Self {
            max_steps: u64::MAX,
            max_branches: u32::MAX,
            max_state_bytes: u64::MAX,
            deadline: None,
            context_loops: false,
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
    /// Paths cut by a loop heuristic, by heuristic.
    pub loop_heuristics: BTreeMap<&'static str, u32>,
}

impl ExecStats {
    /// Number of paths cut short for the given reason.
    pub fn truncated(&self, reason: TruncationReason) -> u32 {
        self.truncations.get(&reason).copied().unwrap_or(0)
    }
}

/// Whether a JUMPI condition reads calldata, and only through `CALLDATASIZE` or the selector
/// (`calldataload(0)` shifted right by 224 bits, or divided by 2^224), on top of constants and
/// pure arithmetic. Such a condition is decided by the calldata of a non-selector entry.
pub fn is_decided_by_dispatch(condition: &WrappedOpcode) -> bool {
    fn constant(input: &WrappedInput) -> Option<U256> {
        match input {
            WrappedInput::Raw(value) => Some(*value),
            WrappedInput::Opcode(op) if (PUSH0..=PUSH32).contains(&op.opcode) => {
                op.inputs.first().map_or(Some(U256::ZERO), constant)
            }
            WrappedInput::Opcode(_) => None,
        }
    }
    fn is_selector_word(op: &WrappedOpcode) -> bool {
        op.opcode == CALLDATALOAD && op.inputs.first().and_then(constant) == Some(U256::ZERO)
    }
    fn walk(op: &WrappedOpcode, reads: &mut bool) -> bool {
        let operand = |i: usize| match op.inputs.get(i) {
            Some(WrappedInput::Opcode(inner)) => Some(&**inner),
            _ => None,
        };
        match op.opcode {
            PUSH0..=PUSH32 => true,
            CALLDATASIZE => {
                *reads = true;
                true
            }
            // calldataload(0) >> 224 and calldataload(0) / 2^224: the selector
            SHR if constant(&op.inputs[0]) == Some(U256::from(224)) &&
                operand(1).is_some_and(is_selector_word) =>
            {
                *reads = true;
                true
            }
            DIV if op.inputs.get(1).and_then(constant) == Some(U256::from(1) << 224) &&
                operand(0).is_some_and(is_selector_word) =>
            {
                *reads = true;
                true
            }
            ADD | SUB | MUL | DIV | MOD | LT | GT | SLT | SGT | EQ | ISZERO | AND | OR | XOR |
            NOT | SHL | SHR => op.inputs.iter().all(|input| match input {
                WrappedInput::Raw(_) => true,
                WrappedInput::Opcode(inner) => walk(inner, reads),
            }),
            _ => false,
        }
    }
    let mut reads = false;
    !condition.is_too_large_to_expand() && walk(condition, &mut reads) && reads
}

/// A JUMPI visit as loop detection with `context_loops` sees it: the jump, the direction the VM
/// takes, the stack depth, and the calling context, i.e. the stack items that are code addresses
/// (JUMPDESTs pushed as constants: return addresses of internal functions). The same helper
/// called from two call sites differs in its return address; one loop iteration and the next do
/// not; data values (counters, amounts) are not part of it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ContextKey {
    pc: u128,
    jumpdest: U256,
    jump_taken: bool,
    stack_depth: usize,
    context: Vec<U256>,
}

impl ContextKey {
    fn new(frame: &JumpFrame, stack: &Stack, bytecode: &[u8]) -> Self {
        let context = stack
            .stack
            .iter()
            .filter(|item| (PUSH1..=PUSH32).contains(&item.operation.opcode))
            .filter(|item| {
                usize::try_from(item.value)
                    .ok()
                    .and_then(|pc| bytecode.get(pc))
                    .is_some_and(|op| *op == JUMPDEST)
            })
            .map(|item| item.value)
            .collect();
        Self {
            pc: frame.pc,
            jumpdest: frame.jumpdest,
            jump_taken: frame.jump_taken,
            stack_depth: frame.stack_depth,
            context,
        }
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
    /// Follow only the side the VM takes at a JUMPI decided by the selector or the calldata size
    /// (see [`VM::symbolic_exec_entry`]).
    follow_dispatch: bool,
    /// JUMPI visits explored so far, by calling context (`context_loops`).
    explored_contexts: hashbrown::HashSet<ContextKey>,
}

impl ExecContext {
    fn new(budget: ExecBudget) -> Self {
        Self {
            budget,
            stats: ExecStats::default(),
            follow_dispatch: false,
            explored_contexts: hashbrown::HashSet::new(),
        }
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

    /// Cut `trace` short at `pc` by the named loop heuristic, and count it.
    fn cut_loop(&mut self, trace: VMTrace, heuristic: &'static str, pc: u128) -> VMTrace {
        *self.stats.loop_heuristics.entry(heuristic).or_insert(0) += 1;
        self.truncate(trace, TruncationReason::LoopHeuristic, pc)
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

    /// Run symbolic execution for a function whose body starts at `entry_pc`, a 0-based bytecode
    /// offset (e.g. evmole's `bytecodeOffset`), with the given calldata.
    ///
    /// The VM first executes concretely from its current position until the next instruction is
    /// the one at `entry_pc`, for at most [`MAX_DISPATCH_STEPS`] steps (they also count against the
    /// step budget). Symbolic execution starts exactly there. If the VM halts, fails or runs out of
    /// steps before, the error says `entry_not_reached`.
    ///
    /// Heimdall's `instruction` counts from 1; `entry_pc` counts from 0, like a program counter.
    pub fn symbolic_exec_from_entry(
        &mut self,
        calldata: &[u8],
        entry_pc: u128,
        budget: ExecBudget,
    ) -> Result<(VMTrace, ExecStats)> {
        self.calldata = calldata.to_vec();
        let mut ctx = ExecContext::new(budget);
        let entry_instruction = entry_pc.saturating_add(1);

        let mut steps = 0usize;
        while self.instruction != entry_instruction {
            if steps >= MAX_DISPATCH_STEPS {
                eyre::bail!("entry_not_reached: no pc {entry_pc:#x} within {MAX_DISPATCH_STEPS} steps");
            }
            if self.bytecode.len() < self.instruction as usize {
                eyre::bail!("entry_not_reached: execution left the code before pc {entry_pc:#x}");
            }
            if let Some(reason) = ctx.exhausted() {
                // nothing of the body was explored; it starts at the entry all the same
                let root = VMTrace { instruction: entry_instruction, ..Default::default() };
                return Ok((ctx.truncate(root, reason, entry_pc), ctx.stats));
            }
            let pc = self.instruction.saturating_sub(1);
            steps += 1;
            ctx.stats.steps += 1;
            if let Err(e) = self.step() {
                eyre::bail!("entry_not_reached: execution failed at pc {pc:#x}: {e}");
            }
            if self.exitcode != 255 || !self.returndata.is_empty() {
                eyre::bail!("entry_not_reached: execution halted at pc {pc:#x}");
            }
        }

        let trace = self.recursive_map(&mut ctx, &mut HashMap::new());
        Ok((trace, ctx.stats))
    }

    /// Performs symbolic execution from pc 0 with the given calldata, for an entry that is not a
    /// selector: empty calldata (what `receive` runs) or a selector the dispatcher does not match
    /// (what `fallback` runs).
    ///
    /// A JUMPI whose condition reads calldata only through `CALLDATASIZE` or the selector
    /// (`calldataload(0) >> 224`) is decided by that calldata, so only the side the VM takes is
    /// followed: the other side belongs to other entries, and it is not a dropped path. Every
    /// other JUMPI forks as usual.
    pub fn symbolic_exec_entry(
        &mut self,
        calldata: &[u8],
        budget: ExecBudget,
    ) -> Result<(VMTrace, ExecStats)> {
        self.calldata = calldata.to_vec();
        let mut ctx = ExecContext::new(budget);
        ctx.follow_dispatch = true;
        let trace = self.recursive_map(&mut ctx, &mut HashMap::new());
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
                    return ctx.cut_loop(vm_trace, "stack_too_large", jumpi_pc);
                }

                // if the stack has over 16 items of the same source, it's probably a loop
                if stack_contains_too_many_of_the_same_item(&vm.stack) {
                    return ctx.cut_loop(vm_trace, "repeated_stack_items", jumpi_pc);
                }

                // if any item on the stack has a depth > 16, it's probably a loop (because of stack
                // too deep)
                if stack_item_source_depth_too_deep(&vm.stack) {
                    return ctx.cut_loop(vm_trace, "expression_too_deep", jumpi_pc);
                }

                if ctx.budget.context_loops {
                    let key = ContextKey::new(&jump_frame, &vm.stack, &vm.bytecode);
                    if !ctx.explored_contexts.insert(key) {
                        return ctx.cut_loop(vm_trace, "explored_context", jumpi_pc);
                    }
                } else {
                // if the jump stack depth is less than the max stack depth of all previous matching
                // jumps, it's probably a loop
                if jump_stack_depth_less_than_max_stack_depth(&jump_frame, handled_jumps) {
                    return ctx.cut_loop(vm_trace, "shallower_stack", jumpi_pc);
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
                            return ctx.cut_loop(vm_trace, "seen_or_changing_condition", jumpi_pc);
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
                            return ctx.cut_loop(vm_trace, "stack_pattern", jumpi_pc);
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
                            return ctx.cut_loop(vm_trace, "similar_stack_diffs", jumpi_pc);
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
                        handled_jumps.insert(jump_frame.clone(), vec![vm.stack.clone()]);
                    }
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

                // dispatch on the calldata of this entry: the other side is another entry
                if ctx.follow_dispatch &&
                    last_instruction.input_operations.get(1).is_some_and(is_decided_by_dispatch)
                {
                    vm_trace.children.push(vm.recursive_map(ctx, handled_jumps));
                    break;
                }

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
    fn entry_exploration_follows_dispatch_but_forks_on_arguments() {
        // 0: CALLDATASIZE ISZERO PUSH1 27 JUMPI            -- empty calldata: receive at 27
        // 5: PUSH1 0 CALLDATALOAD PUSH1 0xe0 SHR PUSH4 11111111 EQ PUSH1 29 JUMPI
        // 20: PUSH1 4 CALLDATALOAD PUSH1 31 JUMPI STOP     -- forks on an argument word
        // 27: JUMPDEST STOP  29: JUMPDEST STOP  31: JUMPDEST STOP
        let code = "3615601b5760003560e01c631111111114601d57600435601f57005b005b005b00";
        let fallback = [0xff, 0xff, 0xff, 0xff];
        let (trace, stats) =
            vm(code).symbolic_exec_entry(&fallback, ExecBudget::unlimited()).unwrap();
        // the two dispatch JUMPIs are followed, only the argument JUMPI forks
        assert_eq!(stats.branches, 1);
        assert_eq!(trace.children.len(), 1);
        assert_eq!(trace.children[0].children.len(), 1);
        assert_eq!(trace.children[0].children[0].children.len(), 2);

        // empty calldata goes to the receive block
        let (trace, stats) = vm(code).symbolic_exec_entry(&[], ExecBudget::unlimited()).unwrap();
        assert_eq!(stats.branches, 0);
        assert_eq!(trace.children.len(), 1);
        assert_eq!(trace.children[0].instruction, 28);
    }

    fn executes(trace: &VMTrace, opcode: u8) -> bool {
        trace.operations.iter().any(|s| s.last_instruction.opcode == opcode) ||
            trace.children.iter().any(|child| executes(child, opcode))
    }

    #[test]
    fn context_loops_explore_a_helper_once_per_call_site() {
        // 0: PUSH1 0xaa PUSH1 7 PUSH1 21 JUMP          -- helper(), returning to 7, stack depth 2
        // 7: JUMPDEST POP PUSH1 14 PUSH1 21 JUMP        -- helper() again, returning to 14, depth 1
        // 14: JUMPDEST PUSH1 1 PUSH1 0 SSTORE STOP      -- only after the second call
        // 21: JUMPDEST PUSH1 0 CALLDATALOAD PUSH1 28 JUMPI  28: JUMPDEST JUMP   -- the helper
        let code = "60aa60076015565b50600e6015565b6001600055005b600035601c575b56";

        // upstream: the second visit of the helper's JUMPI has a shallower stack than the first,
        // so it is taken for a loop, and the code after the second call is never reached
        let (trace, stats) = vm(code).symbolic_exec_with_budget(ExecBudget::unlimited()).unwrap();
        assert!(!executes(&trace, SSTORE));
        assert_eq!(stats.loop_heuristics.get("shallower_stack"), Some(&2));

        // by calling context, the two calls return to different places, so both are explored
        let budget = ExecBudget { context_loops: true, ..ExecBudget::unlimited() };
        let (trace, stats) = vm(code).symbolic_exec_with_budget(budget).unwrap();
        assert!(executes(&trace, SSTORE));
        // the sibling path of the first call reaches the second call in a context already
        // explored: that is the only cut
        assert_eq!(stats.loop_heuristics.keys().collect::<Vec<_>>(), vec![&"explored_context"]);

        // a real loop is still cut, after one pass
        let budget = ExecBudget { context_loops: true, ..ExecBudget::unlimited() };
        let (_, stats) = vm("5b600160005560003560005700").symbolic_exec_with_budget(budget).unwrap();
        assert_eq!(stats.loop_heuristics.get("explored_context"), Some(&1));
    }

    #[test]
    fn an_entry_out_of_budget_still_starts_at_the_entry() {
        // the dispatcher of `entry_exploration_follows_dispatch_but_forks_on_arguments`
        let code = "3615601b5760003560e01c631111111114601d57600435601f57005b005b005b00";
        let budget = ExecBudget { max_steps: 0, ..ExecBudget::unlimited() };
        let (trace, _) = vm(code).symbolic_exec_from_entry(&[0x11; 4], 29, budget).unwrap();
        assert_eq!(trace.instruction, 30);
        assert_eq!(trace.truncated, Some(Truncation { reason: TruncationReason::StepBudget, pc: 29 }));

        let (trace, _) =
            vm(code).symbolic_exec_from_entry(&[0x11; 4], 29, ExecBudget::unlimited()).unwrap();
        assert_eq!(trace.operations[0].last_instruction.instruction, 30);
        // an entry the dispatcher never jumps to for this calldata
        let wrong = vm(code).symbolic_exec_from_entry(&[0x22; 4], 29, ExecBudget::unlimited());
        assert!(wrong.unwrap_err().to_string().contains("entry_not_reached"));
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
