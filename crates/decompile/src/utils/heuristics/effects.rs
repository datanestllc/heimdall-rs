//! Effects of a function, recorded at opcode time from the state of the path that executed the
//! opcode (not from the statements, which the postprocessors rewrite).

use alloy::primitives::U256;
use alloy_dyn_abi::{DynSolType, DynSolValue};
use futures::future::BoxFuture;
use hashbrown::HashMap;
use heimdall_vm::core::{
    opcodes::{self, WrappedInput, WrappedOpcode},
    vm::State,
};

use crate::{
    core::analyze::AnalyzerState,
    interfaces::{AnalyzedFunction, Effect, EffectOperand, RevertReason},
    Error,
};

/// Offsets and sizes past this are not tracked (heimdall's memory model caps at 2^16 anyway).
const MAX_TRACKED_MEMORY: usize = 1 << 20;

/// How deep memory provenance follows MCOPY chains, and mapping roots follow nested keys.
const MAX_FOLLOW_DEPTH: u32 = 8;

/// Where the bytes of one memory write came from.
#[derive(Clone, Copy, Debug)]
enum MemorySource {
    /// Bytes fixed by the bytecode: a constant MSTORE, CODECOPY.
    Constant,
    /// Bytes from calldata, returndata, another account's code or a computed value.
    Dynamic,
    /// MCOPY from this offset, as memory was at this point of the write log.
    Copy { source: usize, log_len: usize },
}

#[derive(Clone, Copy, Debug)]
struct MemoryWrite {
    offset: usize,
    size: usize,
    source: MemorySource,
}

/// Per-function scratch state of the effects heuristic.
#[derive(Clone, Debug, Default)]
pub(crate) struct EffectScratch {
    /// Memory writes of the current path, oldest first. The analyzer truncates it when it leaves
    /// a branch, so it always describes the path being analyzed.
    writes: Vec<MemoryWrite>,
    /// Keccak results seen in this function, with their preimages.
    preimages: HashMap<U256, Vec<u8>>,
}

impl EffectScratch {
    /// Length of the path's write log, to restore with [`EffectScratch::restore`].
    pub(crate) fn mark(&self) -> usize {
        self.writes.len()
    }

    /// Forget the writes made after `mark`.
    pub(crate) fn restore(&mut self, mark: usize) {
        self.writes.truncate(mark);
    }

    fn record_write(&mut self, offset: U256, size: U256, source: MemorySource) {
        let (Some(offset), Some(size)) = (to_tracked(offset), to_tracked(size)) else {
            return;
        };
        if size > 0 {
            self.writes.push(MemoryWrite { offset, size, source });
        }
    }

    /// Whether every byte of `memory[offset..offset + size]` is fixed by the bytecode, looking at
    /// the first `log_len` writes of the path. Bytes the path never wrote are fresh memory (zero),
    /// unless they are non-zero, which means code before the function entry wrote them.
    fn is_constant(&self, offset: usize, size: usize, log_len: usize, memory: &[u8], depth: u32) -> bool {
        let mut pending = vec![(offset, offset.saturating_add(size))];
        for (index, write) in self.writes[..log_len].iter().enumerate().rev() {
            if pending.is_empty() {
                return true;
            }
            let (write_start, write_end) = (write.offset, write.offset.saturating_add(write.size));
            let mut rest = Vec::with_capacity(pending.len());
            for (start, end) in pending {
                let (lo, hi) = (start.max(write_start), end.min(write_end));
                if lo >= hi {
                    rest.push((start, end));
                    continue;
                }
                match write.source {
                    MemorySource::Constant => {}
                    MemorySource::Dynamic => return false,
                    MemorySource::Copy { source, log_len } => {
                        let from = source.saturating_add(lo - write_start);
                        if depth >= MAX_FOLLOW_DEPTH ||
                            !self.is_constant(from, hi - lo, log_len.min(index), memory, depth + 1)
                        {
                            return false;
                        }
                    }
                }
                if start < lo {
                    rest.push((start, lo));
                }
                if hi < end {
                    rest.push((hi, end));
                }
            }
            pending = rest;
        }
        pending.iter().all(|(start, end)| {
            memory.get(*start..(*end).min(memory.len())).is_none_or(|bytes| bytes.iter().all(|b| *b == 0))
        })
    }

    /// The root slot of a mapping or dynamic-array element, following nested keys.
    fn mapping_root(&self, hash: U256, depth: u32) -> Option<U256> {
        let preimage = self.preimages.get(&hash)?;
        let root = match preimage.len() {
            // mapping element: keccak(key . slot)
            64 => U256::from_be_slice(&preimage[32..64]),
            // dynamic array or bytes data: keccak(slot)
            32 => U256::from_be_slice(preimage),
            _ => return None,
        };
        if depth < MAX_FOLLOW_DEPTH {
            if let Some(outer) = self.mapping_root(root, depth + 1) {
                return Some(outer);
            }
        }
        Some(root)
    }
}

fn to_tracked(value: U256) -> Option<usize> {
    usize::try_from(value).ok().filter(|v| *v <= MAX_TRACKED_MEMORY)
}

/// The constant value of an expression, if the bytecode fixes it. The VM folds arithmetic on
/// constants into a single PUSH32, so a PUSH is what a constant looks like.
fn constant(op: &WrappedOpcode) -> Option<U256> {
    match op.opcode {
        opcodes::PUSH0 => Some(U256::ZERO),
        opcodes::PUSH1..=opcodes::PUSH32 => match op.inputs.first() {
            Some(WrappedInput::Raw(value)) => Some(*value),
            Some(WrappedInput::Opcode(inner)) => constant(inner),
            None => None,
        },
        _ => None,
    }
}

fn input_constant(op: &WrappedOpcode, index: usize) -> Option<U256> {
    match op.inputs.get(index)? {
        WrappedInput::Raw(value) => Some(*value),
        WrappedInput::Opcode(inner) => constant(inner),
    }
}

fn input_op(op: &WrappedOpcode, index: usize) -> Option<&WrappedOpcode> {
    match op.inputs.get(index)? {
        WrappedInput::Opcode(inner) => Some(inner),
        WrappedInput::Raw(_) => None,
    }
}

/// Peel masks, shifts and divisions by constants: `and(sload(0), 0xff..ff)`,
/// `shr(8, sload(0))`, `div(sload(0), 0x100)`.
fn strip_masks(mut op: &WrappedOpcode) -> &WrappedOpcode {
    for _ in 0..MAX_FOLLOW_DEPTH {
        let next = match op.opcode {
            opcodes::AND if input_constant(op, 0).is_some() => input_op(op, 1),
            opcodes::AND if input_constant(op, 1).is_some() => input_op(op, 0),
            opcodes::SHR if input_constant(op, 0).is_some() => input_op(op, 1),
            opcodes::DIV if input_constant(op, 1).is_some() => input_op(op, 0),
            _ => None,
        };
        match next {
            Some(next) => op = next,
            None => break,
        }
    }
    op
}

/// What an operand is, from its expression.
fn classify(op: &WrappedOpcode) -> EffectOperand {
    if let Some(value) = constant(op) {
        return EffectOperand::Const(value);
    }
    let core = strip_masks(op);
    match core.opcode {
        opcodes::ADDRESS => EffectOperand::SelfAddress,
        opcodes::CALLER => EffectOperand::Caller,
        opcodes::SLOAD => match input_constant(core, 0) {
            Some(slot) => EffectOperand::Slot(slot),
            None => EffectOperand::Dynamic,
        },
        opcodes::CALLDATALOAD => match input_constant(core, 0) {
            Some(offset) if offset >= U256::from(4) && (offset - U256::from(4)) % U256::from(32) == U256::ZERO => {
                u32::try_from((offset - U256::from(4)) / U256::from(32))
                    .map(EffectOperand::Arg)
                    .unwrap_or(EffectOperand::Dynamic)
            }
            _ => EffectOperand::Dynamic,
        },
        _ => EffectOperand::Dynamic,
    }
}

/// The slot operand of SSTORE / TSTORE: a constant, a mapping element or dynamic.
fn classify_slot(scratch: &EffectScratch, op: &WrappedOpcode, value: U256) -> EffectOperand {
    if let Some(slot) = constant(op) {
        return EffectOperand::Const(slot);
    }
    // keccak(..) or keccak(..) + constant (a struct member of a mapping element)
    let (hash_op, hash) = match op.opcode {
        opcodes::ADD => match (input_constant(op, 0), input_constant(op, 1)) {
            (Some(offset), None) => (input_op(op, 1), value.wrapping_sub(offset)),
            (None, Some(offset)) => (input_op(op, 0), value.wrapping_sub(offset)),
            _ => (None, value),
        },
        _ => (Some(op), value),
    };
    match hash_op {
        Some(hash_op) if hash_op.opcode == opcodes::SHA3 => scratch
            .mapping_root(hash, 0)
            .map(EffectOperand::Mapping)
            .unwrap_or(EffectOperand::Dynamic),
        _ => EffectOperand::Dynamic,
    }
}

fn is_precompile(target: &EffectOperand) -> bool {
    matches!(target, EffectOperand::Const(address) if *address > U256::ZERO && *address <= U256::from(0x100))
}

/// Records the executed PC, tracks memory provenance and keccak preimages of the path, and
/// records an [`Effect`] for every effectful opcode.
pub(crate) fn effects_heuristic<'a>(
    function: &'a mut AnalyzedFunction,
    state: &'a State,
    _analyzer_state: &'a mut AnalyzerState,
) -> BoxFuture<'a, Result<(), Error>> {
    Box::pin(async move {
        let instruction = &state.last_instruction;
        let pc = instruction.instruction.saturating_sub(1);
        function.executed_pcs.insert(pc);

        let inputs = &instruction.inputs;
        let ops = &instruction.input_operations;
        // the VM reports fewer inputs than the opcode takes only for an instruction it did not
        // execute (past the end of the code)
        let arity = |n: usize| inputs.len() >= n && ops.len() >= n;
        let memory = &state.memory.memory;
        let scratch = &mut function.effect_scratch;

        let effect = |opcode: u8| Effect {
            pc,
            opcode,
            target: None,
            selector: None,
            value: None,
            slot: None,
            topic0: None,
            revert: None,
        };

        let recorded = match instruction.opcode {
            // memory provenance of this path
            opcodes::MSTORE | opcodes::MSTORE8 if arity(2) => {
                let size = if instruction.opcode == opcodes::MSTORE { 32 } else { 1 };
                let source = if constant(&ops[1]).is_some() {
                    MemorySource::Constant
                } else {
                    MemorySource::Dynamic
                };
                scratch.record_write(inputs[0], U256::from(size), source);
                None
            }
            opcodes::CALLDATACOPY | opcodes::RETURNDATACOPY if arity(3) => {
                scratch.record_write(inputs[0], inputs[2], MemorySource::Dynamic);
                None
            }
            opcodes::CODECOPY if arity(3) => {
                scratch.record_write(inputs[0], inputs[2], MemorySource::Constant);
                None
            }
            opcodes::EXTCODECOPY if arity(4) => {
                scratch.record_write(inputs[1], inputs[3], MemorySource::Dynamic);
                None
            }
            opcodes::MCOPY if arity(3) => {
                let source = match to_tracked(inputs[1]) {
                    Some(source) => MemorySource::Copy { source, log_len: scratch.mark() },
                    None => MemorySource::Dynamic,
                };
                scratch.record_write(inputs[0], inputs[2], source);
                None
            }
            opcodes::SHA3 if arity(2) => {
                if let (Some(offset), Some(size), Some(hash)) =
                    (to_tracked(inputs[0]), to_tracked(inputs[1]), instruction.outputs.first())
                {
                    if size == 32 || size == 64 {
                        scratch.preimages.insert(*hash, state.memory.read(offset, size));
                    }
                }
                None
            }

            // effects
            opcodes::SSTORE | opcodes::TSTORE if arity(2) => Some(Effect {
                slot: Some(classify_slot(scratch, &ops[0], inputs[0])),
                ..effect(instruction.opcode)
            }),
            opcodes::LOG0..=opcodes::LOG4 if arity(2) => {
                let topic0 = (instruction.opcode > opcodes::LOG0 && arity(3))
                    .then(|| constant(&ops[2]))
                    .flatten();
                Some(Effect { topic0, ..effect(instruction.opcode) })
            }
            opcodes::CALL | opcodes::CALLCODE | opcodes::DELEGATECALL | opcodes::STATICCALL
                if arity(6) =>
            {
                let with_value =
                    matches!(instruction.opcode, opcodes::CALL | opcodes::CALLCODE) && arity(7);
                let (offset_at, size_at) = if with_value { (3, 4) } else { (2, 3) };
                let target = classify(&ops[1]);
                let selector = if is_precompile(&target) || constant(&ops[size_at]) == Some(U256::ZERO) {
                    // a precompile takes no selector; a literal-zero size sends no call data
                    None
                } else if inputs[size_at] < U256::from(4) {
                    Some(EffectOperand::Dynamic)
                } else {
                    match to_tracked(inputs[offset_at]) {
                        Some(offset) if scratch.is_constant(offset, 4, scratch.mark(), memory, 0) => {
                            Some(EffectOperand::Const(U256::from_be_slice(&state.memory.read(offset, 4))))
                        }
                        _ => Some(EffectOperand::Dynamic),
                    }
                };
                Some(Effect {
                    target: Some(target),
                    selector,
                    value: with_value.then(|| classify(&ops[2])),
                    ..effect(instruction.opcode)
                })
            }
            opcodes::CREATE | opcodes::CREATE2 if arity(3) => {
                Some(Effect { value: Some(classify(&ops[0])), ..effect(instruction.opcode) })
            }
            opcodes::SELFDESTRUCT if arity(1) => {
                Some(Effect { target: Some(classify(&ops[0])), ..effect(instruction.opcode) })
            }
            opcodes::REVERT if arity(2) => {
                let revert = if constant(&ops[1]) == Some(U256::ZERO) {
                    Some(RevertReason::Empty)
                } else {
                    match (to_tracked(inputs[0]), to_tracked(inputs[1])) {
                        (Some(offset), Some(size))
                            if size >= 4 && scratch.is_constant(offset, 4, scratch.mark(), memory, 0) =>
                        {
                            let data = state.memory.read(offset, size);
                            let all_constant = scratch.is_constant(offset, size, scratch.mark(), memory, 0);
                            match data[0..4] {
                                // Error(string)
                                [0x08, 0xc3, 0x79, 0xa0] => Some(RevertReason::ErrorString(
                                    all_constant
                                        .then(|| match DynSolType::String.abi_decode(&data[4..]) {
                                            Ok(DynSolValue::String(message)) => Some(message),
                                            _ => None,
                                        })
                                        .flatten(),
                                )),
                                // Panic(uint256)
                                [0x4e, 0x48, 0x7b, 0x71] => (size >= 36 && all_constant)
                                    .then(|| RevertReason::Panic(U256::from_be_slice(&data[4..36]))),
                                [a, b, c, d] => Some(RevertReason::Custom([a, b, c, d])),
                                _ => None,
                            }
                        }
                        // data from returndata or computed: a bubbled or dynamic revert
                        _ => None,
                    }
                };
                Some(Effect { revert, ..effect(instruction.opcode) })
            }
            _ => None,
        };

        if let Some(effect) = recorded {
            function.effects.insert(effect);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(value: u64) -> WrappedOpcode {
        WrappedOpcode::new(opcodes::PUSH32, vec![U256::from(value).into()])
    }

    fn wrap(opcode: u8, inputs: Vec<WrappedOpcode>) -> WrappedOpcode {
        WrappedOpcode::new(
            opcode,
            inputs.into_iter().map(|op| WrappedInput::Opcode(std::sync::Arc::new(op))).collect(),
        )
    }

    #[test]
    fn operands_are_classified_from_their_expressions() {
        assert_eq!(classify(&push(0)), EffectOperand::Const(U256::ZERO));
        assert_eq!(classify(&wrap(opcodes::CALLER, vec![])), EffectOperand::Caller);
        assert_eq!(classify(&wrap(opcodes::ADDRESS, vec![])), EffectOperand::SelfAddress);
        // and(sload(3), 2^160 - 1)
        let mask: U256 = (U256::from(1) << 160) - U256::from(1);
        let owner = wrap(
            opcodes::AND,
            vec![
                WrappedOpcode::new(opcodes::PUSH32, vec![mask.into()]),
                wrap(opcodes::SLOAD, vec![push(3)]),
            ],
        );
        assert_eq!(classify(&owner), EffectOperand::Slot(U256::from(3)));
        assert_eq!(classify(&wrap(opcodes::CALLDATALOAD, vec![push(36)])), EffectOperand::Arg(1));
        assert_eq!(classify(&wrap(opcodes::CALLDATALOAD, vec![push(5)])), EffectOperand::Dynamic);
        assert_eq!(classify(&wrap(opcodes::MLOAD, vec![push(0x80)])), EffectOperand::Dynamic);
    }

    #[test]
    fn memory_provenance_follows_writes_and_copies() {
        let mut scratch = EffectScratch::default();
        let memory = vec![0u8; 0x200];
        // 0x80: constant word; 0x100: calldata; 0x180: mcopy of the constant word
        scratch.record_write(U256::from(0x80), U256::from(32), MemorySource::Constant);
        scratch.record_write(U256::from(0x100), U256::from(64), MemorySource::Dynamic);
        let copy = MemorySource::Copy { source: 0x80, log_len: scratch.mark() };
        scratch.record_write(U256::from(0x180), U256::from(32), copy);
        let len = scratch.mark();
        assert!(scratch.is_constant(0x80, 4, len, &memory, 0));
        assert!(!scratch.is_constant(0x104, 4, len, &memory, 0));
        assert!(scratch.is_constant(0x180, 32, len, &memory, 0));
        // a constant word partly overwritten by calldata is not constant
        scratch.record_write(U256::from(0x82), U256::from(1), MemorySource::Dynamic);
        assert!(!scratch.is_constant(0x80, 4, scratch.mark(), &memory, 0));
        // untouched zero memory is constant; untouched non-zero memory was written before entry
        assert!(scratch.is_constant(0x1c0, 4, len, &memory, 0));
        let mut written_before_entry = memory.clone();
        written_before_entry[0x1c1] = 1;
        assert!(!scratch.is_constant(0x1c0, 4, len, &written_before_entry, 0));
    }

    #[test]
    fn mapping_roots_follow_nested_keys() {
        let mut scratch = EffectScratch::default();
        let word = |v: u64| U256::from(v).to_be_bytes::<32>().to_vec();
        // allowance[a][b] at slot 1: keccak(b . keccak(a . 1))
        let inner = U256::from(0xaaaa);
        scratch.preimages.insert(inner, [word(7), word(1)].concat());
        let outer = U256::from(0xbbbb);
        scratch.preimages.insert(outer, [word(8), inner.to_be_bytes::<32>().to_vec()].concat());
        assert_eq!(scratch.mapping_root(outer, 0), Some(U256::from(1)));
        let sha3 = wrap(opcodes::SHA3, vec![push(0), push(64)]);
        assert_eq!(classify_slot(&scratch, &sha3, outer), EffectOperand::Mapping(U256::from(1)));
        // struct member: keccak(..) + 2
        let member = wrap(opcodes::ADD, vec![sha3.clone(), push(2)]);
        assert_eq!(
            classify_slot(&scratch, &member, outer + U256::from(2)),
            EffectOperand::Mapping(U256::from(1))
        );
        assert_eq!(classify_slot(&scratch, &sha3, U256::from(0xcccc)), EffectOperand::Dynamic);
    }
}
