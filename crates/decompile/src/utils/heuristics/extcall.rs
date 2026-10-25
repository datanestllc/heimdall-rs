use alloy::primitives::U256;
use futures::future::BoxFuture;
use heimdall_common::utils::{hex::ToLowerHex, strings::encode_hex};
use heimdall_decoder::{decode, DecodeArgsBuilder};
use heimdall_vm::{
    core::{
        opcodes::{self, opcode_name, WrappedInput, WrappedOpcode},
        vm::State,
    },
    w_gas, w_push0,
};
use tracing::trace;

use crate::{
    core::{
        analyze::AnalyzerState,
        ir::{BinaryOp, Expr, Statement},
    },
    interfaces::AnalyzedFunction,
    utils::precompile::decode_precompile,
    Error,
};

/// Whether an operand is the literal 0 (PUSH0 or a PUSH of 0), not just a value that is 0 here.
fn is_literal_zero(op: &WrappedOpcode) -> bool {
    match op.opcode {
        opcodes::PUSH0 => true,
        opcodes::PUSH1..=opcodes::PUSH32 => {
            matches!(op.inputs.first(), Some(WrappedInput::Raw(value)) if value.is_zero())
        }
        _ => false,
    }
}

/// The first four bytes of the call data, as hex, read from memory at the call (empty when the
/// call data is shorter).
fn call_selector(state: &State, offset: U256, size: U256) -> String {
    match (usize::try_from(offset), usize::try_from(size)) {
        (Ok(offset), Ok(size)) if size >= 4 => encode_hex(&state.memory.read(offset, 4)),
        _ => String::new(),
    }
}

pub(crate) fn extcall_heuristic<'a>(
    function: &'a mut AnalyzedFunction,
    state: &'a State,
    analyzer_state: &'a mut AnalyzerState,
) -> BoxFuture<'a, Result<(), Error>> {
    Box::pin(async move {
        let instruction = &state.last_instruction;

        match instruction.opcode {
            // CALL / CALLCODE
            0xf1 | 0xf2 => {
                let memory =
                    function.get_memory_range(instruction.inputs[3], instruction.inputs[4]);
                let extcalldata =
                    memory.iter().map(|x| x.value.to_lower_hex()).collect::<Vec<String>>().join("");
                let address = Expr::from_opcode(&instruction.input_operations[1]);
                let value = Expr::from_opcode(&instruction.input_operations[2]);

                // Calls with the stipend or no calldata are value transfers.
                // a value transfer only when the call data size is the literal 0: a size that
                // merely evaluates to 0 comes from data the symbolic calldata left empty
                if is_literal_zero(&instruction.input_operations[4]) {
                    trace!(
                        "instruction {} ({}) indicates a value transfer",
                        instruction.instruction,
                        opcode_name(instruction.opcode)
                    );
                    function.push_statement(Statement::ExternalCall {
                        address,
                        function: "transfer".to_string(),
                        args: vec![value],
                        gas: None,
                        value: None,
                        comment: None,
                    });
                    return Ok(());
                }

                let decoded = decode(
                    DecodeArgsBuilder::new()
                        .target(extcalldata.clone())
                        .raw(true)
                        .skip_resolving(analyzer_state.skip_resolving)
                        .build()
                        .expect("Failed to build DecodeArgs"),
                )
                .await
                .ok();

                if let Some(precompile) = decode_precompile(
                    instruction.inputs[1],
                    &memory,
                    &instruction.input_operations[5],
                ) {
                    function.push_statement(precompile);
                    return Ok(());
                }

                let (name, args) = if let Some(decoded) = decoded {
                    let start_slot = instruction.inputs[3] + U256::from(4);
                    (
                        decoded.decoded.name,
                        decoded
                            .decoded
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(i, _)| {
                                Expr::index(
                                    "memory",
                                    Expr::Literal(start_slot + U256::from(i * 32)),
                                )
                            })
                            .collect(),
                    )
                } else {
                    let start = Expr::from_opcode(&instruction.input_operations[3]);
                    let size = Expr::from_opcode(&instruction.input_operations[4]);
                    (
                        format!(
                            "Unresolved_{}",
                            call_selector(state, instruction.inputs[3], instruction.inputs[4])
                        ),
                        vec![Expr::slice(
                            "msg.data",
                            start.clone(),
                            Expr::binary(BinaryOp::Add, start, size),
                        )],
                    )
                };

                function.push_statement(Statement::ExternalCall {
                    address,
                    function: name,
                    args,
                    gas: (instruction.input_operations[0] != w_gas!())
                        .then(|| Expr::from_opcode(&instruction.input_operations[0])),
                    value: (instruction.input_operations[2] != w_push0!()).then_some(value),
                    comment: Some(opcode_name(instruction.opcode).to_lowercase()),
                });
            }

            // STATICCALL / DELEGATECALL
            0xfa | 0xf4 => {
                let memory =
                    function.get_memory_range(instruction.inputs[2], instruction.inputs[3]);
                let extcalldata = memory
                    .iter()
                    .map(|x| x.value.to_lower_hex().trim_start_matches("0x").to_owned())
                    .collect::<Vec<String>>()
                    .join("");
                let address = Expr::from_opcode(&instruction.input_operations[1]);

                let decoded = decode(
                    DecodeArgsBuilder::new()
                        .target(extcalldata.clone())
                        .raw(true)
                        .skip_resolving(analyzer_state.skip_resolving)
                        .build()
                        .expect("Failed to build DecodeArgs"),
                )
                .await
                .ok();

                if let Some(precompile) = decode_precompile(
                    instruction.inputs[1],
                    &memory,
                    &instruction.input_operations[4],
                ) {
                    function.push_statement(precompile);
                    return Ok(());
                }

                let (name, args) = if let Some(decoded) = decoded {
                    let start_slot = instruction.inputs[2] + U256::from(4);
                    (
                        decoded.decoded.name,
                        decoded
                            .decoded
                            .inputs
                            .iter()
                            .enumerate()
                            .map(|(i, _)| {
                                Expr::index(
                                    "memory",
                                    Expr::Literal(start_slot + U256::from(i * 32)),
                                )
                            })
                            .collect(),
                    )
                } else {
                    let start = Expr::from_opcode(&instruction.input_operations[2]);
                    let size = Expr::from_opcode(&instruction.input_operations[3]);
                    (
                        format!(
                            "Unresolved_{}",
                            call_selector(state, instruction.inputs[2], instruction.inputs[3])
                        ),
                        vec![Expr::slice(
                            "memory",
                            start.clone(),
                            Expr::binary(BinaryOp::Add, start, size),
                        )],
                    )
                };

                function.push_statement(Statement::ExternalCall {
                    address,
                    function: name,
                    args,
                    gas: (instruction.input_operations[0] != w_gas!())
                        .then(|| Expr::from_opcode(&instruction.input_operations[0])),
                    value: None,
                    comment: Some(opcode_name(instruction.opcode).to_lowercase()),
                });
            }

            _ => {}
        };

        Ok(())
    })
}
