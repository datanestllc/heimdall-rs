use crate::{
    core::types::SolidityType,
    interfaces::{AnalyzedFunction, CalldataFrame, SelectorAbi},
};
use hashbrown::HashSet;
use heimdall_common::ether::{signatures::ResolvedFunction, types::parse_function_parameters};
use tracing::{debug, trace};

/// Use caller-supplied types for `function` instead of matching resolved signatures against the
/// inferred arguments.
///
/// The function gets a placeholder name, `func_<selector>`, and a signature built from
/// `abi.inputs`; its argument slots and, when `abi.outputs` is given, its return type follow the
/// supplied types, so the header, the ABI and the casts in the body agree. Returns `false` and
/// changes nothing when the types do not form a parsable signature.
pub(crate) fn apply_selector_abi(function: &mut AnalyzedFunction, abi: &SelectorAbi) -> bool {
    let name = format!("func_{}", function.selector);
    let signature = format!("{name}({})", abi.inputs.join(","));
    if parse_function_parameters(&signature).is_err() {
        debug!("ignoring unparsable supplied signature '{}'", signature);
        function.notices.push(format!("supplied argument types do not parse: {signature}"));
        return false;
    }

    function.resolved_function = Some(ResolvedFunction {
        name,
        signature,
        inputs: abi.inputs.clone(),
        decoded_inputs: None,
    });

    // one argument slot per supplied input, keeping what analysis learned about each slot
    let mut inferred = std::mem::take(&mut function.arguments);
    for (index, ty) in abi.inputs.iter().enumerate() {
        let mut frame = inferred.remove(&index).unwrap_or_else(|| CalldataFrame {
            arg_op: String::new(),
            mask_size: 32,
            heuristics: HashSet::new(),
        });
        frame.mask_size = abi_type_byte_size(ty);
        function.arguments.insert(index, frame);
    }

    apply_selector_abi_returns(function, abi);
    true
}

/// Set the return type from `abi.outputs`, when given. Several outputs keep the inferred type.
pub(crate) fn apply_selector_abi_returns(function: &mut AnalyzedFunction, abi: &SelectorAbi) {
    match abi.outputs.as_deref() {
        Some([]) => function.returns = None,
        Some([output]) => {
            let ty = SolidityType::parse(output);
            function.returns = Some(match ty {
                SolidityType::Bytes | SolidityType::String | SolidityType::Array { .. } => {
                    ty.in_memory()
                }
                ty => ty,
            });
        }
        _ => {}
    }
}

/// The byte width of a value type, as the argument heuristics record it; 32 for everything else.
fn abi_type_byte_size(ty: &str) -> usize {
    let ty = ty.trim();
    match ty {
        "address" => 20,
        "bool" => 1,
        _ => {
            let bits = |prefix: &str| ty.strip_prefix(prefix).and_then(|n| n.parse::<usize>().ok());
            if let Some(bits) = bits("uint").or_else(|| bits("int")) {
                bits / 8
            } else if let Some(bytes) = ty.strip_prefix("bytes").and_then(|n| n.parse().ok()) {
                bytes
            } else {
                32
            }
        }
    }
}

/// Given a list of potential [`ResolvedFunction`]s and a [`Snapshot`], return a list of
/// [`ResolvedFunction`]s (that is, resolved signatures that were found on a 4byte directory) that
/// match the parameters found during symbolic execution for said [`Snapshot`].
pub(crate) fn match_parameters(
    resolved_functions: Vec<ResolvedFunction>,
    function: &AnalyzedFunction,
) -> Vec<ResolvedFunction> {
    let mut matched_functions: Vec<ResolvedFunction> = Vec::new();
    for mut resolved_function in resolved_functions {
        trace!(
            "checking function {} against Unresolved_0x{}({})",
            &resolved_function.signature,
            &function.selector,
            &function
                .arguments
                .values()
                .map(|f| f
                    .potential_types()
                    .first()
                    .cloned()
                    .unwrap_or(SolidityType::FixedBytes(32))
                    .to_string())
                .collect::<Vec<String>>()
                .join(",")
        );
        // skip checking if length of parameters list is less than the resolved functions inputs
        resolved_function.inputs.retain(|x| !x.is_empty());
        if resolved_function.inputs.len() < function.arguments.len() {
            continue;
        }
        let mut matched = true;

        // check each parameter type against a list of potential types
        for (index, input) in resolved_function.inputs.iter().enumerate() {
            trace!("    checking for parameter {} with type {}", &index.to_string(), &input);
            match function.arguments.get(&index) {
                Some(f) => {
                    let input_type = SolidityType::parse(input);
                    // Arrays are typically recorded as bytes by the decompiler's potential types.
                    if matches!(input_type.without_location(), SolidityType::Array { .. }) {
                        if !f.potential_types().contains(&SolidityType::Bytes) {
                            trace!(
                                "        parameter {} does not match type {} for function {}({})",
                                &index.to_string(),
                                &input,
                                &resolved_function.name,
                                &resolved_function.inputs.join(",")
                            );
                            continue;
                        }
                    } else if !f.potential_types().contains(&input_type) {
                        matched = false;
                        trace!(
                            "        parameter {} does not match type {} for function {}({})",
                            &index.to_string(),
                            &input,
                            &resolved_function.name,
                            &resolved_function.inputs.join(",")
                        );
                        break;
                    }
                }
                None => {
                    // parameter not found
                    matched = false;
                    trace!(
                        "        parameter {} not found for function {}({})",
                        &index.to_string(),
                        &resolved_function.name,
                        &resolved_function.inputs.join(",")
                    );
                    break;
                }
            }
        }

        trace!("    matched: {}", &matched.to_string());
        if matched {
            matched_functions.push(resolved_function);
        }
    }

    matched_functions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::analyze::AnalyzerType;

    fn abi(inputs: &[&str], outputs: Option<&[&str]>) -> SelectorAbi {
        SelectorAbi {
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.map(|o| o.iter().map(|s| s.to_string()).collect()),
        }
    }

    #[test]
    fn supplied_types_replace_inference_and_keep_a_placeholder_name() {
        let mut f = AnalyzedFunction::new("deadbeef", false);
        f.analyzer_type = AnalyzerType::Solidity;
        // inference saw one 32-byte argument and a bool return
        f.arguments.insert(
            0,
            CalldataFrame { arg_op: "op0".into(), mask_size: 32, heuristics: HashSet::new() },
        );
        f.returns = Some(SolidityType::Bool);

        let supplied = abi(&["address", "((address,uint256)[],uint8)", "bytes"], None);
        assert!(apply_selector_abi(&mut f, &supplied));

        let resolved = f.resolved_function.as_ref().unwrap();
        assert_eq!(resolved.name, "func_deadbeef");
        assert_eq!(resolved.signature, "func_deadbeef(address,((address,uint256)[],uint8),bytes)");
        assert_eq!(resolved.inputs().len(), 3);
        assert_eq!(f.emitted_name(), "func_deadbeef");
        assert_eq!(f.arguments.len(), 3);
        assert_eq!(f.arguments[&0].arg_op, "op0", "what analysis learned about a slot is kept");
        assert_eq!(f.arguments[&0].mask_size, 20);
        // evmole gives no outputs: the inferred return type stays
        assert_eq!(f.returns, Some(SolidityType::Bool));

        apply_selector_abi_returns(&mut f, &abi(&[], Some(&["string"])));
        assert_eq!(f.returns, Some(SolidityType::String.in_memory()));
    }

    #[test]
    fn unparsable_types_change_nothing() {
        let mut f = AnalyzedFunction::new("deadbeef", false);
        assert!(!apply_selector_abi(&mut f, &abi(&["(address"], None)));
        assert!(f.resolved_function.is_none());
        assert!(f.arguments.is_empty());
    }
}
