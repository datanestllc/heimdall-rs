pub(crate) mod analyze;
mod calldata;
pub(crate) mod control_flow;
mod event_logs;
pub(crate) mod ir;
pub(crate) mod out;
pub(crate) mod postprocess;
pub(crate) mod resolve;
pub(crate) mod types;

use alloy::primitives::Address;
use alloy_dyn_abi::{DynSolType, DynSolValue};
use alloy_json_abi::JsonAbi;
use eyre::eyre;
use hashbrown::HashMap;
use heimdall_common::{
    ether::{
        compiler::detect_compiler,
        signatures::{
            cache_signatures_from_abi, score_signature, ResolvedError, ResolvedFunction,
            ResolvedLog,
        },
        types::to_type,
    },
    utils::strings::{decode_hex, encode_hex, encode_hex_reduced, StringExt},
};
use heimdall_disassembler::{disassemble, DisassemblerArgsBuilder};
use heimdall_vm::{
    core::vm::VM,
    ext::{
        exec::ExecStats,
        selectors::{find_function_selectors, resolve_selectors},
    },
};
use futures::FutureExt;
use std::{
    collections::BTreeMap,
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use crate::{
    core::{
        analyze::{Analyzer, AnalyzerType},
        out::{
            build_abi, build_abi_with_details,
            source::{build_source, StorageVariable},
        },
        postprocess::PostprocessOrchestrator,
        resolve::{apply_selector_abi, apply_selector_abi_returns, match_parameters},
    },
    error::Error,
    interfaces::{
        sort_analyzed_functions, AnalyzedFunction, BodyStatus, DecompilerArgs, FunctionReport,
    },
};
use tracing::{debug, info, warn};

/// How the body of a selector is reached.
#[derive(Clone, Copy, Debug)]
enum Entry {
    /// Heimdall's own walk of the dispatcher found this jump destination.
    Walked(u128),
    /// The caller supplied this 0-based entry offset; the VM steps until the pc is exactly there.
    Supplied(u128),
}

/// Upper bound on VM steps when a constant function is executed for its value.
const MAX_CONSTANT_CALL_STEPS: u64 = 100_000;

/// The message of a caught panic.
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

#[derive(Debug, Clone)]
/// Result of a successful decompile operation
///
/// Contains the decompiled source code (if requested) and the reconstructed ABI
/// of the contract.
pub struct DecompileResult {
    /// The decompiled source code in Solidity or Yul format (if requested)
    pub source: Option<String>,
    /// The reconstructed JSON ABI of the contract
    pub abi: JsonAbi,
    /// The extended ABI with selector and signature information
    pub abi_with_details: serde_json::Value,
    /// One report per function found, ordered by selector (`fallback` and `receive` last)
    pub functions: Vec<FunctionReport>,
}

/// Decompiles EVM bytecode into higher-level Solidity-like code
///
/// This function analyzes the bytecode of a contract through symbolic execution
/// and attempts to reconstruct the original source code or a functionally equivalent
/// representation. It also generates an ABI for the contract.
///
/// # Arguments
///
/// * `args` - Configuration parameters for the decompile operation
///
/// # Returns
///
/// A DecompileResult containing the decompiled source (if requested) and the ABI
pub async fn decompile(args: DecompilerArgs) -> Result<DecompileResult, Error> {
    // init
    let start_time = Instant::now();
    let mut all_resolved_events: HashMap<String, ResolvedLog> = HashMap::new();
    let mut all_resolved_errors: HashMap<String, ResolvedError> = HashMap::new();

    // validate arguments
    if args.include_solidity && args.include_yul {
        return Err(Error::Eyre(eyre!(
            "arguments '--include-sol' and '--include-yul' are mutually exclusive.".to_string(),
        )));
    }
    if args.llm_postprocess && args.openrouter_api_key.is_empty() {
        return Err(Error::Eyre(eyre!(
                "llm postprocessing requires an OpenRouter API key. please provide one using the '--openrouter-api-key' flag."
            )));
    }
    if !args.include_solidity && args.llm_postprocess {
        return Err(Error::Eyre(eyre!(
            "llm postprocessing requires including solidity source code. please enable the '--include-sol' flag."
        )));
    }

    if args.indexed_log_check {
        event_logs::validate_lookup(&args)?;
    }

    let analyzer_type = AnalyzerType::from_args(args.include_solidity, args.include_yul);

    // parse and cache signatures from the ABI, if provided
    if let Some(abi_path) = args.abi.as_ref() {
        cache_signatures_from_abi(abi_path.into())
            .map_err(|e| Error::Eyre(eyre!("caching signatures from ABI failed: {}", e)))?;
    }

    // Resolve hardfork (handles Auto detection if needed)
    let start_hardfork_resolve = Instant::now();
    let hardfork = args.get_hardfork().await;
    debug!("resolved hardfork: {} (took {:?})", hardfork, start_hardfork_resolve.elapsed());

    // get the bytecode from the target
    let start_fetch_time = Instant::now();
    let contract_bytecode = args
        .get_bytecode()
        .await
        .map_err(|e| Error::FetchError(format!("fetching target bytecode failed: {e}")))?;
    debug!("fetching target bytecode took {:?}", start_fetch_time.elapsed());

    if contract_bytecode.is_empty() {
        return Err(Error::Eyre(eyre!(
            "contract bytecode is empty. This may be a self-destructed contract. \
            try passing an Etherscan API key to attempt to fetch creation bytecode: \
            \
            `--etherscan_api_key <YOUR_KEY>`"
        )));
    }

    // perform versioning and compiler heuristics
    let (_compiler, _version) = detect_compiler(&contract_bytecode);

    // create a new EVM instance. we will use this for finding function selectors,
    // performing symbolic execution, and more.
    let mut evm = VM::new(
        &contract_bytecode,
        &[],
        Address::default(),
        Address::default(),
        Address::default(),
        0,
        u128::MAX,
    )
    .with_hardfork(hardfork);

    // disassemble the contract's bytecode
    let assembly = disassemble(
        DisassemblerArgsBuilder::new()
            .target(encode_hex(&contract_bytecode))
            .hardfork(hardfork)
            .build()
            .expect("impossible case: failed to build disassembly arguments"),
    )
    .await
    .map_err(|e| Error::Eyre(eyre!("disassembling contract bytecode failed: {}", e)))?;

    // find all the function selectors in the bytecode
    let start_selectors_time = Instant::now();
    let found = find_function_selectors(&evm, &assembly);
    debug!("finding function selectors took {:?}", start_selectors_time.elapsed());

    // supplied entries win over the dispatcher walk; heimdall's own finds stay for selectors the
    // caller did not supply
    let mut selectors: BTreeMap<String, Entry> =
        found.into_iter().map(|(selector, entry)| (selector, Entry::Walked(entry))).collect();
    for (selector, entry_pc) in args.entry_points.iter().flatten() {
        selectors.insert(selector.clone(), Entry::Supplied(*entry_pc));
    }

    // resolve selectors (if enabled)
    let resolved_selectors = match args.skip_resolving {
        true => HashMap::new(),
        false => resolve_selectors::<ResolvedFunction>(selectors.keys().cloned().collect()).await,
    };

    info!("performing symbolic execution on '{}'", args.target.truncate(64));

    // budgets are counted in VM steps, branches and saved state bytes; wall clock is only an
    // emergency stop for the whole job
    let job_deadline = (args.job_timeout > 0)
        .then(|| start_time.checked_add(Duration::from_millis(args.job_timeout)))
        .flatten();
    let mut job_used = ExecStats::default();
    let mut reports: HashMap<String, FunctionReport> = HashMap::new();

    // ordered, so functions are analyzed (and share postprocessor state) in the same order in
    // every process
    let mut symbolic_execution_maps = BTreeMap::new();
    if selectors.is_empty() {
        warn!("discovered no function selectors in the bytecode.");
        let start_sym_exec_time = Instant::now();
        let (map, stats) = evm
            .symbolic_exec_with_budget(args.function_budget(&job_used, job_deadline))
            .map_err(|e| Error::Eyre(eyre!("symbolic execution failed: {}", e)))?;
        job_used.steps += stats.steps;
        job_used.state_bytes += stats.state_bytes;

        debug!("symbolic execution (fallback) took {:?}", start_sym_exec_time.elapsed());
        debug!("'fallback' has {} unique branches", stats.branches);
        reports.insert("fallback".to_string(), FunctionReport::explored("fallback", &map, stats));
        symbolic_execution_maps.insert("fallback".to_string(), map);
    }

    // explore selectors in sorted order, each on a fresh copy of the initial VM: `reset()` keeps
    // storage, so one function's SSTOREs used to steer the next, in hash-map order
    let selectors = selectors.into_iter().collect::<Vec<_>>();

    // the entries that are not selectors run first: a dispatcher with selectors used to hide
    // them completely
    if args.fallback_entries && !selectors.is_empty() {
        let known = selectors
            .iter()
            .map(|(selector, _)| selector.as_str())
            .chain(args.selector_abi.iter().flat_map(|abi| abi.keys().map(String::as_str)))
            .collect::<std::collections::BTreeSet<_>>();
        let unmatched = (0..=u32::MAX)
            .rev()
            .map(|candidate| format!("{candidate:08x}"))
            .find(|candidate| !known.contains(candidate.as_str()))
            .expect("impossible case: every selector is known");
        let unmatched = decode_hex(&unmatched).expect("impossible case: invalid selector");
        for (entry, calldata) in [("fallback", unmatched), ("receive", Vec::new())] {
            let mut entry_evm = evm.clone();
            let (map, stats) = entry_evm
                .symbolic_exec_entry(&calldata, args.function_budget(&job_used, job_deadline))
                .map_err(|e| Error::Eyre(eyre!("symbolic execution failed: {}", e)))?;
            job_used.steps += stats.steps;
            job_used.state_bytes += stats.state_bytes;
            debug!("'{}' has {} unique branches", entry, stats.branches);
            reports.insert(entry.to_string(), FunctionReport::explored(entry, &map, stats));
            symbolic_execution_maps.insert(entry.to_string(), map);
        }
    }
    let overall_sym_exec_time = Instant::now();
    for (selector, entry) in selectors {
        let start_sym_exec_time = Instant::now();
        let mut function_evm = evm.clone();
        let budget = args.function_budget(&job_used, job_deadline);
        let result = match entry {
            Entry::Walked(entry_point) => {
                function_evm.symbolic_exec_selector_with_budget(&selector, entry_point, budget)
            }
            Entry::Supplied(entry_pc) => {
                // calldata shaped by the argument types when they are known (F9)
                let typed = args
                    .selector_abi
                    .as_ref()
                    .and_then(|abi| abi.get(&selector))
                    .and_then(|abi| calldata::typed_calldata(&selector, &abi.inputs));
                typed
                    .map(Ok)
                    .unwrap_or_else(|| decode_hex(&selector).map_err(Into::into))
                    .and_then(|calldata| {
                        function_evm.symbolic_exec_from_entry(&calldata, entry_pc, budget)
                    })
            }
        };
        let (map, stats) = match result {
            Ok(map) => map,
            Err(e) => {
                warn!("failed to symbolically execute '{}': {}", selector, e);
                reports.insert(
                    selector.clone(),
                    FunctionReport::failed(&selector, format!("symbolic execution failed: {e}")),
                );
                continue;
            }
        };
        job_used.steps += stats.steps;
        job_used.state_bytes += stats.state_bytes;
        debug!("symbolically executed '{}' in {:?}", selector, start_sym_exec_time.elapsed());
        debug!("'{}' has {} unique branches", selector, stats.branches);
        reports.insert(selector.clone(), FunctionReport::explored(&selector, &map, stats));
        symbolic_execution_maps.insert(selector.clone(), map);
    }
    debug!("symbolic execution took {:?}", overall_sym_exec_time.elapsed());
    info!("symbolically executed {} selectors", symbolic_execution_maps.len());

    let start_analysis_time = Instant::now();
    let handles = symbolic_execution_maps.into_iter().map(|(selector, trace_root)| {
        let mut evm_clone = evm.clone();
        let mut function =
            AnalyzedFunction::new(&selector, selector == "fallback" || selector == "receive");
        function.receive = selector == "receive";
        let mut analyzer = Analyzer::new(analyzer_type, args.skip_resolving, function);
        let key = selector.clone();
        let analysis = async move {
            // analyze the symbolic execution trace; keep what was found even if it fails
            let (mut analyzed_function, error) = analyzer.analyze_partial(trace_root).await;
            if error.is_some() {
                return (analyzed_function, error);
            }

            // if the function is constant, we can get the exact val
            if analyzed_function.is_constant() && !analyzed_function.fallback {
                evm_clone.reset();
                let x = match evm_clone.call_bounded(
                    &decode_hex(&selector).expect("invalid selector"),
                    0,
                    MAX_CONSTANT_CALL_STEPS,
                ) {
                    Ok(x) => x,
                    Err(e) => return (analyzed_function, Some(Error::Eyre(e))),
                };

                let returns_param_type = analyzed_function
                    .returns
                    .as_ref()
                    .map(|ret_type| to_type(&ret_type.without_location().to_string()))
                    .unwrap_or(DynSolType::Bytes);

                let decoded = returns_param_type
                    .abi_decode(&x.returndata)
                    .map(|decoded| match decoded {
                        DynSolValue::String(s) => format!("\"{s}\""),
                        DynSolValue::Uint(x, _) => x.to_string(),
                        DynSolValue::Int(x, _) => x.to_string(),
                        token => format!("0x{token:?}"),
                    })
                    .unwrap_or_else(|_| encode_hex(&x.returndata));

                analyzed_function.constant_value = Some(decoded);
            }

            (analyzed_function, None)
        };
        // a failure or a panic in one function marks only that function
        AssertUnwindSafe(analysis).catch_unwind().map(move |outcome| (key, outcome))
    });
    let mut analyzed_functions = Vec::new();
    for (selector, outcome) in futures::future::join_all(handles).await {
        let failure = match outcome {
            Ok((function, error)) => {
                if let Some(report) = reports.get_mut(&selector) {
                    report.effects = function.effects.iter().cloned().collect();
                    report.executed_pcs = function.executed_pcs.iter().copied().collect();
                }
                match error {
                    None => {
                        analyzed_functions.push(function);
                        None
                    }
                    Some(error) => Some(format!("analysis failed: {error}")),
                }
            }
            Err(panic) => Some(format!("analysis panicked: {}", panic_message(&panic))),
        };
        if let (Some(failure), Some(report)) = (failure, reports.get_mut(&selector)) {
            warn!("'{}': {}", selector, failure);
            report.status = BodyStatus::Failed;
            report.failure = Some(failure);
        }
    }

    debug!("analyzing symbolic execution results took {:?}", start_analysis_time.elapsed());
    info!("analyzed {} symbolic execution traces", analyzed_functions.len());

    // resolve event and error selectors
    if !args.skip_resolving {
        // resolve error selectors
        let start_error_resolving_time = Instant::now();
        let mut error_selectors: Vec<String> = analyzed_functions
            .iter()
            .flat_map(|f| f.errors.iter().map(|e| encode_hex_reduced(*e).replacen("0x", "", 1)))
            .collect();
        error_selectors.dedup();
        debug!("resolving {} error signatures", error_selectors.len());
        let resolved_errors: HashMap<String, ResolvedError> =
            resolve_selectors(error_selectors.clone())
                .await
                .into_iter()
                .map(|(k, mut potential_values)| {
                    // sort by score, take the highest
                    potential_values.sort_by(|a: &ResolvedError, b: &ResolvedError| {
                        let a_score = score_signature(&a.signature, None);
                        let b_score = score_signature(&b.signature, None);
                        b_score.cmp(&a_score)
                    });

                    (k, potential_values.remove(0))
                })
                .collect();
        debug!("resolving error signatures took {:?}", start_error_resolving_time.elapsed());
        info!(
            "resolved {} error signatures from {} selectors",
            resolved_errors.len(),
            error_selectors.len()
        );
        all_resolved_errors.extend(resolved_errors);

        // resolve event selectors
        let start_event_resolving_time = Instant::now();
        let mut event_selectors: Vec<String> = analyzed_functions
            .iter()
            .flat_map(|f| f.events.iter().map(|e| encode_hex_reduced(*e).replacen("0x", "", 1)))
            .collect();
        event_selectors.dedup();
        debug!("resolving {} event signatures", event_selectors.len());
        let resolved_events: HashMap<String, ResolvedLog> =
            resolve_selectors(event_selectors.clone())
                .await
                .into_iter()
                .map(|(k, mut potential_values)| {
                    // sort by score, take the highest
                    potential_values.sort_by(|a: &ResolvedLog, b: &ResolvedLog| {
                        let a_score = score_signature(&a.signature, None);
                        let b_score = score_signature(&b.signature, None);
                        b_score.cmp(&a_score)
                    });

                    (k, potential_values.remove(0))
                })
                .collect();
        debug!("resolving event signaturess took {:?}", start_event_resolving_time.elapsed());
        info!(
            "resolved {} event signatures from {} selectors",
            resolved_events.len(),
            event_selectors.len()
        );
        all_resolved_events.extend(resolved_events);
    }

    // match analyzed parameters with resolved signatures for each function
    analyzed_functions.iter_mut().for_each(|f| {
        // caller-supplied types are taken as they are
        if let Some(abi) = args.selector_abi.as_ref().and_then(|abi| abi.get(&f.selector)) {
            if apply_selector_abi(f, abi) {
                return;
            }
        }

        let resolve_function_signatures =
            resolved_selectors.get(&f.selector).unwrap_or(&Vec::new()).to_owned();
        let mut matched_resolved_functions = match_parameters(resolve_function_signatures, f);
        debug!(
            "matched {} resolved functions for '{}'",
            matched_resolved_functions.len(),
            f.selector
        );

        matched_resolved_functions.sort_by(|a, b| {
            let a_score = score_signature(&a.signature, None);
            let b_score = score_signature(&b.signature, None);
            b_score.cmp(&a_score)
        });

        f.resolved_function = matched_resolved_functions.first().cloned();
        debug!(
            "using signature '{}' for '{}'",
            f.resolved_function.as_ref().map(|r| &r.signature).unwrap_or(&String::new()),
            f.selector
        );
    });

    // sort the analyzed functions into a deterministic order, so that repeated decompilation of
    // the same bytecode always produces identical output
    sort_analyzed_functions(&mut analyzed_functions);

    // get a new PostprocessorOrchestrator
    // note: this will do nothing if the include_solidity and include_yul flags are false
    let mut postprocessor = PostprocessOrchestrator::new(analyzer_type)?;
    let states = analyzed_functions
        .iter_mut()
        .filter_map(|f| {
            postprocessor.postprocess(f).map_err(|e| f.notices.push(e.to_string())).ok()
        })
        .collect::<Vec<_>>();

    // postprocessors may refine the return type; a supplied one stays
    for f in analyzed_functions.iter_mut() {
        if let Some(abi) = args.selector_abi.as_ref().and_then(|abi| abi.get(&f.selector)) {
            apply_selector_abi_returns(f, abi);
        }
    }

    let storage_slots = states
        .iter()
        .flat_map(|state| state.storage_root_slots.iter())
        .map(|(name, slot)| {
            let rendered = slot.render();
            (name.clone(), if rendered == "0" { "0x00".to_string() } else { rendered })
        })
        .collect::<HashMap<String, String>>();
    let storage_variables = states
        .iter()
        .flat_map(|state| state.storage_type_map.iter())
        .chain(states.iter().flat_map(|state| state.transient_type_map.iter()))
        .map(|(name, typ)| {
            (
                name.clone(),
                StorageVariable { typ: typ.clone(), slot: storage_slots.get(name).cloned() },
            )
        })
        .collect::<HashMap<_, _>>();

    // construct the abi for the given analyzed functions
    let mut abi = build_abi(&analyzed_functions, &all_resolved_errors, &all_resolved_events)?;
    if args.indexed_log_check && event_logs::recover_event_indexing(&mut abi, &args).await.is_err()
    {
        // Provider errors can contain credentials from the RPC URL.
        warn!("indexed log lookup failed; retaining inferred event layouts");
    }
    let abi_with_details = build_abi_with_details(&abi, &analyzed_functions)?;
    let source = build_source(
        &analyzed_functions,
        &all_resolved_errors,
        &all_resolved_events,
        &abi,
        &storage_variables,
        args.llm_postprocess,
        args.openrouter_api_key,
        args.model,
    )
    .await?;

    debug!("decompilation took {:?}", start_time.elapsed());

    let mut functions = reports.into_values().collect::<Vec<_>>();
    let entry = |selector: &str| matches!(selector, "fallback" | "receive");
    functions.sort_by(|a, b| {
        (entry(&a.selector), &a.selector).cmp(&(entry(&b.selector), &b.selector))
    });

    Ok(DecompileResult { source, abi, abi_with_details, functions })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{DecompilerArgsBuilder, HardFork, SelectorAbi};

    use super::decompile;

    /// `transfer`-like dispatcher: selector 0xa9059cbb jumps to pc 0x10, which stores
    /// calldataload(4) in slot 0 and stops.
    /// 0: PUSH1 0 CALLDATALOAD PUSH1 0xe0 SHR PUSH4 a9059cbb EQ PUSH1 0x10 JUMPI STOP
    /// 16: JUMPDEST PUSH1 4 CALLDATALOAD PUSH1 0 SSTORE STOP
    pub(crate) const TINY: &str = "60003560e01c63a9059cbb14601057005b60043560005500";

    /// Dispatcher for 0xa9059cbb (body at pc 0x11) whose body has one effect of each kind:
    /// pc 23  SSTORE slot 0 <- calldataload(4)
    /// pc 41  SSTORE balances[msg.sender] (mapping at slot 1) <- 5, via SHA3 at pc 37
    /// pc 79  LOG1 Transfer topic0
    /// pc 148 CALL 0x..c0ffee01 balanceOf(..): selector written by a constant MSTORE, value 0
    /// pc 172 CALL calldataload(4) with msg.value: selector copied from calldata at pc 157
    /// pc 223 REVERT CustomError 0x12345678, on the branch where calldataload(0x24) != 0
    const EFFECTS: &str = "60003560e01c63a9059cbb1461001157005b6004356000553360005260016020526040600020600590557fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef60006000a17f70a0823100000000000000000000000000000000000000000000000000000000608052602060006024608060007300000000000000000000000000000000c0ffee015af1506024600461010037600060006024610100346004355af1506024356100b657005b7f123456780000000000000000000000000000000000000000000000000000000060005260046000fd";

    #[tokio::test]
    async fn effects_are_recorded_at_opcode_time() {
        use crate::{BodyStatus, Effect, EffectOperand as Op, RevertReason};
        use alloy::primitives::U256;
        use heimdall_vm::core::opcodes;

        let args = DecompilerArgsBuilder::new()
            .target(EFFECTS.to_string())
            .skip_resolving(true)
            .include_solidity(true)
            .hardfork(HardFork::Fusaka)
            .build()
            .unwrap();
        let result = decompile(args).await.unwrap();
        let report = result.functions.iter().find(|f| f.selector == "a9059cbb").unwrap();
        assert_eq!(report.status, BodyStatus::Done);
        assert_eq!(report.entry_pc, Some(0x11));
        assert_eq!(report.executed_pcs.first(), Some(&0x11), "dispatcher PCs are not included");
        assert!(report.executed_pcs.contains(&223));

        let none = |pc: u128, opcode: u8| Effect {
            pc,
            opcode,
            target: None,
            selector: None,
            value: None,
            slot: None,
            topic0: None,
            revert: None,
        };
        let topic0 = U256::from_str_radix(
            "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
            16,
        )
        .unwrap();
        let expected = vec![
            Effect { slot: Some(Op::Const(U256::ZERO)), ..none(23, opcodes::SSTORE) },
            Effect { slot: Some(Op::Mapping(U256::from(1))), ..none(41, opcodes::SSTORE) },
            Effect { topic0: Some(topic0), ..none(79, opcodes::LOG1) },
            Effect {
                target: Some(Op::Const(U256::from(0xc0ffee01u64))),
                selector: Some(Op::Const(U256::from(0x70a08231u64))),
                value: Some(Op::Const(U256::ZERO)),
                ..none(148, opcodes::CALL)
            },
            Effect {
                target: Some(Op::Arg(0)),
                selector: Some(Op::Dynamic),
                value: Some(Op::Dynamic),
                ..none(172, opcodes::CALL)
            },
            Effect {
                revert: Some(RevertReason::Custom([0x12, 0x34, 0x56, 0x78])),
                ..none(223, opcodes::REVERT)
            },
        ];
        assert_eq!(report.effects, expected);
    }

    #[tokio::test]
    async fn a_failing_function_does_not_fail_the_contract() {
        use crate::BodyStatus;

        // 0x11111111 jumps to pc 28, an endless `JUMPDEST PUSH2 28 JUMP`: a "constant" function
        // whose concrete execution never ends. 0x22222222 jumps to pc 33: `JUMPDEST STOP`.
        let code = "60003560e01c8063111111111461001c5763222222221461002157005b61001c565b00";
        let args = DecompilerArgsBuilder::new()
            .target(code.to_string())
            .skip_resolving(true)
            .include_solidity(true)
            .hardfork(HardFork::Fusaka)
            .step_budget(1000)
            .build()
            .unwrap();
        let result = decompile(args).await.unwrap();
        let status = |sel: &str| result.functions.iter().find(|f| f.selector == sel).unwrap();
        let looping = status("11111111");
        assert_eq!(looping.status, BodyStatus::Failed);
        assert!(looping.failure.as_deref().unwrap().contains("did not finish"), "{looping:?}");
        assert_eq!(looping.executed_pcs, vec![28, 29, 32], "kept although the function failed");
        assert_eq!(status("22222222").status, BodyStatus::Done);
        let source = result.source.unwrap();
        assert!(source.contains("22222222") && !source.contains("11111111"), "{source}");
    }

    #[tokio::test]
    async fn storage_does_not_leak_between_functions() {
        // 0x11111111 (pc 28) stores 41 in slot 0. 0x22222222 (pc 36) jumps to sload(0); only a
        // leaked 41 reaches the LOG0 at pc 41.
        let code = "60003560e01c8063111111111461001c5763222222221461002457005b610029600055005b600054565b60006000a000";
        for _ in 0..4 {
            let args = DecompilerArgsBuilder::new()
                .target(code.to_string())
                .skip_resolving(true)
                .include_solidity(true)
                .hardfork(HardFork::Fusaka)
                .step_budget(2000)
                .build()
                .unwrap();
            let result = decompile(args).await.unwrap();
            let reader = result.functions.iter().find(|f| f.selector == "22222222").unwrap();
            assert!(!reader.executed_pcs.contains(&41), "{reader:?}");
        }
    }

    #[tokio::test]
    async fn fallback_and_receive_are_explored_next_to_selectors() {
        use heimdall_vm::core::opcodes;

        // 0: CALLDATASIZE ISZERO PUSH1 28 JUMPI                         -- receive at 28
        // 5: PUSH1 0 CALLDATALOAD PUSH1 0xe0 SHR PUSH4 11111111 EQ PUSH1 35 JUMPI
        // 20: PUSH1 7 PUSH1 0 SSTORE STOP STOP STOP                     -- fallback body
        // 28: JUMPDEST PUSH1 0 PUSH1 0 LOG0 STOP   35: JUMPDEST STOP
        let code = "3615601c5760003560e01c63111111111460235760076000550000005b60006000a0005b00";
        let args = DecompilerArgsBuilder::new()
            .target(code.to_string())
            .skip_resolving(true)
            .include_solidity(true)
            .hardfork(HardFork::Fusaka)
            .fallback_entries(true)
            .build()
            .unwrap();
        let result = decompile(args).await.unwrap();
        let selectors = result.functions.iter().map(|f| f.selector.as_str()).collect::<Vec<_>>();
        assert_eq!(selectors, vec!["11111111", "fallback", "receive"]);
        let opcodes_of = |name: &str| {
            let f = result.functions.iter().find(|f| f.selector == name).unwrap();
            f.effects.iter().map(|e| e.opcode).collect::<Vec<_>>()
        };
        assert_eq!(opcodes_of("fallback"), vec![opcodes::SSTORE]);
        assert_eq!(opcodes_of("receive"), vec![opcodes::LOG0]);
        let source = result.source.unwrap();
        assert!(source.contains("fallback() external payable {"), "{source}");
        assert!(source.contains("receive() external payable {"), "{source}");
    }

    #[tokio::test]
    async fn the_job_budget_is_shared_in_selector_order() {
        use crate::BodyStatus;

        // as in `a_failing_function_does_not_fail_the_contract`: 0x11111111 is an endless jump
        // loop that uses up the job budget, so 0x22222222, explored after it, gets nothing left
        let code = "60003560e01c8063111111111461001c5763222222221461002157005b61001c565b00";
        let args = DecompilerArgsBuilder::new()
            .target(code.to_string())
            .skip_resolving(true)
            .include_solidity(true)
            .hardfork(HardFork::Fusaka)
            .job_step_budget(500)
            .build()
            .unwrap();
        let result = decompile(args).await.unwrap();
        let steps = |sel: &str| {
            result.functions.iter().find(|f| f.selector == sel).map(|f| f.stats.steps).unwrap()
        };
        assert_eq!(steps("11111111"), 500);
        assert_eq!(steps("22222222"), 0);
        let last = result.functions.iter().find(|f| f.selector == "22222222").unwrap();
        assert_eq!(last.status, BodyStatus::Partial);
    }

    #[tokio::test]
    async fn supplied_entries_reach_selectors_the_dispatcher_walk_misses() {
        use crate::BodyStatus;

        // 0x00fdd58e (pushed as PUSH3 0xfdd58e) jumps to pc 25: SSTORE(0, 1); 0x22222222 to pc 32
        let code = "60003560e01c8062fdd58e14601957632222222214602057005b6001600055005b00";
        let build = |entries: Option<BTreeMap<String, u128>>| {
            DecompilerArgsBuilder::new()
                .target(code.to_string())
                .skip_resolving(true)
                .include_solidity(true)
                .hardfork(HardFork::Fusaka)
                .entry_points(entries)
                .build()
                .unwrap()
        };

        // heimdall's own walk misses the selector with a leading zero byte
        let walked = decompile(build(None)).await.unwrap();
        assert!(walked.functions.iter().all(|f| f.selector != "00fdd58e"));

        // with entries, the body starts exactly at the supplied offset; a wrong entry fails alone
        let entries = BTreeMap::from([
            ("00fdd58e".to_string(), 25u128),
            ("22222222".to_string(), 25u128),
        ]);
        let result = decompile(build(Some(entries))).await.unwrap();
        let report = |sel: &str| result.functions.iter().find(|f| f.selector == sel).unwrap();
        let found = report("00fdd58e");
        assert_eq!(found.status, BodyStatus::Done);
        assert_eq!(found.entry_pc, Some(25));
        assert_eq!(found.executed_pcs.first(), Some(&25));
        assert_eq!(found.effects.len(), 1);
        let wrong = report("22222222");
        assert_eq!(wrong.status, BodyStatus::Failed);
        assert!(wrong.failure.as_deref().unwrap().contains("entry_not_reached"), "{wrong:?}");
    }

    #[tokio::test]
    async fn only_a_literal_zero_size_renders_a_value_transfer() {
        use heimdall_vm::core::opcodes;

        // 0x11111111 (pc 26): call(gas, caller, callvalue, 0, 0, 0, 0) -- a plain ETH transfer
        // 0x22222222 (pc 41): execute(address target, bytes data) -- copies `data` to 0x80 and
        // calls target with it; with selector-only calldata its length evaluates to 0
        let code = "60003560e01c80631111111114601a57632222222214602957005b600060006000600034335af150005b60243560040180359060200181906080376000600082608060006004355af1505000";
        let abi = |inputs: &[&str]| SelectorAbi {
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: None,
        };
        let args = DecompilerArgsBuilder::new()
            .target(code.to_string())
            .skip_resolving(true)
            .include_solidity(true)
            .hardfork(HardFork::Fusaka)
            .selector_abi(Some(BTreeMap::from([
                ("11111111".to_string(), abi(&[])),
                ("22222222".to_string(), abi(&["address", "bytes"])),
            ])))
            .entry_points(Some(BTreeMap::from([
                ("11111111".to_string(), 26),
                ("22222222".to_string(), 41),
            ])))
            .build()
            .unwrap();
        let result = decompile(args).await.unwrap();
        let source = result.source.unwrap();
        let body = |name: &str| {
            let start = source.find(&format!("function {name}(")).unwrap();
            let rest = &source[start + 1..];
            rest[..rest.find("function ").unwrap_or(rest.len())].to_string()
        };
        assert!(body("func_11111111").contains(".transfer("), "{source}");
        assert!(!body("func_22222222").contains(".transfer("), "{source}");

        // typed calldata: `data` is one byte long, so the forwarded size is 1, not 0
        let exec = result.functions.iter().find(|f| f.selector == "22222222").unwrap();
        let call = exec.effects.iter().find(|e| e.opcode == opcodes::CALL).unwrap();
        assert_eq!(call.selector, Some(crate::EffectOperand::Dynamic));
        assert_eq!(call.target, Some(crate::EffectOperand::Arg(0)));
    }

    #[tokio::test]
    async fn supplied_abi_types_the_header_and_skips_signature_matching() {
        let mut selector_abi = BTreeMap::new();
        selector_abi.insert(
            "a9059cbb".to_string(),
            SelectorAbi { inputs: vec!["address".into(), "uint256".into()], outputs: None },
        );
        let args = DecompilerArgsBuilder::new()
            .target(TINY.to_string())
            .skip_resolving(true)
            .include_solidity(true)
            .hardfork(HardFork::Fusaka)
            .selector_abi(Some(selector_abi))
            .build()
            .unwrap();
        let result = decompile(args).await.unwrap();

        // inference alone sees one argument; the supplied ABI has two
        let function = &result.abi.functions["func_a9059cbb"][0];
        let inputs = function.inputs.iter().map(|p| p.ty.as_str()).collect::<Vec<_>>();
        assert_eq!(inputs, vec!["address", "uint256"]);
        let source = result.source.unwrap();
        assert!(
            source.contains("function func_a9059cbb(address arg0, uint256 arg1)"),
            "{source}"
        );
    }
}
