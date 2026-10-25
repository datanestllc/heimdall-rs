use alloy::primitives::Address;
use clap::Parser;
use derive_builder::Builder;
use eyre::Result;
use heimdall_common::ether::bytecode::get_bytecode_from_target;
use heimdall_config::parse_url_arg;
use heimdall_vm::{
    core::hardfork::HardFork,
    ext::exec::{ExecBudget, ExecStats},
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

/// Argument and return types for one selector, from a source the caller trusts more than the
/// decompiler's own inference: an interface extractor such as evmole, or a verified ABI.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectorAbi {
    /// Solidity types of the inputs, in order, e.g. `["address", "(uint256,bytes)[]"]`.
    pub inputs: Vec<String>,
    /// Solidity types of the outputs, when known. `None` keeps the inferred return type.
    pub outputs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Parser, Builder)]
#[clap(
    about = "Decompiles EVM bytecode to human-readable representations",
    after_help = "For more information, read the wiki: https://jbecker.dev/r/heimdall-rs/wiki",
    override_usage = "heimdall decompile <TARGET> [OPTIONS]"
)]
/// Arguments for the decompile operation
///
/// This struct contains all the configuration parameters needed to decompile
/// bytecode into human-readable source code and ABI.
pub struct DecompilerArgs {
    /// The target to decompile, either a file, bytecode, contract address, or ENS name.
    #[clap(required = true)]
    pub target: String,

    /// The RPC provider to use for fetching target bytecode.
    /// This can be an explicit URL or a reference to a MESC endpoint.
    #[clap(long, short, value_parser = parse_url_arg, default_value = "", hide_default_value = true)]
    pub rpc_url: String,

    /// When prompted, always select the default value.
    #[clap(long, short)]
    pub default: bool,

    /// Whether to skip resolving function selectors.
    #[clap(long = "skip-resolving")]
    pub skip_resolving: bool,

    /// Check emitted logs for an unambiguous indexed event layout (disabled by default).
    #[clap(long)]
    #[builder(default)]
    pub indexed_log_check: bool,

    /// The emitting address to check, e.g. a proxy when decompiling its implementation.
    /// Defaults to the target address. Required for bytecode/file targets.
    #[clap(long, requires = "indexed_log_check")]
    #[builder(default)]
    pub indexed_log_address: Option<Address>,

    /// First block to check. Defaults to the last 10,000 blocks ending at the upper bound.
    #[clap(long, requires = "indexed_log_check")]
    #[builder(default)]
    pub indexed_log_from_block: Option<u64>,

    /// Last block to check. Defaults to the latest block. At most 10,000 blocks are checked.
    #[clap(long, requires = "indexed_log_check")]
    #[builder(default)]
    pub indexed_log_to_block: Option<u64>,

    /// Whether to include solidity source code in the output (in beta).
    #[clap(long = "include-sol")]
    pub include_solidity: bool,

    /// Whether to include yul source code in the output (in beta).
    #[clap(long = "include-yul")]
    pub include_yul: bool,

    /// The output directory to write the output to or 'print' to print to the console
    #[clap(long = "output", short = 'o', default_value = "output", hide_default_value = true)]
    pub output: String,

    /// The name for the output file
    #[clap(long, short, default_value = "", hide_default_value = true)]
    pub name: String,

    /// The timeout for each function's symbolic execution in milliseconds.
    #[clap(long, short, default_value = "10000", hide_default_value = true)]
    pub timeout: u64,

    /// Deterministic budget: VM steps per function, across all of its paths (0 = unlimited).
    /// A path that runs out is kept with a `path not explored` marker.
    #[clap(long, default_value = "0", hide_default_value = true)]
    #[builder(default)]
    pub step_budget: u64,

    /// Deterministic budget: VM steps for all functions together (0 = unlimited). Functions are
    /// explored in selector order, so the split is deterministic.
    #[clap(long, default_value = "0", hide_default_value = true)]
    #[builder(default)]
    pub job_step_budget: u64,

    /// Deterministic budget: JUMPI forks per function (0 = unlimited). Past it, only the side
    /// the VM takes is followed.
    #[clap(long, default_value = "0", hide_default_value = true)]
    #[builder(default)]
    pub branch_budget: u32,

    /// Deterministic budget: approximate bytes of saved VM states per function (0 = unlimited).
    #[clap(long, default_value = "0", hide_default_value = true)]
    #[builder(default)]
    pub state_budget_bytes: u64,

    /// Deterministic budget: approximate bytes of saved VM states of all functions together
    /// (0 = unlimited). Every function's trace is kept until analysis, so this bounds the memory
    /// of the job.
    #[clap(long, default_value = "0", hide_default_value = true)]
    #[builder(default)]
    pub job_state_budget_bytes: u64,

    /// Wall-clock emergency stop for the whole decompilation in milliseconds (0 = none). Unlike
    /// the budgets above, it depends on machine load.
    #[clap(long, default_value = "0", hide_default_value = true)]
    #[builder(default)]
    pub job_timeout: u64,

    /// Path to an optional ABI file to use for resolving errors, functions, and events.
    #[clap(long, short, default_value = None, hide_default_value = true)]
    pub abi: Option<String>,

    /// Function entries for library callers: selector (8 lowercase hex characters, no `0x`) →
    /// 0-based bytecode offset of the body, e.g. evmole's `bytecodeOffset`. For these selectors
    /// the VM executes concretely until the pc is exactly the offset, then explores from there; a
    /// body it cannot reach is reported `entry_not_reached`. Selectors found by heimdall itself
    /// and not listed here keep its own dispatcher walk.
    #[clap(skip)]
    #[builder(default)]
    pub entry_points: Option<BTreeMap<String, u128>>,

    /// Also explore the two entries that are not selectors, each from pc 0: empty calldata
    /// (`receive`) and a selector the dispatcher does not match (`fallback`). Dispatch on the
    /// selector or the calldata size follows that calldata; other branches fork as usual.
    #[clap(long)]
    #[builder(default)]
    pub fallback_entries: bool,

    /// Argument and return types per selector (8 lowercase hex characters, no `0x`), for library
    /// callers. For these selectors the given types are used as they are: signature matching is
    /// skipped, the header and the body are typed from them, and the function is named
    /// `func_<selector>`. Nothing is written to the signature cache.
    #[clap(skip)]
    #[builder(default)]
    pub selector_abi: Option<BTreeMap<String, SelectorAbi>>,

    /// Whether to post-process the output using a LLM.
    #[clap(long, short)]
    pub llm_postprocess: bool,

    /// Your OpenRouter API key, used for LLM post-processing.
    #[clap(long, default_value = "", hide_default_value = true)]
    pub openrouter_api_key: String,

    /// The model to use for LLM post-processing (e.g., "openai/gpt-4o-mini",
    /// "anthropic/claude-3-haiku").
    #[clap(long, short = 'm', default_value = "", hide_default_value = true)]
    pub model: String,

    /// Your Etherscan API key, used for fetching creation bytecode of self-destructed contracts.
    #[clap(long, default_value = "", hide_default_value = true)]
    pub etherscan_api_key: String,

    /// The hardfork to use for opcode recognition. Opcodes introduced after this hardfork
    /// will be treated as unknown. Defaults to 'latest'.
    #[clap(long, short = 'f', default_value = "latest")]
    pub hardfork: HardFork,
}

impl DecompilerArgs {
    /// The symbolic execution budget for one function, given what the job has used so far and
    /// the job's wall-clock deadline.
    pub(crate) fn function_budget(
        &self,
        job_used: &ExecStats,
        job_deadline: Option<Instant>,
    ) -> ExecBudget {
        let unlimited = |n: u64| if n == 0 { u64::MAX } else { n };
        let job_left = unlimited(self.job_step_budget).saturating_sub(job_used.steps);
        let job_bytes_left =
            unlimited(self.job_state_budget_bytes).saturating_sub(job_used.state_bytes);
        let function_deadline = Instant::now().checked_add(Duration::from_millis(self.timeout));
        let deadline = match (function_deadline, job_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        ExecBudget {
            max_steps: unlimited(self.step_budget).min(job_left),
            max_branches: if self.branch_budget == 0 { u32::MAX } else { self.branch_budget },
            max_state_bytes: unlimited(self.state_budget_bytes).min(job_bytes_left),
            deadline,
        }
    }

    /// Retrieves the bytecode for the specified target
    ///
    /// This method fetches the bytecode from a file, address, or directly from a hex string,
    /// depending on the target type provided in the arguments.
    ///
    /// For self-destructed contracts, if an Etherscan API key is configured and the chain is
    /// supported, this will attempt to fetch the creation bytecode from the deployment transaction.
    ///
    /// # Returns
    /// The raw bytecode as a vector of bytes
    pub async fn get_bytecode(&self) -> Result<Vec<u8>> {
        get_bytecode_from_target(&self.target, &self.rpc_url, &self.etherscan_api_key).await
    }

    /// Gets the hardfork to use for decompilation.
    ///
    /// If `hardfork` is set to `Auto`, attempts to detect the hardfork based on the
    /// contract's creation block. If detection fails, falls back to `Latest`.
    pub async fn get_hardfork(&self) -> HardFork {
        if self.hardfork != HardFork::Auto {
            return self.hardfork;
        }

        match self.detect_hardfork_from_creation_block().await {
            Some(fork) => fork,
            None => HardFork::Latest,
        }
    }

    /// Attempts to detect the hardfork based on the contract's creation block.
    async fn detect_hardfork_from_creation_block(&self) -> Option<HardFork> {
        if self.rpc_url.is_empty() {
            return None;
        }

        let address: Address = self.target.parse().ok()?;
        let chain_id = heimdall_common::ether::rpc::chain_id(&self.rpc_url).await.ok()?;
        let creation_block = self.get_creation_block(address, chain_id).await?;

        Some(HardFork::from_chain(chain_id, creation_block, None))
    }

    /// Gets the creation block for a contract address.
    async fn get_creation_block(&self, address: Address, chain_id: u64) -> Option<u64> {
        if !self.etherscan_api_key.is_empty() &&
            heimdall_common::ether::etherscan::is_supported_chain(chain_id)
        {
            if let Ok(block) = heimdall_common::ether::etherscan::get_contract_creation_block(
                address,
                &self.rpc_url,
                chain_id,
                &self.etherscan_api_key,
            )
            .await
            {
                return Some(block);
            }
        }

        heimdall_common::ether::rpc::get_contract_creation_block(address, &self.rpc_url).await.ok()
    }
}

impl DecompilerArgsBuilder {
    /// Creates a new DecompilerArgsBuilder with default values
    pub fn new() -> Self {
        Self {
            target: Some(String::new()),
            rpc_url: Some(String::new()),
            default: Some(true),
            skip_resolving: Some(false),
            indexed_log_check: Some(false),
            indexed_log_address: Some(None),
            indexed_log_from_block: Some(None),
            indexed_log_to_block: Some(None),
            include_solidity: Some(false),
            include_yul: Some(false),
            output: Some(String::new()),
            name: Some(String::new()),
            timeout: Some(10000),
            step_budget: Some(0),
            job_step_budget: Some(0),
            branch_budget: Some(0),
            state_budget_bytes: Some(0),
            job_state_budget_bytes: Some(0),
            job_timeout: Some(0),
            abi: Some(None),
            selector_abi: Some(None),
            fallback_entries: Some(false),
            entry_points: Some(None),
            llm_postprocess: Some(false),
            openrouter_api_key: Some(String::new()),
            model: Some(String::new()),
            etherscan_api_key: Some(String::new()),
            hardfork: Some(HardFork::Latest),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_log_lookup_is_opt_in_for_cli_and_library() {
        let cli = DecompilerArgs::try_parse_from(["decompile", "0x00"]).unwrap();
        let library = DecompilerArgsBuilder::new().target("0x00".to_string()).build().unwrap();
        assert!(!cli.indexed_log_check);
        assert!(!library.indexed_log_check);
        assert!(cli.indexed_log_from_block.is_none());
        assert!(library.indexed_log_address.is_none());
        assert!(DecompilerArgs::try_parse_from([
            "decompile",
            "0x00",
            "--indexed-log-from-block",
            "10"
        ])
        .is_err());
        let enabled = DecompilerArgs::try_parse_from([
            "decompile",
            "0x00",
            "--indexed-log-check",
            "--indexed-log-address",
            "0x0000000000000000000000000000000000000001",
            "--indexed-log-from-block",
            "10",
            "--indexed-log-to-block",
            "20",
        ])
        .unwrap();
        assert!(enabled.indexed_log_check);
        assert_eq!(enabled.indexed_log_from_block, Some(10));
        assert_eq!(enabled.indexed_log_to_block, Some(20));
    }
}
