/// OpenRouter API integration for AI-powered analysis.
/// OpenRouter provides access to multiple LLM providers through a unified API.
#[cfg(feature = "network")]
pub mod openrouter;

/// Transpose API integration for blockchain data access.
#[cfg(feature = "network")]
pub mod transpose;
