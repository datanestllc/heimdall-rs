use alloy::primitives::U256;

/// What the decompiler knows about one operand of an [`Effect`], read from the symbolic
/// expression of the operand on the path that executed the opcode.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EffectOperand {
    /// A constant of the bytecode (a PUSH, or arithmetic on PUSHes).
    Const(U256),
    /// The value of this constant storage slot (masks and shifts of a packed slot included).
    Slot(U256),
    /// An element of a mapping or dynamic array rooted at this slot (Solidity key layout).
    Mapping(U256),
    /// Function argument N: the calldata word at `4 + 32 * N`.
    Arg(u32),
    /// `address(this)`.
    SelfAddress,
    /// `msg.sender`.
    Caller,
    /// Computed at run time from calldata, returndata, memory or anything not listed above.
    Dynamic,
}

/// Why a REVERT reverts, read from the revert data in memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RevertReason {
    /// A custom error: its 4-byte selector, with leading zero bytes kept.
    Custom([u8; 4]),
    /// `Error(string)`, with the message when the whole revert data is constant.
    ErrorString(Option<String>),
    /// `Panic(uint256)` with its code.
    Panic(U256),
    /// Empty revert data (the size operand is the literal 0).
    Empty,
}

/// One effectful opcode of a function, recorded when the opcode was analyzed, from the state of
/// the path that executed it.
///
/// Operands are `None` when the opcode has no such operand, or when the decompiler cannot tell
/// (a REVERT whose data comes from returndata, a LOG whose topic0 is not a constant).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Effect {
    /// 0-based program counter of the opcode.
    pub pc: u128,
    /// The opcode: CALL, CALLCODE, DELEGATECALL, STATICCALL, CREATE, CREATE2, SELFDESTRUCT,
    /// SSTORE, TSTORE, LOG0..LOG4 or REVERT.
    pub opcode: u8,
    /// Call target or SELFDESTRUCT beneficiary.
    pub target: Option<EffectOperand>,
    /// First four bytes of the call data, read from memory at the call: `Const` (the selector as
    /// a number) or `Dynamic`. `None` when the call data size is the literal 0, or the target is a
    /// precompile.
    pub selector: Option<EffectOperand>,
    /// Wei sent (CALL, CALLCODE, CREATE, CREATE2).
    pub value: Option<EffectOperand>,
    /// Storage slot (SSTORE, TSTORE).
    pub slot: Option<EffectOperand>,
    /// Full topic0 (LOG1..LOG4), when it is a constant.
    pub topic0: Option<U256>,
    /// Revert reason (REVERT), when it can be read.
    pub revert: Option<RevertReason>,
}
