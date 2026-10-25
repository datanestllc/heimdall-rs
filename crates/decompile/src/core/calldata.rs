//! Typed calldata for symbolic execution (F9).
//!
//! With only the 4-byte selector as calldata, every dynamic argument has length 0: a `bytes`
//! argument copied to memory is empty, so a call that forwards it has an argument size that
//! evaluates to 0, and loops over arrays read a length of 0. This builds calldata from the
//! argument types instead, the way evmole's `CallDataImpl` does: value words are 0, every dynamic
//! value gets a proper offset and a non-zero length (one element; `bytes` and `string` are `"A"`).

use alloy::primitives::U256;
use alloy_dyn_abi::DynSolType;

/// Calldata for `selector` (8 hex characters) called with arguments of `types`, or `None` when a
/// type does not parse. The result is at least `4 + 32 * types.len()` bytes long.
pub(crate) fn typed_calldata(selector: &str, types: &[String]) -> Option<Vec<u8>> {
    let selector = heimdall_common::utils::strings::decode_hex(selector).ok()?;
    let types = types.iter().map(|ty| DynSolType::parse(ty).ok()).collect::<Option<Vec<_>>>()?;
    let mut words = Vec::new();
    let size = encode(&types, 0, &mut words)?;
    let mut data = vec![0u8; size];
    for (offset, value) in words {
        data.get_mut(offset..offset + 32)?.copy_from_slice(&value.to_be_bytes::<32>());
    }
    Some([selector, data].concat())
}

fn is_dynamic(ty: &DynSolType) -> bool {
    match ty {
        DynSolType::Bytes | DynSolType::String | DynSolType::Array(_) => true,
        DynSolType::FixedArray(inner, _) => is_dynamic(inner),
        DynSolType::Tuple(inner) => inner.iter().any(is_dynamic),
        _ => false,
    }
}

/// Encode `types` at `base`, pushing the non-zero words as (absolute offset, value). Returns the
/// encoded size. Offsets of dynamic values are relative to `base`, as in the ABI.
fn encode(types: &[DynSolType], base: usize, words: &mut Vec<(usize, U256)>) -> Option<usize> {
    let mut head = 0;
    let mut dynamic = Vec::new();
    for ty in types {
        if is_dynamic(ty) {
            dynamic.push((head, ty));
            head += 32;
            continue;
        }
        head += match ty {
            DynSolType::FixedArray(inner, n) => {
                encode(&vec![(**inner).clone(); *n], base + head, words)?
            }
            DynSolType::Tuple(inner) => encode(inner, base + head, words)?,
            _ => 32,
        };
    }

    let mut tail = head;
    for (slot, ty) in dynamic {
        words.push((base + slot, U256::from(tail)));
        tail += match ty {
            DynSolType::Bytes | DynSolType::String => {
                // length 1, the byte "A", right-padded
                words.push((base + tail, U256::from(1)));
                words.push((base + tail + 32, U256::from(0x41) << 248));
                64
            }
            DynSolType::Array(inner) => {
                // length 1
                words.push((base + tail, U256::from(1)));
                32 + encode(&[(**inner).clone()], base + tail + 32, words)?
            }
            DynSolType::FixedArray(inner, n) => {
                encode(&vec![(**inner).clone(); *n], base + tail, words)?
            }
            DynSolType::Tuple(inner) => encode(inner, base + tail, words)?,
            _ => return None,
        };
    }
    Some(tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(data: &[u8], index: usize) -> U256 {
        U256::from_be_slice(&data[4 + 32 * index..4 + 32 * (index + 1)])
    }

    #[test]
    fn static_arguments_are_zero_words() {
        let data = typed_calldata("a9059cbb", &["address".into(), "uint256".into()]).unwrap();
        assert_eq!(data.len(), 4 + 64);
        assert_eq!(&data[..4], &[0xa9, 0x05, 0x9c, 0xbb]);
        assert!(data[4..].iter().all(|b| *b == 0));
    }

    #[test]
    fn dynamic_arguments_get_offsets_and_non_zero_lengths() {
        // (address, bytes, uint256[]): head of 3 words, then "A", then one array element
        let types = ["address".to_string(), "bytes".to_string(), "uint256[]".to_string()];
        let data = typed_calldata("12345678", &types).unwrap();
        assert_eq!(data.len(), 4 + 32 * 3 + 64 + 64);
        assert_eq!(word(&data, 1), U256::from(96));
        assert_eq!(word(&data, 2), U256::from(160));
        assert_eq!(word(&data, 3), U256::from(1));
        assert_eq!(word(&data, 4), U256::from(0x41) << 248);
        assert_eq!(word(&data, 5), U256::from(1));
        assert_eq!(word(&data, 6), U256::ZERO);
    }

    #[test]
    fn nested_dynamic_tuples_are_encoded_relative_to_their_base() {
        // ((address,bytes)[]): offset 32, length 1, offset of the tuple 32, tuple head, "A"
        let data = typed_calldata("12345678", &["(address,bytes)[]".into()]).unwrap();
        assert_eq!(word(&data, 0), U256::from(32));
        assert_eq!(word(&data, 1), U256::from(1));
        assert_eq!(word(&data, 2), U256::from(32));
        assert_eq!(word(&data, 4), U256::from(64));
        assert_eq!(word(&data, 5), U256::from(1));
        assert!(typed_calldata("12345678", &["(address".into()]).is_none());
    }
}
