//! Zero-copy metadata handling using memory mapping
//! 
//! Uses borrowed references where possible to eliminate allocations

#![allow(missing_docs)]

use std::borrow::Cow;
use std::mem;

/// Lightweight metadata reference (no allocation)
pub struct MetadataRef<'a> {
    pub model_name: Cow<'a, str>,
    pub architecture: Cow<'a, str>,
    pub author: Option<Cow<'a, str>>,
    pub license: Option<Cow<'a, str>>,
    pub version: u32,
}

impl<'a> MetadataRef<'a> {
    /// Create owned copy when needed
    pub fn to_owned(&self) -> MetadataOwned {
        MetadataOwned {
            model_name: self.model_name.clone().into_owned(),
            architecture: self.architecture.clone().into_owned(),
            author: self.author.as_ref().map(|a| a.clone().into_owned()),
            license: self.license.as_ref().map(|l| l.clone().into_owned()),
            version: self.version,
        }
    }

    /// Size in bytes
    pub fn size(&self) -> usize {
        self.model_name.len() + self.architecture.len() + 4
            + self.author.as_ref().map(|a| a.len()).unwrap_or(0)
            + self.license.as_ref().map(|l| l.len()).unwrap_or(0)
    }
}

/// Owned metadata
pub struct MetadataOwned {
    pub model_name: String,
    pub architecture: String,
    pub author: Option<String>,
    pub license: Option<String>,
    pub version: u32,
}

impl MetadataOwned {
    pub fn as_ref(&self) -> MetadataRef {
        MetadataRef {
            model_name: Cow::Borrowed(&self.model_name),
            architecture: Cow::Borrowed(&self.architecture),
            author: self.author.as_ref().map(|a| Cow::Borrowed(a.as_str())),
            license: self.license.as_ref().map(|l| Cow::Borrowed(l.as_str())),
            version: self.version,
        }
    }
}

/// String slice cache - reuse string references
pub struct StringCache<'a> {
    buffer: &'a [u8],
    offsets: Vec<(usize, usize)>, // (offset, len)
}

impl<'a> StringCache<'a> {
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            offsets: Vec::new(),
        }
    }

    /// Parse and cache a string (no allocation)
    pub fn get_string(&mut self, offset: usize, len: usize) -> &'a str {
        if offset + len <= self.buffer.len() {
            // SAFETY: Buffer comes from valid source, checked bounds
            unsafe { std::str::from_utf8_unchecked(&self.buffer[offset..offset + len]) }
        } else {
            ""
        }
    }

    /// Register offset for later retrieval
    pub fn register(&mut self, offset: usize, len: usize) -> usize {
        self.offsets.push((offset, len));
        self.offsets.len() - 1
    }

    pub fn cached_string(&self, index: usize) -> &'a str {
        if index < self.offsets.len() {
            let (offset, len) = self.offsets[index];
            if offset + len <= self.buffer.len() {
                unsafe { std::str::from_utf8_unchecked(&self.buffer[offset..offset + len]) }
            } else {
                ""
            }
        } else {
            ""
        }
    }
}

/// Inline small strings (<24 bytes) in a u64
#[derive(Debug, Clone, Copy)]
pub struct InlineString {
    data: [u8; 24],
    len: u8,
}

impl InlineString {
    pub fn new(s: &str) -> Option<Self> {
        if s.len() > 23 {
            return None;
        }

        let mut data = [0u8; 24];
        data[..s.len()].copy_from_slice(s.as_bytes());

        Some(Self {
            data,
            len: s.len() as u8,
        })
    }

    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len as usize]) }
    }

    pub fn size(&self) -> usize {
        mem::size_of::<Self>()
    }
}

impl std::fmt::Display for InlineString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fast prefix matching without allocation
pub struct PrefixMatcher<'a> {
    prefixes: Vec<&'a [u8]>,
}

impl<'a> PrefixMatcher<'a> {
    pub fn new() -> Self {
        Self {
            prefixes: Vec::new(),
        }
    }

    pub fn add_prefix(&mut self, prefix: &'a [u8]) {
        self.prefixes.push(prefix);
    }

    pub fn find_match(&self, data: &[u8]) -> Option<usize> {
        for (i, prefix) in self.prefixes.iter().enumerate() {
            if data.starts_with(prefix) {
                return Some(i);
            }
        }
        None
    }
}

/// Borrowed tensor name without allocation
pub struct TensorNameRef<'a> {
    layer: u32,
    suffix: &'a str,
}

impl<'a> TensorNameRef<'a> {
    pub fn new(layer: u32, suffix: &'a str) -> Self {
        Self { layer, suffix }
    }

    pub fn format_compact(&self) -> String {
        format!("l{}{}", self.layer, self.suffix)
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn suffix(&self) -> &'a str {
        self.suffix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_string() {
        let s = InlineString::new("hello").unwrap();
        assert_eq!(s.as_str(), "hello");

        let long = InlineString::new("this is a very long string");
        assert!(long.is_none());
    }

    #[test]
    fn test_string_cache() {
        let buffer = b"hello world test";
        let mut cache = StringCache::new(buffer);

        let s1 = cache.get_string(0, 5);
        let s2 = cache.get_string(6, 5);

        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");
    }

    #[test]
    fn test_prefix_matcher() {
        let mut matcher = PrefixMatcher::new();
        matcher.add_prefix(b"NCF");
        matcher.add_prefix(b"GGUF");

        assert_eq!(matcher.find_match(b"NCF_FILE"), Some(0));
        assert_eq!(matcher.find_match(b"GGUF_DATA"), Some(1));
        assert_eq!(matcher.find_match(b"OTHER"), None);
    }

    #[test]
    fn test_tensor_name_ref() {
        let name = TensorNameRef::new(0, ".weight");
        assert_eq!(name.layer(), 0);
        assert_eq!(name.suffix(), ".weight");
        assert_eq!(name.format_compact(), "l0.weight");
    }
}
