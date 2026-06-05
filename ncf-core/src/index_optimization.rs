//! Advanced index optimization techniques
//! 
//! Features:
//! - B-tree indices for range queries
//! - Hash indices for point lookups
//! - Bloom filters for negative lookups
//! - Index compression

#![allow(missing_docs)]

use std::collections::BTreeMap;

pub type TensorHash = u64;
pub type Offset = u64;

/// Fast hash-based tensor index
pub struct HashIndex {
    index: std::collections::HashMap<TensorHash, Offset>,
}

impl HashIndex {
    pub fn new() -> Self {
        Self {
            index: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, hash: TensorHash, offset: Offset) {
        self.index.insert(hash, offset);
    }

    pub fn lookup(&self, hash: TensorHash) -> Option<Offset> {
        self.index.get(&hash).copied()
    }

    pub fn size(&self) -> usize {
        self.index.len()
    }
}

/// Range-queryable B-tree index
pub struct RangeIndex {
    tree: BTreeMap<String, Offset>,
}

impl RangeIndex {
    pub fn new() -> Self {
        Self {
            tree: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, offset: Offset) {
        self.tree.insert(name, offset);
    }

    pub fn lookup(&self, name: &str) -> Option<Offset> {
        self.tree.get(name).copied()
    }

    pub fn range_query(&self, prefix: &str) -> Vec<(String, Offset)> {
        self.tree
            .range(prefix.to_string()..=format!("{}~", prefix))
            .map(|(k, &v)| (k.clone(), v))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.tree.len()
    }
}

/// Bloom filter for negative lookups
pub struct BloomIndex {
    bits: Vec<u8>,
    size: u32,
    k: u32,
}

impl BloomIndex {
    pub fn new(capacity: usize) -> Self {
        let size = (capacity * 8) as u32; // bits
        let bits = vec![0u8; ((size + 7) / 8) as usize];
        let k = 4; // Number of hash functions

        Self { bits, size, k }
    }

    pub fn insert(&mut self, hash: TensorHash) {
        for i in 0..self.k {
            let pos = ((hash.wrapping_mul(i as u64)) % (self.size as u64)) as usize;
            let byte = pos / 8;
            let bit = pos % 8;
            if byte < self.bits.len() {
                self.bits[byte] |= 1 << bit;
            }
        }
    }

    pub fn might_contain(&self, hash: TensorHash) -> bool {
        for i in 0..self.k {
            let pos = ((hash.wrapping_mul(i as u64)) % (self.size as u64)) as usize;
            let byte = pos / 8;
            let bit = pos % 8;
            if byte >= self.bits.len() || (self.bits[byte] & (1 << bit)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn memory_used(&self) -> usize {
        self.bits.len()
    }
}

/// Two-level index (hash for hot path, B-tree for secondary)
pub struct HybridIndex {
    hash: HashIndex,
    btree: RangeIndex,
    bloom: BloomIndex,
}

impl HybridIndex {
    pub fn new(capacity: usize) -> Self {
        Self {
            hash: HashIndex::new(),
            btree: RangeIndex::new(),
            bloom: BloomIndex::new(capacity),
        }
    }

    pub fn insert_hash(&mut self, hash: TensorHash, offset: Offset) {
        self.hash.insert(hash, offset);
        self.bloom.insert(hash);
    }

    pub fn insert_range(&mut self, name: String, offset: Offset) {
        self.btree.insert(name, offset);
    }

    pub fn lookup_hash(&self, hash: TensorHash) -> Option<Offset> {
        if self.bloom.might_contain(hash) {
            self.hash.lookup(hash)
        } else {
            None
        }
    }

    pub fn lookup_range(&self, name: &str) -> Option<Offset> {
        self.btree.lookup(name)
    }

    pub fn range_query(&self, prefix: &str) -> Vec<(String, Offset)> {
        self.btree.range_query(prefix)
    }

    pub fn memory_used(&self) -> usize {
        let hash_size = self.hash.size() * 24; // Rough estimate
        let btree_size = self.btree.size() * 30;
        let bloom_size = self.bloom.memory_used();

        hash_size + btree_size + bloom_size
    }
}

/// Compressed index - delta encoding offsets
pub struct CompressedIndex {
    names: Vec<String>,
    offsets_delta: Vec<u32>,
}

impl CompressedIndex {
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            offsets_delta: Vec::new(),
        }
    }

    pub fn insert(&mut self, name: String, offset: u64) {
        let delta = if self.offsets_delta.is_empty() {
            offset as u32
        } else {
            (offset - self.last_offset() as u64) as u32
        };

        self.names.push(name);
        self.offsets_delta.push(delta);
    }

    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.get_offset(i))
    }

    fn get_offset(&self, index: usize) -> u64 {
        self.offsets_delta[..=index].iter().map(|&d| d as u64).sum()
    }

    fn last_offset(&self) -> u64 {
        if self.offsets_delta.is_empty() {
            0
        } else {
            self.get_offset(self.offsets_delta.len() - 1)
        }
    }

    pub fn memory_used(&self) -> usize {
        let names_size: usize = self.names.iter().map(|n| n.len()).sum();
        let deltas_size = self.offsets_delta.len() * 4;

        names_size + deltas_size
    }

    pub fn size(&self) -> usize {
        self.names.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_index() {
        let mut index = HashIndex::new();
        index.insert(12345, 1000);
        assert_eq!(index.lookup(12345), Some(1000));
        assert_eq!(index.lookup(99999), None);
    }

    #[test]
    fn test_range_index() {
        let mut index = RangeIndex::new();
        index.insert("layer_0".to_string(), 100);
        index.insert("layer_1".to_string(), 200);
        index.insert("layer_2".to_string(), 300);

        let results = index.range_query("layer");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomIndex::new(100);
        bloom.insert(12345);

        assert!(bloom.might_contain(12345));
        assert!(!bloom.might_contain(99999)); // Probably
    }

    #[test]
    fn test_compressed_index() {
        let mut index = CompressedIndex::new();
        index.insert("tensor_1".to_string(), 1000);
        index.insert("tensor_2".to_string(), 2000);
        index.insert("tensor_3".to_string(), 3000);

        assert_eq!(index.lookup("tensor_1"), Some(1000));
        assert_eq!(index.lookup("tensor_2"), Some(2000));
        assert_eq!(index.lookup("tensor_3"), Some(3000));
    }
}
