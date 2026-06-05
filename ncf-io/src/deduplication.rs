//! Content-addressable storage with deduplication
//!
//! Features:
//! - Content hash-based addressing
//! - Duplicate detection
//! - Reference counting
//! - Space savings tracking

#![allow(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

pub type ContentHash = u64;

#[derive(Debug, Clone)]
pub struct DeduplicatedChunk {
    pub hash: ContentHash,
    pub data: Arc<Vec<u8>>,
    pub references: usize,
}

pub struct DeduplicationStore {
    chunks: Arc<RwLock<HashMap<ContentHash, DeduplicatedChunk>>>,
    saved_bytes: Arc<RwLock<u64>>,
}

impl DeduplicationStore {
    pub fn new() -> Self {
        Self {
            chunks: Arc::new(RwLock::new(HashMap::new())),
            saved_bytes: Arc::new(RwLock::new(0)),
        }
    }

    /// Quick content hash (simple for demo, use blake3 in production)
    pub fn compute_hash(data: &[u8]) -> ContentHash {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Store or retrieve chunk by content
    pub fn store_or_get(&self, data: Arc<Vec<u8>>) -> (ContentHash, bool) {
        let hash = Self::compute_hash(&data);
        let mut chunks = self.chunks.write();

        if chunks.contains_key(&hash) {
            chunks.get_mut(&hash).map(|c| c.references += 1);
            (hash, false) // Already existed (duplicate detected)
        } else {
            chunks.insert(
                hash,
                DeduplicatedChunk {
                    hash,
                    data,
                    references: 1,
                },
            );
            (hash, true) // New chunk
        }
    }

    /// Get data by hash
    pub fn get(&self, hash: ContentHash) -> Option<Arc<Vec<u8>>> {
        self.chunks
            .read()
            .get(&hash)
            .map(|chunk| chunk.data.clone())
    }

    /// Statistics
    pub fn stats(&self) -> DeduplicationStats {
        let chunks = self.chunks.read();
        let total_chunks = chunks.len();
        let total_refs: usize = chunks.values().map(|c| c.references).sum();
        let duplicates = total_refs - total_chunks;
        let total_size: u64 = chunks.values().map(|c| c.data.len() as u64).sum();
        let saved_bytes = *self.saved_bytes.read();

        DeduplicationStats {
            total_chunks,
            total_references: total_refs,
            duplicates_found: duplicates,
            total_unique_bytes: total_size,
            saved_bytes,
            deduplication_ratio: if total_refs > 0 {
                (duplicates as f64 / total_refs as f64) * 100.0
            } else {
                0.0
            },
        }
    }

    /// Release reference to chunk
    pub fn release(&self, hash: ContentHash) {
        let mut chunks = self.chunks.write();
        if let Some(chunk) = chunks.get_mut(&hash) {
            if chunk.references > 1 {
                chunk.references -= 1;
            }
        }
    }

    /// Clear all chunks
    pub fn clear(&self) {
        self.chunks.write().clear();
        *self.saved_bytes.write() = 0;
    }
}

#[derive(Debug, Clone)]
pub struct DeduplicationStats {
    pub total_chunks: usize,
    pub total_references: usize,
    pub duplicates_found: usize,
    pub total_unique_bytes: u64,
    pub saved_bytes: u64,
    pub deduplication_ratio: f64,
}

impl std::fmt::Display for DeduplicationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Dedup(chunks={}, refs={}, dups={}, ratio={:.2}%, saved={} MB)",
            self.total_chunks,
            self.total_references,
            self.duplicates_found,
            self.deduplication_ratio,
            self.saved_bytes / 1_000_000
        )
    }
}

/// Batch deduplicator for large tensor collections
pub struct BatchDeduplicator {
    store: DeduplicationStore,
    batch_size: usize,
}

impl BatchDeduplicator {
    pub fn new(batch_size: usize) -> Self {
        Self {
            store: DeduplicationStore::new(),
            batch_size,
        }
    }

    pub fn deduplicate_batch(&self, chunks: Vec<Arc<Vec<u8>>>) -> Vec<ContentHash> {
        chunks
            .into_iter()
            .map(|data| {
                let (hash, _is_new) = self.store.store_or_get(data);
                hash
            })
            .collect()
    }

    pub fn stats(&self) -> DeduplicationStats {
        self.store.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduplication() {
        let store = DeduplicationStore::new();
        let data1 = Arc::new(vec![1, 2, 3, 4, 5]);
        let data2 = Arc::new(vec![1, 2, 3, 4, 5]); // Same content

        let (hash1, new1) = store.store_or_get(data1);
        let (hash2, new2) = store.store_or_get(data2);

        assert_eq!(hash1, hash2);
        assert!(new1); // First is new
        assert!(!new2); // Second is duplicate
    }

    #[test]
    fn test_dedup_stats() {
        let store = DeduplicationStore::new();
        let data = Arc::new(vec![1, 2, 3]);

        for _ in 0..3 {
            store.store_or_get(data.clone());
        }

        let stats = store.stats();
        assert_eq!(stats.total_chunks, 1);
        assert_eq!(stats.total_references, 3);
        assert_eq!(stats.duplicates_found, 2);
    }
}
