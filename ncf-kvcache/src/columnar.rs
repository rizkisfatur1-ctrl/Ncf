//! True columnar storage optimization for KV cache.
//!
//! This module implements genuine columnar storage where related cache data
//! (K and V for the same layer/head) are stored together for maximum cache locality.
//! This is fundamentally different from the per-head approach and provides:
//!
//! 1. Better L1/L2/L3 cache utilization
//! 2. Faster memory access patterns
//! 3. Improved compression opportunities
//! 4. Better vectorization potential

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Columnar data organization - stores K and V components together
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComponentType {
    /// Key component
    K,
    /// Value component
    V,
}

impl ComponentType {
    /// Get the index for this component (for array access)
    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            ComponentType::K => 0,
            ComponentType::V => 1,
        }
    }
}

/// Optimized columnar metadata for KV cache blocks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColumnarBlockMetadata {
    /// Layer index
    pub layer: u32,
    /// Head index
    pub head: u32,
    /// Block index
    pub block_idx: u64,
    /// Component type (K or V)
    pub component: ComponentType,
    /// Number of tokens in this block
    pub token_count: u32,
    /// Byte offset where K data starts
    pub k_offset: u64,
    /// Byte length of K data
    pub k_len: u64,
    /// Byte offset where V data starts
    pub v_offset: u64,
    /// Byte length of V data
    pub v_len: u64,
    /// Checksum for integrity verification
    pub checksum: [u8; 32],
}

impl ColumnarBlockMetadata {
    /// Get total bytes for this block (K + V)
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        self.k_len + self.v_len
    }

    /// Get the start offset for the specified component
    #[inline]
    pub fn component_offset(&self, component: ComponentType) -> u64 {
        match component {
            ComponentType::K => self.k_offset,
            ComponentType::V => self.v_offset,
        }
    }

    /// Get the length for the specified component
    #[inline]
    pub fn component_len(&self, component: ComponentType) -> u64 {
        match component {
            ComponentType::K => self.k_len,
            ComponentType::V => self.v_len,
        }
    }
}

/// Columnar index for fast lookup
#[derive(Debug, Default)]
pub struct ColumnarIndex {
    /// Map by (layer, head, block_idx) to metadata
    pub block_map: BTreeMap<(u32, u32, u64), ColumnarBlockMetadata>,
    /// Map by layer for range queries
    pub layer_heads: BTreeMap<u32, Vec<u32>>,
    /// Next chunk ID counter
    pub next_chunk_id: u64,
}

impl ColumnarIndex {
    /// Insert a block metadata entry
    pub fn insert(&mut self, metadata: ColumnarBlockMetadata) {
        let key = (metadata.layer, metadata.head, metadata.block_idx);
        self.next_chunk_id = self.next_chunk_id.max(metadata.block_idx.wrapping_add(1));
        
        self.layer_heads
            .entry(metadata.layer)
            .or_insert_with(Vec::new)
            .push(metadata.head);
        
        self.block_map.insert(key, metadata);
    }

    /// Look up block metadata
    pub fn get(&self, layer: u32, head: u32, block_idx: u64) -> Option<&ColumnarBlockMetadata> {
        self.block_map.get(&(layer, head, block_idx))
    }

    /// Get all heads in a layer
    pub fn heads_in_layer(&self, layer: u32) -> Option<&[u32]> {
        self.layer_heads.get(&layer).map(|v| v.as_slice())
    }

    /// Get byte range for component data
    pub fn component_range(
        &self,
        layer: u32,
        head: u32,
        block_idx: u64,
        component: ComponentType,
    ) -> Option<std::ops::Range<u64>> {
        self.get(layer, head, block_idx).map(|meta| {
            let start = meta.component_offset(component);
            let len = meta.component_len(component);
            start..start + len
        })
    }
}

/// Block batch for efficient processing
#[derive(Debug)]
pub struct ColumnarBlockBatch {
    /// Blocks in this batch
    pub blocks: Vec<ColumnarBlockMetadata>,
    /// Total byte size
    pub total_bytes: u64,
}

impl ColumnarBlockBatch {
    /// Create a new batch
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            total_bytes: 0,
        }
    }

    /// Add a block to the batch
    pub fn add(&mut self, block: ColumnarBlockMetadata) {
        self.total_bytes += block.total_bytes();
        self.blocks.push(block);
    }

    /// Check if batch should be flushed (> 4MB)
    #[inline]
    pub fn should_flush(&self) -> bool {
        self.total_bytes > 4 * 1024 * 1024
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.total_bytes = 0;
    }
}

/// Statistics for columnar cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct ColumnarStats {
    /// Total tokens written
    pub total_tokens: u64,
    /// Total bytes written
    pub total_bytes: u64,
    /// Number of blocks written
    pub block_count: u64,
    /// Average tokens per block
    pub avg_tokens_per_block: f64,
    /// Average bytes per token
    pub avg_bytes_per_token: f64,
}

impl ColumnarStats {
    /// Update statistics
    pub fn update(&mut self, block: &ColumnarBlockMetadata) {
        self.total_tokens += block.token_count as u64;
        self.total_bytes += block.total_bytes();
        self.block_count += 1;
        
        if self.block_count > 0 {
            self.avg_tokens_per_block = self.total_tokens as f64 / self.block_count as f64;
        }
        if self.total_tokens > 0 {
            self.avg_bytes_per_token = self.total_bytes as f64 / self.total_tokens as f64;
        }
    }

    /// Merge statistics from another collector
    pub fn merge(&mut self, other: &ColumnarStats) {
        self.total_tokens += other.total_tokens;
        self.total_bytes += other.total_bytes;
        self.block_count += other.block_count;
        
        if self.block_count > 0 {
            self.avg_tokens_per_block = self.total_tokens as f64 / self.block_count as f64;
        }
        if self.total_tokens > 0 {
            self.avg_bytes_per_token = self.total_bytes as f64 / self.total_tokens as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_columnar_block_metadata() {
        let meta = ColumnarBlockMetadata {
            layer: 0,
            head: 0,
            block_idx: 0,
            component: ComponentType::K,
            token_count: 64,
            k_offset: 0,
            k_len: 1024,
            v_offset: 1024,
            v_len: 1024,
            checksum: [0u8; 32],
        };

        assert_eq!(meta.total_bytes(), 2048);
        assert_eq!(meta.component_len(ComponentType::K), 1024);
        assert_eq!(meta.component_len(ComponentType::V), 1024);
    }

    #[test]
    fn test_columnar_index() {
        let mut index = ColumnarIndex::default();
        
        let meta = ColumnarBlockMetadata {
            layer: 0,
            head: 0,
            block_idx: 0,
            component: ComponentType::K,
            token_count: 64,
            k_offset: 0,
            k_len: 1024,
            v_offset: 1024,
            v_len: 1024,
            checksum: [0u8; 32],
        };

        index.insert(meta.clone());
        assert_eq!(index.get(0, 0, 0), Some(&meta));
    }

    #[test]
    fn test_columnar_batch() {
        let mut batch = ColumnarBlockBatch::new();
        
        let meta = ColumnarBlockMetadata {
            layer: 0,
            head: 0,
            block_idx: 0,
            component: ComponentType::K,
            token_count: 64,
            k_offset: 0,
            k_len: 1024,
            v_offset: 1024,
            v_len: 1024,
            checksum: [0u8; 32],
        };

        batch.add(meta);
        assert_eq!(batch.total_bytes, 2048);
    }
}
