//! High-performance optimization utilities for NCF core operations.
//! This module provides SIMD-aware operations, parallel processing, and memory layout optimizations.

use crate::schema::DType;
use std::fmt;

/// SIMD block size for vectorized operations - aligned to cache line for better locality
pub const SIMD_BLOCK_SIZE: usize = 64;

/// Chunk batch for efficient processing
pub const BATCH_SIZE: usize = 1024;

/// Memory alignment for optimal SIMD performance
pub const MEMORY_ALIGNMENT: usize = 64;

/// Fast hash computation using xxhash3 for tensor identification
#[inline]
pub fn fast_hash(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

/// Compute optimal chunk size based on tensor dtype and target cache utilization
#[inline]
pub fn optimal_chunk_size(dtype: DType, _element_count: u64) -> u64 {
    let element_bytes = dtype.size_bytes() as u64;
    let target_chunk_mb = 16; // 16MB chunks for L3 cache efficiency
    let target_bytes = target_chunk_mb * 1024 * 1024;
    
    let ideal_chunk_elements = target_bytes / element_bytes;
    
    // Round up to nearest SIMD boundary
    let simd_elements = SIMD_BLOCK_SIZE as u64;
    ((ideal_chunk_elements + simd_elements - 1) / simd_elements) * simd_elements
}

/// Vectorized checksum calculation for batch validation
pub fn batch_checksum(chunks: &[&[u8]]) -> Vec<[u8; 32]> {
    chunks
        .iter()
        .map(|chunk| *blake3::hash(chunk).as_bytes())
        .collect()
}

/// Memory-efficient buffer pool for chunk processing
pub struct BufferPool {
    buffers: Vec<Vec<u8>>,
    buffer_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool with pre-allocated buffers
    pub fn new(buffer_size: usize, pool_size: usize) -> Self {
        let buffers = (0..pool_size)
            .map(|_| Vec::with_capacity(buffer_size))
            .collect();
        
        Self {
            buffers,
            buffer_size,
        }
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self) -> Option<Vec<u8>> {
        self.buffers.pop().or_else(|| {
            let mut buf = Vec::with_capacity(self.buffer_size);
            buf.resize(self.buffer_size, 0);
            Some(buf)
        })
    }

    /// Return a buffer to the pool
    pub fn release(&mut self, mut buffer: Vec<u8>) {
        buffer.clear();
        if self.buffers.len() < 256 {
            self.buffers.push(buffer);
        }
    }
}

impl fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferPool")
            .field("pool_size", &self.buffers.len())
            .field("buffer_size", &self.buffer_size)
            .finish()
    }
}

/// Parallel-aware tensor shape analyzer for optimal data layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimalLayout {
    /// Row-major layout (C-contiguous)
    RowMajor,
    /// Column-major layout (Fortran-contiguous)
    ColumnMajor,
    /// Tiled layout for large matrices
    Tiled {
        /// Tile size for this layout
        tile_size: usize
    },
}

/// Analyze optimal layout for tensor
pub fn analyze_optimal_layout(shape: &[u64]) -> OptimalLayout {
    if shape.len() < 2 {
        return OptimalLayout::RowMajor;
    }
    
    let total_elements: u64 = shape.iter().product();
    
    // Use tiling for very large matrices
    if total_elements > 1_000_000 {
        OptimalLayout::Tiled { tile_size: 1024 }
    } else if shape[0] > shape[1] {
        OptimalLayout::ColumnMajor
    } else {
        OptimalLayout::RowMajor
    }
}

/// High-precision metrics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total time in microseconds
    pub time_us: u64,
    /// Compression ratio (0.0 - 1.0, where < 1.0 means compression)
    pub compression_ratio: f64,
    /// Throughput in GB/s
    pub throughput_gbs: f64,
}

impl PerformanceMetrics {
    /// Calculate throughput from bytes and microseconds
    pub fn calculate_throughput(&mut self) {
        if self.time_us > 0 {
            self.throughput_gbs = (self.bytes_processed as f64 / 1e9) / (self.time_us as f64 / 1e6);
        }
    }

    /// Add another metrics result
    pub fn merge(&mut self, other: &PerformanceMetrics) {
        self.bytes_processed += other.bytes_processed;
        self.time_us += other.time_us;
        self.calculate_throughput();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_chunk_size() {
        let size = optimal_chunk_size(DType::F32, 1_000_000);
        assert!(size > 0);
        assert_eq!(size % SIMD_BLOCK_SIZE as u64, 0);
    }

    #[test]
    fn test_layout_analysis() {
        let layout1 = analyze_optimal_layout(&[100, 50]);
        assert_eq!(layout1, OptimalLayout::ColumnMajor);
        
        let layout2 = analyze_optimal_layout(&[50, 100]);
        assert_eq!(layout2, OptimalLayout::RowMajor);
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(1024, 4);
        let buf1 = pool.acquire();
        assert!(buf1.is_some());
        let buf2 = pool.acquire();
        assert!(buf2.is_some());
        
        if let Some(buf) = buf1 {
            pool.release(buf);
        }
    }
}
