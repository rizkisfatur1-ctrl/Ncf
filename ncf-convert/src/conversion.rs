//! High-performance conversion utilities for NCF format conversion.
//!
//! This module provides:
//! - Parallel tensor conversion
//! - Optimized dtype mapping
//! - Memory-efficient streaming
//! - Progress tracking

use ncf_core::schema::{DType, Compression};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Progress tracker for conversion operations
pub struct ConversionProgress {
    total_tensors: usize,
    completed_tensors: Arc<AtomicUsize>,
    total_bytes: u64,
    processed_bytes: Arc<AtomicUsize>,
}

impl ConversionProgress {
    /// Create a new progress tracker
    pub fn new(total_tensors: usize, total_bytes: u64) -> Self {
        Self {
            total_tensors,
            completed_tensors: Arc::new(AtomicUsize::new(0)),
            total_bytes,
            processed_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Mark a tensor as completed
    pub fn mark_tensor_done(&self, bytes: usize) {
        self.completed_tensors.fetch_add(1, Ordering::Relaxed);
        self.processed_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get current progress percentage
    pub fn percent(&self) -> f32 {
        let completed = self.completed_tensors.load(Ordering::Relaxed);
        if self.total_tensors == 0 {
            100.0
        } else {
            (completed as f32 / self.total_tensors as f32) * 100.0
        }
    }

    /// Get bytes progress percentage
    pub fn bytes_percent(&self) -> f32 {
        if self.total_bytes == 0 {
            100.0
        } else {
            let processed = self.processed_bytes.load(Ordering::Relaxed);
            (processed as f64 / self.total_bytes as f64 * 100.0) as f32
        }
    }

    /// Check if conversion is complete
    pub fn is_complete(&self) -> bool {
        self.completed_tensors.load(Ordering::Relaxed) >= self.total_tensors
    }
}

/// Optimal compression selection based on tensor characteristics
pub fn select_compression(tensor_size: u64, dtype: DType) -> Compression {
    // Skip compression for very small tensors
    if tensor_size < 4096 {
        return Compression::None;
    }

    // Use Zstd with level 10 for most cases (good balance of speed/compression)
    match dtype {
        DType::F32 | DType::F64 => Compression::Zstd(10),
        DType::I32 | DType::I16 | DType::I8 => Compression::Zstd(11),
        DType::Q4K | DType::Q4_0 | DType::Q8_0 => Compression::Lz4,
        _ => Compression::Zstd(10),
    }
}

/// Memory-efficient buffer for tensor data conversion
pub struct ConversionBuffer {
    data: Vec<u8>,
    capacity: usize,
}

impl ConversionBuffer {
    /// Create a new conversion buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Append data to the buffer
    pub fn append(&mut self, data: &[u8]) -> Result<(), String> {
        if self.data.len() + data.len() > self.capacity {
            return Err(format!(
                "Buffer overflow: {} + {} > {}",
                self.data.len(),
                data.len(),
                self.capacity
            ));
        }
        self.data.extend_from_slice(data);
        Ok(())
    }

    /// Get buffer contents
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Take buffer ownership
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.data)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Tensor statistics for optimization
#[derive(Debug, Clone, Default)]
pub struct TensorStats {
    /// Total tensors converted
    pub tensor_count: u64,
    /// Total bytes converted
    pub total_bytes: u64,
    /// Smallest tensor size
    pub min_size: u64,
    /// Largest tensor size
    pub max_size: u64,
    /// Average tensor size
    pub avg_size: f64,
}

impl TensorStats {
    /// Update statistics with a new tensor
    pub fn update(&mut self, size: u64) {
        self.tensor_count += 1;
        self.total_bytes += size;
        
        if self.tensor_count == 1 {
            self.min_size = size;
            self.max_size = size;
        } else {
            self.min_size = self.min_size.min(size);
            self.max_size = self.max_size.max(size);
        }
        
        self.avg_size = self.total_bytes as f64 / self.tensor_count as f64;
    }
}

/// Streaming converter for large models
pub struct StreamingConverter {
    batch_size: usize,
    compression: Compression,
    progress: Option<Arc<ConversionProgress>>,
}

impl StreamingConverter {
    /// Create a new streaming converter
    pub fn new(batch_size: usize, compression: Compression) -> Self {
        Self {
            batch_size,
            compression,
            progress: None,
        }
    }

    /// Set progress tracker
    pub fn with_progress(mut self, progress: Arc<ConversionProgress>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get compression method
    pub fn compression(&self) -> Compression {
        self.compression
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracking() {
        let progress = ConversionProgress::new(100, 1024 * 1024);
        assert_eq!(progress.percent(), 0.0);
        
        for _ in 0..50 {
            progress.mark_tensor_done(10240);
        }
        assert_eq!(progress.percent(), 50.0);
    }

    #[test]
    fn test_compression_selection() {
        let comp1 = select_compression(10000, DType::F32);
        assert_eq!(comp1, Compression::Zstd(10));
        
        let comp2 = select_compression(1000, DType::F32);
        assert_eq!(comp2, Compression::None);
    }

    #[test]
    fn test_conversion_buffer() {
        let mut buf = ConversionBuffer::new(1024);
        buf.append(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.data(), &[1, 2, 3, 4, 5]);
    }
}
