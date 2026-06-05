//! Parallel I/O optimization utilities for NCF writer operations.
#![allow(dead_code)]

use ncf_core::schema::Compression;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Parallel chunk compression using rayon
pub fn compress_chunks_parallel(
    chunks: Vec<(Vec<u8>, Compression)>,
) -> Result<Vec<(Vec<u8>, u64, u64)>, String> {
    chunks
        .into_par_iter()
        .map(|(data, compression)| {
            let uncompressed_len = data.len() as u64;
            let compressed = compress_payload(&data, compression)?;
            let compressed_len = compressed.len() as u64;
            Ok((compressed, uncompressed_len, compressed_len))
        })
        .collect()
}

/// Compress a single payload with the specified compression method
fn compress_payload(data: &[u8], compression: Compression) -> Result<Vec<u8>, String> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Zstd(level) => {
            zstd::encode_all(data, level.into()).map_err(|e| format!("Zstd error: {}", e))
        }
        Compression::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),
        Compression::Snappy => {
            use snap::raw::Encoder;
            let mut encoder = Encoder::new();
            encoder.compress_vec(data).map_err(|e| format!("Snappy error: {}", e))
        }
    }
}

/// Progress tracker for write operations
pub struct WriteProgress {
    total_chunks: usize,
    completed_chunks: Arc<AtomicUsize>,
}

impl WriteProgress {
    /// Create a new progress tracker
    pub fn new(total_chunks: usize) -> Self {
        Self {
            total_chunks,
            completed_chunks: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Mark a chunk as completed
    pub fn mark_completed(&self) {
        self.completed_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current progress as percentage
    pub fn percent(&self) -> f32 {
        let completed = self.completed_chunks.load(Ordering::Relaxed);
        if self.total_chunks == 0 {
            100.0
        } else {
            (completed as f32 / self.total_chunks as f32) * 100.0
        }
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.completed_chunks.load(Ordering::Relaxed) >= self.total_chunks
    }
}

/// Buffered writer for efficient I/O
pub struct BufferedWriter {
    buffer: Vec<u8>,
    capacity: usize,
}

impl BufferedWriter {
    /// Create a new buffered writer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if buffer should be flushed
    pub fn should_flush(&self) -> bool {
        self.buffer.len() >= self.capacity / 2
    }

    /// Add data to buffer
    pub fn write(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Get buffer contents
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Take buffer contents
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracking() {
        let progress = WriteProgress::new(100);
        assert_eq!(progress.percent(), 0.0);
        
        for _ in 0..50 {
            progress.mark_completed();
        }
        assert_eq!(progress.percent(), 50.0);
        
        for _ in 0..50 {
            progress.mark_completed();
        }
        assert!(progress.is_complete());
        assert_eq!(progress.percent(), 100.0);
    }

    #[test]
    fn test_buffered_writer() {
        let mut writer = BufferedWriter::new(1024);
        assert!(writer.is_empty());
        
        writer.write(&[1, 2, 3, 4, 5]);
        assert_eq!(writer.len(), 5);
        assert!(!writer.is_empty());
        
        let data = writer.take();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
        assert!(writer.is_empty());
    }
}
