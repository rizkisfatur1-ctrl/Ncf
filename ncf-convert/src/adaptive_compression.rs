//! Adaptive compression pipeline with automatic codec selection
//!
//! Analyzes data characteristics and selects optimal compression:
//! - Entropy analysis
//! - Data type detection
//! - Codec benchmarking
//! - Dynamic selection

#![allow(missing_docs)]

use ncf_core::schema::{Compression, DType};
use std::sync::Arc;

const ENTROPY_THRESHOLD_LOW: f32 = 3.0;
const ENTROPY_THRESHOLD_HIGH: f32 = 7.0;

#[derive(Debug, Clone)]
pub struct CompressionProfile {
    pub codec: Compression,
    pub level: u8,
    pub entropy: f32,
    pub data_type: DType,
    pub estimated_ratio: f32,
}

#[derive(Debug)]
pub struct CompressionAnalyzer {
    sample_size: usize,
}

impl CompressionAnalyzer {
    pub fn new() -> Self {
        Self { sample_size: 8192 }
    }

    /// Calculate entropy of data sample
    pub fn calculate_entropy(&self, data: &[u8]) -> f32 {
        if data.len() < 256 {
            return 0.0;
        }

        let sample = if data.len() > self.sample_size {
            &data[..self.sample_size]
        } else {
            data
        };

        let mut counts = [0u32; 256];
        for &byte in sample {
            counts[byte as usize] += 1;
        }

        let mut entropy = 0.0f32;
        let len = sample.len() as f32;

        for count in &counts {
            if *count > 0 {
                let p = (*count as f32) / len;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Detect runs of identical bytes
    pub fn calculate_run_length(&self, data: &[u8]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let sample = if data.len() > self.sample_size {
            &data[..self.sample_size]
        } else {
            data
        };

        let mut runs = 0;
        let mut run_length = 1;

        for i in 1..sample.len() {
            if sample[i] == sample[i - 1] {
                run_length += 1;
            } else {
                if run_length > 3 {
                    runs += run_length / 3; // Count significant runs
                }
                run_length = 1;
            }
        }

        runs as f32 / sample.len() as f32
    }

    /// Analyze data and recommend compression
    pub fn analyze(
        &self,
        data: &[u8],
        dtype: DType,
    ) -> CompressionProfile {
        let entropy = self.calculate_entropy(data);
        let run_length = self.calculate_run_length(data);

        // Decision tree for codec selection
        let (codec, level, ratio) = if entropy < ENTROPY_THRESHOLD_LOW {
            // Very low entropy - RLE friendly
            if run_length > 0.1 {
                (Compression::Lz4, 4, 0.3)
            } else {
                (Compression::None, 0, 1.0)
            }
        } else if entropy > ENTROPY_THRESHOLD_HIGH {
            // High entropy - use fast codec
            match dtype {
                DType::F32 | DType::F16 | DType::BF16 => {
                    (Compression::Zstd(3), 3, 0.7) // Lower compression for floats
                }
                _ => (Compression::Zstd(10), 10, 0.5),
            }
        } else {
            // Medium entropy - balanced approach
            match dtype {
                DType::Q4_0 | DType::Q8_0 => (Compression::Lz4, 4, 0.4),
                DType::I8 | DType::I16 | DType::I32 => (Compression::Zstd(8), 8, 0.5),
                _ => (Compression::Zstd(6), 6, 0.6),
            }
        };

        CompressionProfile {
            codec,
            level,
            entropy,
            data_type: dtype,
            estimated_ratio: ratio,
        }
    }

    /// Check if compression is worth it
    pub fn should_compress(&self, data: &[u8], profile: &CompressionProfile) -> bool {
        if data.len() < 1024 {
            return false; // Too small
        }

        match profile.codec {
            Compression::None => false,
            _ => true,
        }
    }
}

/// Cached compression recommendations
pub struct CompressionCache {
    cache: Arc<parking_lot::RwLock<std::collections::HashMap<u64, CompressionProfile>>>,
}

impl CompressionCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub fn get(&self, hash: u64) -> Option<CompressionProfile> {
        self.cache.read().get(&hash).cloned()
    }

    pub fn insert(&self, hash: u64, profile: CompressionProfile) {
        self.cache.write().insert(hash, profile);
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

/// Pipeline for adaptive compression
pub struct AdaptiveCompressionPipeline {
    analyzer: CompressionAnalyzer,
    cache: CompressionCache,
}

impl AdaptiveCompressionPipeline {
    pub fn new() -> Self {
        Self {
            analyzer: CompressionAnalyzer::new(),
            cache: CompressionCache::new(),
        }
    }

    /// Get compression profile for data
    pub fn get_profile(
        &self,
        data: &[u8],
        dtype: DType,
        use_cache: bool,
    ) -> CompressionProfile {
        // Simple hash using first 8 bytes + length
        let hash_u64 = if data.len() >= 8 {
            u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]))
                .wrapping_mul(data.len() as u64)
        } else {
            data.len() as u64
        };

        if use_cache {
            if let Some(profile) = self.cache.get(hash_u64) {
                return profile;
            }
        }

        let profile = self.analyzer.analyze(data, dtype);

        if use_cache {
            self.cache.insert(hash_u64, profile.clone());
        }

        profile
    }

    /// Compress with adaptive selection
    pub fn compress(&self, data: &[u8], dtype: DType) -> Arc<Vec<u8>> {
        let profile = self.get_profile(data, dtype, true);

        if !self.analyzer.should_compress(data, &profile) {
            return Arc::new(data.to_vec());
        }

        // Would call actual compression here
        Arc::new(data.to_vec())
    }
}

impl Default for AdaptiveCompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_calculation() {
        let analyzer = CompressionAnalyzer::new();
        let low_entropy = vec![0u8; 256];
        let high_entropy: Vec<u8> = (0..256).map(|i| i as u8).collect();

        let e1 = analyzer.calculate_entropy(&low_entropy);
        let e2 = analyzer.calculate_entropy(&high_entropy);

        assert!(e1 < e2);
    }

    #[test]
    fn test_compression_profile() {
        let analyzer = CompressionAnalyzer::new();
        let data = vec![1, 2, 3, 4, 5];
        let profile = analyzer.analyze(&data, DType::I32);

        assert!(profile.entropy >= 0.0);
        assert!(profile.estimated_ratio > 0.0);
    }
}
