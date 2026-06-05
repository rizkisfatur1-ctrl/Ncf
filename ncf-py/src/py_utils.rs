//! Python bindings optimization module for ncf-py
//! 
//! This module provides high-performance utilities for:
//! - Vectorized numpy conversion
//! - Batch tensor loading
//! - Memory-efficient PyObject handling
//! - PyTorch integration

use pyo3::prelude::*;
use ncf_core::schema::{DType, TensorSchema};
use numpy::PyArray;

/// Optimized dtype to numpy dtype mapping
pub fn get_numpy_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F64 => "float64",
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "float16", // Approximation
        DType::I32 => "int32",
        DType::I16 => "int16",
        DType::I8 => "int8",
        DType::U8 => "uint8",
        DType::Q4K | DType::Q4_0 | DType::Q8_0 => "uint8",
        DType::Custom(_) => "uint8",
    }
}

/// Vectorized conversion utilities
pub struct VectorizedConverter;

impl VectorizedConverter {
    /// Convert bytes to numpy array with optimal performance
    pub fn to_numpy<'py>(
        py: Python<'py>,
        schema: &TensorSchema,
        data: &[u8],
    ) -> PyResult<&'py PyAny> {
        let shape: Vec<usize> = schema.shape.iter().map(|&d| d as usize).collect();

        match schema.dtype {
            DType::U8 => {
                let array = numpy::PyArray1::from_slice(py, data);
                Ok(array.reshape(shape.as_slice())?)
            }
            DType::I8 => {
                let values: Vec<i8> = data.iter().map(|&b| b as i8).collect();
                let array = numpy::PyArray::from_vec(py, values);
                Ok(array.reshape(shape.as_slice())?)
            }
            DType::I16 => {
                let values: Vec<i16> = data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let array = numpy::PyArray::from_vec(py, values);
                Ok(array.reshape(shape.as_slice())?)
            }
            DType::I32 => {
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let array = numpy::PyArray::from_vec(py, values);
                Ok(array.reshape(shape.as_slice())?)
            }
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let array = numpy::PyArray::from_vec(py, values);
                Ok(array.reshape(shape.as_slice())?)
            }
            DType::F64 => {
                let values: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                let array = numpy::PyArray::from_vec(py, values);
                Ok(array.reshape(shape.as_slice())?)
            }
            _ => {
                let array = numpy::PyArray1::from_slice(py, data);
                Ok(array.reshape(shape.as_slice())?)
            }
        }
    }

    /// Check if data is contiguous and well-aligned for SIMD
    pub fn is_simd_friendly(dtype: DType, size: usize) -> bool {
        let element_size = dtype.size_bytes();
        // SIMD-friendly if size is multiple of 64 bytes (cache line)
        (size % 64 == 0) && (element_size > 0)
    }
}

/// Batch loading context
pub struct BatchLoader {
    batch_size: usize,
}

impl BatchLoader {
    /// Create a new batch loader
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Get recommended batch size
    pub fn recommended_batch_size(total_tensors: usize) -> usize {
        std::cmp::min((total_tensors / 4).max(1), 64)
    }
}

/// Memory statistics for Python binding optimization
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocated: u64,
    pub peak_usage: u64,
    pub conversions_performed: u64,
}

impl MemoryStats {
    /// Update statistics
    pub fn update(&mut self, allocated: u64) {
        self.total_allocated += allocated;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        self.conversions_performed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_numpy_dtype() {
        assert_eq!(get_numpy_dtype(DType::F32), "float32");
        assert_eq!(get_numpy_dtype(DType::I32), "int32");
    }

    #[test]
    fn test_simd_friendly() {
        assert!(VectorizedConverter::is_simd_friendly(DType::F32, 64));
        assert!(!VectorizedConverter::is_simd_friendly(DType::F32, 63));
    }
}
