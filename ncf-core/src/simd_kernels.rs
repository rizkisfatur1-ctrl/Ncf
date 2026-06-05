//! Advanced SIMD kernels for compression, decompression, and data processing
//! 
//! This module provides hand-optimized SIMD implementations for critical paths:
//! - Fast memcpy with SIMD (AVX2/SSE2)
//! - Delta encoding/decoding
//! - Transpose operations
//! - Horizontal reductions

#![allow(missing_docs)]

use std::arch::x86_64::*;

pub const SIMD_ALIGN: usize = 64; // AVX-512 alignment
pub const CACHE_LINE: usize = 64;

/// SIMD-accelerated memory copy with prefetching
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_memcpy(dst: *mut u8, src: *const u8, len: usize) {
    let mut offset = 0;
    let simd_len = len & !31; // Align to 32 bytes (AVX2)

    while offset < simd_len {
        let v = _mm256_loadu_si256(src.add(offset) as *const __m256i);
        _mm256_storeu_si256(dst.add(offset) as *mut __m256i, v);
        offset += 32;
    }

    // Handle remainder
    while offset < len {
        *dst.add(offset) = *src.add(offset);
        offset += 1;
    }
}

/// SIMD delta encoding for integer data (reduces entropy)
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_delta_encode_i32(data: &[i32], out: &mut [i32]) -> usize {
    if data.len() < 8 {
        return 0; // Too small for SIMD
    }

    out[0] = data[0];
    let mut i = 1;
    let simd_len = (data.len() - 1) & !7;

    while i <= simd_len {
        let v0 = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(data.as_ptr().add(i - 1) as *const __m256i);
        let delta = _mm256_sub_epi32(v0, v1);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, delta);
        i += 8;
    }

    // Handle remainder
    while i < data.len() {
        out[i] = data[i] - data[i - 1];
        i += 1;
    }

    i
}

/// SIMD delta decoding
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_delta_decode_i32(data: &[i32], out: &mut [i32]) -> usize {
    if data.len() < 8 {
        return 0;
    }

    out[0] = data[0];
    let mut acc = _mm256_set1_epi32(data[0]);
    let mut i = 1;

    // Process in blocks with carries
    while i + 8 <= data.len() {
        let deltas = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        acc = _mm256_add_epi32(acc, deltas);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, acc);
        i += 8;
    }

    // Handle remainder
    let last_val = if i > 0 { out[i - 1] } else { data[0] };
    let mut accum = last_val;
    while i < data.len() {
        accum += data[i];
        out[i] = accum;
        i += 1;
    }

    i
}

/// SIMD transpose for 8x8 f32 matrix
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_transpose_8x8_f32(src: &[f32; 64], dst: &mut [f32; 64]) {
    // Load 8 rows of 8 floats each
    let rows: [__m256; 8] = [
        _mm256_loadu_ps(src.as_ptr()),
        _mm256_loadu_ps(src.as_ptr().add(8)),
        _mm256_loadu_ps(src.as_ptr().add(16)),
        _mm256_loadu_ps(src.as_ptr().add(24)),
        _mm256_loadu_ps(src.as_ptr().add(32)),
        _mm256_loadu_ps(src.as_ptr().add(40)),
        _mm256_loadu_ps(src.as_ptr().add(48)),
        _mm256_loadu_ps(src.as_ptr().add(56)),
    ];

    // First pass: interleave floats
    let a0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    let a1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    let a2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    let a3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    let a4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    let a5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    let a6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    let a7 = _mm256_unpackhi_ps(rows[6], rows[7]);

    // Second pass: interleave pairs
    let b0 = _mm256_shuffle_ps::<0x44>(a0, a2);
    let b1 = _mm256_shuffle_ps::<0xee>(a0, a2);
    let b2 = _mm256_shuffle_ps::<0x44>(a1, a3);
    let b3 = _mm256_shuffle_ps::<0xee>(a1, a3);
    let b4 = _mm256_shuffle_ps::<0x44>(a4, a6);
    let b5 = _mm256_shuffle_ps::<0xee>(a4, a6);
    let b6 = _mm256_shuffle_ps::<0x44>(a5, a7);
    let b7 = _mm256_shuffle_ps::<0xee>(a5, a7);

    // Permute to get final result
    let out: [__m256; 8] = [
        _mm256_permute2f128_ps::<0x20>(b0, b4),
        _mm256_permute2f128_ps::<0x20>(b1, b5),
        _mm256_permute2f128_ps::<0x20>(b2, b6),
        _mm256_permute2f128_ps::<0x20>(b3, b7),
        _mm256_permute2f128_ps::<0x31>(b0, b4),
        _mm256_permute2f128_ps::<0x31>(b1, b5),
        _mm256_permute2f128_ps::<0x31>(b2, b6),
        _mm256_permute2f128_ps::<0x31>(b3, b7),
    ];

    for (i, row) in out.iter().enumerate() {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i * 8), *row);
    }
}

/// Count non-zero elements with SIMD
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_count_nonzero_i32(data: &[i32]) -> u32 {
    let mut count: u32 = 0;
    let zero = _mm256_setzero_si256();
    let mut i = 0;

    while i + 8 <= data.len() {
        let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi32(v, zero);
        let mask = _mm256_movemask_epi8(cmp);
        
        // Count set bits in mask (inverted for non-zero)
        count += (0xFF_u32.wrapping_sub((mask & 0xFF) as u32)).count_ones();
        i += 8;
    }

    // Handle remainder
    while i < data.len() {
        if data[i] != 0 {
            count += 1;
        }
        i += 1;
    }

    count
}

/// SIMD-optimized horizontal max for f32
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn simd_hmax_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;

    while i + 8 <= data.len() {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        max_vec = _mm256_max_ps(max_vec, v);
        i += 8;
    }

    // Horizontal max
    let v128 = _mm256_castps256_ps128(max_vec);
    let v64 = _mm_max_ps(v128, _mm256_extractf128_ps::<1>(max_vec));
    let v32 = _mm_max_ps(v64, _mm_shuffle_ps::<0x4E>(v64, v64));
    let v16 = _mm_max_ps(v32, _mm_shuffle_ps::<0xB1>(v32, v32));
    
    let mut result = _mm_cvtss_f32(v16);

    // Handle remainder
    while i < data.len() {
        result = result.max(data[i]);
        i += 1;
    }

    result
}

/// Safe wrapper for simd_memcpy (fallback to regular copy)
pub fn memcpy_simd(dst: &mut [u8], src: &[u8]) {
    if dst.len() < src.len() || src.is_empty() {
        return;
    }

    if is_x86_feature_detected!("avx2") {
        unsafe {
            simd_memcpy(dst.as_mut_ptr(), src.as_ptr(), src.len());
        }
    } else {
        dst[..src.len()].copy_from_slice(src);
    }
}

/// Safe wrapper for delta encoding
pub fn delta_encode_i32_safe(data: &[i32]) -> Vec<i32> {
    let mut out = vec![0i32; data.len()];
    if is_x86_feature_detected!("avx2") && data.len() >= 8 {
        unsafe {
            simd_delta_encode_i32(data, &mut out);
        }
    } else {
        // Fallback
        if !data.is_empty() {
            out[0] = data[0];
            for i in 1..data.len() {
                out[i] = data[i] - data[i - 1];
            }
        }
    }
    out
}

/// Safe wrapper for delta decoding
pub fn delta_decode_i32_safe(data: &[i32]) -> Vec<i32> {
    let mut out = vec![0i32; data.len()];
    if is_x86_feature_detected!("avx2") && data.len() >= 8 {
        unsafe {
            simd_delta_decode_i32(data, &mut out);
        }
    } else {
        // Fallback
        if !data.is_empty() {
            out[0] = data[0];
            for i in 1..data.len() {
                out[i] = out[i - 1] + data[i];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encode_decode() {
        let data = vec![1, 3, 7, 6, 10, 15, 12, 20];
        let encoded = delta_encode_i32_safe(&data);
        let decoded = delta_decode_i32_safe(&encoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_hmax() {
        let data = vec![1.5, 3.2, 2.1, 5.9, 2.3];
        let max = if is_x86_feature_detected!("avx2") {
            unsafe { simd_hmax_f32(&data) }
        } else {
            data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        };
        assert_eq!(max, 5.9);
    }
}
