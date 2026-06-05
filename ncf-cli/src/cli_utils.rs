//! Enhanced CLI utilities with progress tracking and parallelism
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ncf_core::schema::TensorSchema;
use ncf_io::NcfReader;
use std::sync::Arc;

/// Create a progress bar for file operations
pub fn create_progress_bar(total: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a spinner for indeterminate operations
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(message.to_string());
    pb
}

/// Parallel tensor verification with progress
pub fn verify_tensors_parallel(
    reader: Arc<NcfReader>,
    schemas: Vec<TensorSchema>,
) -> anyhow::Result<bool> {
    let multi_progress = MultiProgress::new();
    let total = schemas.len() as u64;
    let pb = multi_progress.add(create_progress_bar(total, "Verifying tensors"));
    
    let results: Vec<bool> = schemas
        .into_iter()
        .map(|schema| {
            let result = reader.verify_tensor(&schema.name).unwrap_or(false);
            pb.inc(1);
            result
        })
        .collect();
    
    pb.finish_with_message("Verification complete");
    
    Ok(results.iter().all(|&v| v))
}

/// Format file size for display
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_idx])
}

/// Display tensor information in a formatted table
pub fn display_tensor_info(schema: &TensorSchema) {
    let elements = schema.shape.iter().product::<u64>();
    let bytes = elements * schema.dtype.size_bytes() as u64;
    
    print!("  {} | ", schema.name);
    print!("{} | ", schema.dtype);
    print!("{:?} | ", schema.shape);
    print!("{} ", format_size(bytes));
    println!();
}

/// Display compression statistics
pub fn display_compression_stats(
    original_bytes: u64,
    compressed_bytes: u64,
) {
    let ratio = if original_bytes > 0 {
        (compressed_bytes as f64 / original_bytes as f64) * 100.0
    } else {
        0.0
    };
    
    let saved = original_bytes.saturating_sub(compressed_bytes);
    
    println!("Compression Statistics:");
    println!("  Original:   {}", format_size(original_bytes));
    println!("  Compressed: {}", format_size(compressed_bytes));
    println!("  Ratio:      {:.2}%", ratio);
    println!("  Saved:      {}", format_size(saved));
}

/// Conversion statistics formatter
#[derive(Debug, Clone, Default)]
pub struct ConversionStats {
    pub input_size: u64,
    pub output_size: u64,
    pub tensor_count: u64,
    pub duration_secs: f64,
}

impl ConversionStats {
    /// Display formatted statistics
    pub fn display(&self) {
        println!("\nConversion Statistics:");
        println!("  Input:       {}", format_size(self.input_size));
        println!("  Output:      {}", format_size(self.output_size));
        println!("  Tensors:     {}", self.tensor_count);
        println!("  Duration:    {:.2}s", self.duration_secs);
        
        if self.duration_secs > 0.0 {
            let throughput_mbps = (self.input_size as f64 / 1024.0 / 1024.0) / self.duration_secs;
            println!("  Throughput:  {:.2} MB/s", throughput_mbps);
        }
        
        let ratio = if self.input_size > 0 {
            (self.output_size as f64 / self.input_size as f64) * 100.0
        } else {
            0.0
        };
        println!("  Size ratio:  {:.2}%", ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_conversion_stats_display() {
        let stats = ConversionStats {
            input_size: 1024 * 1024,
            output_size: 512 * 1024,
            tensor_count: 10,
            duration_secs: 2.0,
        };
        stats.display();
    }
}
