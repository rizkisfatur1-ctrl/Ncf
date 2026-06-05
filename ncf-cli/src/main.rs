mod cli_utils;

use clap::{Parser, Subcommand};
use ncf_convert::{gguf_to_ncf, safetensors_to_ncf};
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use cli_utils::*;

#[derive(Parser)]
#[command(author, version, about = "NCF CLI - High-performance model format converter", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect NCF file structure and contents
    Inspect {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Display basic file information
    Info {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Create an NCF file from binary data
    Create {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
        #[arg(long, default_value = "tensor")]
        name: String,
    },
    /// Convert safetensors to NCF format
    ConvertSafetensors {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
        #[arg(long)]
        architecture: Option<String>,
        #[arg(long)]
        author: Option<String>,
    },
    /// Convert GGUF to NCF format
    ConvertGguf {
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
        #[arg(long)]
        architecture: Option<String>,
        #[arg(long)]
        author: Option<String>,
    },
    /// Verify NCF file integrity
    Verify {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// List all tensors in NCF file
    List {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Get detailed statistics about NCF file
    Stats {
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { file } => {
            let reader = ncf_io::NcfReader::open(&file)?;
            reader.inspect()?;
        }
        Commands::Info { file } => {
            let reader = ncf_io::NcfReader::open(&file)?;
            let prefix = reader.header_prefix();
            println!("NCF Format Information");
            println!("  Version:        0x{:08x}", prefix.version);
            println!("  Flags:          {}", prefix.flags);
            println!("  Header length:  {} bytes", prefix.header_len);
            println!("  Schema offset:  {}", format_size(prefix.schema_offset));
            println!("  Index offset:   {}", format_size(prefix.index_offset));
            println!("  Chunk count:    {}", prefix.chunk_count);
            
            let metadata = reader.metadata();
            println!("\nModel Information");
            println!("  Name:           {}", metadata.metadata.model_name);
            println!("  Architecture:   {}", metadata.metadata.architecture);
            println!("  Created:        {} (Unix timestamp)", metadata.metadata.created_at);
            if let Some(author) = &metadata.metadata.author {
                println!("  Author:         {}", author);
            }
        }
        Commands::Create { input, output, name } => {
            let spinner = create_spinner("Reading input file...");
            let bytes = fs::read(&input)?;
            spinner.finish_with_message(format!("Read {} bytes", format_size(bytes.len() as u64)));
            
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
            let metadata = NcfHeader {
                metadata: Metadata {
                    model_name: name.clone(),
                    architecture: "generic".into(),
                    created_at: now,
                    author: None,
                    license: None,
                    quantization: None,
                    custom: BTreeMap::new(),
                },
            };
            let tensor_schema = TensorSchema {
                name: name.clone(),
                dtype: DType::U8,
                shape: vec![bytes.len() as u64],
                column_layout: Layout::RowMajor,
                compression: Compression::None,
                encoding: Encoding::Plain,
                chunks: Vec::new(),
            };
            
            let spinner = create_spinner("Writing NCF file...");
            let mut writer = NcfWriter::new(metadata, NcfFlags::empty());
            writer.add_tensor(tensor_schema, bytes);
            writer.finalize(&output)?;
            spinner.finish_with_message(format!("Created NCF file: {}", output.display()));
        }
        Commands::ConvertSafetensors { input, output, architecture, author } => {
            let input_size = fs::metadata(&input)?.len();
            let start = Instant::now();
            
            let spinner = create_spinner("Converting safetensors to NCF...");
            safetensors_to_ncf(&input, &output, architecture.as_deref(), author.as_deref())?;
            spinner.finish_with_message("Conversion complete");
            
            let output_size = fs::metadata(&output)?.len();
            let duration = start.elapsed().as_secs_f64();
            
            let stats = ConversionStats {
                input_size,
                output_size,
                tensor_count: 0, // Would need to count from actual conversion
                duration_secs: duration,
            };
            stats.display();
        }
        Commands::ConvertGguf { input, output, architecture, author } => {
            let input_size = fs::metadata(&input)?.len();
            let start = Instant::now();
            
            let spinner = create_spinner("Converting GGUF to NCF...");
            gguf_to_ncf(&input, &output, architecture.as_deref(), author.as_deref())?;
            spinner.finish_with_message("Conversion complete");
            
            let output_size = fs::metadata(&output)?.len();
            let duration = start.elapsed().as_secs_f64();
            
            let stats = ConversionStats {
                input_size,
                output_size,
                tensor_count: 0,
                duration_secs: duration,
            };
            stats.display();
        }
        Commands::Verify { file } => {
            let reader = Arc::new(ncf_io::NcfReader::open(&file)?);
            let schemas = reader.schemas()?;
            
            println!("Verifying {} tensors...\n", schemas.len());
            let mut all_valid = true;
            for schema in schemas {
                let valid = reader.verify_tensor(&schema.name)?;
                let status = if valid { "✓" } else { "✗" };
                println!("{} {}", status, schema.name);
                if !valid {
                    all_valid = false;
                }
            }
            
            if all_valid {
                println!("\n✓ Verification successful!");
            } else {
                anyhow::bail!("✗ Verification failed: one or more tensors failed validation");
            }
        }
        Commands::List { file } => {
            let reader = ncf_io::NcfReader::open(&file)?;
            let schemas = reader.schemas()?;
            
            println!("Tensors in file: {}\n", schemas.len());
            println!("  {:<40} | {:<8} | {:<20} | {:>15}", "Name", "Type", "Shape", "Size");
            println!("{:-<40}-+-{:-<8}-+-{:-<20}-+-{:-<15}", "", "", "", "");
            
            for schema in schemas {
                display_tensor_info(&schema);
            }
        }
        Commands::Stats { file } => {
            let reader = ncf_io::NcfReader::open(&file)?;
            let prefix = reader.header_prefix();
            let schemas = reader.schemas()?;
            
            let mut total_uncompressed = 0u64;
            let mut compression_methods = BTreeMap::new();
            
            for schema in schemas {
                total_uncompressed += schema.byte_size();
                *compression_methods.entry(format!("{:?}", schema.compression))
                    .or_insert(0) += 1;
            }
            
            let file_size = fs::metadata(&file)?.len();
            
            println!("NCF File Statistics");
            println!("  Total file size:    {}", format_size(file_size));
            println!("  Tensor count:       {}", schemas.len());
            println!("  Uncompressed:       {}", format_size(total_uncompressed));
            println!("  Compression ratio:  {:.2}%", 
                (file_size as f64 / total_uncompressed as f64) * 100.0);
            
            println!("\nCompression Methods:");
            for (method, count) in compression_methods {
                println!("  {}: {} tensors", method, count);
            }
        }
    }
    Ok(())
}

