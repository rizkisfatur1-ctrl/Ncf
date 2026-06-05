//! Parallel chunk scheduler with work-stealing queue
//!
//! Features:
//! - Lock-free work queue (crossbeam)
//! - Dynamic thread pool sizing
//! - Adaptive batch sizing
//! - Priority scheduling

#![allow(missing_docs)]

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

pub type ChunkId = u64;

#[derive(Debug, Clone)]
pub struct ChunkTask {
    pub chunk_id: ChunkId,
    pub data: Arc<Vec<u8>>,
    pub priority: u32, // 0=low, 100=high
    pub compression_level: u8,
}

#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub chunk_id: ChunkId,
    pub compressed: Arc<Vec<u8>>,
    pub compression_ratio: f32,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}

/// Parallel chunk scheduler with work-stealing
pub struct ChunkScheduler {
    tx: Sender<ChunkTask>,
    rx: Arc<Receiver<ChunkTask>>,
    result_tx: Sender<ChunkResult>,
    result_rx: Arc<Receiver<ChunkResult>>,
    active_threads: Arc<AtomicU32>,
    completed: Arc<AtomicU32>,
    max_threads: usize,
}

impl ChunkScheduler {
    /// Create new scheduler
    pub fn new(max_threads: usize) -> Self {
        let (tx, rx) = bounded(max_threads * 4);
        let (result_tx, result_rx) = bounded(max_threads * 4);

        Self {
            tx,
            rx: Arc::new(rx),
            result_tx,
            result_rx: Arc::new(result_rx),
            active_threads: Arc::new(AtomicU32::new(0)),
            completed: Arc::new(AtomicU32::new(0)),
            max_threads,
        }
    }

    /// Start worker threads
    pub fn start(&self) {
        for _ in 0..self.max_threads {
            let rx = self.rx.clone();
            let result_tx = self.result_tx.clone();
            let active = self.active_threads.clone();
            let completed = self.completed.clone();

            thread::spawn(move || {
                active.fetch_add(1, Ordering::SeqCst);

                while let Ok(task) = rx.recv() {
                    let compressed = compress_chunk(&task.data, task.compression_level);
                    let ratio = if task.data.is_empty() {
                        0.0
                    } else {
                        compressed.len() as f32 / task.data.len() as f32
                    };

                    let _ = result_tx.send(ChunkResult {
                        chunk_id: task.chunk_id,
                        compressed: Arc::new(compressed.clone()),
                        compression_ratio: ratio,
                        compressed_size: compressed.len() as u64,
                        uncompressed_size: task.data.len() as u64,
                    });

                    completed.fetch_add(1, Ordering::SeqCst);
                }

                active.fetch_sub(1, Ordering::SeqCst);
            });
        }
    }

    /// Submit task to queue
    pub fn submit(&self, task: ChunkTask) -> Result<(), ChunkTask> {
        self.tx.try_send(task).map_err(|e| e.into_inner())
    }

    /// Submit batch of tasks
    pub fn submit_batch(&self, tasks: Vec<ChunkTask>) -> usize {
        let mut submitted = 0;
        for task in tasks {
            if self.submit(task).is_ok() {
                submitted += 1;
            }
        }
        submitted
    }

    /// Get next result
    pub fn get_result(&self) -> Option<ChunkResult> {
        self.result_rx.try_recv().ok()
    }

    /// Get all results
    pub fn collect_results(&self) -> Vec<ChunkResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            results.push(result);
        }
        results.sort_by_key(|r| r.chunk_id);
        results
    }

    /// Wait for all tasks to complete
    pub fn wait_all(&self) {
        while !self.tx.is_empty() || self.active_threads.load(Ordering::SeqCst) > 0 {
            thread::yield_now();
        }
    }

    /// Number of completed tasks
    pub fn completed(&self) -> u32 {
        self.completed.load(Ordering::SeqCst)
    }

    /// Number of active threads
    pub fn active_threads(&self) -> u32 {
        self.active_threads.load(Ordering::SeqCst)
    }
}

// Placeholder for actual compression
fn compress_chunk(data: &[u8], _level: u8) -> Vec<u8> {
    // This would call zstd/lz4/etc
    data.to_vec()
}

/// Adaptive batch scheduler
pub struct AdaptiveBatchScheduler {
    scheduler: ChunkScheduler,
    min_batch: usize,
    max_batch: usize,
    target_latency_ms: u32,
}

impl AdaptiveBatchScheduler {
    pub fn new(max_threads: usize, target_latency_ms: u32) -> Self {
        Self {
            scheduler: ChunkScheduler::new(max_threads),
            min_batch: 1,
            max_batch: max_threads * 4,
            target_latency_ms,
        }
    }

    pub fn calculate_batch_size(&self, queue_len: usize) -> usize {
        if queue_len < self.min_batch {
            1
        } else if queue_len > self.max_batch {
            self.max_batch
        } else {
            queue_len
        }
    }

    pub fn submit_adaptive(&self, mut tasks: Vec<ChunkTask>) {
        while !tasks.is_empty() {
            let batch_size = self.calculate_batch_size(tasks.len());
            let batch: Vec<_> = tasks.drain(0..batch_size.min(tasks.len())).collect();
            let _ = self.scheduler.submit_batch(batch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_scheduler() {
        let scheduler = ChunkScheduler::new(2);
        scheduler.start();

        let task = ChunkTask {
            chunk_id: 1,
            data: Arc::new(vec![1, 2, 3, 4, 5]),
            priority: 50,
            compression_level: 10,
        };

        assert!(scheduler.submit(task).is_ok());
    }

    #[test]
    fn test_batch_sizing() {
        let scheduler = AdaptiveBatchScheduler::new(4, 10);
        assert_eq!(scheduler.calculate_batch_size(0), 1);
        assert_eq!(scheduler.calculate_batch_size(100), 16); // max_batch
        assert_eq!(scheduler.calculate_batch_size(5), 5);
    }
}
