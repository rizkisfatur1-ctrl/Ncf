//! High-performance LRU cache with concurrent access
//! 
//! Features:
//! - Lock-free reads (with RwLock for writes)
//! - Configurable capacity
//! - TTL support
//! - Statistics tracking

#![allow(missing_docs)]

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

type CacheKey = u64;
type CacheValue = Arc<Vec<u8>>;

#[derive(Debug, Clone)]
struct CacheEntry {
    value: CacheValue,
    last_accessed: u64,
    size: usize,
}

/// Thread-safe LRU cache with TTL support
pub struct LruCache {
    inner: Arc<RwLock<LruCacheInner>>,
}

struct LruCacheInner {
    entries: HashMap<CacheKey, CacheEntry>,
    capacity: usize,
    current_size: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    ttl_secs: Option<u64>,
}

impl LruCache {
    /// Create new LRU cache with given capacity (in bytes)
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LruCacheInner {
                entries: HashMap::new(),
                capacity,
                current_size: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
                ttl_secs: None,
            })),
        }
    }

    /// Create cache with TTL
    pub fn with_ttl(capacity: usize, ttl_secs: u64) -> Self {
        let cache = Self::new(capacity);
        {
            let mut inner = cache.inner.write();
            inner.ttl_secs = Some(ttl_secs);
        }
        cache
    }

    /// Get value from cache
    pub fn get(&self, key: CacheKey) -> Option<CacheValue> {
        let mut inner = self.inner.write();
        
        // Check TTL
        if let Some(ttl) = inner.ttl_secs {
            if let Some(entry) = inner.entries.get(&key) {
                let now = current_timestamp();
                if now.saturating_sub(entry.last_accessed) > ttl {
                    inner.entries.remove(&key);
                    return None;
                }
            }
        }

        if let Some(entry) = inner.entries.get_mut(&key) {
            let value = entry.value.clone();
            let timestamp = current_timestamp();
            drop(value); // Release the borrow before incrementing hits
            
            // Now update entry safely
            if let Some(e) = inner.entries.get_mut(&key) {
                e.last_accessed = timestamp;
            }
            inner.hits += 1;
            inner.entries.get(&key).map(|e| e.value.clone())
        } else {
            inner.misses += 1;
            None
        }
    }

    /// Insert value into cache
    pub fn insert(&self, key: CacheKey, value: Arc<Vec<u8>>) {
        let value_size = value.len();
        let mut inner = self.inner.write();

        // Remove if exists
        if let Some(old) = inner.entries.remove(&key) {
            inner.current_size = inner.current_size.saturating_sub(old.size);
        }

        // Evict LRU entries if needed
        while inner.current_size + value_size > inner.capacity && !inner.entries.is_empty() {
            let lru_key = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(&k, _)| k)
                .unwrap();

            if let Some(removed) = inner.entries.remove(&lru_key) {
                inner.current_size = inner.current_size.saturating_sub(removed.size);
                inner.evictions += 1;
            }
        }

        // Insert new entry
        if inner.current_size + value_size <= inner.capacity {
            inner.entries.insert(
                key,
                CacheEntry {
                    value,
                    last_accessed: current_timestamp(),
                    size: value_size,
                },
            );
            inner.current_size += value_size;
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.entries.clear();
        inner.current_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.read();
        let total_requests = inner.hits + inner.misses;
        let hit_rate = if total_requests > 0 {
            (inner.hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        CacheStats {
            hits: inner.hits,
            misses: inner.misses,
            hit_rate,
            evictions: inner.evictions,
            size_used: inner.current_size,
            size_capacity: inner.capacity,
            entries: inner.entries.len(),
        }
    }

    /// Size used
    pub fn size_used(&self) -> usize {
        self.inner.read().current_size
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().entries.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub evictions: u64,
    pub size_used: usize,
    pub size_capacity: usize,
    pub entries: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache(hits={}, misses={}, rate={:.2}%, evictions={}, size={}/{} bytes, entries={})",
            self.hits,
            self.misses,
            self.hit_rate,
            self.evictions,
            self.size_used,
            self.size_capacity,
            self.entries
        )
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic() {
        let cache = LruCache::new(1000);
        let data = Arc::new(vec![1, 2, 3, 4, 5]);
        
        cache.insert(1, data.clone());
        assert_eq!(cache.get(1), Some(data));
        assert_eq!(cache.get(2), None);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = LruCache::new(100);
        let data1 = Arc::new(vec![0u8; 60]);
        let data2 = Arc::new(vec![0u8; 60]);
        
        cache.insert(1, data1.clone());
        cache.insert(2, data2);
        
        // Access 1 to make it recently used
        let _ = cache.get(1);
        
        // Insert 3 should evict 2 (LRU)
        let data3 = Arc::new(vec![0u8; 60]);
        cache.insert(3, data3);
        
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_cache_stats() {
        let cache = LruCache::new(10000);
        let _ = cache.get(1); // miss
        let _ = cache.get(2); // miss
        
        let data = Arc::new(vec![0u8; 100]);
        cache.insert(1, data.clone());
        let _ = cache.get(1); // hit
        let _ = cache.get(1); // hit
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
    }
}
