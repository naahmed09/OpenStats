//! Order-preserving parallel map on standard threads (sequential on `wasm32`, where threads are unavailable).
//!
//! Results come back in index order and every task is a pure function of its index, so a computation built on
//! [`par_map`] gives the same answer with one thread or many; only the wall-clock time changes.

/// Number of worker threads: `OPENSTATS_THREADS` if set (≥ 1), otherwise the available parallelism (at most 16).
pub fn worker_threads() -> usize {
    if let Some(t) = std::env::var("OPENSTATS_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
        return t.clamp(1, 64);
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16)
    }
}

/// `[f(0), f(1), …, f(n − 1)]`, computed on up to [`worker_threads`] threads.
pub fn par_map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let threads = worker_threads().min(n);
    #[cfg(target_arch = "wasm32")]
    let threads = {
        let _ = threads;
        1
    };
    if threads <= 1 {
        return (0..n).map(f).collect();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // contiguous chunks keep the output order trivial to restore
        let chunk = n.div_ceil(threads);
        let f = &f;
        let parts: Vec<Vec<T>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
                    s.spawn(move || (lo..hi).map(f).collect::<Vec<T>>())
                })
                .collect();
            handles
                .into_iter()
                .map(|h| match h.join() {
                    Ok(v) => v,
                    Err(e) => std::panic::resume_unwind(e),
                })
                .collect()
        });
        parts.into_iter().flatten().collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_keep_index_order() {
        let v = par_map(1000, |i| i * i);
        assert!(v.iter().enumerate().all(|(i, &x)| x == i * i));
        assert!(par_map(0, |i| i).is_empty());
        assert_eq!(par_map(1, |i| i + 5), vec![5]);
    }
}
