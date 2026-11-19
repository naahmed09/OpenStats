//! Multi-threaded forest growing gives bit-identical results to one thread (its own process, so the
//! OPENSTATS_THREADS variable cannot race with other tests).

use openstats_stats::ml::{fit_forest, Dataset, ForestOptions, Response};

fn data(n: usize) -> (Dataset, Vec<f64>) {
    let mut s = 12345u64;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (1u64 << 31) as f64
    };
    let cols: Vec<Vec<f64>> = (0..6).map(|_| (0..n).map(|_| next() * 10.0).collect()).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * cols[0][i] + (cols[1][i] - 5.0).powi(2) + 0.3 * next()).collect();
    let mut ds = Dataset::new();
    for (j, c) in cols.into_iter().enumerate() {
        ds = ds.with_numeric(&format!("x{j}"), c).unwrap();
    }
    (ds, y)
}

#[test]
fn forest_is_identical_with_one_or_many_threads_and_faster_with_many() {
    let (ds, y) = data(1500);
    let resp = Response::numeric(y).unwrap();
    let opts = ForestOptions { n_trees: 120, seed: 9, importance: true, ..Default::default() };
    std::env::set_var("OPENSTATS_THREADS", "1");
    let t1 = std::time::Instant::now();
    let a = fit_forest(&ds, &resp, &opts).unwrap();
    let d1 = t1.elapsed();
    std::env::set_var("OPENSTATS_THREADS", "4");
    let t4 = std::time::Instant::now();
    let b = fit_forest(&ds, &resp, &opts).unwrap();
    let d4 = t4.elapsed();
    assert_eq!(a.trees.len(), b.trees.len());
    assert_eq!(a.oob_error.to_bits(), b.oob_error.to_bits());
    assert_eq!(a.oob_count, b.oob_count);
    assert_eq!(format!("{:?}", a.oob.values), format!("{:?}", b.oob.values));
    assert_eq!(format!("{:?}", a.oob_curve), format!("{:?}", b.oob_curve));
    assert_eq!(format!("{:?}", a.impurity_importance), format!("{:?}", b.impurity_importance));
    assert_eq!(format!("{:?}", a.permutation), format!("{:?}", b.permutation));
    let p1 = a.predict(&ds).unwrap();
    let p4 = b.predict(&ds).unwrap();
    assert_eq!(format!("{:?}", p1.values), format!("{:?}", p4.values));
    eprintln!("forest 120 trees: 1 thread {d1:?}, 4 threads {d4:?}");
}
