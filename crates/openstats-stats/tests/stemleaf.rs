//! Stem-and-leaf display checks against hand-worked examples.

use openstats_stats::report::Reportable;
use openstats_stats::stemleaf::stem_and_leaf;

#[test]
fn classic_two_digit_example() {
    // 12 14 15 21 22 22 23 31 38 40 with leaf unit 1: stems 1,2,3,4 (width 10).
    let d = [12.0, 14.0, 15.0, 21.0, 22.0, 22.0, 23.0, 31.0, 38.0, 40.0];
    let s = stem_and_leaf(&d, Some(1.0)).unwrap();
    assert_eq!(s.leaf_unit, 1.0);
    let rows: Vec<(String, String, String)> =
        s.lines.iter().map(|l| (l.depth.clone(), l.stem.clone(), l.leaves.clone())).collect();
    // The leaf unit is 1 and at most max(6, 10) lines; the span 28 needs width 5 lines or more.
    assert_eq!(s.n, 10);
    let total: usize = rows.iter().map(|r| r.2.len()).sum();
    assert_eq!(total, 10);
    // Every observation is recoverable as stem*10 + leaf.
    let mut back: Vec<f64> = Vec::new();
    for l in &s.lines {
        let st: f64 = l.stem.trim_start_matches('-').parse().unwrap();
        for c in l.leaves.chars() {
            back.push(st * 10.0 + f64::from(c.to_digit(10).unwrap()));
        }
    }
    back.sort_by(f64::total_cmp);
    assert_eq!(back, d.to_vec());
    // The median line holds the (n+1)/2-th = 5th smallest value (22) and is shown in parentheses.
    let med = rows.iter().position(|r| r.0.starts_with('(')).unwrap();
    assert!(rows[med].2.contains('2'));
    // Depths count towards the median from both ends.
    let first = rows.iter().find(|r| !r.0.is_empty()).unwrap();
    assert_eq!(first.0, format!("{}", first.2.len()));
    let last = rows.last().unwrap();
    assert_eq!(last.0, format!("{}", last.2.len()));
}

#[test]
fn negative_values_and_decimals() {
    let d = [-0.35, -0.12, -0.05, 0.0, 0.07, 0.21, 0.26, 0.4];
    let s = stem_and_leaf(&d, Some(0.01)).unwrap();
    assert_eq!(s.leaf_unit, 0.01);
    // Truncation toward zero: -0.35 -> -35 -> stem -3 leaf 5; 0.07 -> stem 0 leaf 7.
    let neg: Vec<&str> = s.lines.iter().filter(|l| l.stem.starts_with('-')).map(|l| l.stem.as_str()).collect();
    assert!(neg.contains(&"-3") && neg.contains(&"-0"), "{neg:?}");
    let all: String = s.lines.iter().map(|l| l.leaves.clone()).collect();
    assert_eq!(all.len(), 8);
    // The line for -0.05 is "-0" and holds the leaf 5, and 0.0 sits on "0".
    let l = s.lines.iter().find(|l| l.stem == "-0").unwrap();
    assert!(l.leaves.contains('5'));
}

#[test]
fn automatic_unit_missing_values_and_errors() {
    let mut d: Vec<f64> = (0..100).map(|i| f64::from(i) * 1.7 + 3.0).collect();
    d.push(f64::NAN);
    let s = stem_and_leaf(&d, None).unwrap();
    assert_eq!(s.n, 100);
    assert_eq!(s.n_missing, 1);
    assert!(s.lines.len() <= 21, "{}", s.lines.len());
    let t = s.report_section();
    let mut r = openstats_stats::report::SessionReport::new("t");
    r.push(t);
    let txt = r.render_text();
    assert!(txt.contains("Leaf Unit") && txt.contains("N = 100 (1 missing ignored)"), "{txt}");
    assert!(stem_and_leaf(&[1.0], None).is_err());
    assert!(stem_and_leaf(&[1.0, 2.0, 3.0], Some(0.3)).is_err());
    assert!(stem_and_leaf(&[1.0, 2.0, 3.0], Some(-1.0)).is_err());
    // Constant data still produces a display.
    let c = stem_and_leaf(&[5.0, 5.0, 5.0], None).unwrap();
    assert_eq!(c.lines.iter().map(|l| l.leaves.len()).sum::<usize>(), 3);
}
