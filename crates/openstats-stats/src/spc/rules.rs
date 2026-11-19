//! Tests for special causes (Minitab tests 1–8, i.e. the Western Electric /
//! Nelson rules) applied to standardised chart points.
//!
//! A point's z-score is `(value − centre)/σᵢ`, with `σᵢ` the one-sigma
//! distance of that point (so charts with varying limits use the right zones).
//! A test flags the point that *completes* the pattern, and every later point
//! that keeps it complete. A `NaN` point (missing) breaks every pattern.

use crate::error::{invalid, Result};

/// Which tests to run, with their parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecialCauseTests {
    /// Test 1: one point more than this many σ from the centre line (3).
    pub beyond_sigma: Option<f64>,
    /// Test 2: K points in a row on the same side of the centre line (9).
    pub same_side: Option<usize>,
    /// Test 3: K points in a row all increasing or all decreasing (6).
    pub trend: Option<usize>,
    /// Test 4: K points in a row alternating up and down (14).
    pub alternating: Option<usize>,
    /// Test 5: K out of K+1 points more than 2σ from the centre, same side (K = 2).
    pub two_sigma: Option<usize>,
    /// Test 6: K out of K+1 points more than 1σ from the centre, same side (K = 4).
    pub one_sigma: Option<usize>,
    /// Test 7: K points in a row within 1σ of the centre, either side (15).
    pub stratification: Option<usize>,
    /// Test 8: K points in a row more than 1σ from the centre, either side (8).
    pub mixture: Option<usize>,
}

impl SpecialCauseTests {
    /// No tests.
    pub fn none() -> Self {
        SpecialCauseTests {
            beyond_sigma: None,
            same_side: None,
            trend: None,
            alternating: None,
            two_sigma: None,
            one_sigma: None,
            stratification: None,
            mixture: None,
        }
    }

    /// Test 1 only (Minitab's default).
    pub fn test1_only() -> Self {
        SpecialCauseTests { beyond_sigma: Some(3.0), ..Self::none() }
    }

    /// All eight tests with Minitab's default parameters.
    pub fn all() -> Self {
        SpecialCauseTests {
            beyond_sigma: Some(3.0),
            same_side: Some(9),
            trend: Some(6),
            alternating: Some(14),
            two_sigma: Some(2),
            one_sigma: Some(4),
            stratification: Some(15),
            mixture: Some(8),
        }
    }

    /// Tests 1–4 with default parameters (the tests Minitab offers for attribute charts).
    pub fn first_four() -> Self {
        SpecialCauseTests {
            beyond_sigma: Some(3.0),
            same_side: Some(9),
            trend: Some(6),
            alternating: Some(14),
            ..Self::none()
        }
    }

    /// The Nelson rules: Minitab tests 1–8 with the original Nelson parameters
    /// (9 on one side, 6 trending, 14 alternating, 2 of 3 beyond 2σ, 4 of 5
    /// beyond 1σ, 15 within 1σ, 8 beyond 1σ). Same as [`all`](Self::all).
    pub fn nelson() -> Self {
        Self::all()
    }

    /// The Western Electric zone rules: one point beyond 3σ, 2 of 3 beyond 2σ
    /// (same side), 4 of 5 beyond 1σ (same side) and 8 in a row on one side
    /// of the centre line.
    pub fn western_electric() -> Self {
        SpecialCauseTests {
            beyond_sigma: Some(3.0),
            same_side: Some(8),
            two_sigma: Some(2),
            one_sigma: Some(4),
            ..Self::none()
        }
    }

    /// The Western Electric rules plus the supplementary runs rules (trend of
    /// 6, alternation of 14, stratification of 15).
    pub fn western_electric_supplementary() -> Self {
        SpecialCauseTests {
            trend: Some(6),
            alternating: Some(14),
            stratification: Some(15),
            ..Self::western_electric()
        }
    }

    /// `true` when test 5, 6, 7 or 8 is requested.
    pub fn uses_zone_tests(&self) -> bool {
        self.two_sigma.is_some() || self.one_sigma.is_some() || self.stratification.is_some() || self.mixture.is_some()
    }

    fn validate(&self) -> Result<()> {
        if let Some(k) = self.beyond_sigma {
            if !(k > 0.0 && k.is_finite()) {
                return invalid(format!("test 1 needs a positive σ multiple (got {k})"));
            }
        }
        for (name, v, min) in [
            ("2", self.same_side, 2),
            ("3", self.trend, 2),
            ("4", self.alternating, 3),
            ("5", self.two_sigma, 1),
            ("6", self.one_sigma, 1),
            ("7", self.stratification, 2),
            ("8", self.mixture, 2),
        ] {
            if let Some(k) = v {
                if k < min {
                    return invalid(format!("test {name} needs K ≥ {min} (got {k})"));
                }
            }
        }
        Ok(())
    }
}

impl Default for SpecialCauseTests {
    fn default() -> Self {
        Self::test1_only()
    }
}

/// Points flagged by one test.
#[derive(Debug, Clone, PartialEq)]
pub struct TestViolation {
    /// Test number 1–8.
    pub test: u8,
    /// Flagged points (0-based, increasing).
    pub points: Vec<usize>,
}

/// Human-readable description of a test (with its parameter).
pub fn describe(tests: &SpecialCauseTests, test: u8) -> String {
    match test {
        1 => format!("1 point more than {} standard deviations from center line", tests.beyond_sigma.unwrap_or(3.0)),
        2 => format!("{} points in a row on same side of center line", tests.same_side.unwrap_or(9)),
        3 => format!("{} points in a row, all increasing or all decreasing", tests.trend.unwrap_or(6)),
        4 => format!("{} points in a row, alternating up and down", tests.alternating.unwrap_or(14)),
        5 => {
            let k = tests.two_sigma.unwrap_or(2);
            format!("{k} out of {} points more than 2 standard deviations from center line (same side)", k + 1)
        }
        6 => {
            let k = tests.one_sigma.unwrap_or(4);
            format!("{k} out of {} points more than 1 standard deviation from center line (same side)", k + 1)
        }
        7 => format!(
            "{} points in a row within 1 standard deviation of center line (either side)",
            tests.stratification.unwrap_or(15)
        ),
        _ => format!(
            "{} points in a row more than 1 standard deviation from center line (either side)",
            tests.mixture.unwrap_or(8)
        ),
    }
}

/// Run the requested tests on z-scores. Violations are returned in test order;
/// tests that flag nothing are omitted.
pub fn evaluate(z: &[f64], tests: &SpecialCauseTests) -> Result<Vec<TestViolation>> {
    tests.validate()?;
    let n = z.len();
    let mut out = Vec::new();
    let mut push = |test: u8, pts: Vec<usize>| {
        if !pts.is_empty() {
            out.push(TestViolation { test, points: pts });
        }
    };
    if let Some(k) = tests.beyond_sigma {
        push(1, (0..n).filter(|&i| z[i].abs() > k).collect());
    }
    // A window ending at i, of length len, is complete when no NaN lies inside it.
    let window = |i: usize, len: usize| i + 1 >= len && z[i + 1 - len..=i].iter().all(|v| !v.is_nan());
    if let Some(k) = tests.same_side {
        push(
            2,
            (0..n)
                .filter(|&i| {
                    window(i, k)
                        && (z[i + 1 - k..=i].iter().all(|&v| v > 0.0) || z[i + 1 - k..=i].iter().all(|&v| v < 0.0))
                })
                .collect(),
        );
    }
    if let Some(k) = tests.trend {
        push(
            3,
            (0..n)
                .filter(|&i| {
                    window(i, k) && {
                        let w = &z[i + 1 - k..=i];
                        w.windows(2).all(|p| p[1] > p[0]) || w.windows(2).all(|p| p[1] < p[0])
                    }
                })
                .collect(),
        );
    }
    if let Some(k) = tests.alternating {
        push(
            4,
            (0..n)
                .filter(|&i| {
                    window(i, k) && {
                        let w = &z[i + 1 - k..=i];
                        let d: Vec<f64> = w.windows(2).map(|p| p[1] - p[0]).collect();
                        d.windows(2).all(|p| (p[0] > 0.0 && p[1] < 0.0) || (p[0] < 0.0 && p[1] > 0.0))
                    }
                })
                .collect(),
        );
    }
    // K out of K+1 beyond `limit` on one side; the flagged points are the ones beyond.
    let k_of_next = |k: usize, limit: f64| -> Vec<usize> {
        let len = k + 1;
        let mut flagged = vec![false; n];
        for i in 0..n {
            if !window(i, len) {
                continue;
            }
            let w = &z[i + 1 - len..=i];
            for side in [1.0, -1.0] {
                let beyond: Vec<usize> = (0..len).filter(|&j| !w[j].is_nan() && side * w[j] > limit).collect();
                // Minitab flags the point at which the pattern is completed.
                if beyond.len() >= k && beyond.contains(&(len - 1)) {
                    flagged[i] = true;
                }
            }
        }
        (0..n).filter(|&i| flagged[i]).collect()
    };
    if let Some(k) = tests.two_sigma {
        push(5, k_of_next(k, 2.0));
    }
    if let Some(k) = tests.one_sigma {
        push(6, k_of_next(k, 1.0));
    }
    if let Some(k) = tests.stratification {
        push(7, (0..n).filter(|&i| window(i, k) && z[i + 1 - k..=i].iter().all(|v| v.abs() < 1.0)).collect());
    }
    if let Some(k) = tests.mixture {
        push(8, (0..n).filter(|&i| window(i, k) && z[i + 1 - k..=i].iter().all(|v| v.abs() > 1.0)).collect());
    }
    Ok(out)
}

/// Union of the points flagged by any violation, sorted.
pub fn flagged_points(v: &[TestViolation]) -> Vec<usize> {
    let mut all: Vec<usize> = v.iter().flat_map(|t| t.points.iter().copied()).collect();
    all.sort_unstable();
    all.dedup();
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(z: &[f64], t: SpecialCauseTests) -> Vec<TestViolation> {
        evaluate(z, &t).unwrap()
    }

    #[test]
    fn test1_beyond_three_sigma() {
        let v = run(&[0.0, 3.0, 3.01, -3.5, 2.9], SpecialCauseTests::test1_only());
        assert_eq!(v, vec![TestViolation { test: 1, points: vec![2, 3] }]);
        let v = run(&[0.0, 2.5, -2.6], SpecialCauseTests { beyond_sigma: Some(2.0), ..SpecialCauseTests::none() });
        assert_eq!(v[0].points, vec![1, 2]);
    }

    #[test]
    fn test2_nine_on_one_side() {
        let mut z = vec![0.5; 8];
        assert!(run(&z, SpecialCauseTests { same_side: Some(9), ..SpecialCauseTests::none() }).is_empty());
        z.push(0.1);
        z.push(0.2);
        let v = run(&z, SpecialCauseTests { same_side: Some(9), ..SpecialCauseTests::none() });
        assert_eq!(v[0].points, vec![8, 9]);
        // A point on the centre line breaks the run; so does a NaN.
        let mut z2 = vec![0.5; 9];
        z2[4] = 0.0;
        assert!(run(&z2, SpecialCauseTests { same_side: Some(9), ..SpecialCauseTests::none() }).is_empty());
        z2[4] = f64::NAN;
        assert!(run(&z2, SpecialCauseTests { same_side: Some(9), ..SpecialCauseTests::none() }).is_empty());
    }

    #[test]
    fn test3_and_4_trend_and_alternation() {
        let up = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert_eq!(run(&up, SpecialCauseTests { trend: Some(6), ..SpecialCauseTests::none() })[0].points, vec![5]);
        let down = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1];
        let v = run(&down, SpecialCauseTests { trend: Some(6), ..SpecialCauseTests::none() });
        assert_eq!(v[0].points, vec![5]); // the repeated value ends the run
        let alt: Vec<f64> = (0..14).map(|i| if i % 2 == 0 { 0.3 } else { -0.3 }).collect();
        assert_eq!(
            run(&alt, SpecialCauseTests { alternating: Some(14), ..SpecialCauseTests::none() })[0].points,
            vec![13]
        );
        assert!(run(&alt[..13], SpecialCauseTests { alternating: Some(14), ..SpecialCauseTests::none() }).is_empty());
    }

    #[test]
    fn tests_5_to_8_zone_rules() {
        // Two of three beyond 2σ on the same side.
        let z = [0.0, 2.5, 0.2, 2.2];
        let v = run(&z, SpecialCauseTests { two_sigma: Some(2), ..SpecialCauseTests::none() });
        assert_eq!(v[0].points, vec![3]);
        // Opposite sides do not count.
        assert!(
            run(&[2.5, 0.0, -2.5], SpecialCauseTests { two_sigma: Some(2), ..SpecialCauseTests::none() }).is_empty()
        );
        // Four of five beyond 1σ.
        let z = [1.2, 0.3, 1.5, 1.1, 1.3];
        assert_eq!(run(&z, SpecialCauseTests { one_sigma: Some(4), ..SpecialCauseTests::none() })[0].points, vec![4]);
        // Fifteen within 1σ; eight beyond 1σ on either side.
        assert_eq!(
            run(&[0.2; 15], SpecialCauseTests { stratification: Some(15), ..SpecialCauseTests::none() })[0].points,
            vec![14]
        );
        let z: Vec<f64> = (0..8).map(|i| if i % 2 == 0 { 1.4 } else { -1.4 }).collect();
        assert_eq!(run(&z, SpecialCauseTests { mixture: Some(8), ..SpecialCauseTests::none() })[0].points, vec![7]);
    }

    #[test]
    fn invalid_parameters_and_helpers() {
        assert!(evaluate(&[0.0], &SpecialCauseTests { same_side: Some(1), ..SpecialCauseTests::none() }).is_err());
        assert!(evaluate(&[0.0], &SpecialCauseTests { beyond_sigma: Some(-1.0), ..SpecialCauseTests::none() }).is_err());
        let v = run(&[4.0, 0.0, 0.0], SpecialCauseTests::all());
        assert_eq!(flagged_points(&v), vec![0]);
        assert!(SpecialCauseTests::all().uses_zone_tests() && !SpecialCauseTests::first_four().uses_zone_tests());
        assert!(describe(&SpecialCauseTests::all(), 2).starts_with("9 points"));
    }

    #[test]
    fn presets_have_documented_parameters() {
        assert_eq!(SpecialCauseTests::nelson(), SpecialCauseTests::all());
        let we = SpecialCauseTests::western_electric();
        assert_eq!((we.beyond_sigma, we.same_side, we.two_sigma, we.one_sigma), (Some(3.0), Some(8), Some(2), Some(4)));
        assert!(we.trend.is_none() && we.alternating.is_none() && we.stratification.is_none() && we.mixture.is_none());
        let wes = SpecialCauseTests::western_electric_supplementary();
        assert_eq!((wes.trend, wes.alternating, wes.stratification), (Some(6), Some(14), Some(15)));
        assert!(wes.mixture.is_none());
    }

    #[test]
    fn western_electric_eight_in_a_row() {
        let z = [0.5; 8];
        let v = evaluate(&z, &SpecialCauseTests::western_electric()).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].test, 2);
        assert_eq!(v[0].points, vec![7]);
        // Nelson needs nine.
        assert!(evaluate(&z, &SpecialCauseTests::nelson()).unwrap().is_empty());
    }
}
