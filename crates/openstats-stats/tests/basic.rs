//! Goodness-of-fit chi-square against SciPy (tests/gen/basic_ref.py).

mod common;
use common::basic_data::*;
use openstats_stats::crosstab::{chi_square_gof, GofExpected};
use openstats_stats::report::{Reportable, SessionReport};

fn labels(n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("c{i}")).collect()
}

#[test]
fn equal_proportions_match_scipy() {
    let r = chi_square_gof(&labels(4), GOF_EQ_OBS, &GofExpected::Equal, 0).unwrap();
    assert!((r.statistic - GOF_EQ_STAT).abs() < 1e-12);
    assert!((r.p_value - GOF_EQ_P).abs() < 1e-12);
    assert_eq!(r.df, 3);
    for (a, b) in r.expected.iter().zip(GOF_EQ_EXP) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn proportions_and_counts_match_scipy() {
    let r = chi_square_gof(&labels(4), GOF_PR_OBS, &GofExpected::Proportions(vec![0.5, 0.3, 0.15, 0.05]), 0).unwrap();
    assert!((r.statistic - GOF_PR_STAT).abs() < 1e-12 && (r.p_value - GOF_PR_P).abs() < 1e-12);
    // proportions are rescaled
    let r2 = chi_square_gof(&labels(4), GOF_PR_OBS, &GofExpected::Proportions(vec![50.0, 30.0, 15.0, 5.0]), 0).unwrap();
    assert!((r2.statistic - r.statistic).abs() < 1e-12);
    let c = chi_square_gof(&labels(3), GOF_CT_OBS, &GofExpected::Counts(vec![10.0, 30.0, 50.0]), 0).unwrap();
    assert!((c.statistic - GOF_CT_STAT).abs() < 1e-12 && (c.p_value - GOF_CT_P).abs() < 1e-12);
    assert!((c.contribution.iter().sum::<f64>() - c.statistic).abs() < 1e-12);
}

#[test]
fn estimated_parameters_reduce_df_and_errors_are_reported() {
    let r = chi_square_gof(&labels(4), GOF_EQ_OBS, &GofExpected::Equal, 1).unwrap();
    assert_eq!(r.df, 2);
    assert!(chi_square_gof(&labels(4), GOF_EQ_OBS, &GofExpected::Equal, 3).is_err());
    assert!(chi_square_gof(&labels(1), &[5.0], &GofExpected::Equal, 0).is_err());
    assert!(chi_square_gof(&labels(3), &[1.0, -1.0, 3.0], &GofExpected::Equal, 0).is_err());
    assert!(chi_square_gof(&labels(3), &[0.0, 0.0, 0.0], &GofExpected::Equal, 0).is_err());
    assert!(chi_square_gof(&labels(3), &[1.0, 2.0, 3.0], &GofExpected::Counts(vec![1.0, 0.0, 1.0]), 0).is_err());
    assert!(chi_square_gof(&labels(3), &[1.0, 2.0, 3.0], &GofExpected::Proportions(vec![1.0, 1.0]), 0).is_err());
    assert!(chi_square_gof(&labels(2), &[1.0, 2.0, 3.0], &GofExpected::Equal, 0).is_err());
}

#[test]
fn report_renders_and_round_trips() {
    let r = chi_square_gof(&labels(4), GOF_EQ_OBS, &GofExpected::Equal, 0).unwrap();
    let mut rep = SessionReport::new("gof");
    rep.push(r.report_section());
    rep.validate().unwrap();
    assert!(rep.render_text().contains("Contribution to Chi-Square"));
    assert_eq!(SessionReport::from_json(&rep.to_json()).unwrap().to_json(), rep.to_json());
}
