//! Reference tests: OpenStats against SciPy 1.18 / NumPy (values in
//! `common/data.rs`, produced by `gen/stats_ref.py`) and against exact
//! rational enumeration (Fisher r×c).

mod common;
use common::data::*;
use openstats_stats::correlation::*;
use openstats_stats::crosstab::*;
use openstats_stats::descriptive::describe;
use openstats_stats::graphical::*;
use openstats_stats::quantile::*;

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let ok = if want == 0.0 { got.abs() <= rel } else { ((got - want) / want).abs() <= rel };
    assert!(ok, "{what}: got {got:e}, want {want:e}");
}

#[test]
fn descriptives_against_numpy_scipy() {
    for (k, d) in DESC.iter().enumerate() {
        let r = describe(d.data).unwrap();
        let w = |s: &str| format!("dataset {k} {s}");
        assert_eq!(r.n, d.n);
        close(r.mean.unwrap(), d.mean, 1e-13, &w("mean"));
        close(r.std_dev.unwrap(), d.sd, 1e-12, &w("sd"));
        close(r.se_mean.unwrap(), d.se, 1e-12, &w("se"));
        close(r.coef_var.unwrap(), d.cv, 1e-12, &w("cv"));
        close(r.skewness.unwrap(), d.skew, 1e-10, &w("skew"));
        close(r.kurtosis.unwrap(), d.kurt, 1e-9, &w("kurt"));
        close(r.q1.unwrap(), d.q1, 1e-13, &w("q1"));
        close(r.median.unwrap(), d.med, 1e-13, &w("median"));
        close(r.q3.unwrap(), d.q3, 1e-13, &w("q3"));
        close(r.mssd.unwrap(), d.mssd, 1e-10, &w("mssd"));
        close(r.trimmed_mean.unwrap(), d.trim, 1e-13, &w("trimmed"));
        assert_eq!(r.modes, d.mode, "{}", w("mode"));
        assert_eq!(r.mode_count, d.mode_n);
        close(r.iqr.unwrap(), d.q3 - d.q1, 1e-12, &w("iqr"));
    }
}

#[test]
fn all_nine_quantile_types_match_numpy() {
    use QuantileMethod::*;
    let types = [Type1, Type2, Type3, Type4, Type5, Type6, Type7, Type8, Type9];
    for (t, m) in types.iter().enumerate() {
        for (i, &p) in Q_PS.iter().enumerate() {
            let got = quantile(Q_DATA, p, *m).unwrap();
            let want = Q_EXPECT[t][i];
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "type {} p={p}: got {got}, want {want}", t + 1);
        }
    }
}

#[test]
fn graphical_summary_against_scipy() {
    let ad = anderson_darling(G_DATA).unwrap();
    close(ad.a2, G_A2, 1e-12, "A2");
    close(ad.a2_adjusted, G_ADJ, 1e-12, "A2*");
    close(ad.p_value, G_P, 1e-10, "AD p");
    let ci = ci_mean(G_DATA, 0.95).unwrap();
    close(ci.lower, G_CI_MEAN.0, 1e-12, "ci mean lo");
    close(ci.upper, G_CI_MEAN.1, 1e-12, "ci mean hi");
    let sd = ci_std_dev(G_DATA, 0.95).unwrap();
    close(sd.lower, G_CI_SD.0, 1e-11, "ci sd lo");
    close(sd.upper, G_CI_SD.1, 1e-11, "ci sd hi");
    let m = ci_median(G_DATA, 0.95).unwrap();
    assert_eq!((m.lower, m.upper, m.rank), (G_CI_MED.0, G_CI_MED.1, G_CI_MED.3));
    close(m.achieved, G_CI_MED.2, 1e-12, "achieved");
}

fn table(counts: &[&[u64]]) -> Crosstab {
    Crosstab::from_counts(counts.iter().map(|r| r.to_vec()).collect()).unwrap()
}

#[test]
fn chi_square_family_against_scipy() {
    for (k, t) in TABS.iter().enumerate() {
        let tb = table(t.counts);
        let p = pearson_chi_square(&tb).unwrap();
        close(p.statistic, t.pearson, 1e-12, &format!("table {k} pearson"));
        close(p.p_value, t.pearson_p, 1e-10, &format!("table {k} pearson p"));
        assert_eq!(p.df, t.df);
        let g = likelihood_ratio_chi_square(&tb).unwrap();
        close(g.statistic, t.g2, 1e-12, &format!("table {k} G2"));
        close(g.p_value, t.g2_p, 1e-10, &format!("table {k} G2 p"));
        let m = linear_by_linear(&tb).unwrap();
        close(m.statistic, t.mh, 1e-11, &format!("table {k} MH"));
        close(m.p_value, t.mh_p, 1e-9, &format!("table {k} MH p"));
    }
}

#[test]
fn fisher_2x2_against_scipy() {
    for (t, two, le, gr) in FISHER2 {
        let tb = table(&[&t[0], &t[1]]);
        let f = fisher_exact(&tb).unwrap();
        close(f.p_value, *two, 1e-9, &format!("{t:?} two-sided"));
        close(f.p_left.unwrap(), *le, 1e-9, &format!("{t:?} less"));
        close(f.p_right.unwrap(), *gr, 1e-9, &format!("{t:?} greater"));
    }
}

#[test]
fn fisher_rxc_against_exact_rationals() {
    for (t, want) in FISHER_RXC {
        let f = fisher_exact(&table(t)).unwrap();
        close(f.p_value, *want, 1e-11, &format!("{t:?}"));
    }
}

#[test]
fn mcnemar_against_scipy() {
    for &(b, c, p, pc, pe, stat) in MCNEMAR {
        let m = mcnemar(&table(&[&[10, b], &[c, 10]])).unwrap();
        close(m.statistic.unwrap(), stat, 1e-12, "stat");
        close(m.p_value.unwrap(), p, 1e-10, "p");
        close(m.p_value_corrected.unwrap(), pc, 1e-10, "p corrected");
        close(m.p_value_exact, pe, 1e-10, "exact p");
    }
}

#[test]
fn correlations_against_scipy() {
    let p = pearson(CX, CY, 0.95).unwrap();
    close(p.r, PEARSON.0, 1e-13, "r");
    close(p.p_value.unwrap(), PEARSON.1, 1e-10, "p");
    let ci = p.ci.unwrap();
    close(ci.lower, PEARSON.2, 1e-11, "ci lo");
    close(ci.upper, PEARSON.3, 1e-11, "ci hi");

    let s = spearman(TX, TY).unwrap();
    close(s.rho, SPEARMAN_TIES.0, 1e-13, "spearman rho (ties)");
    close(s.p_value.unwrap(), SPEARMAN_TIES.1, 1e-9, "spearman p");

    let k = kendall(TX, TY).unwrap();
    assert!(!k.exact);
    close(k.tau_b, KENDALL_TIES.0, 1e-13, "tau-b");
    close(k.p_value, KENDALL_TIES.1, 1e-10, "tau-b p");
    close(k.tau_c, KENDALL_TIES.2, 1e-13, "tau-c");

    let k = kendall(&CX[..25], &CY[..25]).unwrap();
    assert!(k.exact);
    close(k.tau_b, KENDALL_EXACT.0, 1e-13, "tau exact");
    close(k.p_value, KENDALL_EXACT.1, 1e-11, "p exact");

    let pc = partial_correlation(CX, CY, &[CZ.to_vec()], 0.95).unwrap();
    close(pc.r, PARTIAL.0, 1e-11, "partial r");
    close(pc.p_value, PARTIAL.1, 1e-9, "partial p");
    let ci = pc.ci.unwrap();
    close(ci.lower, PARTIAL.2, 1e-9, "partial ci lo");
    close(ci.upper, PARTIAL.3, 1e-9, "partial ci hi");

    let c = covariance_matrix(&[CX.to_vec(), CY.to_vec(), CZ.to_vec()], MissingData::Pairwise).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            close(c.values[i][j], COV[i][j], 1e-12, &format!("cov[{i}][{j}]"));
        }
    }
}
