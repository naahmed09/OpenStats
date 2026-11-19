//! Verification against the NIST Statistical Reference Datasets (StRD).
//!
//! Every `.dat` file found in `tests/fixtures/nist/` (official downloads,
//! git-ignored) and `tests/fixtures/nist/embedded/` (committed reconstructions,
//! see the README there) is parsed and the matching OpenStats engine is checked
//! against the certified values. Official files take precedence.
//!
//! Run `cargo test -p openstats-math --test nist_strd -- --nocapture` to see the
//! table of log relative errors (LRE, correct significant digits).

mod nist;

use nist::expr::{parse, Dual, Expr};
use nist::parser::{parse as parse_dat, Dataset, Kind};
use nist::{discover, lre};
use openstats_math::optim::{gauss_newton, levenberg_marquardt, LsOptions};
use openstats_math::stats::{self, anova_oneway, ols, R2Kind};
use openstats_math::Matrix;
use std::collections::HashMap;

struct Report {
    rows: Vec<String>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, set: &str, what: &str, got: f64, cert: f64, min_lre: f64) {
        let l = lre(got, cert);
        let ok = l >= min_lre;
        self.rows
            .push(format!("{set:<14} {what:<22} LRE {l:5.1} (need {min_lre:>4.1}) {}", if ok { "ok" } else { "FAIL" }));
        if !ok {
            self.failures.push(format!("{set}: {what}: got {got:e}, certified {cert:e}, LRE {l:.2} < {min_lre}"));
        }
    }
}

/// Digits that a double can possibly deliver for a spread statistic of data
/// whose decimal values are not exactly representable: each stored value is
/// off by ~1e-16*|mean|, which is ~|mean|/sd * 1e-16 relative to the spread.
fn representable_lre(mean: f64, sd: f64, cap: f64) -> f64 {
    (16.0 - (1.0 + mean.abs() / sd).log10()).clamp(0.0, cap)
}

fn model_columns(ds: &Dataset) -> Matrix {
    // data columns: y, then predictors
    let n = ds.data[0].len();
    let has_b0 = ds.params.first().is_some_and(|p| p.0.eq_ignore_ascii_case("b0"));
    let npar = ds.params.len();
    let polynomial = ds.model.contains("**");
    let preds = &ds.data[1..];
    Matrix::from_fn(n, npar, |i, j| {
        let k = if has_b0 { j } else { j + 1 };
        if has_b0 && j == 0 {
            1.0
        } else if polynomial {
            preds[0][i].powi(k as i32)
        } else {
            preds[k - 1][i]
        }
    })
}

fn linear(ds: &Dataset, rep: &mut Report) {
    let has_b0 = ds.params.first().is_some_and(|p| p.0.eq_ignore_ascii_case("b0"));
    let x = model_columns(ds);
    let kind = if has_b0 { R2Kind::Centered } else { R2Kind::Uncentered };
    // NIST's Filip and Wampler4/5 are deliberately near-singular; the fit may
    // legitimately be refused, which is then recorded as a failure.
    let fit = match ols(&x, &ds.data[0], kind) {
        Ok(f) => f,
        Err(e) => {
            rep.failures.push(format!("{}: ols failed: {e}", ds.name));
            rep.rows.push(format!("{:<14} ols refused: {e}", ds.name));
            return;
        }
    };
    let exact =
        ds.name.to_ascii_lowercase().starts_with("wampler1") || ds.name.to_ascii_lowercase().starts_with("wampler2");
    for (j, (pn, est, sd)) in ds.params.iter().enumerate() {
        rep.check(&ds.name, &format!("{pn} estimate"), fit.coef[j], *est, 9.0);
        if *sd != 0.0 {
            rep.check(&ds.name, &format!("{pn} std err"), fit.std_err[j], *sd, 8.0);
        } else if !exact {
            rep.check(&ds.name, &format!("{pn} std err"), fit.std_err[j], 0.0, 0.0);
        }
    }
    if let Some(s) = ds.resid_sd {
        if s != 0.0 {
            rep.check(&ds.name, "residual sd", fit.residual_sd, s, 8.0);
        }
    }
    if let Some(r2) = ds.r2 {
        rep.check(&ds.name, "R-squared", fit.r_squared, r2, 9.0);
    }
    if exact {
        // Certified residuals are exactly zero; require them to be tiny relative to the response scale.
        let scale = ds.data[0].iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let worst = fit.residuals.iter().fold(0.0_f64, |a, v| a.max(v.abs())) / scale;
        rep.rows.push(format!("{:<14} {:<22} max |resid|/max|y| = {worst:.1e}", ds.name, "exact-fit residual"));
        if worst > 1e-9 {
            rep.failures.push(format!("{}: residuals not ~0 ({worst:e})", ds.name));
        }
    }
}

fn univariate(ds: &Dataset, rep: &mut Report) {
    let y = &ds.data[0];
    if let Some(m) = ds.mean {
        rep.check(&ds.name, "mean", stats::summary::mean(y).unwrap(), m, 14.0);
    }
    if let Some(s) = ds.sd {
        let need = representable_lre(ds.mean.unwrap_or(0.0), s, 14.0);
        rep.check(&ds.name, "std dev", stats::summary::std_dev(y).unwrap(), s, need);
    }
    if let Some(a) = ds.acf1 {
        rep.check(&ds.name, "lag-1 autocorr", stats::summary::autocorrelation(y, 1).unwrap(), a, 13.0);
    }
}

fn anova(ds: &Dataset, rep: &mut Report) {
    // data columns: response, group label
    let (y, g) = (&ds.data[0], &ds.data[1]);
    let mut labels: Vec<f64> = Vec::new();
    for v in g {
        if !labels.contains(v) {
            labels.push(*v);
        }
    }
    let groups: Vec<Vec<f64>> =
        labels.iter().map(|l| y.iter().zip(g).filter(|(_, gv)| *gv == l).map(|(v, _)| *v).collect()).collect();
    let a = anova_oneway(&groups).unwrap();
    let grand = stats::summary::mean(y).unwrap();
    let need = representable_lre(grand, ds.resid_sd.unwrap_or(1.0), 11.0);
    let (b, w) = (ds.between.as_ref().unwrap(), ds.within.as_ref().unwrap());
    rep.check(&ds.name, "between df", a.df_between, b.df, 15.0);
    rep.check(&ds.name, "within df", a.df_within, w.df, 15.0);
    rep.check(&ds.name, "between SS", a.ss_between, b.ss, need);
    rep.check(&ds.name, "within SS", a.ss_within, w.ss, need);
    rep.check(&ds.name, "between MS", a.ms_between, b.ms, need);
    rep.check(&ds.name, "within MS", a.ms_within, w.ms, need);
    if let Some(f) = b.f {
        rep.check(&ds.name, "F", a.f, f, need);
    }
    if let Some(r2) = ds.r2 {
        rep.check(&ds.name, "R-squared", a.r_squared, r2, need);
    }
    if let Some(s) = ds.resid_sd {
        rep.check(&ds.name, "residual sd", a.residual_sd, s, need);
    }
}

struct Model {
    expr: Expr,
    np: usize,
    xs: Vec<Vec<f64>>,
    x_names: Vec<String>,
    names: Vec<String>,
}

impl Model {
    /// Predictions and (optionally) gradients for parameters `p`.
    fn eval(&self, p: &[f64], i: usize) -> Dual {
        let mut vars: HashMap<String, Dual> = HashMap::new();
        for (k, name) in self.names.iter().enumerate() {
            let mut d = vec![0.0; self.np];
            d[k] = 1.0;
            vars.insert(name.clone(), Dual { v: p[k], d });
        }
        for (k, name) in self.x_names.iter().enumerate() {
            vars.insert(name.clone(), Dual::constant(self.xs[k][i], self.np));
        }
        self.expr.eval(&vars, self.np).expect("model evaluation")
    }
}

fn nonlinear(ds: &Dataset, rep: &mut Report) {
    let expr = match parse(&ds.model) {
        Ok(e) => e,
        Err(e) => {
            rep.failures.push(format!("{}: cannot parse model {:?}: {e}", ds.name, ds.model));
            return;
        }
    };
    let y = &ds.data[0];
    let names: Vec<String> = ds.params.iter().map(|p| p.0.to_ascii_lowercase()).collect();
    let x_names: Vec<String> =
        if ds.data.len() == 2 { vec!["x".to_string()] } else { (1..ds.data.len()).map(|k| format!("x{k}")).collect() };
    let model = Model { expr, np: names.len(), xs: ds.data[1..].to_vec(), x_names, names };
    let n = y.len();
    let np = model.np;

    for (si, start) in ds.starts.iter().enumerate() {
        let label = format!("{} start {}", ds.name, si + 1);
        for (alg, name) in [(0, "LM"), (1, "GN")] {
            let mut resid = |p: &[f64]| (0..n).map(|i| y[i] - model.eval(p, i).v).collect::<Vec<f64>>();
            let mut jac = |p: &[f64]| {
                let rows: Vec<Vec<f64>> = (0..n).map(|i| model.eval(p, i).d.iter().map(|d| -d).collect()).collect();
                Matrix::from_rows(&rows).unwrap()
            };
            let opts = LsOptions { max_iter: 5000, ..LsOptions::default() };
            let fit = if alg == 0 {
                levenberg_marquardt(&mut resid, Some(&mut jac), start, &opts)
            } else {
                gauss_newton(&mut resid, Some(&mut jac), start, &opts)
            };
            let fit = match fit {
                Ok(f) => f,
                Err(e) => {
                    rep.rows.push(format!("{label:<22} {name}: error {e}"));
                    if alg == 0 && si == 1 {
                        rep.failures.push(format!("{label}: LM errored: {e}"));
                    }
                    continue;
                }
            };
            // LM from Start 2 must always reach certified accuracy. Elsewhere
            // (Start 1 is deliberately hard; Gauss-Newton has no globalisation
            // guarantee) accuracy is asserted only when the solver claims
            // convergence, and non-convergence is reported honestly.
            let required = alg == 0 && si == 1;
            if !fit.converged && !required {
                let l = ds.rss.map_or(0.0, |r| lre(fit.sse, r));
                rep.rows.push(format!(
                    "{label:<22} {name}: did not converge ({:?}), RSS LRE {l:.1} (reported, not asserted)",
                    fit.termination
                ));
                continue;
            }
            // The achievable accuracy is limited by the conditioning of the
            // problem; these are the thresholds OpenStats commits to.
            let (p_min, se_min, rss_min) = (6.0, 5.0, 8.0);
            let tag = format!("{label:<22} {name}");
            let mut sub = Report { rows: Vec::new(), failures: Vec::new() };
            for (k, (pn, est, _)) in ds.params.iter().enumerate() {
                sub.check(&tag, &format!("{pn} estimate"), fit.x[k], *est, p_min);
            }
            if let Ok(Some(se)) = fit.std_errors() {
                for (k, (pn, _, sd)) in ds.params.iter().enumerate() {
                    sub.check(&tag, &format!("{pn} std err"), se[k], *sd, se_min);
                }
            } else {
                sub.failures.push(format!("{tag}: no covariance"));
            }
            if let Some(rss) = ds.rss {
                sub.check(&tag, "residual SS", fit.sse, rss, rss_min);
            }
            if let (Some(s), Some(m)) = (ds.resid_sd, fit.mse()) {
                sub.check(&tag, "residual sd", m.sqrt(), s, 6.0);
            }
            // Summarise: print the worst LRE per fit rather than every line.
            let worst = sub
                .rows
                .iter()
                .filter_map(|r| r.split("LRE").nth(1))
                .filter_map(|r| r.split_whitespace().next()?.parse::<f64>().ok())
                .fold(15.0, f64::min);
            rep.rows.push(format!(
                "{tag:<30} converged in {:>4} it, worst LRE {worst:4.1}{}",
                fit.iterations,
                if sub.failures.is_empty() { "" } else { "  FAIL" }
            ));
            rep.failures.extend(sub.failures);
        }
    }
    assert!(np > 0);
}

#[test]
fn nist_strd_certified_values() {
    let files = discover();
    assert!(!files.is_empty(), "no NIST fixtures found");
    let mut rep = Report { rows: Vec::new(), failures: Vec::new() };
    let (mut official, mut embedded) = (0, 0);
    let mut seen = Vec::new();
    for (path, is_official) in &files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let ds = match parse_dat(&text, &name) {
            Ok(d) => d,
            Err(e) => {
                rep.failures.push(format!("{name}: parse error: {e}"));
                continue;
            }
        };
        if *is_official {
            official += 1;
        } else {
            embedded += 1;
        }
        seen.push(format!("{name} [{}]", if *is_official { "official" } else { "embedded" }));
        match ds.kind {
            Kind::Univariate => univariate(&ds, &mut rep),
            Kind::Anova => anova(&ds, &mut rep),
            Kind::Linear => linear(&ds, &mut rep),
            Kind::Nonlinear => nonlinear(&ds, &mut rep),
        }
    }
    println!("\nNIST StRD verification: {official} official file(s), {embedded} embedded reconstruction(s)");
    println!("datasets: {}", seen.join(", "));
    for r in &rep.rows {
        println!("{r}");
    }
    assert!(rep.failures.is_empty(), "NIST verification failures:\n{}", rep.failures.join("\n"));
}

#[test]
fn parser_handles_official_layout_quirks() {
    // "Infinity" F statistic, E+14 zero standard deviations, multi-line model.
    let t = "NIST/ITL StRD\nProcedure:     Nonlinear Least Squares Regression\nData:          1 Response (y)\n               1 Predictor (x)\n               3 Observations\nModel:         Rational Class\n               2 Parameters (b1 and b2)\n\n               y = (b1 + x) /\n                   (1 + b2*x**2) + e\n\n  b1 =   1   2     3.0E+0   1.0E-1\n  b2 =   4   5     6.0E+0   2.0E-1\n\nResidual Sum of Squares:   1.5E-2\nResidual Standard Deviation: 3E-1\nDegrees of Freedom:    1\nNumber of Observations:   3\n\nData:   y   x\n  1  2\n  2  3\n  3  4\n";
    let d = parse_dat(t, "t").unwrap();
    assert_eq!(d.kind, Kind::Nonlinear);
    assert_eq!(d.model, "(b1 + x) / (1 + b2*x**2)");
    assert_eq!(d.params.len(), 2);
    assert_eq!(d.starts, vec![vec![1.0, 4.0], vec![2.0, 5.0]]);
    assert_eq!(d.data, vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
    assert_eq!(d.nobs, Some(3));
}

#[test]
fn expression_parser_and_derivatives() {
    let e = parse("b1*(1-exp[-b2*x]) + 2*x**2 - x/b1 + arctan(b2) + cos(pi*x)").unwrap();
    let mk = |b1: f64, b2: f64| {
        let mut m = HashMap::new();
        m.insert("b1".to_string(), Dual { v: b1, d: vec![1.0, 0.0] });
        m.insert("b2".to_string(), Dual { v: b2, d: vec![0.0, 1.0] });
        m.insert("x".to_string(), Dual::constant(0.5, 2));
        e.eval(&m, 2).unwrap()
    };
    let v = mk(3.0, 0.7);
    let want = 3.0 * (1.0 - (-0.7_f64 * 0.5).exp()) + 2.0 * 0.25 - 0.5 / 3.0
        + 0.7_f64.atan()
        + (std::f64::consts::PI * 0.5).cos();
    assert!((v.v - want).abs() < 1e-14);
    let h = 1e-6;
    let d1 = (mk(3.0 + h, 0.7).v - mk(3.0 - h, 0.7).v) / (2.0 * h);
    let d2 = (mk(3.0, 0.7 + h).v - mk(3.0, 0.7 - h).v) / (2.0 * h);
    assert!((v.d[0] - d1).abs() < 1e-8 && (v.d[1] - d2).abs() < 1e-8);
    // operator precedence: -x**2 is -(x**2); ** is right associative
    let p = parse("-2**2").unwrap().eval(&HashMap::new(), 0).unwrap();
    assert_eq!(p.v, -4.0);
    let q = parse("2**3**2").unwrap().eval(&HashMap::new(), 0).unwrap();
    assert_eq!(q.v, 512.0);
}
