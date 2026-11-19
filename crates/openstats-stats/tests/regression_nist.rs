//! NIST StRD linear-regression datasets through the formula layer.
//! The fixture files are OpenStats reconstructions (see their headers).

use openstats_stats::model::{Data, DesignOptions};
use openstats_stats::regression::{regress, RegressionOptions};
use std::path::Path;

struct Nist {
    params: Vec<(f64, f64)>,
    resid_sd: f64,
    r2: f64,
    ssr: f64,
    sse: f64,
    f: f64,
    cols: Vec<Vec<f64>>,
}

fn load(name: &str) -> Nist {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/nist/embedded").join(name);
    let text = std::fs::read_to_string(p).unwrap();
    let num = |t: &str| -> Option<f64> {
        if t.eq_ignore_ascii_case("infinity") {
            Some(f64::INFINITY)
        } else {
            t.parse().ok()
        }
    };
    let (mut params, mut cols) = (Vec::new(), Vec::new());
    let (mut resid_sd, mut r2, mut ssr, mut sse, mut f) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut in_data = false;
    for l in text.lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        if in_data {
            let v: Vec<f64> = t.iter().filter_map(|x| num(x)).collect();
            if cols.is_empty() {
                cols = vec![Vec::new(); v.len()];
            }
            for (c, x) in cols.iter_mut().zip(v) {
                c.push(x);
            }
        } else if t[0] == "Data:" && t.get(1) == Some(&"y") {
            in_data = true;
        } else if t[0].starts_with('B') && t.len() == 3 && t[0][1..].chars().all(|c| c.is_ascii_digit()) {
            params.push((num(t[1]).unwrap(), num(t[2]).unwrap()));
        } else if t[0] == "Standard" && t[1] == "Deviation" && t.len() == 3 {
            resid_sd = num(t[2]).unwrap();
        } else if t[0] == "R-Squared" {
            r2 = num(t[1]).unwrap();
        } else if t[0] == "Regression" {
            ssr = num(t[2]).unwrap();
            f = num(t[4]).unwrap();
        } else if t[0] == "Residual" && t.len() == 4 {
            sse = num(t[2]).unwrap();
        }
    }
    Nist { params, resid_sd, r2, ssr, sse, f, cols }
}

fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a.abs()
    } else {
        ((a - b) / b).abs()
    }
}

fn lre(a: f64, b: f64) -> f64 {
    let r = rel(a, b);
    if r == 0.0 {
        16.0
    } else {
        -r.log10()
    }
}

#[test]
fn longley() {
    let d = load("Longley.dat");
    let mut data = Data::new();
    for (i, c) in d.cols.iter().enumerate() {
        let name = if i == 0 { "y".to_string() } else { format!("x{i}") };
        data = data.with_numeric(&name, c.clone()).unwrap();
    }
    let fit =
        regress("y ~ x1+x2+x3+x4+x5+x6", &data, &DesignOptions::default(), &RegressionOptions::default()).unwrap();
    assert_eq!(fit.coefficients.len(), 7);
    for (c, (est, sd)) in fit.coefficients.iter().zip(&d.params) {
        assert!(lre(c.estimate, *est) > 9.0, "{} est lre {}", c.name, lre(c.estimate, *est));
        assert!(lre(c.std_err, *sd) > 9.0, "{} se lre {}", c.name, lre(c.std_err, *sd));
    }
    assert!(lre(fit.s, d.resid_sd) > 10.0);
    assert!(lre(fit.r2, d.r2) > 10.0);
    assert!(lre(fit.ssr, d.ssr) > 10.0);
    assert!(lre(fit.sse, d.sse) > 10.0);
    assert!(lre(fit.f.unwrap(), d.f) > 9.0);
    // Longley is the textbook ill-conditioned case: VIFs are huge.
    assert!(fit.coefficients[1..].iter().all(|c| c.vif.unwrap() > 1.0));
    assert!(fit.coefficients.iter().map(|c| c.vif.unwrap_or(0.0)).fold(0.0, f64::max) > 100.0);
}

#[test]
fn wampler1_exact_fit() {
    let d = load("Wampler1.dat");
    let data = Data::new().with_numeric("y", d.cols[0].clone()).unwrap().with_numeric("x", d.cols[1].clone()).unwrap();
    let fit = regress(
        "y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5)",
        &data,
        &DesignOptions::default(),
        &RegressionOptions::default(),
    )
    .unwrap();
    for c in &fit.coefficients {
        assert!(lre(c.estimate, 1.0) > 8.0, "{} lre {}", c.name, lre(c.estimate, 1.0));
    }
    assert!(fit.r2 > 1.0 - 1e-12);
    assert!(fit.s / 1e6 < 1e-6, "s = {}", fit.s);
}

#[test]
fn wampler2() {
    let d = load("Wampler2.dat");
    let data = Data::new().with_numeric("y", d.cols[0].clone()).unwrap().with_numeric("x", d.cols[1].clone()).unwrap();
    let fit = regress(
        "y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5)",
        &data,
        &DesignOptions::default(),
        &RegressionOptions::default(),
    )
    .unwrap();
    for (c, (est, _)) in fit.coefficients.iter().zip(&d.params) {
        assert!(lre(c.estimate, *est) > 8.0, "{} lre {}", c.name, lre(c.estimate, *est));
    }
    assert!(lre(fit.r2, d.r2) > 8.0 || fit.r2 > 1.0 - 1e-12);
}

#[test]
fn no_intercept() {
    for name in ["NoInt1.dat", "NoInt2.dat"] {
        let d = load(name);
        let data =
            Data::new().with_numeric("y", d.cols[0].clone()).unwrap().with_numeric("x", d.cols[1].clone()).unwrap();
        let fit = regress("y ~ x - 1", &data, &DesignOptions::default(), &RegressionOptions::default()).unwrap();
        assert_eq!(fit.coefficients.len(), 1);
        let (est, sd) = d.params[0];
        assert!(lre(fit.coefficients[0].estimate, est) > 10.0, "{name}");
        assert!(lre(fit.coefficients[0].std_err, sd) > 10.0, "{name}");
        assert!(lre(fit.r2, d.r2) > 10.0, "{name} r2 {} vs {}", fit.r2, d.r2);
        assert!(lre(fit.f.unwrap(), d.f) > 9.0, "{name} F");
    }
}
