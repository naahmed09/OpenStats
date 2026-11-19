//! Phase 2k: design of experiments, checked against independent NumPy/SciPy computations.
mod common;

use common::data10::*;
use openstats_stats::doe::*;
use openstats_stats::report::{Reportable, SessionReport};

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}: got {a}, want {b}");
}

fn same_json(r: &SessionReport) {
    r.validate().unwrap();
    assert_eq!(SessionReport::from_json(&r.to_json()).unwrap().to_json(), r.to_json());
    assert!(!r.render_text().is_empty());
}

fn label(mask: usize) -> String {
    (0..8).filter(|j| mask >> j & 1 == 1).map(|j| (b'A' + j as u8) as char).collect()
}

/// Effect-row label of a term mask ("A*B*C").
fn term(mask: usize) -> String {
    label(mask).chars().map(String::from).collect::<Vec<_>>().join("*")
}

/// Response of the design point `row` from `(A.., y)` records of width `w`.
fn lookup(data: &[f64], w: usize, row: &[f64]) -> f64 {
    let k = row.len();
    data.chunks(w)
        .find(|c| c[..k].iter().zip(row).all(|(a, b)| (a - b).abs() < 1e-9))
        .unwrap_or_else(|| panic!("point {row:?} not in the data"))[k]
}

fn effect<'a>(a: &'a FactorialAnalysis, term: &str) -> &'a EffectRow {
    a.effects.iter().find(|e| e.term == term).unwrap_or_else(|| panic!("no term {term}"))
}

// ------------------------------------------------------------------ factorial

#[test]
fn pilot_plant_full_factorial_effects_and_lenth() {
    let d = full_factorial(3, &FactorialOptions::default()).unwrap();
    assert_eq!(d.n_runs(), 8);
    assert!(d.columns_orthogonal(1e-12));
    // Standard order: A fastest.
    assert_eq!(d.column(0)[..4], [-1.0, 1.0, -1.0, 1.0]);
    let a = analyze_factorial(&d, PP_Y, &FactorialAnalysisOptions::default()).unwrap();
    for (i, m) in (1..8usize).enumerate() {
        close(effect(&a, &term(m)).effect.unwrap(), PP_EFFECTS[i], 1e-10, &label(m));
    }
    assert_eq!(a.df_error, 0);
    let l = a.lenth.as_ref().unwrap();
    close(l.pse, PP_LENTH[0], 1e-10, "pse");
    close(l.df, PP_LENTH[1], 1e-12, "df");
    close(l.margin, PP_LENTH[2], 1e-9, "me");
    close(l.simultaneous_margin, PP_LENTH[3], 1e-9, "sme");
    // The published BHH effects.
    close(effect(&a, "A").effect.unwrap(), 23.0, 1e-12, "T");
    close(effect(&a, "B").effect.unwrap(), -5.0, 1e-12, "C");
    close(effect(&a, "A*C").effect.unwrap(), 10.0, 1e-12, "TK");
    same_json(&a.to_report("pilot plant"));
}

#[test]
fn filtration_2_4_effects_and_regression_match_numpy() {
    let d = full_factorial(4, &FactorialOptions::default()).unwrap();
    let y: Vec<f64> = d.coded.iter().map(|r| lookup(FIL_DATA, 5, r)).collect();
    let opts = FactorialAnalysisOptions { max_order: Some(2), ..Default::default() };
    let a = analyze_factorial(&d, &y, &opts).unwrap();
    for m in 1..16usize {
        if label(m).len() > 2 {
            continue;
        }
        close(effect(&a, &term(m)).effect.unwrap(), FIL_EFFECTS[m - 1], 1e-10, &term(m));
    }
    close(effect(&a, "A").effect.unwrap(), 21.625, 1e-12, "A");
    close(effect(&a, "A*C").effect.unwrap(), -18.125, 1e-12, "AC");
    close(effect(&a, "A*D").effect.unwrap(), 16.625, 1e-12, "AD");
    assert_eq!(a.df_error, 5);
    close(a.s.unwrap(), FIL2_FIT[2], 1e-9, "S");
    close(a.r2, FIL2_FIT[3], 1e-10, "R2");
    close(a.r2_adj.unwrap(), FIL2_FIT[4], 1e-10, "R2adj");
    close(a.r2_pred.unwrap(), FIL2_FIT[5], 1e-9, "R2pred");
    // Coefficient table (constant, then the terms of order <= 2 by mask).
    close(a.constant.coef, FIL2_COEF[0], 1e-10, "const");
    close(a.constant.p.unwrap(), FIL2_COEF[3], 1e-8, "const p");
    for (i, &m) in FIL2_MASKS.iter().enumerate() {
        let tm = term(m as usize);
        let e = effect(&a, &tm);
        let r = &FIL2_COEF[4 * (i + 1)..4 * (i + 1) + 4];
        close(e.coef, r[0], 1e-10, &tm);
        close(e.se_coef.unwrap(), r[1], 1e-10, &tm);
        close(e.t.unwrap(), r[2], 1e-9, &tm);
        close(e.p.unwrap(), r[3], 1e-8, &tm);
    }
    let model = a.anova.iter().find(|g| g.source == "Model").unwrap();
    close(model.df as f64, FIL2_MODEL_F[1], 0.0, "model df");
    close(model.adj_ss, FIL2_MODEL_F[0], 1e-10, "model ss");
    close(model.f.unwrap(), FIL2_MODEL_F[2], 1e-9, "model F");
    close(model.p.unwrap(), FIL2_MODEL_F[3], 1e-8, "model p");
    same_json(&a.to_report("filtration"));
}

#[test]
fn half_fraction_aliasing_and_effects() {
    let d = fractional_factorial(4, 1, Some(&["D=ABC"]), &FactorialOptions::default()).unwrap();
    assert_eq!(d.n_runs(), 8);
    assert!(d.columns_orthogonal(1e-12));
    let al = d.alias_structure(3).unwrap();
    assert_eq!(al.resolution, Some(4));
    assert_eq!(al.defining_relation, vec!["ABCD".to_string()]);
    let class = |t: &str| al.classes.iter().find(|c| c.iter().any(|x| x == t)).unwrap().clone();
    assert!(class("A").contains(&"BCD".to_string()));
    assert!(class("AB").contains(&"CD".to_string()));
    assert!(class("AC").contains(&"BD".to_string()));
    assert!(class("AD").contains(&"BC".to_string()));
    let y: Vec<f64> = d.coded.iter().map(|r| lookup(HALF_DATA, 5, r)).collect();
    let a = analyze_factorial(&d, &y, &FactorialAnalysisOptions::default()).unwrap();
    for (i, t) in ["A", "B", "C", "D", "A*B", "A*C", "A*D"].iter().enumerate() {
        close(effect(&a, t).effect.unwrap(), HALF_EFFECTS[i], 1e-10, t);
    }
    close(a.lenth.as_ref().unwrap().pse, HALF_LENTH[0], 1e-10, "half pse");
    same_json(&al.to_report("alias"));
    same_json(&d.to_report("design"));
}

#[test]
fn default_generators_give_known_resolutions() {
    let res = |k, p| {
        let d = fractional_factorial(k, p, None, &FactorialOptions::default()).unwrap();
        (d.n_runs(), d.defining.as_ref().unwrap().resolution)
    };
    assert_eq!(res(4, 1), (8, 4));
    assert_eq!(res(5, 1), (16, 5));
    assert_eq!(res(5, 2), (8, 3));
    assert_eq!(res(6, 2), (16, 4));
    assert_eq!(res(7, 3), (16, 4));
    assert_eq!(res(7, 4), (8, 3));
    assert_eq!(res(6, 1), (32, 6));
    assert_eq!(res(8, 4), (16, 4));
}

#[test]
fn centre_points_blocks_and_randomisation() {
    let opts = FactorialOptions { replicates: 2, center_points: 3, ..Default::default() };
    let d = full_factorial(3, &opts).unwrap();
    assert_eq!(d.n_runs(), 16 + 3);
    assert_eq!(d.point_type.iter().filter(|p| **p == PointType::Center).count(), 3);
    let r = d.clone().randomized(7);
    assert_eq!(r.n_runs(), d.n_runs());
    let mut a: Vec<usize> = r.std_order.clone();
    a.sort_unstable();
    let mut b = d.std_order.clone();
    b.sort_unstable();
    assert_eq!(a, b);
    assert_ne!(r.std_order, d.std_order);
    assert_eq!(d.clone().randomized(7).std_order, r.std_order);
    let blocked =
        full_factorial(3, &FactorialOptions { block_generators: vec!["ABC".into()], ..Default::default() }).unwrap();
    assert_eq!(blocked.blocks, 2);
    for b in 1..=2 {
        let rows: Vec<_> = (0..8).filter(|&i| blocked.block[i] == b).collect();
        assert_eq!(rows.len(), 4);
        let s: f64 = rows.iter().map(|&i| blocked.coded[i].iter().product::<f64>()).sum();
        assert!((s.abs() - 4.0).abs() < 1e-12); // ABC constant within a block
    }
}

#[test]
fn plackett_burman_and_hadamard() {
    for n in [4usize, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48] {
        let h = hadamard::hadamard(n).unwrap_or_else(|| panic!("no hadamard {n}"));
        assert!(hadamard::is_hadamard(&h), "n={n}");
    }
    let d = plackett_burman(12, 11, &FactorialOptions::default()).unwrap();
    assert_eq!((d.n_runs(), d.n_factors()), (12, 11));
    assert!(d.columns_orthogonal(1e-12));
    for j in 0..11 {
        assert_eq!(d.column(j).iter().sum::<f64>(), 0.0);
    }
    let d = plackett_burman(20, 15, &FactorialOptions::default()).unwrap();
    assert_eq!((d.n_runs(), d.n_factors()), (20, 15));
    assert!(d.columns_orthogonal(1e-12));
    assert!(plackett_burman(12, 12, &FactorialOptions::default()).is_err());
    assert!(plackett_burman(10, 3, &FactorialOptions::default()).is_err());
}

#[test]
fn factorial_error_cases() {
    assert!(full_factorial(1, &FactorialOptions::default()).is_err());
    assert!(fractional_factorial(4, 1, Some(&["D=AB"]), &FactorialOptions::default()).is_ok());
    assert!(fractional_factorial(4, 1, Some(&["E=ABC"]), &FactorialOptions::default()).is_err());
    assert!(fractional_factorial(4, 4, None, &FactorialOptions::default()).is_err());
    let d = full_factorial(2, &FactorialOptions::default()).unwrap();
    assert!(analyze_factorial(&d, &[1.0, 2.0], &FactorialAnalysisOptions::default()).is_err());
    assert!(d.clone().with_names(&["x"]).is_err());
}

// ------------------------------------------------------------------ RSM

#[test]
fn ccd_structure() {
    let r = central_composite(3, &CcdOptions::new(CcdKind::Circumscribed)).unwrap();
    let alpha = r.coded.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    close(alpha, 8f64.powf(0.25), 1e-12, "rotatable alpha");
    let f = central_composite(3, &CcdOptions::new(CcdKind::FaceCentered)).unwrap();
    assert!(f.coded.iter().flatten().all(|v| v.abs() <= 1.0 + 1e-12));
    let i = central_composite(3, &CcdOptions::new(CcdKind::Inscribed)).unwrap();
    close(i.coded.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs())), 1.0, 1e-12, "inscribed max");
    let axial = r.point_type.iter().filter(|p| **p == PointType::Axial).count();
    assert_eq!(axial, 6);
    let cube = r.point_type.iter().filter(|p| **p == PointType::Factorial).count();
    assert_eq!(cube, 8);
    // Rotatable: all axial points are alpha from the centre.
    for (row, t) in r.coded.iter().zip(&r.point_type) {
        if *t == PointType::Axial {
            close(row.iter().map(|v| v * v).sum::<f64>().sqrt(), alpha, 1e-12, "axial radius");
        }
    }
    let half = central_composite(5, &CcdOptions { fraction_p: 1, ..CcdOptions::new(CcdKind::Circumscribed) }).unwrap();
    assert_eq!(half.point_type.iter().filter(|p| **p == PointType::Factorial).count(), 16);
}

#[test]
fn box_behnken_structure() {
    let d = box_behnken(3, 3, None).unwrap();
    assert_eq!(d.n_runs(), 12 + 3);
    for row in &d.coded {
        assert!(row.iter().filter(|v| **v == 0.0).count() >= 1);
        assert!(row.iter().all(|v| v.abs() <= 1.0));
    }
    assert_eq!(box_behnken(4, 3, None).unwrap().n_runs(), 24 + 3);
    assert_eq!(box_behnken(5, 6, None).unwrap().n_runs(), 40 + 6);
    assert!(box_behnken(6, 3, None).is_err());
    assert!(box_behnken(2, 3, None).is_err());
}

#[test]
fn ccd_quadratic_fit_matches_numpy() {
    let mut opts = CcdOptions::new(CcdKind::Circumscribed);
    opts.center_factorial = 3;
    opts.center_axial = 2;
    let d = central_composite(2, &opts).unwrap();
    assert_eq!(d.n_runs(), 13);
    let mut centers = CCD_DATA.chunks(3).filter(|c| c[0] == 0.0 && c[1] == 0.0).map(|c| c[2]);
    let y: Vec<f64> = d
        .coded
        .iter()
        .zip(&d.point_type)
        .map(|(r, t)| if *t == PointType::Center { centers.next().unwrap() } else { lookup(CCD_DATA, 3, r) })
        .collect();
    let a = analyze_surface(&d, &y, &SurfaceOptions::default()).unwrap();
    assert_eq!(a.fit.df_error, 7);
    close(a.fit.sse, CCD_FIT[0], 1e-9, "sse");
    close(a.fit.s, CCD_FIT[2], 1e-9, "s");
    close(a.fit.r2, CCD_FIT[3], 1e-10, "r2");
    close(a.fit.r2_adj, CCD_FIT[4], 1e-10, "r2adj");
    close(a.fit.r2_pred, CCD_FIT[5], 1e-8, "r2pred");
    for (i, name) in ["Constant", "A", "B", "A*A", "B*B", "A*B"].iter().enumerate() {
        let c = a.fit.coefficients.iter().find(|c| c.name == *name).unwrap_or_else(|| panic!("no coefficient {name}"));
        let r = &CCD_COEF[4 * i..4 * i + 4];
        close(c.estimate, r[0], 1e-9, name);
        close(c.std_err, r[1], 1e-9, name);
        close(c.t, r[2], 1e-8, name);
        close(c.p_value, r[3], 1e-7, name);
    }
    let lof = a.anova.iter().find(|r| r.source == "Lack-of-Fit").unwrap();
    let pe = a.anova.iter().find(|r| r.source == "Pure Error").unwrap();
    close(lof.adj_ss, CCD_LOF[0], 1e-8, "lof ss");
    assert_eq!(lof.df as f64, CCD_LOF[1]);
    close(pe.adj_ss, CCD_LOF[2], 1e-8, "pe ss");
    assert_eq!(pe.df as f64, CCD_LOF[3]);
    close(lof.f.unwrap(), CCD_LOF[4], 1e-8, "lof F");
    close(lof.p.unwrap(), CCD_LOF[5], 1e-7, "lof p");
    let c = a.canonical.as_ref().unwrap();
    close(c.point_coded[0], CCD_CANON[0], 1e-9, "xs1");
    close(c.point_coded[1], CCD_CANON[1], 1e-9, "xs2");
    close(c.response, CCD_CANON[2], 1e-9, "ys");
    close(c.eigenvalues[0], CCD_CANON[3], 1e-9, "eig1");
    close(c.eigenvalues[1], CCD_CANON[4], 1e-9, "eig2");
    assert_eq!(c.kind, StationaryKind::Maximum);
    close(a.model.predict(&[0.3, -0.4]), CCD_PRED[0], 1e-9, "pred");
    close(a.model.se_fit(&[0.3, -0.4]).unwrap(), CCD_PRED[1], 1e-8, "se fit");
    same_json(&a.to_report("ccd"));

    // Contour and surface data.
    let plot = SurfacePlot {
        model: &a.model,
        factors: &d.factors,
        ix: 0,
        iy: 1,
        fixed: &[0.0, 0.0],
        range: ([-1.5, 1.5], [-1.5, 1.5]),
        grid_points: 31,
    };
    let c = contour_chart(&plot, 6).unwrap();
    assert!(!c.series.is_empty());
    let s = surface_chart(&plot).unwrap();
    assert_eq!(s.series.len(), 31);
    assert_eq!(s.series[0].x.len(), 31);
    // Every point of a contour lies on its level (checked through the model).
    for ser in &c.series {
        let level: f64 = ser.name.parse().unwrap();
        for (x, y) in ser.x.iter().zip(&ser.y) {
            if x.is_nan() {
                continue;
            }
            let v = a.model.predict(&[d.factors[0].to_coded(*x), d.factors[1].to_coded(*y)]);
            assert!((v - level).abs() < 0.25, "contour point off level: {v} vs {level}");
        }
    }
    let mut r = SessionReport::new("plots");
    let mut sec = openstats_stats::report::Section::new("Plots");
    sec.chart(c).chart(s);
    r.push(sec);
    same_json(&r);
}

#[test]
fn surface_model_choices_and_errors() {
    let d = box_behnken(3, 3, None).unwrap();
    let y: Vec<f64> =
        d.coded.iter().map(|r| 10.0 + r[0] - 2.0 * r[1] + 0.5 * r[2] - r[0] * r[0] + 0.3 * r[1] * r[2]).collect();
    let a = analyze_surface(&d, &y, &SurfaceOptions::default()).unwrap();
    close(a.fit.r2, 1.0, 1e-9, "exact quadratic");
    let lin = analyze_surface(&d, &y, &SurfaceOptions { model: SurfaceModel::Linear, ..Default::default() }).unwrap();
    assert!(lin.fit.r2 < 1.0);
    assert!(lin.canonical.is_none());
    assert!(analyze_surface(&d, &y[..5], &SurfaceOptions::default()).is_err());
}

// ------------------------------------------------------------------ Taguchi

#[test]
fn orthogonal_arrays_are_orthogonal() {
    for (name, runs, cols) in [
        ("L4", 4, 3),
        ("L8", 8, 7),
        ("L9", 9, 4),
        ("L12", 12, 11),
        ("L16", 16, 15),
        ("L18", 18, 8),
        ("L25", 25, 6),
        ("L27", 27, 13),
        ("L32", 32, 31),
        ("L81", 81, 40),
    ] {
        let a = orthogonal_array(name).unwrap();
        assert_eq!((a.runs(), a.columns()), (runs, cols), "{name}");
        assert!(a.is_orthogonal(), "{name}");
    }
    assert!(orthogonal_array("L11").is_err());
    assert_eq!(orthogonal_array("L18").unwrap().levels, vec![2, 3, 3, 3, 3, 3, 3, 3]);
}

#[test]
fn taguchi_l9_matches_numpy() {
    let arr = orthogonal_array("L9").unwrap();
    // The generator builds the same L9 (normalised GF(3) columns, lexicographic).
    for r in 0..9 {
        for c in 0..4 {
            assert_eq!(arr.data[r][c] as f64, TG_L9[r * 4 + c], "L9 entry {r},{c}");
        }
    }
    let design = TaguchiDesign::new(arr, &[("A", 3), ("B", 3), ("C", 3)]).unwrap();
    let y: Vec<Vec<f64>> = TG_Y.chunks(3).map(|c| c.to_vec()).collect();
    let nom = analyze_taguchi(&design, &y, SignalToNoise::NominalIsBest).unwrap();
    for r in 0..9 {
        close(nom.run_sn[r], TG_SN_NOM[r], 1e-10, "nominal");
    }
    let chk = |t: SignalToNoise, want: &[f64]| {
        let a = analyze_taguchi(&design, &y, t).unwrap();
        for r in 0..9 {
            close(a.run_sn[r], want[r], 1e-10, t.name());
        }
    };
    chk(SignalToNoise::SmallerIsBetter, TG_SN_SMALL);
    chk(SignalToNoise::LargerIsBetter, TG_SN_LARGE);
    chk(SignalToNoise::NominalIsBestVariance, TG_SN_NOMV);
    for j in 0..3 {
        for l in 0..3 {
            close(nom.sn_table[j].level_means[l], TG_SN_TABLE[j * 3 + l], 1e-10, "sn table");
            close(nom.mean_table[j].level_means[l], TG_MEAN_TABLE[j * 3 + l], 1e-10, "mean table");
        }
    }
    for (anova, want) in [(&nom.sn_anova, TG_SN_ANOVA), (&nom.mean_anova, TG_MEAN_ANOVA)] {
        for j in 0..3 {
            close(anova[j].ss, want[j], 1e-9, "anova ss");
            close(anova[j].f.unwrap(), want[5 + j], 1e-8, "anova F");
            close(anova[j].p.unwrap(), want[8 + j], 1e-7, "anova p");
        }
        let err = anova.iter().find(|r| r.source == "Residual Error").unwrap();
        close(err.ss, want[3], 1e-8, "sse");
        close(anova.last().unwrap().ss, want[4], 1e-9, "sst");
        let pct: f64 = anova[..anova.len() - 1].iter().map(|r| r.percent).sum();
        close(pct, 100.0, 1e-9, "percent");
    }
    for j in 0..3 {
        assert_eq!(nom.best_levels[j] as f64, TG_BEST[j]);
    }
    close(nom.predicted_sn, TG_BEST[3], 1e-10, "predicted sn");
    assert_eq!(nom.charts().len(), 2);
    same_json(&nom.to_report("taguchi"));
}

#[test]
fn taguchi_errors_and_ratios() {
    let a = orthogonal_array("L8").unwrap();
    assert!(TaguchiDesign::new(a.clone(), &[("A", 3)]).is_err());
    assert!(TaguchiDesign::new(a.clone(), &[("A", 2), ("A", 2)]).is_err());
    let seven: Vec<(String, usize)> = (0..8).map(|i| (format!("F{i}"), 2)).collect();
    let refs: Vec<(&str, usize)> = seven.iter().map(|(n, l)| (n.as_str(), *l)).collect();
    assert!(TaguchiDesign::new(a, &refs).is_err());
    close(SignalToNoise::SmallerIsBetter.ratio(&[1.0, 1.0]).unwrap(), 0.0, 1e-12, "smaller");
    close(SignalToNoise::LargerIsBetter.ratio(&[10.0, 10.0]).unwrap(), 20.0, 1e-12, "larger");
    assert!(SignalToNoise::LargerIsBetter.ratio(&[0.0, 1.0]).is_err());
    assert!(SignalToNoise::NominalIsBest.ratio(&[3.0]).is_err());
    assert!(SignalToNoise::NominalIsBest.ratio(&[3.0, 3.0]).is_err());
}

// ------------------------------------------------------------------ optimiser

fn model(names: &[&str], terms: &[Vec<(usize, u32)>], coef: &[f64]) -> PolynomialModel {
    PolynomialModel {
        factor_names: names.iter().map(|s| s.to_string()).collect(),
        terms: terms.iter().map(|p| PolyTerm { powers: p.clone() }).collect(),
        coef: coef.to_vec(),
        cov: None,
        mse: None,
        df_error: 0,
    }
}

#[test]
fn desirability_functions() {
    let mx = Goal::Maximize { lower: 10.0, target: 20.0, shape: 1.0 };
    assert_eq!(mx.desirability(9.0), 0.0);
    close(mx.desirability(15.0), 0.5, 1e-15, "max mid");
    assert_eq!(mx.desirability(25.0), 1.0);
    let mn = Goal::Minimize { target: 1.0, upper: 5.0, shape: 2.0 };
    assert_eq!(mn.desirability(0.5), 1.0);
    close(mn.desirability(3.0), 0.25, 1e-15, "min mid");
    assert_eq!(mn.desirability(6.0), 0.0);
    let tg = Goal::Target { lower: 0.0, target: 10.0, upper: 30.0, shape_low: 1.0, shape_high: 2.0 };
    close(tg.desirability(5.0), 0.5, 1e-15, "target low");
    close(tg.desirability(20.0), 0.25, 1e-15, "target high");
    assert_eq!(tg.desirability(10.0), 1.0);
    assert_eq!(tg.desirability(31.0), 0.0);
}

#[test]
fn optimizer_matches_closed_form_and_brute_force() {
    let factors = vec![
        FactorSpec { name: "T".into(), low: 100.0, high: 200.0 },
        FactorSpec { name: "P".into(), low: 1.0, high: 3.0 },
    ];
    // y1 = 50 + 8 x1 + 4 x2 - 3 x1^2 - 2 x2^2 (maximise), y2 = 10 + 2 x1 - 3 x2 + x1 x2 (target 9).
    let terms: Vec<Vec<(usize, u32)>> = vec![vec![], vec![(0, 1)], vec![(1, 1)], vec![(0, 2)], vec![(1, 2)]];
    let m1 = model(&["T", "P"], &terms, &[50.0, 8.0, 4.0, -3.0, -2.0]);
    let t2: Vec<Vec<(usize, u32)>> = vec![vec![], vec![(0, 1)], vec![(1, 1)], vec![(0, 1), (1, 1)]];
    let m2 = model(&["T", "P"], &t2, &[10.0, 2.0, -3.0, 1.0]);
    let goals = vec![
        ResponseGoal {
            name: "Y1".into(),
            model: m1.clone(),
            goal: Goal::Maximize { lower: 40.0, target: 60.0, shape: 1.0 },
            importance: 1.0,
        },
        ResponseGoal {
            name: "Y2".into(),
            model: m2.clone(),
            goal: Goal::Target { lower: 4.0, target: 9.0, upper: 14.0, shape_low: 1.0, shape_high: 1.0 },
            importance: 2.0,
        },
    ];
    let res = optimize_responses(&goals, &factors, &OptimizeOptions::default()).unwrap();
    let best = res.best();
    // Brute force over a fine grid.
    let mut bd = 0.0;
    let n = 400;
    for i in 0..=n {
        for j in 0..=n {
            let x = [-1.0 + 2.0 * i as f64 / n as f64, -1.0 + 2.0 * j as f64 / n as f64];
            let s = composite_desirability(&goals, &factors, &x);
            if s.composite > bd {
                bd = s.composite;
            }
        }
    }
    assert!(best.composite >= bd - 1e-12, "optimizer {} vs grid {}", best.composite, bd);
    assert!(best.composite <= bd + 1e-3);
    // Composite formula.
    let d1 = goals[0].goal.desirability(best.fit[0]);
    let d2 = goals[1].goal.desirability(best.fit[1]);
    close(best.composite, (d1 * d2 * d2).powf(1.0 / 3.0), 1e-12, "geometric mean");
    close(best.natural[0], 150.0 + 50.0 * best.coded[0], 1e-9, "natural T");
    assert!(best.coded.iter().all(|v| v.abs() <= 1.0 + 1e-12));
    for w in res.solutions.windows(2) {
        assert!(w[0].composite >= w[1].composite);
    }
    same_json(&res.to_report("optimizer"));
    assert_eq!(res.charts(None, 11).len(), 2);

    // Single response, target with a unique interior optimum: y = 10 + 4x1 + 2x2, target 10 -> D=1 on the line 2x1 + x2 = 0.
    let lin = model(&["T", "P"], &[vec![], vec![(0, 1)], vec![(1, 1)]], &[10.0, 4.0, 2.0]);
    let g = vec![ResponseGoal {
        name: "Y".into(),
        model: lin,
        goal: Goal::Target { lower: 4.0, target: 10.0, upper: 16.0, shape_low: 1.0, shape_high: 1.0 },
        importance: 1.0,
    }];
    let r = optimize_responses(&g, &factors, &OptimizeOptions::default()).unwrap();
    close(r.best().composite, 1.0, 1e-6, "on the target line");
    close(2.0 * r.best().coded[0] + r.best().coded[1], 0.0, 1e-4, "constraint");

    // Fixed factor.
    let fixed = OptimizeOptions { fixed: vec![Some(100.0), None], ..Default::default() };
    let r = optimize_responses(&goals, &factors, &fixed).unwrap();
    close(r.best().coded[0], -1.0, 1e-12, "fixed T");
    // Restricted range.
    let rng = OptimizeOptions { ranges: Some(vec![(-0.2, 0.2), (-1.0, 1.0)]), ..Default::default() };
    let r = optimize_responses(&goals, &factors, &rng).unwrap();
    assert!(r.best().coded[0].abs() <= 0.2 + 1e-12);
}

#[test]
fn optimizer_errors() {
    let factors = vec![FactorSpec { name: "T".into(), low: 0.0, high: 1.0 }];
    let m = model(&["T"], &[vec![], vec![(0, 1)]], &[1.0, 1.0]);
    let mk = |goal, importance| ResponseGoal { name: "Y".into(), model: m.clone(), goal, importance };
    let ok = Goal::Maximize { lower: 0.0, target: 1.0, shape: 1.0 };
    assert!(optimize_responses(&[], &factors, &OptimizeOptions::default()).is_err());
    assert!(optimize_responses(&[mk(ok, 0.0)], &factors, &OptimizeOptions::default()).is_err());
    assert!(optimize_responses(
        &[mk(Goal::Maximize { lower: 1.0, target: 1.0, shape: 1.0 }, 1.0)],
        &factors,
        &OptimizeOptions::default()
    )
    .is_err());
    let bad = OptimizeOptions { fixed: vec![Some(5.0)], ..Default::default() };
    assert!(optimize_responses(&[mk(ok, 1.0)], &factors, &bad).is_err());
}
