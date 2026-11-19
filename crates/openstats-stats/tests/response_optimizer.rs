//! Worksheet-level multi-response optimisation (Phase 4c): design reconstruction from columns, one
//! model per response, goal parsing and the composite-desirability search against a brute-force grid.

use openstats_stats::doe::*;
use openstats_stats::report::Reportable;

fn ccd2() -> (Design, Vec<String>, Vec<Vec<f64>>) {
    let d = central_composite(2, &CcdOptions::new(CcdKind::Circumscribed))
        .unwrap()
        .with_levels(&[(100.0, 200.0), (10.0, 30.0)])
        .unwrap();
    let names = vec!["Temp".to_string(), "Time".to_string()];
    let nat = d.natural();
    let cols = vec![nat.iter().map(|r| r[0]).collect(), nat.iter().map(|r| r[1]).collect()];
    (d, names, cols)
}

fn y1(x: &[f64]) -> f64 {
    50.0 + 8.0 * x[0] + 3.0 * x[1] - 4.0 * x[0] * x[0] - 2.0 * x[1] * x[1] + x[0] * x[1]
}

fn y2(x: &[f64]) -> f64 {
    20.0 - 5.0 * x[0] + 2.0 * x[1] + 3.0 * x[1] * x[1]
}

fn responses(d: &Design) -> Vec<(String, Vec<f64>)> {
    vec![
        ("Yield".into(), d.coded.iter().map(|r| y1(r)).collect()),
        ("Cost".into(), d.coded.iter().map(|r| y2(r)).collect()),
    ]
}

#[test]
fn design_is_rebuilt_from_natural_columns() {
    let (d, names, cols) = ccd2();
    let r = design_from_columns(&names, &cols).unwrap();
    assert_eq!(r.n_runs(), d.n_runs());
    // Range coding: the axial points are the extremes, so the cube corners sit at ±1/α.
    let alpha = 2f64.sqrt();
    let corner = r.coded.iter().find(|row| row[0] > 0.0 && row[1] > 0.0 && row[0] < 0.9).unwrap();
    assert!((corner[0] - 1.0 / alpha).abs() < 1e-12);
    assert!((r.factors[0].low - (150.0 - 50.0 * alpha)).abs() < 1e-9);
    assert!(r.point_type.contains(&PointType::Center));
    assert!(design_from_columns(&names, &[cols[0].clone()]).is_err());
    assert!(design_from_columns(&names, &[cols[0].clone(), vec![1.0; cols[0].len()]]).is_err());
    assert!(design_from_columns(&names, &[cols[0].clone(), cols[1][..3].to_vec()]).is_err());
    let mut bad = cols.clone();
    bad[1][0] = f64::NAN;
    assert!(design_from_columns(&names, &bad).is_err());
    assert!(design_from_columns(&[], &[]).is_err());
}

#[test]
fn models_recover_the_generating_polynomials() {
    let (d, names, cols) = ccd2();
    let r = design_from_columns(&names, &cols).unwrap();
    // Responses are exact quadratics in the cube-coded units, so they are exact quadratics in the
    // range-coded ones too; compare predictions at arbitrary points through natural units.
    let fits = fit_responses(&r, &responses(&d), ModelOrder::Quadratic).unwrap();
    assert_eq!(fits.len(), 2);
    for f in &fits {
        assert!(f.r2 > 0.999_999, "{} R² {}", f.name, f.r2);
        assert!(f.dropped.is_empty());
    }
    for x in [[0.3, -0.7], [-1.0, 0.2], [0.9, 0.9], [0.0, 0.0]] {
        let nat: Vec<f64> = (0..2).map(|j| d.factors[j].to_natural(x[j])).collect();
        let c: Vec<f64> = (0..2).map(|j| r.factors[j].to_coded(nat[j])).collect();
        assert!((fits[0].model.predict(&c) - y1(&x)).abs() < 1e-8);
        assert!((fits[1].model.predict(&c) - y2(&x)).abs() < 1e-8);
    }
    let s = model_summary_section(&fits);
    assert!(s.to_report_text().contains("Yield"));
}

trait Txt {
    fn to_report_text(&self) -> String;
}

impl Txt for openstats_stats::report::Section {
    fn to_report_text(&self) -> String {
        let mut r = openstats_stats::report::SessionReport::new("t");
        r.push(self.clone());
        r.render_text()
    }
}

#[test]
fn linear_and_interaction_orders_and_errors() {
    let (d, names, cols) = ccd2();
    let r = design_from_columns(&names, &cols).unwrap();
    let lin = fit_responses(&r, &responses(&d), ModelOrder::Linear).unwrap();
    assert_eq!(lin[0].model.terms.len(), 3);
    let inter = fit_responses(&r, &responses(&d), ModelOrder::Interactions).unwrap();
    assert_eq!(inter[0].model.terms.len(), 4);
    assert!(lin[0].r2 < 0.99, "a linear model cannot fit a quadratic response");
    assert!(fit_responses(&r, &[], ModelOrder::Linear).is_err());
    let mut y = responses(&d);
    y[1].1[2] = f64::NAN;
    let e = fit_responses(&r, &y, ModelOrder::Linear).unwrap_err().to_string();
    assert!(e.contains("Cost"), "{e}");
    // A two-level factorial cannot support squares: they are aliased and reported, not fitted.
    let f = full_factorial(2, &FactorialOptions { replicates: 1, center_points: 3, ..Default::default() }).unwrap();
    let ys: Vec<f64> =
        f.coded.iter().map(|r| 5.0 + r[0] + 2.0 * r[1] + 0.5 * r[0] * r[1] + 0.1 * (r[0] * 7.0).sin()).collect();
    let q = fit_responses(&f, &[("Y".into(), ys)], ModelOrder::Quadratic).unwrap();
    assert!(!q[0].dropped.is_empty());
}

#[test]
fn goals_parse_with_defaults_and_errors() {
    assert_eq!(parse_goal("max 70 90").unwrap(), Goal::Maximize { lower: 70.0, target: 90.0, shape: 1.0 });
    assert_eq!(parse_goal("Maximize 70 90 2").unwrap(), Goal::Maximize { lower: 70.0, target: 90.0, shape: 2.0 });
    assert_eq!(parse_goal("min 5 20 0.5").unwrap(), Goal::Minimize { target: 5.0, upper: 20.0, shape: 0.5 });
    assert_eq!(
        parse_goal("target 40 50 60").unwrap(),
        Goal::Target { lower: 40.0, target: 50.0, upper: 60.0, shape_low: 1.0, shape_high: 1.0 }
    );
    assert_eq!(
        parse_goal("target 40 50 60 2").unwrap(),
        Goal::Target { lower: 40.0, target: 50.0, upper: 60.0, shape_low: 2.0, shape_high: 2.0 }
    );
    assert_eq!(
        parse_goal("target 40 50 60 2 3").unwrap(),
        Goal::Target { lower: 40.0, target: 50.0, upper: 60.0, shape_low: 2.0, shape_high: 3.0 }
    );
    for bad in ["", "max 70", "max 70 90 1 1", "min x 3", "target 1 2", "wiggle 1 2", "m 1 2"] {
        assert!(parse_goal(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn composite_desirability_optimum_matches_a_brute_force_grid() {
    let (d, names, cols) = ccd2();
    let r = design_from_columns(&names, &cols).unwrap();
    let fits = fit_responses(&r, &responses(&d), ModelOrder::Quadratic).unwrap();
    let goals = vec![
        ResponseGoal {
            name: "Yield".into(),
            model: fits[0].model.clone(),
            goal: parse_goal("max 40 55 1").unwrap(),
            importance: 2.0,
        },
        ResponseGoal {
            name: "Cost".into(),
            model: fits[1].model.clone(),
            goal: parse_goal("min 12 30 1").unwrap(),
            importance: 1.0,
        },
    ];
    let res = optimize_responses(&goals, &r.factors, &OptimizeOptions::default()).unwrap();
    let best = res.best();
    // Brute force over the factor box in coded units.
    let mut grid_best: f64 = 0.0;
    let steps = 400;
    for i in 0..=steps {
        for j in 0..=steps {
            let x = [-1.0 + 2.0 * i as f64 / steps as f64, -1.0 + 2.0 * j as f64 / steps as f64];
            grid_best = grid_best.max(composite_desirability(&goals, &r.factors, &x).composite);
        }
    }
    assert!(best.composite >= grid_best - 1e-6, "{} vs grid {}", best.composite, grid_best);
    assert!(best.composite <= grid_best + 1e-3);
    // Composite D is the importance-weighted geometric mean of the individual d's.
    let (d1, d2) = (best.d[0], best.d[1]);
    assert!(((d1.powf(2.0) * d2).powf(1.0 / 3.0) - best.composite).abs() < 1e-12);
    // Each reported d equals the goal's own desirability at the predicted value.
    for (i, g) in goals.iter().enumerate() {
        assert!((g.goal.desirability(best.fit[i]) - best.d[i]).abs() < 1e-12);
    }
    // Natural settings lie inside the experimental region and map back to the coded ones.
    for j in 0..2 {
        assert!((r.factors[j].to_coded(best.natural[j]) - best.coded[j]).abs() < 1e-12);
    }
    let text = res.to_report("opt").render_text();
    assert!(text.contains("Weight") && text.contains("Importance") && text.contains("composite desirability"));
}

#[test]
fn weights_and_importance_change_the_optimum() {
    let (d, names, cols) = ccd2();
    let r = design_from_columns(&names, &cols).unwrap();
    let fits = fit_responses(&r, &responses(&d), ModelOrder::Quadratic).unwrap();
    let run = |imp_yield: f64, imp_cost: f64| {
        let goals = vec![
            ResponseGoal {
                name: "Yield".into(),
                model: fits[0].model.clone(),
                goal: Goal::Maximize { lower: 30.0, target: 60.0, shape: 1.0 },
                importance: imp_yield,
            },
            ResponseGoal {
                name: "Cost".into(),
                model: fits[1].model.clone(),
                goal: Goal::Minimize { target: 10.0, upper: 40.0, shape: 1.0 },
                importance: imp_cost,
            },
        ];
        optimize_responses(&goals, &r.factors, &OptimizeOptions::default()).unwrap().solutions[0].fit.clone()
    };
    let favour_yield = run(10.0, 0.1);
    let favour_cost = run(0.1, 10.0);
    assert!(favour_yield[0] > favour_cost[0] + 1.0, "yield {} vs {}", favour_yield[0], favour_cost[0]);
    assert!(favour_cost[1] < favour_yield[1] - 1.0, "cost {} vs {}", favour_cost[1], favour_yield[1]);
}
