//! Analysis of two-level factorial and Plackett–Burman experiments: effects,
//! coefficients, ANOVA by term order, curvature, Lenth's pseudo standard error.

use super::design::{Design, PointType};
use super::factorial::AliasStructure;
use super::model::{block_columns, fit_cols, term_values, Col, PolyTerm, PolynomialModel};
use crate::error::{invalid, Result};
use crate::regression::RegressionFit;
use openstats_math::dist::{ContinuousDistribution, FisherF, StudentT};

/// Options of a factorial analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialAnalysisOptions {
    /// Highest interaction order in the model (default: min(k, 3)).
    pub max_order: Option<usize>,
    /// Confidence level (0.95).
    pub level: f64,
}

impl Default for FactorialAnalysisOptions {
    fn default() -> Self {
        FactorialAnalysisOptions { max_order: None, level: 0.95 }
    }
}

/// One estimated term.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectRow {
    /// Term label ("A", "A*B", "Ct Pt", "Constant", "Block 1").
    pub term: String,
    /// Effect = 2 × coefficient for two-level terms (`None` for the constant, blocks and curvature).
    pub effect: Option<f64>,
    /// Coefficient.
    pub coef: f64,
    /// Standard error of the coefficient.
    pub se_coef: Option<f64>,
    /// t statistic.
    pub t: Option<f64>,
    /// p-value.
    pub p: Option<f64>,
    /// Effects confounded with this term (fractional factorials, letters notation).
    pub aliases: Vec<String>,
}

/// One row of the ANOVA table.
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaGroup {
    /// Source.
    pub source: String,
    /// Degrees of freedom.
    pub df: usize,
    /// Sequential sum of squares.
    pub seq_ss: Option<f64>,
    /// Adjusted sum of squares.
    pub adj_ss: f64,
    /// Adjusted mean square.
    pub adj_ms: Option<f64>,
    /// F statistic.
    pub f: Option<f64>,
    /// p-value.
    pub p: Option<f64>,
}

/// Lenth's method for unreplicated designs.
#[derive(Debug, Clone, PartialEq)]
pub struct Lenth {
    /// Pseudo standard error of an effect.
    pub pse: f64,
    /// Degrees of freedom `m/3`.
    pub df: f64,
    /// Individual margin of error `t_{1−α/2, df}·PSE`.
    pub margin: f64,
    /// Simultaneous margin of error `t_{γ, df}·PSE`, `γ = (1 + (1−α)^{1/m})/2`.
    pub simultaneous_margin: f64,
    /// `effect/PSE` per term.
    pub t_pse: Vec<(String, f64)>,
}

/// Result of a factorial analysis.
#[derive(Debug, Clone)]
pub struct FactorialAnalysis {
    /// The design analysed (in the order of the response).
    pub design: Design,
    /// Response values.
    pub y: Vec<f64>,
    /// Confidence level.
    pub level: f64,
    /// The constant.
    pub constant: EffectRow,
    /// Block terms.
    pub blocks: Vec<EffectRow>,
    /// Factor effects and interactions, by order.
    pub effects: Vec<EffectRow>,
    /// Curvature (centre points versus corner points).
    pub curvature: Option<EffectRow>,
    /// Terms dropped because they are confounded with earlier ones.
    pub dropped: Vec<String>,
    /// ANOVA table: Blocks, Model, each order, Curvature, Error, Lack-of-Fit, Pure Error, Total.
    pub anova: Vec<AnovaGroup>,
    /// Residual standard deviation (`None` without error df).
    pub s: Option<f64>,
    /// R².
    pub r2: f64,
    /// Adjusted R² (`None` without error df).
    pub r2_adj: Option<f64>,
    /// Predicted R² (`None` without error df).
    pub r2_pred: Option<f64>,
    /// Error degrees of freedom.
    pub df_error: usize,
    /// Lenth's analysis (when there is no error df).
    pub lenth: Option<Lenth>,
    /// Alias structure (fractional factorials).
    pub alias: Option<AliasStructure>,
    /// The regression behind the numbers (`None` when saturated).
    pub fit: Option<RegressionFit>,
    /// Prediction model (polynomial terms only).
    pub model: PolynomialModel,
}

fn subsets(k: usize, max_order: usize) -> Vec<Vec<usize>> {
    let mut v: Vec<Vec<usize>> = (1u32..(1 << k))
        .filter(|m| (m.count_ones() as usize) <= max_order)
        .map(|m| (0..k).filter(|i| m >> i & 1 == 1).collect())
        .collect();
    v.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    v
}

fn group_name(order: usize) -> &'static str {
    match order {
        1 => "Main Effects",
        2 => "2-Way Interactions",
        3 => "3-Way Interactions",
        4 => "4-Way Interactions",
        _ => "Higher-Order Interactions",
    }
}

pub(crate) fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Lenth's PSE of a set of effects.
pub fn lenth_pse(effects: &[f64]) -> Option<f64> {
    if effects.is_empty() {
        return None;
    }
    let mut abs: Vec<f64> = effects.iter().map(|e| e.abs()).collect();
    let s0 = 1.5 * median(&mut abs.clone());
    abs.retain(|&a| a < 2.5 * s0);
    if abs.is_empty() {
        return None;
    }
    Some(1.5 * median(&mut abs))
}

/// Analyse a two-level (corner + optional centre point) design.
pub fn analyze_factorial(design: &Design, y: &[f64], opts: &FactorialAnalysisOptions) -> Result<FactorialAnalysis> {
    let k = design.n_factors();
    if y.len() != design.n_runs() {
        return invalid(format!("the response has {} values for {} runs", y.len(), design.n_runs()));
    }
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
    }
    if design.point_type.iter().any(|t| matches!(t, PointType::Axial | PointType::Edge)) {
        return invalid("factorial analysis is for corner and centre points; use the response-surface analysis");
    }
    let max_order = opts.max_order.unwrap_or(k.min(3));
    if max_order == 0 || max_order > k {
        return invalid(format!("the model order must be 1..={k}"));
    }
    let names: Vec<String> = design.factors.iter().map(|f| f.name.clone()).collect();
    let mut cols: Vec<Col> = block_columns(design);
    for s in subsets(k, max_order) {
        let t = PolyTerm { powers: s.iter().map(|&i| (i, 1)).collect() };
        cols.push(Col {
            name: t.label(&names),
            values: term_values(design, &t),
            term: Some(t),
            group: group_name(s.len()),
        });
    }
    let has_center = design.point_type.contains(&PointType::Center);
    let has_corner = design.point_type.contains(&PointType::Factorial);
    if has_center && has_corner {
        cols.push(Col {
            name: "Ct Pt".into(),
            values: design.point_type.iter().map(|&t| f64::from(u8::from(t == PointType::Center))).collect(),
            term: None,
            group: "Curvature",
        });
    }
    let fc = fit_cols(cols.clone(), y, opts.level)?;
    let alias = match &design.defining {
        Some(_) => Some(design.alias_structure(max_order.min(3))?),
        None => None,
    };
    let aliases_of = |label: &str| -> Vec<String> {
        let Some(a) = &alias else { return Vec::new() };
        // Convert the label to letters for lookup.
        let parts: Vec<&str> = label.split('*').collect();
        let mut mask = 0u32;
        for p in parts {
            match names.iter().position(|n| n == p) {
                Some(i) => mask |= 1 << i,
                None => return Vec::new(),
            }
        }
        let letters = super::design::word_label(mask);
        a.classes
            .iter()
            .find(|c| c.contains(&letters))
            .map(|c| c.iter().filter(|w| **w != letters).cloned().collect())
            .unwrap_or_default()
    };
    // Rows.
    let se_of = |i: usize| -> (Option<f64>, Option<f64>, Option<f64>) {
        match &fc.fit {
            Some(f) => (Some(f.coefficients[i].std_err), Some(f.coefficients[i].t), Some(f.coefficients[i].p_value)),
            None => (None, None, None),
        }
    };
    let mut constant = None;
    let mut blocks = Vec::new();
    let mut effects = Vec::new();
    let mut curvature = None;
    for (i, name) in fc.names.iter().enumerate() {
        let (se, t, p) = se_of(i);
        let coef = fc.coef[i];
        let group = fc.groups[i];
        let is_effect = !matches!(group, "Constant" | "Blocks" | "Curvature");
        let row = EffectRow {
            term: name.clone(),
            effect: is_effect.then_some(2.0 * coef),
            coef,
            se_coef: se,
            t,
            p,
            aliases: if is_effect { aliases_of(name) } else { Vec::new() },
        };
        match group {
            "Constant" => constant = Some(row),
            "Blocks" => blocks.push(row),
            "Curvature" => curvature = Some(row),
            _ => effects.push(row),
        }
    }
    let constant = constant.ok_or_else(|| crate::error::StatsError::InvalidArgument("no constant".into()))?;
    // ANOVA.
    let n = y.len();
    let nf = n as f64;
    let ybar = y.iter().sum::<f64>() / nf;
    let sst: f64 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum();
    let sse = fc.sse;
    let df_error = fc.df_error;
    let mse = (df_error > 0).then(|| sse / df_error as f64);
    let mut order: Vec<&'static str> = Vec::new();
    for g in &fc.groups[1..] {
        if !order.contains(g) {
            order.push(g);
        }
    }
    let count = |g: &str| fc.groups[1..].iter().filter(|x| **x == g).count();
    let fit_without = |skip: &dyn Fn(&str) -> bool| -> Result<f64> {
        let kept: Vec<Col> = fc.cols.iter().filter(|c| !skip(c.group)).cloned().collect();
        Ok(fit_cols(kept, y, opts.level)?.sse)
    };
    let mut anova: Vec<AnovaGroup> = Vec::new();
    let mk = |source: &str, df: usize, seq: Option<f64>, adj: f64| -> AnovaGroup {
        let ms = adj / df as f64;
        let f = mse.map(|m| ms / m);
        let p = f.and_then(|f| FisherF::new(df as f64, df_error as f64).ok().map(|d| d.sf(f)));
        AnovaGroup { source: source.into(), df, seq_ss: seq, adj_ss: adj, adj_ms: Some(ms), f, p }
    };
    let model_df = fc.groups.len() - 1;
    anova.push(mk("Model", model_df, Some(sst - sse), sst - sse));
    let mut prev_sse = {
        let sse0 = y.iter().map(|v| (v - ybar) * (v - ybar)).sum::<f64>();
        sse0
    };
    for g in &order {
        let adj = fit_without(&|x| x == *g)? - sse;
        // Sequential: columns in model order up to and including this group.
        let upto: Vec<Col> = {
            let pos = order.iter().position(|x| x == g).unwrap_or(0);
            fc.cols.iter().filter(|c| order[..=pos].contains(&c.group)).cloned().collect()
        };
        let sse_here = fit_cols(upto, y, opts.level)?.sse;
        let seq = prev_sse - sse_here;
        prev_sse = sse_here;
        anova.push(mk(g, count(g), Some(seq), adj.max(0.0)));
    }
    if df_error > 0 {
        let ms = sse / df_error as f64;
        anova.push(AnovaGroup {
            source: "Error".into(),
            df: df_error,
            seq_ss: Some(sse),
            adj_ss: sse,
            adj_ms: Some(ms),
            f: None,
            p: None,
        });
        if let Some(l) = fc.fit.as_ref().and_then(|f| f.lack_of_fit.as_ref()) {
            let lms = l.ss_lof / l.df_lof as f64;
            anova.push(AnovaGroup {
                source: "Lack-of-Fit".into(),
                df: l.df_lof,
                seq_ss: Some(l.ss_lof),
                adj_ss: l.ss_lof,
                adj_ms: Some(lms),
                f: Some(l.f),
                p: Some(l.p_value),
            });
            anova.push(AnovaGroup {
                source: "Pure Error".into(),
                df: l.df_pe,
                seq_ss: Some(l.ss_pe),
                adj_ss: l.ss_pe,
                adj_ms: Some(l.ss_pe / l.df_pe as f64),
                f: None,
                p: None,
            });
        }
    }
    anova.push(AnovaGroup {
        source: "Total".into(),
        df: n - 1,
        seq_ss: Some(sst),
        adj_ss: sst,
        adj_ms: None,
        f: None,
        p: None,
    });
    // Lenth.
    let lenth = if df_error == 0 {
        let eff: Vec<(String, f64)> = effects.iter().filter_map(|e| e.effect.map(|v| (e.term.clone(), v))).collect();
        let vals: Vec<f64> = eff.iter().map(|e| e.1).collect();
        match lenth_pse(&vals) {
            Some(pse) if pse > 0.0 => {
                let m = vals.len() as f64;
                let d = m / 3.0;
                let td = StudentT::new(d)?;
                let a = 1.0 - opts.level;
                let gamma = (1.0 + (1.0 - a).powf(1.0 / m)) / 2.0;
                Some(Lenth {
                    pse,
                    df: d,
                    margin: td.quantile(1.0 - a / 2.0) * pse,
                    simultaneous_margin: td.quantile(gamma) * pse,
                    t_pse: eff.into_iter().map(|(t, v)| (t, v / pse)).collect(),
                })
            }
            _ => None,
        }
    } else {
        None
    };
    let model = fc.model(&names);
    let (r2_adj, r2_pred, s) = match &fc.fit {
        Some(f) => (Some(f.r2_adj), Some(f.r2_pred), Some(f.s)),
        None => (None, None, None),
    };
    Ok(FactorialAnalysis {
        design: design.clone(),
        y: y.to_vec(),
        level: opts.level,
        constant,
        blocks,
        effects,
        curvature,
        dropped: fc.dropped.clone(),
        anova,
        s,
        r2: if sst > 0.0 { 1.0 - sse / sst } else { 1.0 },
        r2_adj,
        r2_pred,
        df_error,
        lenth,
        alias,
        fit: fc.fit,
        model,
    })
}
