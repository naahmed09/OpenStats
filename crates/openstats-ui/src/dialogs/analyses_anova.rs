//! ANOVA with random factors and mixed models: Balanced ANOVA, random factors in the General Linear Model,
//! Fully Nested ANOVA, Mixed Effects Model, Repeated Measures ANOVA, Analysis of Covariance, and the main
//! effects and interactions plots.

use super::analyses::{add, out};
use super::analyses_adv::ident;
use super::analysis::{col_title, expand, labels, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::Worksheet;
use openstats_stats::anova::{balanced_anova, glm, glm_random, nested_anova, GlmOptions, RandomOptions};
use openstats_stats::mixed::{mixed_model, MixedMethod, MixedOptions, RandomSpec};
use openstats_stats::model::Data;
use openstats_stats::regression::RegressionOptions;
use openstats_stats::report::{
    Axis, Cell, ChartKind, ChartSpec, Column as RCol, ColumnKind, ReferenceLine, Reportable, Section, Series, Table,
};
use std::collections::BTreeSet;

// ------------------------------------------------------------------------------------ helpers

/// Worksheet columns exposed to a model under formula-safe names.
pub(crate) struct Prep {
    pub(crate) data: Data,
    y: String,
    y_title: String,
    /// `(column, identifier, title)` of every factor and covariate.
    atoms: Vec<(usize, String, String)>,
    factor_cols: Vec<usize>,
}

pub(crate) fn prepare(ws: &Worksheet, y: usize, factors: &[usize], covs: &[usize]) -> Res<Prep> {
    if factors.iter().chain(covs).any(|&c| c == y) {
        return Err("the response cannot also be a model term".into());
    }
    if factors.iter().any(|f| covs.contains(f)) {
        return Err("a column cannot be both a factor and a covariate".into());
    }
    let mut taken: Vec<String> = Vec::new();
    let yn = ident(ws, y, &taken);
    taken.push(yn.clone());
    let mut data = Data::new().with_numeric(&yn, numeric(ws, y)?)?;
    let mut atoms = Vec::new();
    for &c in factors {
        let n = ident(ws, c, &taken);
        taken.push(n.clone());
        data = data.with_factor(&n, &labels(ws, c)?)?;
        atoms.push((c, n, col_title(ws, c)));
    }
    for &c in covs {
        let n = ident(ws, c, &taken);
        taken.push(n.clone());
        data = data.with_numeric(&n, numeric(ws, c)?)?;
        atoms.push((c, n, col_title(ws, c)));
    }
    Ok(Prep { data, y: yn, y_title: col_title(ws, y), atoms, factor_cols: factors.to_vec() })
}

impl Prep {
    fn ident_of(&self, col: usize) -> &str {
        self.atoms.iter().find(|a| a.0 == col).map_or("", |a| a.1.as_str())
    }

    fn title_of_ident(&self, id: &str) -> String {
        self.atoms.iter().find(|a| a.1 == id).map_or_else(|| id.to_string(), |a| a.2.clone())
    }

    fn is_factor(&self, id: &str) -> bool {
        self.atoms.iter().any(|a| a.1 == id && self.factor_cols.contains(&a.0))
    }

    /// Replace column titles by their identifiers in a model text such as `Machine Operator Machine*Operator`.
    pub(crate) fn translate(&self, text: &str) -> Res<String> {
        let mut order: Vec<usize> = (0..self.atoms.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(self.atoms[k].2.len()));
        let mut s = text.to_string();
        for &k in &order {
            if !self.atoms[k].2.is_empty() {
                s = s.replace(&self.atoms[k].2, &format!("\u{1}{k}\u{2}"));
            }
        }
        for (k, a) in self.atoms.iter().enumerate() {
            s = s.replace(&format!("\u{1}{k}\u{2}"), &a.1);
        }
        let known: BTreeSet<&str> = self.atoms.iter().map(|a| a.1.as_str()).collect();
        for tok in s.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            if tok.chars().next().is_some_and(char::is_alphabetic) && !known.contains(tok) {
                return Err(format!("'{tok}' in the model is not one of the selected factors or covariates").into());
            }
        }
        Ok(s)
    }

    /// The default right-hand side: all terms up to interaction `order`.
    pub(crate) fn default_rhs(&self, order: usize) -> String {
        let names: Vec<&str> = self.atoms.iter().map(|a| a.1.as_str()).collect();
        if order > 1 && names.len() > 1 {
            format!("({})^{}", names.join(" + "), order.min(names.len()))
        } else {
            names.join(" + ")
        }
    }

    /// Random terms from single listed columns and a `;`-separated text such as `Subject; Subject*Block`.
    fn random_specs(&self, listed: &[usize], text: &str) -> Res<Vec<RandomSpec>> {
        let mut out: Vec<RandomSpec> = Vec::new();
        for &c in listed {
            let id = self.ident_of(c);
            if !self.is_factor(id) {
                return Err(format!("random factor {} must also be selected as a factor", col_title_of(self, c)).into());
            }
            out.push(RandomSpec { label: self.title_of_ident(id), factors: vec![id.to_string()] });
        }
        for term in text.split(';').map(str::trim).filter(|t| !t.is_empty()) {
            let mut ids = Vec::new();
            let mut titles = Vec::new();
            for piece in term.split('*').map(str::trim) {
                let a = self
                    .atoms
                    .iter()
                    .find(|a| a.2 == piece && self.factor_cols.contains(&a.0))
                    .ok_or_else(|| format!("'{piece}' in the random terms is not one of the selected factors"))?;
                ids.push(a.1.clone());
                titles.push(a.2.clone());
            }
            let spec = RandomSpec { label: titles.join("*"), factors: ids };
            if !out.contains(&spec) {
                out.push(spec);
            }
        }
        Ok(out)
    }
}

fn col_title_of(p: &Prep, c: usize) -> String {
    p.atoms.iter().find(|a| a.0 == c).map_or_else(|| format!("C{}", c + 1), |a| a.2.clone())
}

/// Replace identifiers by titles in the labels of a section's text (term labels are identifiers inside the library).
pub(crate) fn retitle(sec: &mut Section, p: &Prep) {
    use openstats_stats::report::Block;
    let mut subs: Vec<(String, String)> = p.atoms.iter().map(|a| (a.1.clone(), a.2.clone())).collect();
    subs.push((p.y.clone(), p.y_title.clone()));
    subs.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
    let fix = |t: &str| -> String {
        let mut out = String::new();
        let mut word = String::new();
        let flush = |w: &mut String, out: &mut String| {
            if !w.is_empty() {
                match subs.iter().find(|s| s.0 == *w) {
                    Some(s) => out.push_str(&s.1),
                    None => out.push_str(w),
                }
                w.clear();
            }
        };
        for ch in t.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                word.push(ch);
            } else {
                flush(&mut word, &mut out);
                out.push(ch);
            }
        }
        flush(&mut word, &mut out);
        out
    };
    for b in &mut sec.blocks {
        match b {
            Block::Text(t) => *t = fix(t),
            Block::Table(tb) => {
                for row in &mut tb.rows {
                    for c in row.iter_mut() {
                        if let Cell::Text(s) = c {
                            *s = fix(s);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// --------------------------------------------------------------------- Balanced ANOVA / GLM random

fn random_core(ws: &Worksheet, v: &Values, balanced: bool) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    if factors.is_empty() {
        return Err("select at least one factor".into());
    }
    let random_cols = v.vars("random").to_vec();
    if random_cols.is_empty() {
        return Err("select at least one random factor (without one, use General Linear Model or One-Way ANOVA)".into());
    }
    let prep = prepare(ws, y, &factors, &covs)?;
    for r in &random_cols {
        if !factors.contains(r) {
            return Err(format!("random factor {} must also be listed under Factors", col_title(ws, *r)).into());
        }
    }
    let model_text = v.text("model").trim().to_string();
    let rhs =
        if model_text.is_empty() { prep.default_rhs(v.count("order").max(1)) } else { prep.translate(&model_text)? };
    let formula = format!("{} ~ {rhs}", prep.y);
    let random: Vec<String> = random_cols.iter().map(|&c| prep.ident_of(c).to_string()).collect();
    let o = RandomOptions { restricted: v.flag("restricted"), ..RandomOptions::default() };
    let r = if balanced {
        balanced_anova(&formula, &random, &prep.data, &o)?
    } else {
        glm_random(&formula, &random, &prep.data, &o)?
    };
    let mut s = r.report_section();
    retitle(&mut s, &prep);
    let rand_titles: Vec<String> = random_cols.iter().map(|&c| col_title(ws, c)).collect();
    s.heading = format!(
        "{}: {} versus {} (random: {})",
        if balanced { "Balanced ANOVA" } else { "General Linear Model (random factors)" },
        col_title(ws, y),
        factors.iter().chain(&covs).map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", "),
        rand_titles.join(", ")
    );
    out(s)
}

fn balanced_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    random_core(ws, v, true)
}

/// General Linear Model with random factors (called from the GLM dialog).
pub(crate) fn glm_random_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    random_core(ws, v, false)
}

// ------------------------------------------------------------------------------------ nested

fn nested_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    if factors.is_empty() {
        return Err("select the factors from the highest level to the lowest".into());
    }
    if factors.contains(&y) {
        return Err("the response cannot also be a factor".into());
    }
    let ys = numeric(ws, y)?;
    let cols: Vec<Vec<String>> = factors.iter().map(|&c| labels(ws, c)).collect::<Res<_>>()?;
    let titles: Vec<String> = factors.iter().map(|&c| col_title(ws, c)).collect();
    // A lower factor's labels may repeat across parents (1, 2, 3 in every group): make them unique by path.
    let paths: Vec<Vec<String>> = (0..factors.len())
        .map(|k| {
            (0..ys.len()).map(|i| (0..=k).map(|q| cols[q][i].as_str()).collect::<Vec<_>>().join("\u{1f}")).collect()
        })
        .collect();
    let spec: Vec<(&str, &[String])> = titles.iter().zip(&paths).map(|(t, p)| (t.as_str(), p.as_slice())).collect();
    let r = nested_anova(&ys, &spec)?;
    let mut s = r.report_section();
    s.heading = format!("Nested ANOVA: {} versus {}", col_title(ws, y), titles.join(", "));
    // Show the level labels, not the internal path keys, in the means tables.
    use openstats_stats::report::Block;
    for b in &mut s.blocks {
        if let Block::Table(t) = b {
            for row in &mut t.rows {
                if let Some(Cell::Text(x)) = row.first_mut() {
                    *x = x.replace('\u{1f}', " / ");
                }
            }
        }
    }
    out(s)
}

// ------------------------------------------------------------------------------------- mixed

fn mixed_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    let prep = prepare(ws, y, &factors, &covs)?;
    let specs = prep.random_specs(v.vars("random"), v.text("random_terms"))?;
    if specs.is_empty() {
        return Err("select a random factor or type a random term".into());
    }
    let text = v.text("model").trim().to_string();
    let rhs = if text.is_empty() {
        let fixed: Vec<&str> =
            prep.atoms.iter().filter(|a| !v.vars("random").contains(&a.0)).map(|a| a.1.as_str()).collect();
        if fixed.is_empty() {
            "1".to_string()
        } else if v.count("order") > 1 && fixed.len() > 1 {
            format!("({})^{}", fixed.join(" + "), v.count("order").min(fixed.len()))
        } else {
            fixed.join(" + ")
        }
    } else {
        prep.translate(&text)?
    };
    let formula = format!("{} ~ {rhs}", prep.y);
    let o = MixedOptions {
        method: if v.choice("method") == 1 { MixedMethod::Ml } else { MixedMethod::Reml },
        level: level(v)?,
        ..MixedOptions::default()
    };
    let fit = mixed_model(&formula, &specs, &prep.data, &o)?;
    let mut s = fit.report_section();
    retitle(&mut s, &prep);
    s.heading = format!("Mixed Effects Model: {}", prep.y_title);
    // Random effect predictions.
    if v.flag("show_blups") {
        for b in &fit.blups {
            let mut t = Table::new(
                format!("Predicted random effects: {}", b.term),
                vec![RCol::new("Level", ColumnKind::Text), RCol::new("Effect", ColumnKind::Significant(6))],
            );
            for (l, e) in &b.effects {
                t.push_row(vec![Cell::text(l), Cell::Num(*e)])?;
            }
            s.table(t);
        }
    }
    if v.flag("resid_plot") {
        s.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: "Residuals Versus Fits".into(),
            x_label: "Fitted value".into(),
            y_label: "Residual".into(),
            categories: Vec::new(),
            series: vec![Series { name: "Residuals".into(), x: fit.fitted.clone(), y: fit.residuals.clone() }],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: 0.0, label: "0".into() }],
        });
    }
    let n = ws.column(y)?.len();
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        stored.push(expand("FITS", n, &fit.rows_used, &fit.fitted));
    }
    if v.flag("store_marginal") {
        stored.push(expand("MFITS", n, &fit.rows_used, &fit.fitted_marginal));
    }
    if v.flag("store_resid") {
        stored.push(expand("RESI", n, &fit.rows_used, &fit.residuals));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ------------------------------------------------------------------------- repeated measures

fn repeated_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let subject = v.var("subject").ok_or("select the subject column")?;
    let between = v.vars("between").to_vec();
    let within = v.vars("within").to_vec();
    if within.is_empty() {
        return Err("select at least one within-subject factor".into());
    }
    if between.iter().chain(&within).any(|&c| c == subject) {
        return Err("the subject column cannot also be a factor".into());
    }
    if between.iter().any(|b| within.contains(b)) {
        return Err("a factor cannot be both between and within subjects".into());
    }
    let mut factors = vec![subject];
    factors.extend(&between);
    factors.extend(&within);
    let prep = prepare(ws, y, &factors, &[])?;
    let subj = prep.ident_of(subject).to_string();
    let names: Vec<&str> = between.iter().chain(&within).map(|&c| prep.ident_of(c)).collect();
    let order = v.count("order").max(1).min(names.len());
    // Subjects are nested in the between-subject cells (their labels may be unique or repeat across groups).
    let subj_term = if between.is_empty() {
        subj.clone()
    } else {
        format!("{subj}({})", between.iter().map(|&c| prep.ident_of(c)).collect::<Vec<_>>().join(" "))
    };
    let rhs = if order > 1 && names.len() > 1 {
        format!("({})^{order} + {subj_term}", names.join(" + "))
    } else {
        format!("{} + {subj_term}", names.join(" + "))
    };
    let formula = format!("{} ~ {rhs}", prep.y);
    let o = RandomOptions { restricted: v.flag("restricted"), ..RandomOptions::default() };
    let r = glm_random(&formula, std::slice::from_ref(&subj), &prep.data, &o)?;
    // Subjects must be measured under every within-subject condition: warn when the design is unbalanced.
    let mut s = r.report_section();
    retitle(&mut s, &prep);
    s.heading = format!(
        "Repeated Measures ANOVA: {} (subject: {}; between: {}; within: {})",
        col_title(ws, y),
        col_title(ws, subject),
        if between.is_empty() {
            "none".to_string()
        } else {
            between.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
        },
        within.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    );
    s.text(
        "Subjects are a random factor; within-subject effects are tested against the subject by condition residual, \
         between-subject effects against the subject mean square. Sphericity is assumed and not tested.",
    );
    if !r.balanced {
        s.warn("The design is not balanced (a subject is missing a condition, or subjects differ in number); F tests use Satterthwaite-synthesized denominators.");
    }
    out(s)
}

// ------------------------------------------------------------------------------------ ANCOVA

fn ancova_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    if factors.is_empty() || covs.is_empty() {
        return Err("select at least one factor and at least one covariate".into());
    }
    let prep = prepare(ws, y, &factors, &covs)?;
    let lvl = level(v)?;
    let o = GlmOptions { regression: RegressionOptions { level: lvl, ..Default::default() }, ..GlmOptions::default() };
    let fids: Vec<String> = factors.iter().map(|&c| prep.ident_of(c).to_string()).collect();
    let cids: Vec<String> = covs.iter().map(|&c| prep.ident_of(c).to_string()).collect();
    let main = format!("{} ~ {}", prep.y, fids.iter().chain(&cids).cloned().collect::<Vec<_>>().join(" + "));
    let fit = glm(&main, &prep.data, &o)?;
    let mut s = fit.report_section();
    retitle(&mut s, &prep);
    s.heading = format!(
        "Analysis of Covariance: {} versus {} with covariate{} {}",
        col_title(ws, y),
        factors.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", "),
        if covs.len() > 1 { "s" } else { "" },
        covs.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    );
    // Homogeneity of slopes: add each factor-by-covariate interaction in turn.
    if v.flag("slopes") {
        let mut t = Table::new(
            "Test for Equal Slopes (factor*covariate interaction)",
            vec![
                RCol::new("Interaction", ColumnKind::Text),
                RCol::new("DF", ColumnKind::Integer),
                RCol::new("Adj SS", ColumnKind::Significant(6)),
                RCol::new("F-Value", ColumnKind::Significant(6)),
                RCol::new("P-Value", ColumnKind::PValue),
            ],
        );
        for (fi, f) in fids.iter().enumerate() {
            for (ci, c) in cids.iter().enumerate() {
                let full = glm(&format!("{main} + {f}*{c}"), &prep.data, &o)?;
                let ddf = fit.fit.df_error.saturating_sub(full.fit.df_error);
                if ddf == 0 {
                    continue;
                }
                let ss = fit.fit.sse - full.fit.sse;
                let f_val = (ss / ddf as f64) / full.fit.mse;
                let dist = openstats_math::dist::FisherF::new(ddf as f64, full.fit.df_error as f64)
                    .map_err(|e| format!("{e}"))?;
                let p = openstats_math::dist::ContinuousDistribution::sf(&dist, f_val);
                t.push_row(vec![
                    Cell::text(format!("{}*{}", col_title(ws, factors[fi]), col_title(ws, covs[ci]))),
                    Cell::Int(i64::try_from(ddf).unwrap_or(0)),
                    Cell::Num(ss),
                    Cell::Num(f_val),
                    Cell::Num(p),
                ])?;
            }
        }
        s.table(t);
        s.text("A significant interaction means the covariate's slope differs between groups; the common-slope ANCOVA above is then not appropriate.");
    }
    // Plot: first factor against first covariate, parallel fitted lines (other terms at their reference).
    if v.flag("plot") {
        let (f0, c0) = (&fids[0], &cids[0]);
        let fcol = prep.data.factor(f0).unwrap_or(&[]).to_vec();
        let ccol = prep.data.numeric(c0).unwrap_or(&[]).to_vec();
        let ycol = prep.data.numeric(&prep.y).unwrap_or(&[]).to_vec();
        let mut levels: Vec<String> =
            fcol.iter().filter(|s| !s.is_empty()).cloned().collect::<BTreeSet<_>>().into_iter().collect();
        levels.sort_by(|a, b| match (a.trim().parse::<f64>(), b.trim().parse::<f64>()) {
            (Ok(x), Ok(y)) => x.total_cmp(&y),
            _ => a.cmp(b),
        });
        let finite: Vec<f64> = ccol.iter().copied().filter(|x| x.is_finite()).collect();
        if !levels.is_empty() && !finite.is_empty() {
            let (lo, hi) = finite.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
            let mut series = Vec::new();
            for l in &levels {
                let (mut xs, mut ys) = (Vec::new(), Vec::new());
                for i in 0..fcol.len() {
                    if &fcol[i] == l && ccol[i].is_finite() && ycol[i].is_finite() {
                        xs.push(ccol[i]);
                        ys.push(ycol[i]);
                    }
                }
                series.push(Series { name: l.clone(), x: xs, y: ys });
            }
            // Prediction grid: two covariate values per level; other covariates at their means, other factors at their first level.
            let mut gd = Data::new();
            let m = levels.len() * 2;
            for f in &fids {
                let col: Vec<String> = if f == f0 {
                    levels.iter().flat_map(|l| [l.clone(), l.clone()]).collect()
                } else {
                    let first =
                        prep.data.factor(f).and_then(|c| c.iter().find(|s| !s.is_empty())).cloned().unwrap_or_default();
                    vec![first; m]
                };
                gd = gd.with_factor(f, &col)?;
            }
            for c in &cids {
                let col: Vec<f64> = if c == c0 {
                    levels.iter().flat_map(|_| [lo, hi]).collect()
                } else {
                    let vals: Vec<f64> =
                        prep.data.numeric(c).unwrap_or(&[]).iter().copied().filter(|x| x.is_finite()).collect();
                    vec![vals.iter().sum::<f64>() / vals.len().max(1) as f64; m]
                };
                gd = gd.with_numeric(c, col)?;
            }
            gd = gd.with_numeric(&prep.y, vec![0.0; m])?;
            let preds = fit.fit.predict(&gd, lvl)?;
            for (k, l) in levels.iter().enumerate() {
                series.push(Series {
                    name: format!("Fit: {l}"),
                    x: vec![lo, hi],
                    y: vec![preds[2 * k].fit, preds[2 * k + 1].fit],
                });
            }
            s.chart(ChartSpec {
                kind: ChartKind::Scatter,
                title: format!(
                    "{} versus {} by {}",
                    col_title(ws, y),
                    col_title(ws, covs[0]),
                    col_title(ws, factors[0])
                ),
                x_label: col_title(ws, covs[0]),
                y_label: col_title(ws, y),
                categories: Vec::new(),
                series,
                reference_lines: Vec::new(),
            });
        }
    }
    out(s)
}

// ------------------------------------------------------------------------------- effects plots

fn level_order(values: &[String]) -> Vec<String> {
    let mut l: Vec<String> =
        values.iter().filter(|s| !s.is_empty()).cloned().collect::<BTreeSet<_>>().into_iter().collect();
    if l.iter().all(|s| s.trim().parse::<f64>().is_ok()) {
        l.sort_by(|a, b| a.trim().parse::<f64>().unwrap_or(0.0).total_cmp(&b.trim().parse::<f64>().unwrap_or(0.0)));
    }
    l
}

fn cell_mean(y: &[f64], conds: &[(&[String], &str)]) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for (i, &v) in y.iter().enumerate() {
        if v.is_finite() && conds.iter().all(|(c, l)| c[i] == *l) {
            s += v;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

type EffectInputs = (usize, Vec<usize>, Vec<f64>, Vec<Vec<String>>);

fn effect_inputs(ws: &Worksheet, v: &Values, min: usize) -> Res<EffectInputs> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    if factors.len() < min {
        return Err(format!("select at least {min} factor{}", if min > 1 { "s" } else { "" }).into());
    }
    if factors.contains(&y) {
        return Err("the response cannot also be a factor".into());
    }
    let ys = numeric(ws, y)?;
    let cols = factors.iter().map(|&c| labels(ws, c)).collect::<Res<Vec<_>>>()?;
    Ok((y, factors, ys, cols))
}

fn main_effects_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, factors, ys, cols) = effect_inputs(ws, v, 1)?;
    let grand = {
        let f: Vec<f64> = ys.iter().copied().filter(|x| x.is_finite()).collect();
        f.iter().sum::<f64>() / f.len().max(1) as f64
    };
    let mut s = Section::new(format!("Main Effects Plot for {}", col_title(ws, y)));
    s.text("Points are the means of the response at each level of the factor; the line is the grand mean.");
    for (k, &fc) in factors.iter().enumerate() {
        let levels = level_order(&cols[k]);
        let means: Vec<f64> = levels.iter().map(|l| cell_mean(&ys, &[(&cols[k], l)])).collect();
        let mut t = Table::new(
            format!("Means: {}", col_title(ws, fc)),
            vec![RCol::new("Level", ColumnKind::Text), RCol::new("Mean", ColumnKind::Significant(6))],
        );
        for (l, m) in levels.iter().zip(&means) {
            t.push_row(vec![Cell::text(l), Cell::Num(*m)])?;
        }
        s.table(t);
        s.chart(ChartSpec {
            kind: ChartKind::Line,
            title: format!("Main Effects Plot for {}", col_title(ws, y)),
            x_label: col_title(ws, fc),
            y_label: format!("Mean of {}", col_title(ws, y)),
            categories: levels.clone(),
            series: vec![Series { name: "Mean".into(), x: (1..=levels.len()).map(|i| i as f64).collect(), y: means }],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: grand, label: "Grand mean".into() }],
        });
    }
    out(s)
}

fn interaction_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, factors, ys, cols) = effect_inputs(ws, v, 2)?;
    let mut s = Section::new(format!("Interaction Plot for {}", col_title(ws, y)));
    s.text("Each line joins the means of the response across the levels of one factor for a fixed level of another; non-parallel lines suggest an interaction.");
    for a in 0..factors.len() {
        for b in (a + 1)..factors.len() {
            let la = level_order(&cols[a]);
            let lb = level_order(&cols[b]);
            let series: Vec<Series> = lb
                .iter()
                .map(|l2| Series {
                    name: format!("{} = {l2}", col_title(ws, factors[b])),
                    x: (1..=la.len()).map(|i| i as f64).collect(),
                    y: la.iter().map(|l1| cell_mean(&ys, &[(&cols[a], l1), (&cols[b], l2)])).collect(),
                })
                .collect();
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: format!("Interaction Plot for {}", col_title(ws, y)),
                x_label: col_title(ws, factors[a]),
                y_label: format!("Mean of {}", col_title(ws, y)),
                categories: la,
                series,
                reference_lines: Vec::new(),
            });
        }
    }
    out(s)
}

// -------------------------------------------------------------------------------- registration

pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options, Storage};
    use VarKind::{Any, Numeric};
    add(
        r,
        DialogSpec::new(
            "stat.anova.balanced",
            "Balanced ANOVA",
            "Analysis of variance of a balanced design with fixed and random factors: expected mean squares, F tests against the right error terms and variance components.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors", Any, 1, 20)
        .vars("random", "Random factors (a subset of the factors)", Any, 1, 20)
        .vars("covariates", "Covariates", Numeric, 0, 10)
        .text(Main, "model", "Model terms (blank = all interactions up to the order below)", "")
        .integer(Options, "order", "Interactions up to order (when no model is typed)", 2.0, 1.0, 6.0)
        .check(Options, "restricted", "Use the restricted form of the mixed model", true),
        &["Stat", "ANOVA", "Balanced ANOVA..."],
        Runner::Report(balanced_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.anova.nested",
            "Fully Nested ANOVA",
            "Variance components of a hierarchical design: every factor is nested in the one before it and all are random.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors (highest level first)", Any, 1, 8),
        &["Stat", "ANOVA", "Fully Nested ANOVA..."],
        Runner::Report(nested_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.anova.mixed",
            "Mixed Effects Model",
            "Linear mixed model with random intercepts (crossed or nested) fitted by REML or ML, with Satterthwaite tests of the fixed effects.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors", Any, 0, 20)
        .vars("covariates", "Covariates", Numeric, 0, 20)
        .vars("random", "Random factors", Any, 0, 20)
        .text(Main, "random_terms", "Other random terms (separate with ;  use * for combinations, e.g. Subject*Day)", "")
        .text(Main, "model", "Fixed terms (blank = main effects of the non-random columns)", "")
        .integer(Options, "order", "Fixed interactions up to order (when no model is typed)", 1.0, 1.0, 6.0)
        .choice(Options, "method", "Method", &["REML", "Maximum likelihood"], 0)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Graphs, "resid_plot", "Residuals versus fits", false)
        .check(Options, "show_blups", "Show predicted random effects", false)
        .check(Storage, "store_fits", "Fits (including random effects)", false)
        .check(Storage, "store_marginal", "Fits (fixed effects only)", false)
        .check(Storage, "store_resid", "Residuals", false),
        &["Stat", "ANOVA", "Mixed Effects Model..."],
        Runner::Report(mixed_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.anova.repeated",
            "Repeated Measures ANOVA",
            "Subjects measured under several conditions: the subject is a random factor, within-subject effects are tested against the subject by condition residual.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("subject", "Subject", Any, 1, 1)
        .vars("between", "Between-subject factors", Any, 0, 5)
        .vars("within", "Within-subject factors", Any, 1, 5)
        .integer(Options, "order", "Interactions up to order", 2.0, 1.0, 6.0)
        .check(Options, "restricted", "Use the restricted form of the mixed model", true),
        &["Stat", "ANOVA", "Repeated Measures ANOVA..."],
        Runner::Report(repeated_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.anova.ancova",
            "Analysis of Covariance",
            "Compare group means after adjusting for covariates; test whether the slopes are equal and plot the groups with their fitted lines.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors", Any, 1, 5)
        .vars("covariates", "Covariates", Numeric, 1, 5)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Options, "slopes", "Test for equal slopes (factor by covariate interactions)", true)
        .check(Graphs, "plot", "Plot response versus the first covariate with fitted lines", true),
        &["Stat", "ANOVA", "Analysis of Covariance..."],
        Runner::Report(ancova_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.anova.maineffects",
            "Main Effects Plot",
            "Plot the mean response at each level of each factor.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors", Any, 1, 9),
        &["Stat", "ANOVA", "Main Effects Plot..."],
        Runner::Report(main_effects_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.anova.interactions",
            "Interactions Plot",
            "Plot the mean response across the levels of one factor, one line per level of another.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors (two or more)", Any, 2, 9),
        &["Stat", "ANOVA", "Interactions Plot..."],
        Runner::Report(interaction_run),
    );
}
