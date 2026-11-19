//! DOE completion dialogs (Phase 5o): mixture designs and their analysis, optimal (custom)
//! designs and split-plot designs. Each dialog wraps a routine of `openstats_stats::doe`.

use super::analyses::add;
use super::analyses_doe::{numbers, words};
use super::analyses_ts::parse_groups;
use super::analysis::{col_title, complete_rows, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::doe::{
    analyze_variability, definitive_screening, design_from_columns, extreme_vertices, fit_mixture, fit_responses,
    optimal_design, overlaid_contour, simplex_centroid, simplex_lattice, split_plot, steepest_path, Criterion,
    MixtureDesign, MixtureModel, ModelOrder, OptModel, OptimalOptions, OverlayResponse,
};
use openstats_stats::report::{
    Cell, ChartKind, ChartSpec, Column as TCol, ColumnKind as TKind, Section, Series, Table,
};

use Page::{Graphs, Main, Options, Storage};
use VarKind::Numeric;

const MAX_SHOWN: usize = 300;

fn tcols(h: &[(&str, TKind)]) -> Vec<TCol> {
    h.iter().map(|(n, k)| TCol::new(*n, *k)).collect()
}

fn component_names(v: &Values, default_q: usize) -> Res<Vec<String>> {
    let w = words(v.text("names"));
    if w.is_empty() {
        let q = v.count("components").max(default_q);
        Ok((0..q).map(|i| format!("{}", (b'A' + i as u8 % 26) as char)).collect())
    } else {
        Ok(w)
    }
}

fn seed(v: &Values) -> Option<u64> {
    v.flag("randomize").then(|| v.count("seed") as u64)
}

fn mixture_output(d: &MixtureDesign, v: &Values) -> Res<Output> {
    let mut s = Section::new(format!("{}: {} components, {} runs", d.kind, d.components.len(), d.runs()));
    let mut heads: Vec<(String, TKind)> = vec![("Run".into(), TKind::Integer), ("PtType".into(), TKind::Integer)];
    heads.extend(d.components.iter().map(|c| (c.clone(), TKind::Fixed(4))));
    let mut t = Table::new("Design Table", heads.iter().map(|(h, k)| TCol::new(h.clone(), *k)).collect());
    for (i, (p, ty)) in d.points.iter().zip(&d.point_types).enumerate().take(MAX_SHOWN) {
        let mut row = vec![Cell::Int(i as i64 + 1), Cell::Int(*ty)];
        row.extend(p.iter().map(|&x| Cell::Num(x)));
        t.push_row(row)?;
    }
    if d.runs() > MAX_SHOWN {
        t.footnotes
            .push(format!("Only the first {MAX_SHOWN} of {} runs are shown; store the design to see all.", d.runs()));
    }
    s.text("Point types: 1 = pure component or vertex, 2 = blend, 0 = overall centroid, -1 = axial point.");
    s.table(t);
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric("RunOrder", (1..=d.runs()).map(|x| x as f64).collect()));
        stored.push(Column::numeric("PtType", d.point_types.iter().map(|&x| x as f64).collect()));
        for (j, c) in d.components.iter().enumerate() {
            stored.push(Column::numeric(c.clone(), d.points.iter().map(|p| p[j]).collect()));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn creation_tail(s: DialogSpec) -> DialogSpec {
    s.integer(Options, "replicates", "Replicates", 1.0, 1.0, 100.0)
        .check(Options, "randomize", "Randomize runs", false)
        .integer(Options, "seed", "Base for random data", 1.0, 0.0, 1e9)
        .check(Storage, "store", "Store design in worksheet", true)
}

fn lattice_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let names = component_names(v, 3)?;
    let d = simplex_lattice(
        &names,
        v.count("degree"),
        v.flag("center"),
        v.flag("axial"),
        v.count("replicates").max(1),
        seed(v),
    )?;
    mixture_output(&d, v)
}

fn centroid_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let names = component_names(v, 3)?;
    let d = simplex_centroid(&names, v.flag("axial"), v.count("replicates").max(1), seed(v))?;
    mixture_output(&d, v)
}

fn vertices_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let names = component_names(v, 3)?;
    let lo = numbers(v.text("lower"), "lower bounds")?;
    let hi = numbers(v.text("upper"), "upper bounds")?;
    if lo.len() != names.len() || hi.len() != names.len() {
        return Err(format!("enter {} lower and {} upper bounds (one per component)", names.len(), names.len()).into());
    }
    let d = extreme_vertices(&names, &lo, &hi, v.flag("center"), v.count("replicates").max(1), seed(v))?;
    mixture_output(&d, v)
}

fn mixture_model(v: &Values) -> MixtureModel {
    match v.choice("model") {
        1 => MixtureModel::Quadratic,
        2 => MixtureModel::SpecialCubic,
        3 => MixtureModel::FullCubic,
        _ => MixtureModel::Linear,
    }
}

fn analyze_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let rc = v.var("response").ok_or("select the response column")?;
    let y = numeric(ws, rc)?;
    let comps = v.vars("components");
    if comps.len() < 2 {
        return Err("select at least two component columns".into());
    }
    let cols: Vec<Vec<f64>> = comps.iter().map(|&c| numeric(ws, c)).collect::<Result<_, _>>()?;
    let names: Vec<String> = comps.iter().map(|&c| col_title(ws, c)).collect();
    let keep: Vec<usize> = (0..y.len()).filter(|&i| !y[i].is_nan() && cols.iter().all(|c| !c[i].is_nan())).collect();
    let x: Vec<Vec<f64>> = keep.iter().map(|&i| cols.iter().map(|c| c[i]).collect()).collect();
    let yy: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
    let fit = fit_mixture(&x, &yy, &names, mixture_model(v))?;
    let lev = level(v)?;
    let mut s = Section::new(format!("Mixture Regression: {} versus {}", col_title(ws, rc), names.join(", ")));
    s.text(format!(
        "Model: Scheffé {} (no constant)\nS = {:.6}   R-Sq = {:.2}%   R-Sq(adj) = {:.2}%   Runs: {}",
        fit.model.name().to_lowercase(),
        fit.s,
        fit.r2 * 100.0,
        fit.r2_adj * 100.0,
        x.len()
    ));
    let tq = openstats_math::dist::StudentT::new(fit.df_error as f64)
        .map(|d| openstats_math::dist::ContinuousDistribution::isf(&d, (1.0 - lev) / 2.0))
        .unwrap_or(f64::NAN);
    let mut ct = Table::new(
        "Estimated Regression Coefficients",
        tcols(&[
            ("Term", TKind::Text),
            ("Coef", TKind::Significant(6)),
            ("SE Coef", TKind::Significant(5)),
            ("T", TKind::Fixed(2)),
            ("P", TKind::PValue),
            ("Lower", TKind::Significant(6)),
            ("Upper", TKind::Significant(6)),
        ]),
    );
    for i in 0..fit.coef.len() {
        ct.push_row(vec![
            Cell::text(&fit.terms[i]),
            Cell::Num(fit.coef[i]),
            Cell::Num(fit.se[i]),
            Cell::Num(fit.t[i]),
            Cell::Num(fit.p[i]),
            Cell::Num(fit.coef[i] - tq * fit.se[i]),
            Cell::Num(fit.coef[i] + tq * fit.se[i]),
        ])?;
    }
    s.table(ct);
    let mut at = Table::new(
        "Analysis of Variance (sequential)",
        tcols(&[
            ("Source", TKind::Text),
            ("DF", TKind::Integer),
            ("Seq SS", TKind::Significant(6)),
            ("Seq MS", TKind::Significant(6)),
            ("F", TKind::Fixed(2)),
            ("P", TKind::PValue),
        ]),
    );
    for r in &fit.anova {
        let opt = |x: f64| if x.is_nan() { Cell::Missing } else { Cell::Num(x) };
        at.push_row(vec![
            Cell::text(&r.source),
            Cell::Int(r.df as i64),
            Cell::Num(r.ss),
            opt(r.ms),
            opt(r.f),
            opt(r.p),
        ])?;
    }
    s.table(at);
    let pred = numbers(v.text("predict"), "prediction point")?;
    if !pred.is_empty() {
        if pred.len() != names.len() || (pred.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return Err(format!("give {} proportions that sum to 1 for the prediction", names.len()).into());
        }
        s.text(format!(
            "Predicted response at ({}): {:.6}",
            pred.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", "),
            fit.predict(&pred)
        ));
    }
    if v.flag("charts") {
        s.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: "Fitted versus Observed".into(),
            x_label: "Fitted".into(),
            y_label: col_title(ws, rc),
            categories: Vec::new(),
            series: vec![Series { name: "Runs".into(), x: fit.fitted.clone(), y: yy.clone() }],
            reference_lines: Vec::new(),
        });
    }
    let n = y.len();
    let expand = |name: &str, vals: &[f64]| {
        let mut o = vec![f64::NAN; n];
        for (&i, &v) in keep.iter().zip(vals) {
            o[i] = v;
        }
        Column::numeric(name, o)
    };
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        stored.push(expand("FITS", &fit.fitted));
    }
    if v.flag("store_resid") {
        stored.push(expand("RESI", &fit.residuals));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn optimal_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let k = v.count("factors");
    let lv = numbers(v.text("levels"), "levels")?;
    let levels: Vec<usize> = if lv.is_empty() {
        vec![if v.choice("model") == 2 { 3 } else { 2 }; k]
    } else {
        lv.iter().map(|&x| x as usize).collect()
    };
    if levels.len() != k {
        return Err(format!("enter {k} level counts (one per factor), or leave the box blank").into());
    }
    let model = match v.choice("model") {
        1 => OptModel::Interactions,
        2 => OptModel::Quadratic,
        _ => OptModel::Linear,
    };
    let criterion = if v.choice("criterion") == 1 { Criterion::I } else { Criterion::D };
    let d = optimal_design(&OptimalOptions {
        runs: v.count("runs"),
        criterion,
        model,
        levels,
        restarts: v.count("restarts").max(1),
        seed: v.count("seed") as u64,
    })?;
    let mut names = words(v.text("names"));
    if names.is_empty() {
        names = (0..k).map(|i| format!("{}", (b'A' + i as u8 % 26) as char)).collect();
    }
    if names.len() != k {
        return Err(format!("enter {k} factor names or leave the box blank").into());
    }
    let mut s = Section::new(format!(
        "{}-Optimal Design: {} factors, {} runs",
        if criterion == Criterion::D { "D" } else { "I" },
        k,
        d.points.len()
    ));
    let mut q = Table::new("Design Quality", tcols(&[("Measure", TKind::Text), ("Value", TKind::Significant(6))]));
    q.push_row(vec![Cell::text("Model terms"), Cell::Int(d.n_terms as i64)])?;
    q.push_row(vec![Cell::text("Candidate points searched"), Cell::Int(d.candidates as i64)])?;
    q.push_row(vec![Cell::text("D-efficiency (%)"), Cell::Num(d.d_efficiency)])?;
    q.push_row(vec![Cell::text("ln |X'X|"), Cell::Num(d.log_det)])?;
    q.push_row(vec![Cell::text("Average prediction variance / sigma^2"), Cell::Num(d.avg_prediction_variance)])?;
    q.push_row(vec![Cell::text("Maximum prediction variance / sigma^2"), Cell::Num(d.max_prediction_variance)])?;
    s.table(q);
    let mut heads: Vec<(String, TKind)> = vec![("Run".into(), TKind::Integer)];
    heads.extend(names.iter().map(|n| (n.clone(), TKind::Fixed(3))));
    let mut t = Table::new("Design Table", heads.iter().map(|(h, k)| TCol::new(h.clone(), *k)).collect());
    for (i, p) in d.points.iter().enumerate().take(MAX_SHOWN) {
        let mut row = vec![Cell::Int(i as i64 + 1)];
        row.extend(p.iter().map(|&x| Cell::Num(x)));
        t.push_row(row)?;
    }
    s.table(t);
    s.text("The search is a heuristic exchange algorithm with random restarts; it does not guarantee the global optimum. Levels are coded -1 to +1.");
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric("RunOrder", (1..=d.points.len()).map(|x| x as f64).collect()));
        for (j, n) in names.iter().enumerate() {
            stored.push(Column::numeric(n.clone(), d.points.iter().map(|p| p[j]).collect()));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn split_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let (kw, ks) = (v.count("whole"), v.count("sub"));
    let sp = split_plot(kw, ks, v.count("replicates").max(1), seed(v))?;
    let mut names = words(v.text("names"));
    if names.is_empty() {
        names = (0..kw + ks).map(|i| format!("{}", (b'A' + i as u8 % 26) as char)).collect();
    }
    if names.len() != kw + ks {
        return Err(format!("enter {} factor names (whole-plot factors first) or leave the box blank", kw + ks).into());
    }
    let mut s =
        Section::new(format!("Split-Plot Design: {kw} whole-plot and {ks} subplot factors, {} runs", sp.runs.len()));
    s.text(format!(
        "{} whole plots of {} runs each. Whole-plot factors: {}. Subplot factors: {}.\nAnalyze with Stat > ANOVA > Mixed Effects Model, entering the WholePlot column as a random factor (the whole-plot error), so that whole-plot factors are tested against it.",
        sp.runs.len() >> ks,
        1usize << ks,
        names[..kw].join(", "),
        names[kw..].join(", ")
    ));
    let mut heads: Vec<(String, TKind)> = vec![("Run".into(), TKind::Integer), ("WholePlot".into(), TKind::Integer)];
    heads.extend(names.iter().map(|n| (n.clone(), TKind::Fixed(0))));
    let mut t = Table::new("Design Table", heads.iter().map(|(h, k)| TCol::new(h.clone(), *k)).collect());
    for (i, r) in sp.runs.iter().enumerate().take(MAX_SHOWN) {
        let mut row = vec![Cell::Int(i as i64 + 1), Cell::Int(r.whole_plot as i64)];
        row.extend(r.levels.iter().map(|&x| Cell::Num(x)));
        t.push_row(row)?;
    }
    s.table(t);
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric("RunOrder", (1..=sp.runs.len()).map(|x| x as f64).collect()));
        stored.push(Column::numeric("WholePlot", sp.runs.iter().map(|r| r.whole_plot as f64).collect()));
        for (j, n) in names.iter().enumerate() {
            stored.push(Column::numeric(n.clone(), sp.runs.iter().map(|r| r.levels[j]).collect()));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn order_of(v: &Values) -> ModelOrder {
    match v.choice("model") {
        1 => ModelOrder::Interactions,
        2 => ModelOrder::Quadratic,
        _ => ModelOrder::Linear,
    }
}

fn dsd_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let k = v.count("factors");
    let d = definitive_screening(k, v.count("extra_center"), seed(v))?;
    let mut names = words(v.text("names"));
    if names.is_empty() {
        names = (0..k).map(|i| format!("{}", (b'A' + i as u8 % 26) as char)).collect();
    }
    if names.len() != k {
        return Err(format!("enter {k} factor names or leave the box blank").into());
    }
    let lows = numbers(v.text("lows"), "low levels")?;
    let highs = numbers(v.text("highs"), "high levels")?;
    let natural = !lows.is_empty() || !highs.is_empty();
    if natural && (lows.len() != k || highs.len() != k) {
        return Err(format!("enter {k} low and {k} high levels, or leave both boxes blank for coded units").into());
    }
    let conv = |j: usize, c: f64| if natural { 0.5 * (lows[j] + highs[j]) + 0.5 * (highs[j] - lows[j]) * c } else { c };
    let mut s = Section::new(format!("Definitive Screening Design: {k} factors, {} runs", d.runs.len()));
    s.text(format!(
        "Built from a conference matrix of order {} ({} column(s) dropped); main effects are {} with each other, and no main effect is aliased with any two-factor interaction or a squared term (max |correlation| between main-effect columns: {:.4}).",
        d.conference_order,
        d.dropped_columns,
        if d.max_main_correlation() < 1e-9 { "orthogonal" } else { "nearly orthogonal" },
        d.max_main_correlation()
    ));
    s.text("Fit with Stat > Regression > Fit Regression Model, or use Stat > DOE > Response Surface Analysis (quadratic) once the number of significant factors is small.");
    let mut heads: Vec<(String, TKind)> = vec![("Run".into(), TKind::Integer)];
    heads.extend(names.iter().map(|n| (n.clone(), TKind::Fixed(3))));
    let mut t = Table::new("Design Table", heads.iter().map(|(h, k)| TCol::new(h.clone(), *k)).collect());
    for (i, r) in d.runs.iter().enumerate().take(MAX_SHOWN) {
        let mut row = vec![Cell::Int(i as i64 + 1)];
        row.extend(r.iter().enumerate().map(|(j, &c)| Cell::Num(conv(j, c))));
        t.push_row(row)?;
    }
    s.table(t);
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric("RunOrder", (1..=d.runs.len()).map(|x| x as f64).collect()));
        for (j, n) in names.iter().enumerate() {
            stored.push(Column::numeric(n.clone(), d.runs.iter().map(|r| conv(j, r[j])).collect()));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

/// Factor and response data of a designed experiment: (complete rows, names, factor columns, response columns).
type DesignData = (Vec<usize>, Vec<String>, Vec<Vec<f64>>, Vec<(String, Vec<f64>)>);

fn design_data(ws: &Worksheet, v: &Values) -> Res<DesignData> {
    let fcols = v.vars("factors").to_vec();
    let rcols: Vec<usize> = if v.vars("responses").is_empty() {
        v.var("response").into_iter().collect()
    } else {
        v.vars("responses").to_vec()
    };
    if fcols.is_empty() || rcols.is_empty() {
        return Err("select the factor columns and a response".into());
    }
    if fcols.iter().any(|c| rcols.contains(c)) {
        return Err("a column cannot be both a factor and a response".into());
    }
    let all: Vec<Vec<f64>> = fcols.iter().chain(&rcols).map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let refs: Vec<&[f64]> = all.iter().map(Vec::as_slice).collect();
    let (rows, data) = complete_rows(&refs);
    let names: Vec<String> = fcols.iter().map(|&c| col_title(ws, c)).collect();
    let resp = rcols.iter().enumerate().map(|(i, &c)| (col_title(ws, c), data[fcols.len() + i].clone())).collect();
    Ok((rows, names, data[..fcols.len()].to_vec(), resp))
}

fn steepest_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (_, names, fdata, resp) = design_data(ws, v)?;
    let design = design_from_columns(&names, &fdata)?;
    let fits = fit_responses(&design, &resp, ModelOrder::Linear)?;
    let ascent = v.choice("direction") == 0;
    let path = steepest_path(&fits[0].model, &design.factors, ascent, v.num("step"), v.count("steps"))?;
    let mut s =
        Section::new(format!("Path of Steepest {}: {}", if ascent { "Ascent" } else { "Descent" }, fits[0].name));
    s.text(format!(
        "First-order model fitted to {} runs: R-Sq = {:.2}%, S = {:.6}. The factor with the largest coefficient moves {} coded unit(s) per step; the others move in proportion to their coefficients.",
        fdata[0].len(),
        fits[0].r2 * 100.0,
        fits[0].s,
        v.num("step")
    ));
    let mut ct = Table::new(
        "Linear Coefficients (coded units)",
        tcols(&[("Term", TKind::Text), ("Coef", TKind::Significant(6))]),
    );
    let tn = |i: usize| fits[0].model.terms[i].label(&names);
    for (i, c) in fits[0].model.coef.iter().enumerate() {
        ct.push_row(vec![Cell::text(if i == 0 { "Constant".to_string() } else { tn(i) }), Cell::Num(*c)])?;
    }
    s.table(ct);
    let mut heads: Vec<(String, TKind)> = vec![("Step".into(), TKind::Integer)];
    heads.extend(names.iter().map(|n| (format!("{n} (coded)"), TKind::Fixed(3))));
    heads.extend(names.iter().map(|n| (n.clone(), TKind::Significant(6))));
    heads.push((format!("Predicted {}", fits[0].name), TKind::Significant(6)));
    let mut t = Table::new("Steps", heads.iter().map(|(h, k)| TCol::new(h.clone(), *k)).collect());
    for p in &path {
        let mut row = vec![Cell::Int(p.step as i64)];
        row.extend(p.coded.iter().map(|&x| Cell::Num(x)));
        row.extend(p.natural.iter().map(|&x| Cell::Num(x)));
        row.push(Cell::Num(p.predicted));
        t.push_row(row)?;
    }
    s.table(t);
    if v.flag("charts") {
        s.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: format!("Predicted {} along the path", fits[0].name),
            x_label: "Step".into(),
            y_label: fits[0].name.clone(),
            categories: Vec::new(),
            series: vec![Series {
                name: "Fit".into(),
                x: path.iter().map(|p| p.step as f64).collect(),
                y: path.iter().map(|p| p.predicted).collect(),
            }],
            reference_lines: Vec::new(),
        });
    }
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric("PathStep", path.iter().map(|p| p.step as f64).collect()));
        for (j, n) in names.iter().enumerate() {
            stored.push(Column::numeric(format!("Path{n}"), path.iter().map(|p| p.natural[j]).collect()));
        }
        stored.push(Column::numeric("PathFit", path.iter().map(|p| p.predicted).collect()));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn variability_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (_, names, fdata, resp) = design_data(ws, v)?;
    let design = design_from_columns(&names, &fdata)?;
    if design.coded.iter().flatten().any(|c| (c.abs() - 1.0).abs() > 1e-9) {
        return Err("analyze variability needs two-level factors (each column with exactly two values)".into());
    }
    let a = analyze_variability(&design.coded, &resp[0].1, &names, v.flag("interactions"))?;
    let mut s = Section::new(format!("Analysis of Variability: {} versus {}", resp[0].0, names.join(", ")));
    s.text(format!(
        "ln(StDev) regressed on {} replicated factor combinations ({} error DF): R-Sq = {:.2}%. A StDev ratio above 1 means the spread is larger at the factor's high level.",
        a.groups.len(),
        a.df_error,
        a.r2 * 100.0
    ));
    let mut t = Table::new(
        "Estimated Effects on ln(StDev)",
        tcols(&[
            ("Term", TKind::Text),
            ("Effect", TKind::Significant(5)),
            ("Coef", TKind::Significant(5)),
            ("SE Coef", TKind::Significant(5)),
            ("T", TKind::Fixed(2)),
            ("P", TKind::PValue),
            ("StDev Ratio", TKind::Fixed(4)),
        ]),
    );
    t.push_row(vec![
        Cell::text("Constant"),
        Cell::Missing,
        Cell::Num(a.constant),
        Cell::Missing,
        Cell::Missing,
        Cell::Missing,
        Cell::Missing,
    ])?;
    for x in &a.terms {
        t.push_row(vec![
            Cell::text(&x.name),
            Cell::Num(x.effect),
            Cell::Num(x.coef),
            Cell::Num(x.se),
            Cell::Num(x.t),
            Cell::Num(x.p),
            Cell::Num(x.ratio),
        ])?;
    }
    s.table(t);
    let mut g = Table::new("Standard Deviation of each Factor Combination", {
        let mut h: Vec<(String, TKind)> = names.iter().map(|n| (n.clone(), TKind::Fixed(0))).collect();
        h.extend([
            ("StDev".to_string(), TKind::Significant(6)),
            ("DF".to_string(), TKind::Integer),
            ("Fitted StDev".to_string(), TKind::Significant(6)),
        ]);
        h.iter().map(|(n, k)| TCol::new(n.clone(), *k)).collect()
    });
    for (i, (lv, sd, df)) in a.groups.iter().enumerate() {
        let mut row: Vec<Cell> = lv.iter().map(|&c| Cell::Num(c)).collect();
        row.extend([Cell::Num(*sd), Cell::Int(*df as i64), Cell::Num(a.fitted_sd[i])]);
        g.push_row(row)?;
    }
    s.table(g);
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

fn overlay_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (_, names, fdata, resp) = design_data(ws, v)?;
    let design = design_from_columns(&names, &fdata)?;
    let k = names.len();
    if k < 2 {
        return Err("an overlaid contour plot needs at least two factors".into());
    }
    let fits = fit_responses(&design, &resp, order_of(v))?;
    let groups = parse_groups(v.text("bounds").trim(), "bounds")?;
    if groups.len() != fits.len() || groups.iter().any(|g| g.len() != 2) {
        return Err(
            format!("enter \"lower upper\" (or *) for each of the {} responses, separated by ';'", fits.len()).into()
        );
    }
    let (ix, iy) = (v.count("x_factor").saturating_sub(1), v.count("y_factor").saturating_sub(1));
    if ix >= k || iy >= k || ix == iy {
        return Err(format!("choose two different factor numbers between 1 and {k}").into());
    }
    let mut fixed = vec![0.0; k];
    let hold = v.text("hold").trim();
    if !hold.is_empty() {
        let flat: Vec<Option<f64>> = parse_groups(hold, "held values")?.into_iter().flatten().collect();
        if flat.len() != k {
            return Err(format!("enter one held value (or *) per factor ({k})").into());
        }
        for (j, h) in flat.into_iter().enumerate() {
            if let Some(x) = h {
                fixed[j] = design.factors[j].to_coded(x);
            }
        }
    }
    let ov: Vec<OverlayResponse<'_>> = fits
        .iter()
        .zip(&groups)
        .map(|(f, g)| OverlayResponse {
            name: f.name.clone(),
            model: &f.model,
            lower: g[0].unwrap_or(f64::NEG_INFINITY),
            upper: g[1].unwrap_or(f64::INFINITY),
        })
        .collect();
    let range = ([-1.0, 1.0], [-1.0, 1.0]);
    let o = overlaid_contour(&ov, &design.factors, ix, iy, &fixed, range, v.count("grid").max(3))?;
    let mut s = Section::new(format!(
        "Overlaid Contour Plot: {} (x = {}, y = {})",
        fits.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", "),
        names[ix],
        names[iy]
    ));
    s.text(format!(
        "Models: {}. Feasible region (all responses within their bounds): {:.1}% of the plotted area.{}",
        match order_of(v) {
            ModelOrder::Linear => "linear",
            ModelOrder::Interactions => "linear with interactions",
            ModelOrder::Quadratic => "full quadratic",
        },
        o.feasible_fraction * 100.0,
        o.best_point
            .map(|(x, y)| format!(
                " A point near the middle of the feasible region: {} = {x:.6}, {} = {y:.6}.",
                names[ix], names[iy]
            ))
            .unwrap_or_else(|| " No feasible region was found.".into())
    ));
    s.chart(o.chart);
    if o.best_point.is_some() {
        s.chart(o.feasible_chart);
    }
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

pub(crate) fn register(r: &mut Registry) {
    let names = |s: DialogSpec| {
        s.integer(Main, "components", "Number of components (when no names are typed)", 3.0, 2.0, 24.0).text(
            Main,
            "names",
            "Component names (blank = A, B, C...)",
            "",
        )
    };
    add(
        r,
        creation_tail(
            names(DialogSpec::new(
                "stat.doe.mix_lattice",
                "Simplex Lattice Design",
                "Mixture design with proportions that are multiples of 1/degree.",
            ))
            .integer(Main, "degree", "Lattice degree", 2.0, 1.0, 10.0)
            .check(Options, "center", "Add the overall centroid", true)
            .check(Options, "axial", "Add axial points", false),
        ),
        &["Stat", "DOE", "Mixture", "Simplex Lattice Design..."],
        Runner::Report(lattice_run),
    );
    add(
        r,
        creation_tail(
            names(DialogSpec::new(
                "stat.doe.mix_centroid",
                "Simplex Centroid Design",
                "Mixture design with the centroid of every subset of components.",
            ))
            .check(Options, "axial", "Add axial points", false),
        ),
        &["Stat", "DOE", "Mixture", "Simplex Centroid Design..."],
        Runner::Report(centroid_run),
    );
    add(
        r,
        creation_tail(
            names(DialogSpec::new(
                "stat.doe.mix_vertices",
                "Extreme Vertices Design",
                "Mixture design for components with lower and upper bounds.",
            ))
            .text(Main, "lower", "Lower bounds (one per component)", "")
            .text(Main, "upper", "Upper bounds (one per component)", "")
            .check(Options, "center", "Add the centroid of the vertices", true),
        ),
        &["Stat", "DOE", "Mixture", "Extreme Vertices Design..."],
        Runner::Report(vertices_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.mix_analyze",
            "Analyze Mixture Design",
            "Scheffé regression (no constant) of a response on the component proportions.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("components", "Components", Numeric, 2, 24)
        .choice(Options, "model", "Model", &["Linear", "Quadratic", "Special Cubic", "Full Cubic"], 1)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .text(Options, "predict", "Predict at these proportions (optional; one per component)", "")
        .check(Graphs, "charts", "Fitted versus observed plot", true)
        .check(Storage, "store_fits", "Fits", false)
        .check(Storage, "store_resid", "Residuals", false),
        &["Stat", "DOE", "Mixture", "Analyze Mixture Design..."],
        Runner::Report(analyze_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.optimal",
            "Create Optimal Design",
            "D- or I-optimal design chosen from a candidate grid by the exchange algorithm.",
        )
        .integer(Main, "factors", "Number of factors", 3.0, 1.0, 8.0)
        .integer(Main, "runs", "Number of runs", 12.0, 2.0, 2000.0)
        .choice(Main, "criterion", "Criterion", &["D-optimal", "I-optimal"], 0)
        .choice(Main, "model", "Model", &["Linear", "Linear + interactions", "Quadratic"], 1)
        .text(Options, "levels", "Candidate levels per factor (2-5 each; blank = 2, or 3 for quadratic)", "")
        .text(Options, "names", "Factor names (blank = A, B, C...)", "")
        .integer(Options, "restarts", "Random restarts", 10.0, 1.0, 1000.0)
        .integer(Options, "seed", "Base for random data", 1.0, 0.0, 1e9)
        .check(Storage, "store", "Store design in worksheet", true),
        &["Stat", "DOE", "Custom Design", "Create Optimal Design..."],
        Runner::Report(optimal_run),
    );
    add(
        r,
        creation_tail(
            DialogSpec::new(
                "stat.doe.splitplot",
                "Create Split-Plot Design",
                "Two-level full-factorial split-plot design: whole-plot factors in whole plots, subplot factors inside them.",
            )
            .integer(Main, "whole", "Number of whole-plot factors", 2.0, 1.0, 6.0)
            .integer(Main, "sub", "Number of subplot factors", 2.0, 1.0, 6.0)
            .text(Options, "names", "Factor names, whole-plot factors first (blank = A, B, C...)", ""),
        ),
        &["Stat", "DOE", "Split-Plot", "Create Split-Plot Design..."],
        Runner::Report(split_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.dsd",
            "Create Definitive Screening Design",
            "Three-level screening design (Jones and Nachtsheim) with orthogonal main effects, estimating curvature in any active factor.",
        )
        .integer(Main, "factors", "Number of factors", 6.0, 3.0, 30.0)
        .text(Options, "names", "Factor names (blank = A, B, C...)", "")
        .text(Options, "lows", "Low levels (blank = coded -1)", "")
        .text(Options, "highs", "High levels (blank = coded +1)", "")
        .integer(Options, "extra_center", "Extra centre runs", 0.0, 0.0, 20.0)
        .check(Options, "randomize", "Randomize runs", false)
        .integer(Options, "seed", "Base for random data", 1.0, 0.0, 1e9)
        .check(Storage, "store", "Store design in worksheet", true),
        &["Stat", "DOE", "Screening", "Create Definitive Screening Design..."],
        Runner::Report(dsd_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.steepest",
            "Steepest Ascent",
            "Fit a first-order model and step along the path of steepest ascent or descent.",
        )
        .vars("factors", "Factors (natural units)", Numeric, 2, 15)
        .vars("response", "Response", Numeric, 1, 1)
        .choice(
            Options,
            "direction",
            "Direction",
            &["Ascent (increase the response)", "Descent (decrease the response)"],
            0,
        )
        .number(Options, "step", "Step size of the largest factor (coded units)", 1.0, 0.001, 100.0)
        .integer(Options, "steps", "Number of steps", 10.0, 1.0, 100.0)
        .check(Graphs, "charts", "Predicted response along the path", true)
        .check(Storage, "store", "Store the path in the worksheet", false),
        &["Stat", "DOE", "Factorial", "Steepest Ascent..."],
        Runner::Report(steepest_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.variability",
            "Analyze Variability",
            "Dispersion effects: regress ln(standard deviation) of replicated factor combinations on the two-level factors.",
        )
        .vars("factors", "Factors", Numeric, 1, 8)
        .vars("response", "Response", Numeric, 1, 1)
        .check(Options, "interactions", "Include two-factor interactions", false),
        &["Stat", "DOE", "Factorial", "Analyze Variability..."],
        Runner::Report(variability_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.overlay",
            "Overlaid Contour Plot",
            "Show where several fitted responses are all within their acceptable bounds.",
        )
        .vars("factors", "Factors (natural units)", Numeric, 2, 15)
        .vars("responses", "Responses", Numeric, 1, 10)
        .text(Main, "bounds", "Bounds per response: \"lower upper\" (* = none), ';' separated", "")
        .integer(Main, "x_factor", "Factor for the x axis (its number in the list)", 1.0, 1.0, 15.0)
        .integer(Main, "y_factor", "Factor for the y axis (its number in the list)", 2.0, 1.0, 15.0)
        .choice(
            Options,
            "model",
            "Model for every response",
            &["Linear", "Linear and interactions", "Full quadratic"],
            1,
        )
        .text(Options, "hold", "Hold other factors at (natural units, * = centre)", "")
        .integer(Options, "grid", "Grid points per axis", 60.0, 3.0, 400.0),
        &["Stat", "DOE", "Overlaid Contour Plot..."],
        Runner::Report(overlay_run),
    );
}
