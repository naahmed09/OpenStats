//! Quality completion dialogs (Phase 5m): acceptance sampling, multi-vari and run charts, the
//! gage run chart, attribute and multi-variable capability, and the Johnson and Box-Cox
//! transformations. Each dialog wraps a routine of `openstats_stats`.

use super::analyses::{add, out};
use super::analyses_tables::levels;
use super::analysis::{col_title, labels, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::acceptance::{
    attribute_decision_section, attribute_report, decide_variables, double_report, find_attribute_plan,
    find_variables_plan, variables_decision_section, variables_report, AttributePlan, DoublePlan, LimitSide, LotModel,
    VariablesPlan,
};
use openstats_stats::attrcap::{binomial_capability, poisson_capability};
use openstats_stats::graphical::{anderson_darling, histogram, Binning};
use openstats_stats::multivari::multi_vari;
use openstats_stats::report::{
    Axis, Cell, ChartKind, ChartSpec, Column as TCol, ColumnKind as TKind, ReferenceLine, Section, Series, Table,
};
use openstats_stats::runchart::{run_chart, subgroup_points, Center, SubgroupStat};
use openstats_stats::spc::capability::{capability_individuals, capability_subgroups, CapabilityOptions, Specs};
use openstats_stats::spc::transform::{box_cox_interval, box_cox_profile, fit_johnson, BoxCox, LambdaChoice};

use Page::{Main, Options, Storage};
use VarKind::{Any, Numeric};

const MODELS: [&str; 3] = ["Binomial", "Poisson", "Hypergeometric (needs a lot size)"];
const SIGMA: [&str; 2] = ["Unknown (use the sample standard deviation)", "Known"];
const SIDES: [&str; 2] = ["Upper specification limit", "Lower specification limit"];

fn tcols(heads: &[(&str, TKind)]) -> Vec<TCol> {
    heads.iter().map(|(h, k)| TCol::new(*h, *k)).collect()
}

fn model_of(v: &Values) -> LotModel {
    match v.choice("model") {
        1 => LotModel::Poisson,
        2 => LotModel::Hypergeometric,
        _ => LotModel::Binomial,
    }
}

fn lot_of(v: &Values) -> Res<Option<u64>> {
    let l = v.num("lot");
    if !l.is_finite() {
        return Ok(None);
    }
    if l < 1.0 || l.fract() != 0.0 {
        return Err("the lot size must be a whole number of at least 1".into());
    }
    Ok(Some(l as u64))
}

fn optional_level(v: &Values, name: &str) -> Option<f64> {
    Some(v.num(name)).filter(|x| x.is_finite())
}

fn parse_list(text: &str, what: &str) -> Res<Vec<f64>> {
    let mut out = Vec::new();
    for tok in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';').filter(|t| !t.is_empty()) {
        out.push(tok.parse::<f64>().map_err(|_| format!("'{tok}' in {what} is not a number"))?);
    }
    if out.is_empty() {
        return Err(format!("enter {what}").into());
    }
    Ok(out)
}

fn level_pct(v: &Values) -> Res<f64> {
    let l = v.num("level") / 100.0;
    if l > 0.0 && l < 1.0 {
        Ok(l)
    } else {
        Err("the confidence level must be between 0 and 100".into())
    }
}

fn quality_levels(spec: DialogSpec) -> DialogSpec {
    spec.optional_number(Main, "aql", "Acceptable quality level (AQL, proportion defective; blank = none)")
        .optional_number(Main, "rql", "Rejectable quality level (RQL, proportion defective; blank = none)")
}

// ------------------------------------------------------------------------ attribute sampling

fn attr_create_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let model = model_of(v);
    let lot = lot_of(v)?;
    let (aql, rql, al, be) = (v.num("aql"), v.num("rql"), v.num("alpha"), v.num("beta"));
    let found = find_attribute_plan(aql, rql, al, be, model, lot)?;
    let mut s = attribute_report(&[("Plan".to_string(), found.plan)], model, lot, Some(aql), Some(rql))?;
    s.heading = "Create Acceptance Sampling Plan (Attributes)".into();
    s.text(format!(
        "Sample {} items and accept the lot when at most {} are defective. Probability of accepting at the AQL: {:.4} \
         (producer's risk {:.4}); at the RQL: {:.4} (consumer's risk, at most {:.4}).",
        found.plan.n,
        found.plan.c,
        found.pa_aql,
        1.0 - found.pa_aql,
        found.pa_rql,
        be
    ));
    out(s)
}

fn attr_compare_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let model = model_of(v);
    let lot = lot_of(v)?;
    let ns = parse_list(v.text("ns"), "the sample sizes")?;
    let cs = parse_list(v.text("cs"), "the acceptance numbers")?;
    if ns.len() != cs.len() {
        return Err("give one acceptance number for each sample size".into());
    }
    let mut plans = Vec::new();
    for (n, c) in ns.iter().zip(&cs) {
        if *n < 1.0 || n.fract() != 0.0 || *c < 0.0 || c.fract() != 0.0 {
            return Err("sample sizes and acceptance numbers must be whole numbers (n at least 1)".into());
        }
        plans.push((format!("n={n}, c={c}"), AttributePlan { n: *n as u64, c: *c as u64 }));
    }
    let mut s = attribute_report(&plans, model, lot, optional_level(v, "aql"), optional_level(v, "rql"))?;
    s.heading = "Compare Acceptance Sampling Plans (Attributes)".into();
    out(s)
}

fn int_field(v: &Values, name: &str) -> u64 {
    v.num(name).max(0.0) as u64
}

fn double_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let model = match v.choice("model") {
        1 => LotModel::Poisson,
        _ => LotModel::Binomial,
    };
    let plan = DoublePlan {
        n1: int_field(v, "n1"),
        c1: int_field(v, "c1"),
        r1: int_field(v, "r1"),
        n2: int_field(v, "n2"),
        c2: int_field(v, "c2"),
    };
    let lot = lot_of(v)?;
    out(double_report(&plan, model, lot, optional_level(v, "aql"), optional_level(v, "rql"))?)
}

fn attr_decision_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let plan = AttributePlan { n: int_field(v, "n"), c: int_field(v, "c") };
    if plan.n == 0 {
        return Err("the sample size must be at least 1".into());
    }
    let d = int_field(v, "defectives");
    if d > plan.n {
        return Err("the defectives found cannot exceed the sample size".into());
    }
    out(attribute_decision_section(&plan, d))
}

// ------------------------------------------------------------------------ variables sampling

fn side_of(v: &Values) -> LimitSide {
    if v.choice("side") == 1 {
        LimitSide::Lower
    } else {
        LimitSide::Upper
    }
}

fn var_create_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let (aql, rql, al, be) = (v.num("aql"), v.num("rql"), v.num("alpha"), v.num("beta"));
    let known = v.choice("sigma") == 1;
    let found = find_variables_plan(aql, rql, al, be, known)?;
    let mut s = variables_report(&[("Plan".to_string(), found.plan)], side_of(v), Some(aql), Some(rql))?;
    s.heading = "Create Acceptance Sampling Plan (Variables)".into();
    s.text(format!(
        "Sample {} items; accept the lot when the quality index is at least k = {:.6}. Probability of accepting at the \
         AQL: {:.4}; at the RQL: {:.4} (consumer's risk, at most {:.4}).",
        found.plan.n, found.plan.k, found.pa_aql, found.pa_rql, be
    ));
    out(s)
}

fn var_compare_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let known = v.choice("sigma") == 1;
    let ns = parse_list(v.text("ns"), "the sample sizes")?;
    let ks = parse_list(v.text("ks"), "the acceptability constants")?;
    if ns.len() != ks.len() {
        return Err("give one acceptability constant for each sample size".into());
    }
    let mut plans = Vec::new();
    for (n, k) in ns.iter().zip(&ks) {
        if *n < 2.0 || n.fract() != 0.0 {
            return Err("sample sizes must be whole numbers of at least 2".into());
        }
        plans.push((format!("n={n}, k={k}"), VariablesPlan { n: *n as u64, k: *k, sigma_known: known }));
    }
    let mut s = variables_report(&plans, side_of(v), optional_level(v, "aql"), optional_level(v, "rql"))?;
    s.heading = "Compare Acceptance Sampling Plans (Variables)".into();
    out(s)
}

fn var_decision_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select the measurement column")?;
    let data = numeric(ws, c)?;
    let n = data.iter().filter(|x| !x.is_nan()).count() as u64;
    let known = v.choice("sigma") == 1;
    let plan = VariablesPlan { n, k: v.num("k"), sigma_known: known };
    let limit = v.num("limit");
    if !limit.is_finite() {
        return Err("enter the specification limit".into());
    }
    let sigma = Some(v.num("sigma_value")).filter(|x| x.is_finite());
    let d = decide_variables(&plan, &data, limit, side_of(v), sigma)?;
    let mut s = variables_decision_section(&d, side_of(v), limit);
    s.heading = format!("Accept/Reject Based on Variables: {}", col_title(ws, c));
    out(s)
}

// ----------------------------------------------------------------------------- multi-vari

fn multivari_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("y").ok_or("select the response column")?;
    let fs = v.vars("factors");
    if fs.is_empty() || fs.len() > 4 {
        return Err("choose between 1 and 4 factors".into());
    }
    if fs.contains(&y) {
        return Err("the response cannot also be a factor".into());
    }
    let mut cols = Vec::new();
    let mut names = Vec::new();
    let mut lv = Vec::new();
    for &f in fs {
        cols.push(labels(ws, f)?);
        names.push(col_title(ws, f));
        lv.push(levels(ws, f)?);
    }
    let r = multi_vari(&numeric(ws, y)?, &cols, &names, &lv)?;
    out(r.section(&col_title(ws, y)))
}

// ------------------------------------------------------------------------------- run chart

fn run_chart_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let size = v.count("size").max(1);
    let stat = if v.choice("stat") == 1 { SubgroupStat::Median } else { SubgroupStat::Mean };
    let data = numeric(ws, c)?;
    let pts = if size > 1 { subgroup_points(&data, size, stat)? } else { data };
    let (center, label) = match v.choice("center") {
        1 => (Center::Mean, "Mean".to_string()),
        2 => {
            let x = v.num("center_value");
            if !x.is_finite() {
                return Err("enter the value of the center line".into());
            }
            (Center::Value(x), "Center".to_string())
        }
        _ => (Center::Median, "Median".to_string()),
    };
    let r = run_chart(&pts, center)?;
    out(r.section(&col_title(ws, c), &label))
}

// ----------------------------------------------------------------------------- gage run chart

fn gage_run_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (m, p) =
        (v.var("measurement").ok_or("select the measurement column")?, v.var("part").ok_or("select the part column")?);
    let y = numeric(ws, m)?;
    let part = labels(ws, p)?;
    let plev = levels(ws, p)?;
    let (ops, olev) = match v.var("operator") {
        Some(o) => (labels(ws, o)?, levels(ws, o)?),
        None => (vec!["All".to_string(); y.len()], vec!["All".to_string()]),
    };
    if part.len() != y.len() || ops.len() != y.len() {
        return Err("the measurement, part and operator columns must have the same length".into());
    }
    let mut series = Vec::new();
    let mut rows: Vec<Vec<Cell>> = plev.iter().map(|l| vec![Cell::text(l)]).collect();
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for o in &olev {
        let (mut px, mut py, mut mx, mut my) = (vec![], vec![], vec![], vec![]);
        for (pi, pl) in plev.iter().enumerate() {
            let vals: Vec<f64> =
                (0..y.len()).filter(|&i| !y[i].is_nan() && part[i] == *pl && ops[i] == *o).map(|i| y[i]).collect();
            for &x in &vals {
                px.push((pi + 1) as f64);
                py.push(x);
                sum += x;
                cnt += 1;
            }
            if vals.is_empty() {
                rows[pi].push(Cell::Missing);
            } else {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                mx.push((pi + 1) as f64);
                my.push(mean);
                rows[pi].push(Cell::Num(mean));
            }
        }
        series.push(Series { name: o.clone(), x: px, y: py });
        series.push(Series { name: format!("Fit: {o}"), x: mx, y: my });
    }
    if cnt == 0 {
        return Err("there are no measurements to plot".into());
    }
    let mut heads: Vec<(String, TKind)> = vec![("Part".into(), TKind::Text)];
    heads.extend(olev.iter().map(|o| (format!("Mean, {o}"), TKind::Significant(6))));
    let mut t = Table::new(
        "Mean Measurement by Part and Operator",
        heads.iter().map(|(h, k)| TCol::new(h.clone(), *k)).collect(),
    );
    for r in rows {
        t.push_row(r)?;
    }
    let mut s = Section::new(format!("Gage Run Chart of {}", col_title(ws, m)));
    s.table(t);
    s.chart(ChartSpec {
        kind: ChartKind::Scatter,
        title: format!("Gage Run Chart of {} by {}", col_title(ws, m), col_title(ws, p)),
        x_label: "Part (in the order of the table)".into(),
        y_label: col_title(ws, m),
        categories: Vec::new(),
        series,
        reference_lines: vec![ReferenceLine { axis: Axis::Y, value: sum / cnt as f64, label: "Mean".into() }],
    });
    out(s)
}

// ----------------------------------------------------------------------- attribute capability

fn binomial_cap_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("defectives").ok_or("select the defectives column")?;
    let d = numeric(ws, c)?;
    let sizes = match v.var("size_col") {
        Some(sc) => numeric(ws, sc)?,
        None => {
            let n = v.num("size");
            if !n.is_finite() {
                return Err("enter a sample size or choose a column of sample sizes".into());
            }
            vec![n; d.len()]
        }
    };
    let target = optional_level(v, "target").map(|t| t / 100.0);
    let cap = binomial_capability(&d, &sizes, level_pct(v)?, target)?;
    out(cap.section(&col_title(ws, c)))
}

fn poisson_cap_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("defects").ok_or("select the defects column")?;
    let d = numeric(ws, c)?;
    let units = match v.var("units_col") {
        Some(uc) => numeric(ws, uc)?,
        None => vec![v.num("units"); d.len()],
    };
    let cap = poisson_capability(&d, &units, level_pct(v)?, optional_level(v, "target"))?;
    out(cap.section(&col_title(ws, c)))
}

// ------------------------------------------------------------------ multi-variable capability

fn spec_list(text: &str, k: usize, what: &str) -> Res<Vec<Option<f64>>> {
    let toks: Vec<&str> =
        text.split(|c: char| c.is_whitespace() || c == ',' || c == ';').filter(|t| !t.is_empty()).collect();
    if toks.is_empty() {
        return Ok(vec![None; k]);
    }
    let mut vals = Vec::new();
    for t in &toks {
        vals.push(if *t == "*" {
            None
        } else {
            Some(t.parse::<f64>().map_err(|_| format!("'{t}' in {what} is not a number"))?)
        });
    }
    if vals.len() == 1 {
        return Ok(vec![vals[0]; k]);
    }
    if vals.len() != k {
        return Err(format!("give one {what} for every variable (or a single value for all of them)").into());
    }
    Ok(vals)
}

fn multi_cap_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let vars = v.vars("vars");
    if vars.is_empty() {
        return Err("select at least one variable".into());
    }
    let lsl = spec_list(v.text("lsl"), vars.len(), "lower specification")?;
    let usl = spec_list(v.text("usl"), vars.len(), "upper specification")?;
    let size = v.count("size").max(1);
    let opts = CapabilityOptions { level: level_pct(v)?, ..Default::default() };
    let mut t = Table::new(
        "Capability Summary",
        tcols(&[
            ("Variable", TKind::Text),
            ("N", TKind::Integer),
            ("Mean", TKind::Significant(6)),
            ("StDev (Within)", TKind::Significant(6)),
            ("StDev (Overall)", TKind::Significant(6)),
            ("Cp", TKind::Significant(4)),
            ("Cpk", TKind::Significant(4)),
            ("Pp", TKind::Significant(4)),
            ("Ppk", TKind::Significant(4)),
            ("Z.Bench", TKind::Significant(4)),
            ("PPM Total (Overall)", TKind::Significant(6)),
        ]),
    );
    let mut notes = Vec::new();
    for (i, &c) in vars.iter().enumerate() {
        let specs = Specs { lsl: lsl[i], target: None, usl: usl[i] };
        if specs.lsl.is_none() && specs.usl.is_none() {
            notes.push(format!("{}: no specification limits, skipped.", col_title(ws, c)));
            continue;
        }
        let data = numeric(ws, c)?;
        let cap = if size > 1 {
            let groups: Vec<Vec<f64>> = data
                .chunks(size)
                .map(|ch| ch.iter().copied().filter(|x| !x.is_nan()).collect::<Vec<f64>>())
                .filter(|g| !g.is_empty())
                .collect();
            capability_subgroups(&groups, &specs, &opts)
        } else {
            capability_individuals(&data, &specs, &opts)
        };
        match cap {
            Ok(cap) => t.push_row(vec![
                Cell::text(col_title(ws, c)),
                Cell::Int(cap.n as i64),
                Cell::Num(cap.mean),
                Cell::Num(cap.sd_within),
                Cell::Num(cap.sd_overall),
                cap.cp.map_or(Cell::Missing, Cell::Num),
                Cell::Num(cap.cpk),
                cap.pp.map_or(Cell::Missing, Cell::Num),
                Cell::Num(cap.ppk),
                Cell::Num(cap.expected_overall.z_bench),
                Cell::Num(cap.expected_overall.ppm_total),
            ])?,
            Err(e) => notes.push(format!("{}: {e}", col_title(ws, c))),
        }
    }
    if t.rows.is_empty() {
        return Err(format!("no variable could be analysed. {}", notes.join(" ")).into());
    }
    let mut s = Section::new("Process Capability of Several Variables (Normal)");
    s.table(t);
    for n in notes {
        s.warn(n);
    }
    out(s)
}

// ---------------------------------------------------------------------------- transformations

fn hist_chart(data: &[f64], title: String, xl: &str) -> Res<ChartSpec> {
    let h = histogram(data, &Binning::Sturges)?;
    Ok(ChartSpec {
        kind: ChartKind::Histogram,
        title,
        x_label: xl.into(),
        y_label: "Frequency".into(),
        categories: Vec::new(),
        series: vec![Series { name: xl.into(), x: h.edges.clone(), y: h.counts.iter().map(|&c| c as f64).collect() }],
        reference_lines: Vec::new(),
    })
}

fn ad_row(label: &str, data: &[f64]) -> Res<Vec<Cell>> {
    let ad = anderson_darling(data)?;
    Ok(vec![Cell::text(label), Cell::Num(ad.a2), Cell::Num(ad.p_value)])
}

fn johnson_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let name = col_title(ws, c);
    let data = numeric(ws, c)?;
    let valid: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let fit = fit_johnson(&valid)?;
    let transformed: Vec<f64> = data.iter().map(|&x| if x.is_nan() { f64::NAN } else { fit.forward(x) }).collect();
    let tv: Vec<f64> = transformed.iter().copied().filter(|x| x.is_finite()).collect();
    let d = &fit.dist;
    let mut s = Section::new(format!("Johnson Transformation of {name}"));
    let mut p = Table::new(
        "Johnson Transformation Function",
        tcols(&[
            ("Family", TKind::Text),
            ("Gamma", TKind::Significant(6)),
            ("Delta", TKind::Significant(6)),
            ("Xi", TKind::Significant(6)),
            ("Lambda", TKind::Significant(6)),
            ("Z", TKind::Significant(4)),
        ]),
    );
    p.push_row(vec![
        Cell::text(format!("{:?}", d.family)),
        Cell::Num(d.gamma),
        Cell::Num(d.delta),
        Cell::Num(d.xi),
        Cell::Num(d.lambda),
        Cell::Num(fit.z),
    ])?;
    s.table(p);
    let mut ad = Table::new(
        "Anderson-Darling Normality Test",
        tcols(&[("Data", TKind::Text), ("A-Squared", TKind::Significant(6)), ("P-Value", TKind::PValue)]),
    );
    ad.push_row(ad_row("Original", &valid)?)?;
    ad.push_row(ad_row("Transformed", &tv)?)?;
    s.table(ad);
    let (lsl, usl) = (optional_level(v, "lsl"), optional_level(v, "usl"));
    if lsl.is_some() || usl.is_some() {
        let mut t = Table::new(
            "Transformed Specification Limits",
            tcols(&[
                ("Limit", TKind::Text),
                ("Original", TKind::Significant(6)),
                ("Transformed", TKind::Significant(6)),
            ]),
        );
        for (lab, x) in [("Lower", lsl), ("Upper", usl)] {
            if let Some(x) = x {
                t.push_row(vec![Cell::text(lab), Cell::Num(x), Cell::Num(fit.forward(x))])?;
            }
        }
        s.table(t);
    }
    if tv.len() != valid.len() {
        s.warn(format!(
            "{} value(s) fall outside the support of the fitted curve and were left missing.",
            valid.len() - tv.len()
        ));
    }
    s.chart(hist_chart(&valid, format!("Histogram of {name} (original)"), &name)?);
    s.chart(hist_chart(&tv, format!("Histogram of {name} (transformed)"), &format!("{name} (Johnson)"))?);
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric(format!("{name}_J"), transformed));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn boxcox_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let name = col_title(ws, c);
    let data = numeric(ws, c)?;
    let valid: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    let choice = match v.choice("method") {
        1 => LambdaChoice::Rounded,
        2 => {
            let l = v.num("lambda");
            if !l.is_finite() {
                return Err("enter the value of lambda".into());
            }
            LambdaChoice::Fixed(l)
        }
        _ => LambdaChoice::Optimal,
    };
    let level = level_pct(v)?;
    let fit = BoxCox::fit(&valid, choice)?;
    let (lo, hi) = box_cox_interval(&valid, level)?;
    let transformed: Vec<f64> = data.iter().map(|&x| if x.is_nan() { f64::NAN } else { fit.forward(x) }).collect();
    let tv: Vec<f64> = transformed.iter().copied().filter(|x| x.is_finite()).collect();
    let mut s = Section::new(format!("Box-Cox Transformation of {name}"));
    let mut t = Table::new(
        format!("Lambda ({:.1}% confidence interval)", level * 100.0),
        tcols(&[
            ("Lambda Used", TKind::Significant(6)),
            ("Optimal Lambda", TKind::Significant(6)),
            ("Lower Limit", TKind::Significant(6)),
            ("Upper Limit", TKind::Significant(6)),
            ("Log-Likelihood", TKind::Significant(8)),
        ]),
    );
    t.push_row(vec![
        Cell::Num(fit.lambda),
        Cell::Num(fit.optimal_lambda),
        Cell::Num(lo),
        Cell::Num(hi),
        Cell::Num(fit.log_likelihood),
    ])?;
    s.table(t);
    let mut ad = Table::new(
        "Anderson-Darling Normality Test",
        tcols(&[("Data", TKind::Text), ("A-Squared", TKind::Significant(6)), ("P-Value", TKind::PValue)]),
    );
    ad.push_row(ad_row("Original", &valid)?)?;
    ad.push_row(ad_row("Transformed", &tv)?)?;
    s.table(ad);
    let grid: Vec<f64> = (0..=100).map(|i| -5.0 + 0.1 * i as f64).collect();
    let ll = box_cox_profile(&valid, &grid)?;
    s.chart(ChartSpec {
        kind: ChartKind::Line,
        title: format!("Box-Cox Plot of {name}"),
        x_label: "Lambda".into(),
        y_label: "Log-likelihood".into(),
        categories: Vec::new(),
        series: vec![Series { name: "Log-likelihood".into(), x: grid, y: ll }],
        reference_lines: vec![
            ReferenceLine { axis: Axis::X, value: lo, label: "Lower".into() },
            ReferenceLine { axis: Axis::X, value: hi, label: "Upper".into() },
            ReferenceLine { axis: Axis::X, value: fit.lambda, label: "Lambda".into() },
        ],
    });
    s.chart(hist_chart(&valid, format!("Histogram of {name} (original)"), &name)?);
    s.chart(hist_chart(&tv, format!("Histogram of {name} (transformed)"), &format!("{name} (Box-Cox)"))?);
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric(format!("{name}_BC"), transformed));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// --------------------------------------------------------------------------- registration

fn risks(spec: DialogSpec) -> DialogSpec {
    spec.number(Main, "alpha", "Producer's risk (alpha)", 0.05, 0.0001, 0.9999).number(
        Main,
        "beta",
        "Consumer's risk (beta)",
        0.10,
        0.0001,
        0.9999,
    )
}

pub(crate) fn register(r: &mut Registry) {
    let attr = |leaf: &'static str| -> Vec<&'static str> {
        vec!["Stat", "Quality Tools", "Acceptance Sampling by Attributes", leaf]
    };
    let vari = |leaf: &'static str| -> Vec<&'static str> {
        vec!["Stat", "Quality Tools", "Acceptance Sampling by Variables", leaf]
    };
    add(
        r,
        risks(
            DialogSpec::new(
                "quality.accept.attr_create",
                "Create Sampling Plan (Attributes)",
                "Smallest single plan with the stated producer's and consumer's risks.",
            )
            .choice(Main, "model", "Distribution of defectives", &MODELS, 0)
            .number(Main, "aql", "Acceptable quality level (AQL, proportion defective)", 0.01, 0.0, 1.0)
            .number(Main, "rql", "Rejectable quality level (RQL, proportion defective)", 0.08, 0.0, 1.0)
            .optional_number(Main, "lot", "Lot size (blank = infinite)"),
        ),
        &attr("Create Sampling Plan..."),
        Runner::Report(attr_create_run),
    );
    add(
        r,
        quality_levels(
            DialogSpec::new(
                "quality.accept.attr_compare",
                "Compare Sampling Plans (Attributes)",
                "OC, AOQ and ATI curves of one or more single plans.",
            )
            .choice(Main, "model", "Distribution of defectives", &MODELS, 0)
            .text(Main, "ns", "Sample sizes (separated by spaces)", "50 80 125")
            .text(Main, "cs", "Acceptance numbers (separated by spaces)", "1 2 3")
            .optional_number(Main, "lot", "Lot size (blank = infinite)"),
        ),
        &attr("Compare Sampling Plans..."),
        Runner::Report(attr_compare_run),
    );
    add(
        r,
        quality_levels(
            DialogSpec::new(
                "quality.accept.double",
                "Double Sampling Plan (Attributes)",
                "OC curve, average sample number, AOQ and ATI of a double sampling plan.",
            )
            .choice(Main, "model", "Distribution of defectives", &["Binomial", "Poisson"], 0)
            .integer(Main, "n1", "First sample size", 50.0, 1.0, 100000.0)
            .integer(Main, "c1", "First acceptance number", 1.0, 0.0, 100000.0)
            .integer(Main, "r1", "First rejection number", 4.0, 1.0, 100001.0)
            .integer(Main, "n2", "Second sample size", 100.0, 1.0, 100000.0)
            .integer(Main, "c2", "Cumulative acceptance number", 4.0, 0.0, 100000.0)
            .optional_number(Main, "lot", "Lot size (blank = infinite)"),
        ),
        &attr("Double Sampling Plan..."),
        Runner::Report(double_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.accept.attr_decision",
            "Accept/Reject Based on Attributes",
            "Compare the defectives found in a sample with the acceptance number.",
        )
        .integer(Main, "n", "Sample size", 65.0, 1.0, 1e9)
        .integer(Main, "c", "Acceptance number", 2.0, 0.0, 1e9)
        .integer(Main, "defectives", "Defectives found", 0.0, 0.0, 1e9),
        &attr("Accept/Reject Based on Attributes..."),
        Runner::Report(attr_decision_run),
    );
    add(
        r,
        risks(
            DialogSpec::new(
                "quality.accept.var_create",
                "Create Sampling Plan (Variables)",
                "Smallest single plan on one specification limit with the stated risks.",
            )
            .choice(Main, "side", "Specification limit", &SIDES, 0)
            .choice(Main, "sigma", "Process standard deviation", &SIGMA, 0)
            .number(Main, "aql", "Acceptable quality level (AQL, proportion nonconforming)", 0.01, 0.0, 1.0)
            .number(
                Main,
                "rql",
                "Rejectable quality level (RQL, proportion nonconforming)",
                0.08,
                0.0,
                1.0,
            ),
        ),
        &vari("Create Sampling Plan..."),
        Runner::Report(var_create_run),
    );
    add(
        r,
        quality_levels(
            DialogSpec::new(
                "quality.accept.var_compare",
                "Compare Sampling Plans (Variables)",
                "OC curves of one or more variables plans.",
            )
            .choice(Main, "side", "Specification limit", &SIDES, 0)
            .choice(Main, "sigma", "Process standard deviation", &SIGMA, 0)
            .text(Main, "ns", "Sample sizes (separated by spaces)", "10 20")
            .text(Main, "ks", "Acceptability constants k (separated by spaces)", "1.8 1.9"),
        ),
        &vari("Compare Sampling Plans..."),
        Runner::Report(var_compare_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.accept.var_decision",
            "Accept/Reject Based on Variables",
            "Compute the quality index of a sample and compare it with k.",
        )
        .vars("var", "Measurements", Numeric, 1, 1)
        .choice(Main, "side", "Specification limit", &SIDES, 0)
        .number(Main, "limit", "Specification limit value", 0.0, f64::MIN, f64::MAX)
        .number(Main, "k", "Acceptability constant k", 1.8, f64::MIN, f64::MAX)
        .choice(Main, "sigma", "Process standard deviation", &SIGMA, 0)
        .optional_number(Main, "sigma_value", "Known standard deviation"),
        &vari("Accept/Reject Based on Variables..."),
        Runner::Report(var_decision_run),
    );
    add(
        r,
        DialogSpec::new("quality.multivari", "Multi-Vari Chart", "Observations and means across up to four factors.")
            .vars("y", "Response", Numeric, 1, 1)
            .vars("factors", "Factors (first varies slowest)", Any, 1, 4),
        &["Stat", "Quality Tools", "Multi-Vari Chart..."],
        Runner::Report(multivari_run),
    );
    add(
        r,
        DialogSpec::new("quality.runchart", "Run Chart", "Tests for clustering, mixtures, trends and oscillation.")
            .vars("var", "Observations", Numeric, 1, 1)
            .integer(Main, "size", "Subgroup size (1 = individual observations)", 1.0, 1.0, 1000.0)
            .choice(Options, "stat", "Subgroup statistic", &["Mean", "Median"], 1)
            .choice(Options, "center", "Center line", &["Median", "Mean", "Specified value"], 0)
            .optional_number(Options, "center_value", "Specified center value"),
        &["Stat", "Quality Tools", "Run Chart..."],
        Runner::Report(run_chart_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.gage_run",
            "Gage Run Chart",
            "Measurements by part, with a line through each operator's part means.",
        )
        .vars("measurement", "Measurements", Numeric, 1, 1)
        .vars("part", "Part numbers", Any, 1, 1)
        .vars("operator", "Operators (optional)", Any, 0, 1),
        &["Stat", "Quality Tools", "Gage Study", "Gage Run Chart..."],
        Runner::Report(gage_run_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.capability_binomial",
            "Binomial Capability Analysis",
            "Pooled proportion defective with its interval, Process Z and a test of a common proportion.",
        )
        .vars("defectives", "Defectives", Numeric, 1, 1)
        .vars("size_col", "Sample sizes (column; blank = constant)", Numeric, 0, 1)
        .optional_number(Main, "size", "Constant sample size")
        .optional_number(Options, "target", "Target %defective")
        .number(Options, "level", "Confidence level (%)", 95.0, 0.0, 100.0),
        &["Stat", "Quality Tools", "Capability Analysis", "Binomial..."],
        Runner::Report(binomial_cap_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.capability_poisson",
            "Poisson Capability Analysis",
            "Pooled defects per unit with its interval, Process Z and a test of a common DPU.",
        )
        .vars("defects", "Defects", Numeric, 1, 1)
        .vars("units_col", "Units inspected (column; blank = constant)", Numeric, 0, 1)
        .number(Main, "units", "Constant number of units per sample", 1.0, 0.0, 1e12)
        .optional_number(Options, "target", "Target DPU")
        .number(Options, "level", "Confidence level (%)", 95.0, 0.0, 100.0),
        &["Stat", "Quality Tools", "Capability Analysis", "Poisson..."],
        Runner::Report(poisson_cap_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.capability_multi",
            "Capability Analysis (Several Variables)",
            "Normal capability of each variable against its own specification limits.",
        )
        .vars("vars", "Variables", Numeric, 1, 50)
        .text(Main, "lsl", "Lower spec(s): one value, or one per variable; * for none", "")
        .text(Main, "usl", "Upper spec(s): one value, or one per variable; * for none", "")
        .integer(Main, "size", "Subgroup size (1 = individuals)", 1.0, 1.0, 1000.0)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.0, 100.0),
        &["Stat", "Quality Tools", "Capability Analysis", "Multiple Variables (Normal)..."],
        Runner::Report(multi_cap_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.johnson",
            "Johnson Transformation",
            "Fit the Johnson curve that makes the data most normal and transform the column.",
        )
        .vars("var", "Variable", Numeric, 1, 1)
        .optional_number(Options, "lsl", "Lower specification to transform")
        .optional_number(Options, "usl", "Upper specification to transform")
        .check(Storage, "store", "Store the transformed data", true),
        &["Stat", "Quality Tools", "Johnson Transformation..."],
        Runner::Report(johnson_run),
    );
    add(
        r,
        DialogSpec::new(
            "quality.boxcox",
            "Box-Cox Transformation",
            "Choose lambda by maximum likelihood, with its confidence interval and the Box-Cox plot.",
        )
        .vars("var", "Variable (positive values)", Numeric, 1, 1)
        .choice(Options, "method", "Lambda", &["Optimal", "Optimal, rounded to the nearest 0.5", "Specified"], 0)
        .optional_number(Options, "lambda", "Specified lambda")
        .number(Options, "level", "Confidence level (%)", 95.0, 0.0, 100.0)
        .check(Storage, "store", "Store the transformed data", true),
        &["Stat", "Control Charts", "Box-Cox Transformation..."],
        Runner::Report(boxcox_run),
    );
}
