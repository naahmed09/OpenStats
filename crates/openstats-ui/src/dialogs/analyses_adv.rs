//! Advanced analyses: general linear model, factorial-design analysis, Gage R&R and the Hotelling
//! T² chart. They follow the same pattern as `analyses`: a [`DialogSpec`] plus a report function.

use super::analyses::{add, out};
use super::analysis::{col_title, complete_rows, labels, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::Worksheet;
use openstats_stats::anova::{glm, GlmOptions};
use openstats_stats::doe::{
    analyze_factorial, cube_chart, interaction_charts, main_effects_charts, normal_effects_chart, pareto_chart, Design,
    DesignKind, FactorSpec, FactorialAnalysisOptions, PointType,
};
use openstats_stats::model::Data;
use openstats_stats::msa::{CrossedStudy, GageOptions, Interaction, Tolerance};
use openstats_stats::regression::RegressionOptions;
use openstats_stats::report::Reportable;
use openstats_stats::spc::multivariate::{
    hotelling_individuals, hotelling_subgroups, CovarianceEstimate, HotellingOptions,
};

/// A formula-safe name for column `c`: letters, digits and underscores, unique among `taken`.
pub(crate) fn ident(ws: &Worksheet, c: usize, taken: &[String]) -> String {
    let mut s: String = col_title(ws, c).chars().map(|ch| if ch.is_alphanumeric() { ch } else { '_' }).collect();
    if s.chars().next().is_none_or(|ch| !ch.is_alphabetic()) {
        s.insert(0, 'V');
    }
    let base = s.clone();
    let mut k = 2;
    while taken.contains(&s) {
        s = format!("{base}_{k}");
        k += 1;
    }
    s
}

// --------------------------------------------------------------------------- GLM

fn glm_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    if !v.vars("random").is_empty() {
        return super::analyses_anova::glm_random_run(ws, v);
    }
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    if factors.is_empty() && covs.is_empty() {
        return Err("select at least one factor or covariate".into());
    }
    if factors.iter().chain(&covs).any(|&c| c == y) {
        return Err("the response cannot also be a model term".into());
    }
    if factors.iter().any(|f| covs.contains(f)) {
        return Err("a column cannot be both a factor and a covariate".into());
    }
    let mut names: Vec<String> = Vec::new();
    let mut push = |c: usize| {
        let n = ident(ws, c, &names);
        names.push(n.clone());
        n
    };
    let yn = push(y);
    let mut data = Data::new().with_numeric(&yn, numeric(ws, y)?)?;
    let mut fnames = Vec::new();
    for &c in &factors {
        let n = push(c);
        data = data.with_factor(&n, &labels(ws, c)?)?;
        fnames.push(n);
    }
    let mut cnames = Vec::new();
    for &c in &covs {
        let n = push(c);
        data = data.with_numeric(&n, numeric(ws, c)?)?;
        cnames.push(n);
    }
    let order = v.count("order").max(1);
    let atoms: Vec<&str> = fnames.iter().chain(&cnames).map(String::as_str).collect();
    let rhs = if order > 1 && atoms.len() > 1 {
        format!("({})^{}", atoms.join(" + "), order.min(atoms.len()))
    } else {
        atoms.join(" + ")
    };
    let formula = format!("{yn} ~ {rhs}");
    let o = GlmOptions {
        regression: RegressionOptions { level: level(v)?, ..Default::default() },
        ..GlmOptions::default()
    };
    let r = glm(&formula, &data, &o)?;
    let mut s = r.report_section();
    s.heading = format!("General Linear Model: {} versus {}", col_title(ws, y), {
        let mut t: Vec<String> = factors.iter().chain(&covs).map(|&c| col_title(ws, c)).collect();
        if order > 1 && t.len() > 1 {
            t.push(format!("(interactions to order {})", order.min(t.len())));
        }
        t.join(", ")
    });
    out(s)
}

// --------------------------------------------------------------------------- DOE analysis

fn doe_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let cols = v.vars("factors").to_vec();
    if cols.contains(&y) {
        return Err("the response cannot also be a factor".into());
    }
    let resp = numeric(ws, y)?;
    let n = resp.len();
    let mut specs = Vec::new();
    let mut columns = Vec::new();
    for &c in &cols {
        let x = numeric(ws, c)?;
        if x.len() != n {
            return Err("columns have different lengths".into());
        }
        if x.iter().any(|a| a.is_nan()) {
            return Err(format!("{} has missing values", col_title(ws, c)).into());
        }
        let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &a| (l.min(a), h.max(a)));
        if hi <= lo {
            return Err(format!("{} has only one level", col_title(ws, c)).into());
        }
        specs.push(FactorSpec { name: col_title(ws, c), low: lo, high: hi });
        columns.push(x);
    }
    if resp.iter().any(|a| a.is_nan()) {
        return Err(format!("{} has missing values", col_title(ws, y)).into());
    }
    let k = cols.len();
    let coded: Vec<Vec<f64>> = (0..n).map(|i| (0..k).map(|j| specs[j].to_coded(columns[j][i])).collect()).collect();
    let mut point_type = Vec::with_capacity(n);
    for row in &coded {
        let corner = row.iter().all(|c| (c.abs() - 1.0).abs() < 1e-9);
        let centre = row.iter().all(|c| c.abs() < 1e-9);
        if corner {
            point_type.push(PointType::Factorial);
        } else if centre {
            point_type.push(PointType::Center);
        } else {
            return Err(
                "every run must be at a low or high level (or all factors at the centre): this is a two-level analysis"
                    .into(),
            );
        }
    }
    let corners = point_type.iter().filter(|p| **p == PointType::Factorial).count();
    let kind = if corners >= (1usize << k) { DesignKind::FullFactorial } else { DesignKind::FractionalFactorial };
    let design = Design {
        kind,
        factors: specs,
        coded,
        point_type,
        block: vec![1; n],
        blocks: 1,
        std_order: (1..=n).collect(),
        defining: None,
        block_words: Vec::new(),
    };
    let mut o = FactorialAnalysisOptions { level: level(v)?, ..Default::default() };
    let mo = v.count("maxorder");
    if mo > 0 {
        o.max_order = Some(mo);
    }
    let a = analyze_factorial(&design, &resp, &o)?;
    let mut s = a.report_section();
    s.heading = format!(
        "Factorial Fit: {} versus {}",
        col_title(ws, y),
        design.factors.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", ")
    );
    if v.flag("main_plot") {
        for c in main_effects_charts(&a) {
            s.chart(c);
        }
    }
    if v.flag("int_plot") && k >= 2 {
        for c in interaction_charts(&a) {
            s.chart(c);
        }
    }
    if v.flag("cube_plot") {
        if k < 3 {
            return Err("a cube plot needs at least three factors".into());
        }
        s.chart(cube_chart(&a, [0, 1, 2])?);
    }
    if v.flag("pareto_plot") {
        s.chart(pareto_chart(&a)?);
    }
    if v.flag("normal_plot") {
        s.chart(normal_effects_chart(&a));
    }
    out(s)
}

// --------------------------------------------------------------------------- MSA

fn gage_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (m, p, o) = (
        v.var("measurement").ok_or("select the measurement column")?,
        v.var("part").ok_or("select the part column")?,
        v.var("operator").ok_or("select the operator column")?,
    );
    if m == p || m == o || p == o {
        return Err("measurement, part and operator must be three different columns".into());
    }
    let study = CrossedStudy::from_stacked(&numeric(ws, m)?, &labels(ws, p)?, &labels(ws, o)?)?;
    let tol = v.num("tolerance");
    let opts = GageOptions {
        study_sigma: v.num("sigma"),
        tolerance: tol.is_finite().then_some(Tolerance::Width(tol)),
        alpha_interaction: v.num("alpha") / 100.0,
        interaction: match v.choice("interaction") {
            1 => Interaction::Keep,
            2 => Interaction::Drop,
            _ => Interaction::Auto,
        },
        operator_fixed: v.flag("op_fixed"),
    };
    let r = if v.choice("method") == 0 { study.anova(&opts)? } else { study.xbar_r(&opts)? };
    let mut s = r.report_section();
    s.heading =
        format!("Gage R&R Study ({}): {}", if v.choice("method") == 0 { "ANOVA" } else { "Xbar-R" }, col_title(ws, m));
    out(s)
}

// --------------------------------------------------------------------------- Multivariate

fn hotelling_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars").to_vec();
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let refs: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let (_, kept) = complete_rows(&refs);
    let n = kept.first().map_or(0, Vec::len);
    let rows: Vec<Vec<f64>> = (0..n).map(|i| kept.iter().map(|c| c[i]).collect()).collect();
    let size = v.count("size").max(1);
    let mut o = HotellingOptions { alpha: v.num("alpha"), ..Default::default() };
    if v.choice("cov") == 1 {
        o.covariance = CovarianceEstimate::SuccessiveDifferences;
    }
    let chart = if size <= 1 {
        hotelling_individuals(&rows, &o)?
    } else {
        let groups: Vec<Vec<Vec<f64>>> = rows.chunks_exact(size).map(<[Vec<f64>]>::to_vec).collect();
        if groups.len() < 2 {
            return Err(format!("{n} complete rows make fewer than two subgroups of {size}").into());
        }
        hotelling_subgroups(&groups, &o)?
    };
    let mut s = chart.report_section();
    s.heading = format!("T² Chart of {}", cols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", "));
    out(s)
}

/// Register the advanced analyses.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Main, Options};
    use VarKind::{Any, Numeric};
    add(
        r,
        DialogSpec::new(
            "stat.anova.glm",
            "General Linear Model",
            "Fit a linear model with categorical factors and numeric covariates (sum-to-zero coding, Type I and III sums of squares, least-squares means).",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors", Any, 0, 20)
        .vars("covariates", "Covariates", Numeric, 0, 20)
        .vars("random", "Random factors (a subset of the factors; expected mean squares and variance components)", Any, 0, 20)
        .integer(Options, "order", "Interactions up to order", 1.0, 1.0, 6.0)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Options, "restricted", "Use the restricted form of the mixed model (random factors only)", false),
        &["Stat", "ANOVA", "General Linear Model..."],
        Runner::Report(glm_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.analyze",
            "Analyze Factorial Design",
            "Effects, ANOVA, curvature and Lenth's PSE for a two-level design stored in the worksheet (one column per factor, one response).",
        )
        .vars("factors", "Factors (two levels each)", Numeric, 2, 15)
        .vars("response", "Response", Numeric, 1, 1)
        .integer(Options, "maxorder", "Highest interaction order (0 = automatic)", 0.0, 0.0, 15.0)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Page::Graphs, "main_plot", "Main effects plots", false)
        .check(Page::Graphs, "int_plot", "Interaction plots", false)
        .check(Page::Graphs, "cube_plot", "Cube plot (first three factors)", false)
        .check(Page::Graphs, "pareto_plot", "Pareto chart of effects", false)
        .check(Page::Graphs, "normal_plot", "Normal plot of effects", false),
        &["Stat", "DOE", "Factorial", "Analyze Factorial Design..."],
        Runner::Report(doe_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.quality.gage",
            "Gage R&R Study (Crossed)",
            "Repeatability and reproducibility of a measurement system from a balanced crossed study.",
        )
        .vars("measurement", "Measurement data", Numeric, 1, 1)
        .vars("part", "Part numbers", Any, 1, 1)
        .vars("operator", "Operators", Any, 1, 1)
        .choice(Main, "method", "Method of analysis", &["ANOVA", "Xbar and R"], 0)
        .number(Options, "sigma", "Study variation (number of standard deviations)", 6.0, 1.0, 10.0)
        .optional_number(Options, "tolerance", "Process tolerance (USL − LSL)")
        .number(Options, "alpha", "Interaction level (%)", 25.0, 0.1, 99.9)
        .choice(
            Options,
            "interaction",
            "Operator*Part interaction",
            &["Remove if not significant", "Keep", "Remove"],
            0,
        )
        .check(Options, "op_fixed", "Operators are a fixed factor (ANOVA method)", false),
        &["Stat", "Quality Tools", "Gage Study", "Gage R&R Study (Crossed)..."],
        Runner::Report(gage_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.multivariate.hotelling",
            "Hotelling T² Chart",
            "Multivariate control chart for two or more correlated process variables.",
        )
        .vars("vars", "Variables", Numeric, 2, 20)
        .integer(Main, "size", "Subgroup size (1 = individual observations)", 1.0, 1.0, 1000.0)
        .number(Options, "alpha", "Alpha (probability above the limit)", 0.00135, 1e-9, 0.5)
        .choice(Options, "cov", "Covariance estimate", &["Sample", "Successive differences"], 0),
        &["Stat", "Multivariate", "Hotelling T² Chart..."],
        Runner::Report(hotelling_run),
    );
}
