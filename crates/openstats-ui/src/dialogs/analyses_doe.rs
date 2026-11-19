//! Design of experiments: create fractional factorial, Plackett-Burman, central composite,
//! Box-Behnken and Taguchi designs; analyze response surface and Taguchi experiments.

use super::analyses::{add, out};
use super::analysis::{col_title, level, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::doe::response_optimizer::design_from_columns;
use openstats_stats::doe::{
    analyze_surface, analyze_taguchi, box_behnken, central_composite, contour_chart, fractional_factorial,
    full_factorial, orthogonal_array, plackett_burman, surface_chart, AlphaRule, CcdKind, CcdOptions, Design,
    FactorialOptions, OrthogonalArray, PointType, SignalToNoise, SurfaceModel, SurfaceOptions, SurfacePlot,
    TaguchiDesign,
};
use openstats_stats::report::{Reportable, Section};

const PAGE_NAMES: &str = "Factor names (blank = A, B, C...)";

/// Numbers in a box, separated by spaces, commas or semicolons.
pub(super) fn numbers(text: &str, what: &str) -> Res<Vec<f64>> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite())
                .ok_or_else(|| DialogError(format!("'{t}' in the {what} box is not a number")))
        })
        .collect()
}

pub(super) fn words(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

/// Names and natural levels typed into a creation dialog.
fn decorate(mut d: Design, v: &Values) -> Res<Design> {
    let names = words(v.text("names"));
    if !names.is_empty() {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        d = d.with_names(&refs)?;
    }
    let lows = numbers(v.text("lows"), "low levels")?;
    let highs = numbers(v.text("highs"), "high levels")?;
    if !lows.is_empty() || !highs.is_empty() {
        if lows.len() != d.n_factors() || highs.len() != d.n_factors() {
            return Err(format!(
                "give {} low and {} high levels (one pair per factor); got {} and {}",
                d.n_factors(),
                d.n_factors(),
                lows.len(),
                highs.len()
            )
            .into());
        }
        let pairs: Vec<(f64, f64)> = lows.into_iter().zip(highs).collect();
        d = d.with_levels(&pairs)?;
    }
    Ok(d)
}

/// The design as worksheet columns.
fn design_columns(d: &Design, natural: bool) -> Vec<Column> {
    let n = d.n_runs();
    let mut cols = vec![
        Column::numeric("StdOrder", d.std_order.iter().map(|&x| x as f64).collect()),
        Column::numeric("RunOrder", (1..=n).map(|x| x as f64).collect()),
        Column::numeric("Blocks", d.block.iter().map(|&x| x as f64).collect()),
        Column::numeric("PtType", d.point_type.iter().map(|p| p.code() as f64).collect()),
    ];
    let nat = d.natural();
    for (j, f) in d.factors.iter().enumerate() {
        let vals = if natural { nat.iter().map(|r| r[j]).collect() } else { d.column(j) };
        cols.push(Column::numeric(f.name.clone(), vals));
    }
    cols
}

fn design_output(mut s: Section, d: &Design, v: &Values) -> Res<Output> {
    s.heading = format!("{}: {} factors, {} runs", d.kind.name(), d.n_factors(), d.n_runs());
    let stored = if v.flag("store") { design_columns(d, v.flag("natural")) } else { Vec::new() };
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn factorial_options(v: &Values) -> Res<FactorialOptions> {
    let mut o = FactorialOptions {
        replicates: v.count("replicates").max(1),
        center_points: v.count("center"),
        block_generators: words(v.text("blocks")),
        ..Default::default()
    };
    if v.flag("randomize") {
        o.randomize = Some(v.count("seed") as u64);
    }
    Ok(o)
}

pub(super) fn factorial_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let k = v.count("factors");
    let d = decorate(full_factorial(k, &factorial_options(v)?)?, v)?;
    let s = d.report_section();
    design_output(s, &d, v)
}

/// Fields shared by the creation dialogs, after the design-specific ones.
pub(super) fn creation_fields(spec: DialogSpec) -> DialogSpec {
    use Page::{Options, Storage};
    spec.text(Options, "names", PAGE_NAMES, "")
        .text(Options, "lows", "Low levels (natural units, one per factor)", "")
        .text(Options, "highs", "High levels (natural units, one per factor)", "")
        .check(Options, "randomize", "Randomize runs", false)
        .integer(Options, "seed", "Base for random data", 1.0, 0.0, 1e9)
        .check(Storage, "store", "Store design in worksheet", true)
        .check(Storage, "natural", "Store natural units instead of coded levels", false)
}

pub(super) fn extend_factorial(spec: DialogSpec) -> DialogSpec {
    use Page::Options;
    let spec = spec.text(Options, "blocks", "Block generators (words such as ABC, separated by spaces)", "");
    creation_fields_extra(spec)
}

/// The creation fields that the original full factorial dialog does not have yet.
fn creation_fields_extra(spec: DialogSpec) -> DialogSpec {
    use Page::{Options, Storage};
    spec.text(Options, "names", PAGE_NAMES, "")
        .text(Options, "lows", "Low levels (natural units, one per factor)", "")
        .text(Options, "highs", "High levels (natural units, one per factor)", "")
        .check(Storage, "natural", "Store natural units instead of coded levels", false)
}

fn fractional_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let k = v.count("factors");
    let p = v.count("fraction");
    let gens = words(v.text("generators"));
    let refs: Vec<&str> = gens.iter().map(String::as_str).collect();
    let g = if refs.is_empty() { None } else { Some(refs.as_slice()) };
    let d = decorate(fractional_factorial(k, p, g, &factorial_options(v)?)?, v)?;
    let mut s = d.report_section();
    let order = v.count("alias_order");
    if order >= 2 {
        let a = d.alias_structure(order)?;
        let extra = a.report_section();
        s.blocks.extend(extra.blocks);
        s.warnings.extend(extra.warnings);
    }
    design_output(s, &d, v)
}

fn pb_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let d = decorate(plackett_burman(v.count("runs"), v.count("factors"), &factorial_options(v)?)?, v)?;
    let s = d.report_section();
    design_output(s, &d, v)
}

fn ccd_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let k = v.count("factors");
    let kind = match v.choice("kind") {
        1 => CcdKind::Inscribed,
        2 => CcdKind::FaceCentered,
        _ => CcdKind::Circumscribed,
    };
    let mut o = CcdOptions::new(kind);
    o.alpha = match v.choice("alpha") {
        1 => AlphaRule::Orthogonal,
        2 => {
            let a = v.num("alpha_value");
            if !(a.is_finite() && a > 0.0) {
                return Err("enter a positive value for alpha".into());
            }
            AlphaRule::Custom(a)
        }
        _ => AlphaRule::Rotatable,
    };
    o.fraction_p = v.count("fraction");
    o.center_factorial = v.count("center_cube");
    o.center_axial = v.count("center_axial");
    o.blocked = v.flag("blocked");
    if v.flag("randomize") {
        o.randomize = Some(v.count("seed") as u64);
    }
    let d = decorate(central_composite(k, &o)?, v)?;
    let s = d.report_section();
    design_output(s, &d, v)
}

fn bb_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let rnd = v.flag("randomize").then(|| v.count("seed") as u64);
    let d = decorate(box_behnken(v.count("factors"), v.count("center"), rnd)?, v)?;
    let s = d.report_section();
    design_output(s, &d, v)
}

fn taguchi_levels(text: &str) -> Res<Vec<usize>> {
    numbers(text, "levels")?
        .into_iter()
        .map(|x| {
            if x.fract() == 0.0 && (2.0..=9.0).contains(&x) {
                Ok(x as usize)
            } else {
                Err(DialogError(format!("{x} is not a number of levels between 2 and 9")))
            }
        })
        .collect()
}

fn taguchi_create(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let array = orthogonal_array(
        openstats_stats::doe::available_arrays().get(v.choice("array")).ok_or("choose an orthogonal array")?,
    )?;
    let levels = taguchi_levels(v.text("levels"))?;
    if levels.is_empty() {
        return Err("give the number of levels of each factor".into());
    }
    let names = words(v.text("names"));
    let names: Vec<String> = if names.is_empty() {
        (0..levels.len()).map(|i| openstats_stats::doe::design::letter(i).to_string()).collect()
    } else if names.len() == levels.len() {
        names
    } else {
        return Err(format!("{} names for {} factors", names.len(), levels.len()).into());
    };
    let factors: Vec<(&str, usize)> = names.iter().map(String::as_str).zip(levels).collect();
    let d = TaguchiDesign::new(array, &factors)?;
    let mut s = Section::new(format!("Taguchi Design: {}", d.array.name));
    s.text(format!(
        "Array {}: {} runs. Factors: {}.",
        d.array.name,
        d.runs(),
        names.iter().zip(&d.n_levels).map(|(n, l)| format!("{n} ({l} levels)")).collect::<Vec<_>>().join(", ")
    ));
    let stored = if v.flag("store") {
        let mut cols = vec![Column::numeric("RunOrder", (1..=d.runs()).map(|x| x as f64).collect())];
        for (j, n) in names.iter().enumerate() {
            cols.push(Column::numeric(n.clone(), d.levels.iter().map(|r| r[j] as f64).collect()));
        }
        cols
    } else {
        Vec::new()
    };
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ---------------------------------------------------------------------------------- analysis

fn factor_data(ws: &Worksheet, cols: &[usize], resp: &[f64]) -> Res<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for &c in cols {
        let x = numeric(ws, c)?;
        if x.len() != resp.len() {
            return Err("columns have different lengths".into());
        }
        if x.iter().any(|a| a.is_nan()) {
            return Err(format!("{} has missing values", col_title(ws, c)).into());
        }
        out.push(x);
    }
    Ok(out)
}

fn surface_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let cols = v.vars("factors").to_vec();
    if cols.contains(&y) {
        return Err("the response cannot also be a factor".into());
    }
    let resp = numeric(ws, y)?;
    if resp.iter().any(|a| a.is_nan()) {
        return Err(format!("{} has missing values", col_title(ws, y)).into());
    }
    let data = factor_data(ws, &cols, &resp)?;
    let names: Vec<String> = cols.iter().map(|&c| col_title(ws, c)).collect();
    let design = design_from_columns(&names, &data)?;
    let model = match v.choice("model") {
        0 => SurfaceModel::Linear,
        1 => SurfaceModel::LinearInteractions,
        _ => SurfaceModel::FullQuadratic,
    };
    let a = analyze_surface(&design, &resp, &SurfaceOptions { model, level: level(v)? })?;
    let mut s = a.report_section();
    s.heading = format!("Response Surface Regression: {} versus {}", col_title(ws, y), names.join(", "));
    if v.flag("contour") || v.flag("surface") {
        let k = design.n_factors();
        let (ix, iy) = (v.count("plot_x").saturating_sub(1), v.count("plot_y").saturating_sub(1));
        if ix >= k || iy >= k || ix == iy {
            return Err(format!("the plot factors must be two different numbers between 1 and {k}").into());
        }
        let fixed = vec![0.0; k];
        let p = SurfacePlot {
            model: &a.model,
            factors: &design.factors,
            ix,
            iy,
            fixed: &fixed,
            range: ([-1.0, 1.0], [-1.0, 1.0]),
            grid_points: 25,
        };
        if v.flag("contour") {
            s.chart(contour_chart(&p, 8)?);
        }
        if v.flag("surface") {
            s.chart(surface_chart(&p)?);
        }
    }
    let stored = if v.flag("store_fits") {
        let fits: Vec<f64> = design.coded.iter().map(|r| a.model.predict(r)).collect();
        let resid: Vec<f64> = resp.iter().zip(&fits).map(|(y, f)| y - f).collect();
        vec![Column::numeric("RSFITS", fits), Column::numeric("RSRESI", resid)]
    } else {
        Vec::new()
    };
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn sn_type(i: usize) -> SignalToNoise {
    match i {
        1 => SignalToNoise::LargerIsBetter,
        2 => SignalToNoise::NominalIsBest,
        3 => SignalToNoise::NominalIsBestVariance,
        _ => SignalToNoise::SmallerIsBetter,
    }
}

fn taguchi_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let resp_cols = v.vars("responses").to_vec();
    let fcols = v.vars("factors").to_vec();
    if resp_cols.iter().any(|c| fcols.contains(c)) {
        return Err("a response column cannot also be a factor".into());
    }
    let n = numeric(ws, resp_cols[0])?.len();
    let resp: Vec<Vec<f64>> = resp_cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    if resp.iter().any(|r| r.len() != n) {
        return Err("columns have different lengths".into());
    }
    let mut levels = vec![vec![0usize; fcols.len()]; n];
    let mut n_levels = Vec::new();
    for (j, &c) in fcols.iter().enumerate() {
        let x = numeric(ws, c)?;
        if x.len() != n {
            return Err("columns have different lengths".into());
        }
        let mut top = 0;
        for (i, &a) in x.iter().enumerate() {
            if !(a.is_finite() && a >= 1.0 && a.fract() == 0.0) {
                return Err(format!("{} must hold level numbers 1, 2, 3, ... (row {})", col_title(ws, c), i + 1).into());
            }
            levels[i][j] = a as usize;
            top = top.max(a as usize);
        }
        n_levels.push(top);
    }
    let array = OrthogonalArray {
        name: "Worksheet design".into(),
        levels: n_levels.clone(),
        data: levels.iter().map(|r| r.iter().map(|l| l - 1).collect()).collect(),
    };
    let design = TaguchiDesign {
        array,
        factors: fcols.iter().map(|&c| col_title(ws, c)).collect(),
        columns: (0..fcols.len()).collect(),
        levels,
        n_levels,
    };
    let rows: Vec<Vec<f64>> = (0..n).map(|i| resp.iter().map(|r| r[i]).collect()).collect();
    if rows.iter().flatten().any(|x| x.is_nan()) {
        return Err("the response columns have missing values".into());
    }
    let sn = sn_type(v.choice("sn"));
    let a = analyze_taguchi(&design, &rows, sn)?;
    let mut s = a.report_section();
    s.heading = format!(
        "Taguchi Analysis: {} versus {}",
        resp_cols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", "),
        design.factors.join(", ")
    );
    if v.flag("plots") {
        for c in a.charts() {
            s.chart(c);
        }
    }
    let stored = if v.flag("store_sn") {
        let mut cols = vec![Column::numeric("SNRA", a.run_sn.clone()), Column::numeric("MEAN", a.run_mean.clone())];
        if a.run_sd.iter().all(Option::is_some) {
            cols.push(Column::numeric("STDE", a.run_sd.iter().map(|s| s.unwrap_or(f64::NAN)).collect()));
        }
        cols
    } else {
        Vec::new()
    };
    Ok(Output { section: s, stored, models: Vec::new() })
}

/// Register the DOE dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Main, Options, Storage};
    use VarKind::Numeric;
    let _ = out;
    add(
        r,
        creation_fields(
            DialogSpec::new(
                "stat.doe.fractional",
                "Create Fractional Factorial Design",
                "Two-level 2^(k-p) fraction with chosen or automatic generators, blocks, centre points and the alias structure.",
            )
            .integer(Main, "factors", "Number of factors", 5.0, 3.0, 15.0)
            .integer(Main, "fraction", "Fraction exponent p (2^(k-p) runs)", 1.0, 1.0, 10.0)
            .text(Options, "generators", "Generators (such as D=ABC E=ABD; blank = automatic)", "")
            .integer(Main, "replicates", "Number of replicates", 1.0, 1.0, 100.0)
            .integer(Main, "center", "Center points per block", 0.0, 0.0, 100.0)
            .text(Options, "blocks", "Block generators (words such as ABC)", "")
            .integer(Options, "alias_order", "Show aliases up to interaction order (0 = none)", 3.0, 0.0, 6.0),
        ),
        &["Stat", "DOE", "Factorial", "Create Fractional Factorial Design..."],
        Runner::Report(fractional_run),
    );
    add(
        r,
        creation_fields(
            DialogSpec::new(
                "stat.doe.pb",
                "Create Plackett-Burman Design",
                "Screening design from a Hadamard matrix: 4 to 48 runs (multiple of 4).",
            )
            .integer(Main, "runs", "Number of runs", 12.0, 4.0, 48.0)
            .integer(Main, "factors", "Number of factors", 7.0, 2.0, 47.0)
            .integer(Main, "replicates", "Number of replicates", 1.0, 1.0, 100.0)
            .integer(Main, "center", "Center points", 0.0, 0.0, 100.0),
        ),
        &["Stat", "DOE", "Screening", "Create Plackett-Burman Design..."],
        Runner::Report(pb_run),
    );
    add(
        r,
        creation_fields(
            DialogSpec::new(
                "stat.doe.ccd",
                "Create Central Composite Design",
                "Response-surface design with axial points: circumscribed, inscribed or face-centered.",
            )
            .integer(Main, "factors", "Number of factors", 3.0, 2.0, 10.0)
            .choice(Main, "kind", "Type", &["Circumscribed", "Inscribed", "Face centered"], 0)
            .integer(Main, "fraction", "Cube fraction exponent p (0 = full)", 0.0, 0.0, 6.0)
            .choice(Options, "alpha", "Axial distance", &["Rotatable", "Orthogonal", "Custom"], 0)
            .optional_number(Options, "alpha_value", "Custom alpha")
            .integer(Options, "center_cube", "Center points in the cube", 4.0, 0.0, 100.0)
            .integer(Options, "center_axial", "Center points in the axial block", 2.0, 0.0, 100.0)
            .check(Options, "blocked", "Two blocks (cube and axial points)", false),
        ),
        &["Stat", "DOE", "Response Surface", "Create Central Composite Design..."],
        Runner::Report(ccd_run),
    );
    add(
        r,
        creation_fields(
            DialogSpec::new(
                "stat.doe.bb",
                "Create Box-Behnken Design",
                "Three-level response-surface design without corner runs (3, 4, 5 or 7 factors).",
            )
            .integer(Main, "factors", "Number of factors", 3.0, 3.0, 7.0)
            .integer(Main, "center", "Center points", 3.0, 0.0, 100.0),
        ),
        &["Stat", "DOE", "Response Surface", "Create Box-Behnken Design..."],
        Runner::Report(bb_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.taguchi_create",
            "Create Taguchi Design",
            "Assign factors to the columns of an orthogonal array (L4 ... L81).",
        )
        .choice(Main, "array", "Orthogonal array", openstats_stats::doe::available_arrays(), 1)
        .text(Main, "levels", "Number of levels of each factor (such as 3 3 3 3)", "2 2 2")
        .text(Options, "names", PAGE_NAMES, "")
        .check(Storage, "store", "Store design in worksheet", true),
        &["Stat", "DOE", "Taguchi", "Create Taguchi Design..."],
        Runner::Report(taguchi_create),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.surface",
            "Analyze Response Surface Design",
            "Fit a response-surface model to a worksheet design; ANOVA by term group, lack of fit, canonical analysis and optional plots.",
        )
        .vars("factors", "Factors", Numeric, 2, 10)
        .vars("response", "Response", Numeric, 1, 1)
        .choice(Options, "model", "Model", &["Linear", "Linear and interactions", "Full quadratic"], 2)
        .number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
        .check(Page::Graphs, "contour", "Contour plot", false)
        .check(Page::Graphs, "surface", "Surface plot", false)
        .integer(Page::Graphs, "plot_x", "Factor on the x axis (number in the list)", 1.0, 1.0, 10.0)
        .integer(Page::Graphs, "plot_y", "Factor on the y axis (number in the list)", 2.0, 1.0, 10.0)
        .check(Storage, "store_fits", "Fits and residuals", false),
        &["Stat", "DOE", "Response Surface", "Analyze Response Surface Design..."],
        Runner::Report(surface_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.doe.taguchi",
            "Analyze Taguchi Design",
            "Signal-to-noise and mean response tables, ANOVA and best levels from factor level numbers (1, 2, 3...) and one or more response columns.",
        )
        .vars("factors", "Factor level columns", Numeric, 1, 20)
        .vars("responses", "Responses (one column per replicate)", Numeric, 1, 20)
        .choice(
            Main,
            "sn",
            "Signal-to-noise ratio",
            &["Smaller is better", "Larger is better", "Nominal is best (mean^2/variance)", "Nominal is best (variance)"],
            0,
        )
        .check(Page::Graphs, "plots", "Main effects plots", true)
        .check(Storage, "store_sn", "Signal-to-noise ratios, means and standard deviations", false),
        &["Stat", "DOE", "Taguchi", "Analyze Taguchi Design..."],
        Runner::Report(taguchi_run),
    );
    let _ = PointType::Factorial;
}
