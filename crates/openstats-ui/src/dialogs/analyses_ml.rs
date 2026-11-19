//! Predictive analytics: classification and regression trees, random forests and gradient
//! boosting (Stat ▸ Predictive Analytics). Each dialog takes a response, continuous predictors and
//! categorical predictors; a text response (or "Classification") fits a classifier, a numeric one
//! a regression model. Missing predictor values are allowed (trees route them through surrogate
//! splits); rows with a missing response are left out.

use super::analyses::add;
use super::analysis::{col_title, labels, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::{Column, ColumnData};
use openstats_core::Worksheet;
use openstats_stats::ml::{
    expand_indicators, fit_cart, fit_forest, fit_gbdt, fit_mars, fit_mars_binary, mars_cross_validate, CartOptions,
    Criterion, CvResponse, Dataset, ForestOptions, GbdtLoss, GbdtOptions, MarsLink, MarsObject, MarsOptions,
    Predictions, PredictiveModel, Pruning, Response, SavedModel,
};
use openstats_stats::report::{Block, Reportable, Section};

/// The model inputs read from the worksheet.
pub(super) struct Inputs {
    /// Worksheet rows used (those with a response).
    pub(super) rows: Vec<usize>,
    /// Length of the worksheet columns.
    pub(super) n_total: usize,
    pub(super) ds: Dataset,
    pub(super) resp: Response,
    pub(super) response_name: String,
    pub(super) predictor_names: Vec<String>,
}

fn is_text(ws: &Worksheet, c: usize) -> bool {
    ws.column(c).is_ok_and(|col| matches!(col.data, ColumnData::Text(_)))
}

pub(super) fn gather(ws: &Worksheet, v: &Values) -> Res<Inputs> {
    let y = v.var("response").ok_or("select a response")?;
    let cont = v.vars("cont").to_vec();
    let cat = v.vars("cat").to_vec();
    if cont.is_empty() && cat.is_empty() {
        return Err("select at least one predictor".into());
    }
    if cont.iter().chain(&cat).any(|&c| c == y) {
        return Err("the response cannot also be a predictor".into());
    }
    if cont.iter().any(|c| cat.contains(c)) {
        return Err("a column cannot be both a continuous and a categorical predictor".into());
    }
    let class = match v.choice("rtype") {
        1 => false,
        2 => true,
        _ => is_text(ws, y),
    };
    let n_total = ws.column(y)?.len();
    let (rows, resp) = if class {
        let all = labels(ws, y)?;
        let rows: Vec<usize> = (0..all.len()).filter(|&i| !all[i].is_empty()).collect();
        let kept: Vec<String> = rows.iter().map(|&i| all[i].clone()).collect();
        (rows, Response::from_labels(&kept)?)
    } else {
        let all = numeric(ws, y)?;
        let rows: Vec<usize> = (0..all.len()).filter(|&i| !all[i].is_nan()).collect();
        let kept: Vec<f64> = rows.iter().map(|&i| all[i]).collect();
        (rows, Response::numeric(kept)?)
    };
    if rows.len() < 4 {
        return Err(format!("{} has only {} usable values; at least 4 are needed", col_title(ws, y), rows.len()).into());
    }
    let mut ds = Dataset::new();
    let mut predictor_names = Vec::new();
    for &c in &cont {
        let all = numeric(ws, c)?;
        let vals: Vec<f64> = rows.iter().map(|&i| all.get(i).copied().unwrap_or(f64::NAN)).collect();
        let name = col_title(ws, c);
        ds = ds.with_numeric(&name, vals)?;
        predictor_names.push(name);
    }
    for &c in &cat {
        let all = labels(ws, c)?;
        let vals: Vec<String> = rows.iter().map(|&i| all.get(i).cloned().unwrap_or_default()).collect();
        let name = col_title(ws, c);
        ds = ds.with_categorical(&name, &vals)?;
        predictor_names.push(name);
    }
    Ok(Inputs { rows, n_total, ds, resp, response_name: col_title(ws, y), predictor_names })
}

pub(super) fn seed(v: &Values) -> u64 {
    let s = v.num("seed");
    if s.is_finite() && s >= 0.0 {
        s as u64
    } else {
        1
    }
}

/// Remove the charts whose title fails `keep`.
fn filter_charts(s: &mut Section, keep: impl Fn(&str) -> bool) {
    s.blocks.retain(|b| match b {
        Block::Chart(c) => keep(&c.title),
        _ => true,
    });
}

fn heading(kind: &str, inp: &Inputs) -> String {
    format!("{kind}: {} versus {}", inp.response_name, inp.predictor_names.join(", "))
}

/// A numeric column of length `n_total` holding `values` at `rows` and missing elsewhere.
fn numeric_col(name: &str, n_total: usize, rows: &[usize], values: &[f64]) -> Column {
    super::analysis::expand(name, n_total, rows, values)
}

fn text_col(name: &str, n_total: usize, rows: &[usize], values: &[String]) -> Column {
    let mut v = vec![String::new(); n_total];
    for (&r, x) in rows.iter().zip(values) {
        v[r] = x.clone();
    }
    Column::text(name, v)
}

/// Stored columns shared by all three models, from a set of predictions.
#[allow(clippy::too_many_arguments)]
fn store_predictions(
    out: &mut Vec<Column>,
    inp: &Inputs,
    levels: &[String],
    pred: &Predictions,
    usable: Option<&[bool]>,
    fits: bool,
    resid: bool,
    prob: bool,
    prefix: &str,
) {
    let ok = |i: usize| usable.is_none_or(|u| u[i]);
    match &inp.resp {
        Response::Numeric(y) => {
            let fitted: Vec<f64> = (0..y.len()).map(|i| if ok(i) { pred.values[i] } else { f64::NAN }).collect();
            if fits {
                out.push(numeric_col(&format!("{prefix}FITS"), inp.n_total, &inp.rows, &fitted));
            }
            if resid {
                let r: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
                out.push(numeric_col(&format!("{prefix}RESI"), inp.n_total, &inp.rows, &r));
            }
        }
        Response::Class { .. } => {
            if fits {
                let cls: Vec<String> = (0..pred.classes.len())
                    .map(|i| if ok(i) { levels[pred.classes[i]].clone() } else { String::new() })
                    .collect();
                out.push(text_col(&format!("{prefix}PRED"), inp.n_total, &inp.rows, &cls));
            }
            if prob {
                for (k, l) in levels.iter().enumerate() {
                    let p: Vec<f64> = (0..pred.probabilities.len())
                        .map(|i| if ok(i) { pred.probabilities[i][k] } else { f64::NAN })
                        .collect();
                    out.push(numeric_col(&format!("{prefix}PROB_{l}"), inp.n_total, &inp.rows, &p));
                }
            }
        }
    }
}

/// The model to save in the project when the dialog names one.
fn saved(v: &Values, inp: &Inputs, model: PredictiveModel) -> Vec<(String, String)> {
    let name = v.text("save_as").trim();
    if name.is_empty() {
        return Vec::new();
    }
    let json = SavedModel { response: inp.response_name.clone(), n_train: inp.rows.len(), model }.to_json();
    vec![(name.to_string(), json)]
}

// ----------------------------------------------------------------------------- CART

fn cart_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let inp = gather(ws, v)?;
    let class = inp.resp.is_class();
    let o = CartOptions {
        criterion: class.then(|| if v.choice("criterion") == 1 { Criterion::Entropy } else { Criterion::Gini }),
        max_depth: v.count("max_depth").max(1),
        min_split: v.count("min_split").max(2),
        min_leaf: v.count("min_leaf").max(1),
        cp: v.num("cp"),
        max_surrogates: v.count("surrogates"),
        pruning: match v.choice("pruning") {
            0 => Pruning::MinCv,
            1 => Pruning::OneSe,
            2 => Pruning::None,
            3 => Pruning::Alpha(v.num("alpha")),
            _ => Pruning::Leaves(v.count("max_leaves").max(1)),
        },
        cv_folds: v.count("folds"),
        seed: seed(v),
    };
    let m = fit_cart(&inp.ds, &inp.resp, &o)?;
    let mut s = m.report_section();
    s.heading = heading(if class { "Classification Tree" } else { "Regression Tree" }, &inp);
    let (imp, cv) = (v.flag("importance_plot"), v.flag("cv_plot"));
    filter_charts(&mut s, |t| (imp || t != "Variable Importance") && (cv || !t.starts_with("Cross-Validated")));
    let mut stored = Vec::new();
    store_predictions(
        &mut stored,
        &inp,
        &m.levels,
        &m.fitted,
        None,
        v.flag("store_fits"),
        v.flag("store_resid"),
        v.flag("store_prob"),
        "",
    );
    if v.flag("store_node") {
        let node: Vec<f64> = m.terminal_node.iter().map(|&k| k as f64).collect();
        stored.push(numeric_col("NODE", inp.n_total, &inp.rows, &node));
    }
    let models = saved(v, &inp, PredictiveModel::from_cart(&m));
    Ok(Output { section: s, stored, models })
}

// ----------------------------------------------------------------------------- forest

fn forest_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let inp = gather(ws, v)?;
    let class = inp.resp.is_class();
    let positive = |name: &str| {
        let c = v.count(name);
        (c > 0).then_some(c)
    };
    let intervals = !class && v.flag("intervals");
    let o = ForestOptions {
        n_trees: v.count("trees").max(1),
        mtry: positive("mtry"),
        min_leaf: positive("min_leaf"),
        max_depth: v.count("max_depth").max(1),
        criterion: class.then(|| if v.choice("criterion") == 1 { Criterion::Entropy } else { Criterion::Gini }),
        sample_fraction: v.num("sample") / 100.0,
        importance: v.flag("perm_importance"),
        interval_level: intervals.then(|| v.num("level") / 100.0),
        seed: seed(v),
    };
    let m = fit_forest(&inp.ds, &inp.resp, &o)?;
    let mut s = m.report_section();
    s.heading = heading(if class { "Random Forest Classification" } else { "Random Forest Regression" }, &inp);
    let (imp, curve) = (v.flag("importance_plot"), v.flag("oob_plot"));
    filter_charts(&mut s, |t| (imp || t != "Variable Importance") && (curve || !t.starts_with("Out-of-Bag Error")));
    let oob = v.choice("stored") == 0;
    let (pred, usable) = if oob { (&m.oob, Some(m.has_oob.as_slice())) } else { (&m.fitted, None) };
    let mut stored = Vec::new();
    store_predictions(
        &mut stored,
        &inp,
        &m.levels,
        pred,
        usable,
        v.flag("store_fits"),
        v.flag("store_resid"),
        v.flag("store_prob"),
        if oob { "OOB_" } else { "" },
    );
    if v.flag("store_pi") && !m.oob_interval.is_empty() {
        let lo: Vec<f64> = m.oob_interval.iter().map(|p| p.0).collect();
        let hi: Vec<f64> = m.oob_interval.iter().map(|p| p.1).collect();
        stored.push(numeric_col("PI_LOWER", inp.n_total, &inp.rows, &lo));
        stored.push(numeric_col("PI_UPPER", inp.n_total, &inp.rows, &hi));
    }
    let models = saved(v, &inp, PredictiveModel::from_forest(&m));
    Ok(Output { section: s, stored, models })
}

// ----------------------------------------------------------------------------- boosting

fn gbdt_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let inp = gather(ws, v)?;
    let class = inp.resp.is_class();
    let o = GbdtOptions {
        n_trees: v.count("trees").max(1),
        learning_rate: v.num("rate"),
        max_depth: v.count("max_depth").max(1),
        min_leaf: v.count("min_leaf").max(1),
        subsample: v.num("subsample") / 100.0,
        loss: match (class, v.choice("loss")) {
            (true, _) => Some(GbdtLoss::Deviance),
            (false, 1) => Some(GbdtLoss::Huber),
            (false, _) => Some(GbdtLoss::SquaredError),
        },
        huber_alpha: v.num("huber") / 100.0,
        validation_fraction: v.num("validation") / 100.0,
        early_stopping_rounds: v.count("early"),
        seed: seed(v),
    };
    let m = fit_gbdt(&inp.ds, &inp.resp, &o)?;
    let mut s = m.report_section();
    s.heading = heading(if class { "Gradient Boosting Classification" } else { "Gradient Boosting Regression" }, &inp);
    let (imp, curve) = (v.flag("importance_plot"), v.flag("loss_plot"));
    filter_charts(&mut s, |t| (imp || t != "Variable Importance") && (curve || !t.starts_with("Loss versus")));
    let mut stored = Vec::new();
    store_predictions(
        &mut stored,
        &inp,
        &m.levels,
        &m.fitted,
        None,
        v.flag("store_fits"),
        v.flag("store_resid"),
        v.flag("store_prob"),
        "",
    );
    let models = saved(v, &inp, PredictiveModel::from_gbdt(&m));
    Ok(Output { section: s, stored, models })
}

/// Settings candidates from a text such as `1 2 3` or `1, 2.5`.
fn number_list(text: &str, what: &str) -> Res<Vec<f64>> {
    text.split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite())
                .ok_or_else(|| format!("{what}: \"{t}\" is not a number").into())
        })
        .collect()
}

/// MARS: a numeric response (least squares) or a two-class response (logistic link on the chosen
/// basis). Continuous predictors and 0/1 indicators for the levels of the categorical predictors
/// (the first level in sorted order is the reference). Rows with a missing value in any used column
/// are left out.
/// Class codes and level names of a two-class response.
type BinaryCodes = (Vec<usize>, Vec<String>);

fn mars_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let inp = gather(ws, v)?;
    let keep: Vec<usize> =
        (0..inp.ds.n_rows()).filter(|&i| (0..inp.ds.n_features()).all(|f| !inp.ds.is_missing(f, i))).collect();
    if keep.len() < 8 {
        return Err(format!("MARS needs at least 8 rows without missing values; {} remain", keep.len()).into());
    }
    let d = inp.ds.select_rows(&keep);
    let rows: Vec<usize> = keep.iter().map(|&i| inp.rows[i]).collect();
    let (x, names) = expand_indicators(&d);
    if x.is_empty() {
        return Err("the predictors have no usable columns".into());
    }
    let opt_num = |id: &str| {
        let t = v.num(id);
        (t.is_finite() && t > 0.0).then_some(t)
    };
    let mut opts = MarsOptions {
        max_terms: opt_num("max_terms").map(|t| t as usize),
        max_degree: v.count("degree"),
        penalty: v.text("penalty").trim().parse::<f64>().ok().filter(|p| p.is_finite()),
        min_span: opt_num("min_span").map(|t| t as usize),
        end_span: opt_num("end_span").map(|t| t as usize),
        threshold: v.num("threshold"),
        allow_linear: v.flag("linear"),
        prune: v.flag("prune"),
    };
    // The response: numbers, or two classes (the event is the second level, or the one named).
    let (yv, binary): (Vec<f64>, Option<BinaryCodes>) = match &inp.resp {
        Response::Numeric(y) => (keep.iter().map(|&i| y[i]).collect(), None),
        Response::Class { codes, levels } => {
            if levels.len() != 2 {
                return Err(format!(
                    "{} has {} classes; MARS handles a numeric response or two classes (use CART, Random Forests or Gradient Boosting for more)",
                    inp.response_name,
                    levels.len()
                )
                .into());
            }
            let ev = v.text("event").trim();
            let event = if ev.is_empty() {
                1
            } else {
                levels
                    .iter()
                    .position(|l| l == ev)
                    .ok_or_else(|| format!("the event \"{ev}\" is not one of: {}", levels.join(", ")))?
            };
            let c: Vec<usize> = keep.iter().map(|&i| usize::from(codes[i] == event)).collect();
            let lv = vec![levels[1 - event].clone(), levels[event].clone()];
            (Vec::new(), Some((c, lv)))
        }
    };
    let mut s = Section::new(format!("MARS: {} versus {}", inp.response_name, names.join(", ")));
    // Cross-validation over interaction degree and penalty, then (optionally) refit with the best.
    let folds = v.count("cv_folds");
    if folds > 0 {
        let degrees: Vec<usize> = {
            let l = number_list(v.text("cv_degrees"), "degrees")?;
            if l.is_empty() {
                vec![opts.max_degree]
            } else {
                l.iter().map(|&d| d as usize).collect()
            }
        };
        if degrees.iter().any(|&d| d == 0 || d > 5) {
            return Err("interaction degrees must be between 1 and 5".into());
        }
        let default_pen = |d: usize| opts.penalty.unwrap_or(if d > 1 { 3.0 } else { 2.0 });
        let pens = number_list(v.text("cv_penalties"), "penalties")?;
        let cv = match &binary {
            None => {
                let pens = if pens.is_empty() { vec![default_pen(degrees[0])] } else { pens };
                mars_cross_validate(&x, &names, CvResponse::Numeric(&yv), &opts, &degrees, &pens, folds, seed(v))?
            }
            Some((c, lv)) => {
                let pens = if pens.is_empty() { vec![default_pen(degrees[0])] } else { pens };
                mars_cross_validate(
                    &x,
                    &names,
                    CvResponse::Binary { codes: c, levels: lv },
                    &opts,
                    &degrees,
                    &pens,
                    folds,
                    seed(v),
                )?
            }
        };
        s.blocks.extend(cv.report_section().blocks);
        if v.flag("cv_refit") {
            let b = &cv.rows[cv.best];
            opts.max_degree = b.degree;
            opts.penalty = Some(b.penalty);
            s.text(format!(
                "The model below is fitted with the best setting: degree {}, penalty {}.",
                b.degree, b.penalty
            ));
        }
    }
    let (want_gcv, want_imp, want_fit) = (v.flag("gcv_plot"), v.flag("importance_plot"), v.flag("fit_plot"));
    let mut stored = Vec::new();
    let n_total = inp.n_total;
    let model = match &binary {
        None => {
            let m = fit_mars(&x, &names, &yv, &opts)?;
            let mut r = m.report_section();
            filter_charts(&mut r, |t| {
                (want_gcv || !t.starts_with("GCV"))
                    && (want_imp || !t.contains("Importance"))
                    && (want_fit || !t.starts_with("Fitted"))
            });
            s.blocks.extend(r.blocks);
            if v.flag("store_fits") {
                stored.push(numeric_col("MARS_FITS", n_total, &rows, &m.fitted));
            }
            if v.flag("store_resid") {
                let r: Vec<f64> = yv.iter().zip(&m.fitted).map(|(a, b)| a - b).collect();
                stored.push(numeric_col("MARS_RESI", n_total, &rows, &r));
            }
            PredictiveModel::Mars(MarsObject::from_model(&m, &d, MarsLink::Identity))
        }
        Some((c, lv)) => {
            let m = fit_mars_binary(&x, &names, c, lv, &opts)?;
            s.blocks.extend(m.report_section().blocks);
            let mut basis = m.basis_model.report_section();
            filter_charts(&mut basis, |t| (want_gcv && t.starts_with("GCV")) || (want_imp && t.contains("Importance")));
            basis.blocks.retain(|b| !matches!(b, Block::Table(t) if !(t.title.starts_with("Backward") || t.title.starts_with("Variable"))));
            if want_gcv || want_imp {
                s.text("Basis selection: the basis functions come from least-squares MARS on the 0/1 response.");
                s.blocks.extend(basis.blocks.into_iter().filter(|b| !matches!(b, Block::Text(_))));
            }
            if v.flag("store_fits") {
                let cls: Vec<String> = m.probabilities.iter().map(|&p| lv[usize::from(p > 0.5)].clone()).collect();
                stored.push(text_col("MARS_PRED", n_total, &rows, &cls));
            }
            if v.flag("store_prob") {
                stored.push(numeric_col(&format!("MARS_PROB_{}", lv[1]), n_total, &rows, &m.probabilities));
            }
            PredictiveModel::Mars(MarsObject::from_binary(&m, &d))
        }
    };
    let models = saved(v, &inp, model);
    Ok(Output { section: s, stored, models })
}

// ----------------------------------------------------------------------------- registration

pub(super) fn base(spec: DialogSpec) -> DialogSpec {
    spec.vars("response", "Response", VarKind::Any, 1, 1)
        .vars("cont", "Continuous predictors", VarKind::Numeric, 0, 100)
        .vars("cat", "Categorical predictors", VarKind::Any, 0, 100)
        .choice(
            Page::Main,
            "rtype",
            "Type of model",
            &["Automatic (text response: classification, numeric: regression)", "Regression", "Classification"],
            0,
        )
}

/// Register the predictive-analytics dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options, Storage};
    let menu = |last: &'static str| ["Stat", "Predictive Analytics", last];

    add(
        r,
        base(DialogSpec::new(
            "stat.ml.cart",
            "CART (Classification and Regression Tree)",
            "Grow a binary decision tree by recursive partitioning (variance reduction for a numeric response, Gini or entropy for classes), prune it by cost-complexity and choose its size by cross-validation. Missing predictor values are handled with surrogate splits.",
        ))
        .choice(Options, "criterion", "Classification criterion", &["Gini", "Entropy"], 0)
        .integer(Options, "max_depth", "Maximum tree depth", 30.0, 1.0, 100.0)
        .integer(Options, "min_split", "Minimum observations to split a node", 10.0, 2.0, 100000.0)
        .integer(Options, "min_leaf", "Minimum observations in a terminal node", 3.0, 1.0, 100000.0)
        .number(Options, "cp", "Minimum improvement to split (fraction of the root deviance)", 0.0, 0.0, 0.99)
        .integer(Options, "surrogates", "Surrogate splits per node", 5.0, 0.0, 20.0)
        .choice(
            Options,
            "pruning",
            "Tree size",
            &[
                "Minimum cross-validated error",
                "Smallest tree within 1 SE of the minimum",
                "No pruning (largest tree)",
                "Complexity parameter below",
                "Terminal node limit below",
            ],
            0,
        )
        .number(Options, "alpha", "Complexity parameter (relative to the root error)", 0.01, 0.0, 1.0)
        .integer(Options, "max_leaves", "Maximum terminal nodes", 10.0, 1.0, 100000.0)
        .integer(Options, "folds", "Cross-validation folds (0 = none)", 10.0, 0.0, 100.0)
        .integer(Options, "seed", "Random seed (fold assignment)", 1.0, 0.0, 4.0e9)
        .check(Graphs, "importance_plot", "Variable importance chart", true)
        .check(Graphs, "cv_plot", "Cross-validated error versus tree size", true)
        .check(Storage, "store_fits", "Fits (predicted values or classes)", false)
        .check(Storage, "store_resid", "Residuals (regression)", false)
        .check(Storage, "store_prob", "Predicted class probabilities", false)
        .check(Storage, "store_node", "Terminal node numbers", false)
        .text(Storage, "save_as", "Save model in the project as (blank = do not save)", ""),
        &menu("CART..."),
        Runner::Report(cart_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ml.forest",
            "Random Forests",
            "Average many trees grown on bootstrap samples with a random subset of predictors tried at each split. Reports out-of-bag error, permutation and impurity importance and, for a numeric response, quantile-forest prediction intervals.",
        ))
        .integer(Options, "trees", "Number of trees", 500.0, 1.0, 20000.0)
        .integer(Options, "mtry", "Predictors tried per split (0 = automatic)", 0.0, 0.0, 10000.0)
        .integer(Options, "min_leaf", "Minimum observations in a terminal node (0 = automatic)", 0.0, 0.0, 100000.0)
        .integer(Options, "max_depth", "Maximum tree depth", 100.0, 1.0, 1000.0)
        .number(Options, "sample", "Bootstrap sample size (% of the observations)", 100.0, 1.0, 100.0)
        .choice(Options, "criterion", "Classification criterion", &["Gini", "Entropy"], 0)
        .check(Options, "perm_importance", "Compute permutation importance", true)
        .check(Options, "intervals", "Prediction intervals (numeric response)", false)
        .number(Options, "level", "Interval level (%)", 95.0, 1.0, 99.9)
        .integer(Options, "seed", "Random seed", 1.0, 0.0, 4.0e9)
        .check(Graphs, "importance_plot", "Variable importance chart", true)
        .check(Graphs, "oob_plot", "Out-of-bag error versus number of trees", true)
        .choice(Storage, "stored", "Stored predictions are", &["Out-of-bag", "From all trees (in-sample)"], 0)
        .check(Storage, "store_fits", "Fits (predicted values or classes)", false)
        .check(Storage, "store_resid", "Residuals (regression)", false)
        .check(Storage, "store_prob", "Predicted class probabilities", false)
        .check(Storage, "store_pi", "Out-of-bag prediction interval limits", false)
        .text(Storage, "save_as", "Save model in the project as (blank = do not save)", ""),
        &menu("Random Forests..."),
        Runner::Report(forest_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ml.gbdt",
            "Gradient Boosting",
            "Build an additive model of small trees, each fitted to the gradient of the loss (squared error or Huber for a numeric response, deviance for classes), with shrinkage, stochastic subsampling and validation-based choice of the number of trees.",
        ))
        .choice(Options, "loss", "Loss (numeric response)", &["Squared error", "Huber"], 0)
        .integer(Options, "trees", "Maximum number of trees", 100.0, 1.0, 20000.0)
        .number(Options, "rate", "Learning rate", 0.1, 0.0001, 1.0)
        .integer(Options, "max_depth", "Maximum tree depth", 3.0, 1.0, 30.0)
        .integer(Options, "min_leaf", "Minimum observations in a terminal node", 5.0, 1.0, 100000.0)
        .number(Options, "subsample", "Subsample (% of training rows per tree)", 50.0, 1.0, 100.0)
        .number(Options, "huber", "Huber quantile (%)", 90.0, 1.0, 99.0)
        .number(Options, "validation", "Validation rows (% of the observations)", 20.0, 0.0, 50.0)
        .integer(Options, "early", "Stop after this many trees without improvement (0 = never)", 0.0, 0.0, 10000.0)
        .integer(Options, "seed", "Random seed", 1.0, 0.0, 4.0e9)
        .check(Graphs, "importance_plot", "Variable importance chart", true)
        .check(Graphs, "loss_plot", "Loss versus number of trees", true)
        .check(Storage, "store_fits", "Fits (predicted values or classes)", false)
        .check(Storage, "store_resid", "Residuals (regression)", false)
        .check(Storage, "store_prob", "Predicted class probabilities", false)
        .text(Storage, "save_as", "Save model in the project as (blank = do not save)", ""),
        &menu("Gradient Boosting..."),
        Runner::Report(gbdt_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ml.mars",
            "MARS Regression",
            "Multivariate adaptive regression splines (Friedman 1991): a forward pass adds mirrored hinge functions at data-driven knots (optionally multiplied into earlier terms to form interactions), a backward pass deletes terms one at a time and the model with the smallest generalized cross-validation (GCV) score is kept. A two-class response gets a logistic link: the basis is chosen as above on the 0/1 response and its coefficients are then re-estimated by maximum likelihood. Categorical predictors enter as 0/1 indicators; rows with missing values are left out.",
        ))
        .text(Main, "event", "Event level for a two-class response (blank = the last level in sorted order)", "")
        .integer(Options, "max_terms", "Maximum terms in the forward pass (0 = automatic)", 0.0, 0.0, 1000.0)
        .integer(Options, "degree", "Maximum interaction degree", 1.0, 1.0, 5.0)
        .text(Options, "penalty", "GCV penalty per knot (blank = 2, or 3 with interactions)", "")
        .integer(Options, "min_span", "Minimum observations between knots (0 = automatic)", 0.0, 0.0, 100000.0)
        .integer(Options, "end_span", "Observations excluded at each end (0 = automatic)", 0.0, 0.0, 100000.0)
        .number(Options, "threshold", "Forward-pass stopping threshold (fraction of R-squared)", 0.001, 0.0, 0.5)
        .check(Options, "linear", "Allow linear terms", true)
        .check(Options, "prune", "Backward elimination by GCV", true)
        .integer(Options, "cv_folds", "Cross-validation folds (0 = none)", 0.0, 0.0, 100.0)
        .text(Options, "cv_degrees", "Interaction degrees to compare by cross-validation, e.g. 1 2 3 (blank = the degree above)", "")
        .text(Options, "cv_penalties", "GCV penalties to compare, e.g. 2 3 4 (blank = the penalty above)", "")
        .check(Options, "cv_refit", "Fit the final model with the best cross-validated setting", true)
        .integer(Options, "seed", "Random seed (fold assignment)", 1.0, 0.0, 4.0e9)
        .check(Graphs, "gcv_plot", "GCV versus number of terms", true)
        .check(Graphs, "importance_plot", "Variable importance chart", true)
        .check(Graphs, "fit_plot", "Fitted versus observed (numeric response)", true)
        .check(Storage, "store_fits", "Fits (predicted values or classes)", false)
        .check(Storage, "store_resid", "Residuals (numeric response)", false)
        .check(Storage, "store_prob", "Event probabilities (two-class response)", false)
        .text(Storage, "save_as", "Save model in the project as (blank = do not save)", ""),
        &menu("MARS Regression..."),
        Runner::Report(mars_run),
    );
}
