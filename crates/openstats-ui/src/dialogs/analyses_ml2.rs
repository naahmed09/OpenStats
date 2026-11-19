//! Predictive analytics, continued (Phase 5i): scoring new data with a model saved in the project,
//! partial dependence and ICE plots, model comparison by cross-validation and the list of saved
//! models. Fit dialogs save their model with "Save model in the project as"; the model lives in the
//! `.otb` file with the worksheets.

use super::analyses::add;
use super::analyses_ml::{base, gather, seed};
use super::analysis::{col_title, labels, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use crate::grid::GridController;
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::ml::{
    compare_models, partial_dependence, partial_dependence_2d, CartOptions, Dataset, Feature, ForestOptions,
    GbdtOptions, MarsOptions, ModelSpec, PdOptions, PredictiveModel, Pruning, SavedModel,
};
use openstats_stats::report::{Cell, Column as RCol, ColumnKind, Reportable, Section, Table};

fn load(grid: &GridController, name: &str) -> Res<SavedModel> {
    let name = name.trim();
    if name.is_empty() {
        return Err("enter the name of a saved model".into());
    }
    let json = grid.model_json(name).ok_or_else(|| {
        let have = grid.model_names();
        if have.is_empty() {
            format!("there is no saved model named \"{name}\"; the project has no saved models (use \"Save model in the project as\" when fitting)")
        } else {
            format!("there is no saved model named \"{name}\"; saved models: {}", have.join(", "))
        }
    })?;
    SavedModel::from_json(json).map_err(|e| format!("model \"{name}\" cannot be read: {e}").into())
}

/// The worksheet's columns for a model's predictors: the selected ones in the model's order, or the
/// columns that carry the predictors' names.
fn predictor_columns(ws: &Worksheet, schema: &Dataset, picked: &[usize]) -> Res<Vec<usize>> {
    if !picked.is_empty() {
        if picked.len() != schema.n_features() {
            return Err(format!(
                "the model has {} predictors ({}) but {} columns were selected",
                schema.n_features(),
                schema.names.join(", "),
                picked.len()
            )
            .into());
        }
        return Ok(picked.to_vec());
    }
    let mut cols = Vec::new();
    let mut missing = Vec::new();
    for name in &schema.names {
        match ws.columns.iter().position(|c| c.name() == name) {
            Some(i) => cols.push(i),
            None => missing.push(name.clone()),
        }
    }
    if missing.is_empty() {
        Ok(cols)
    } else {
        Err(format!(
            "the worksheet has no column named {} (the model's predictors are {}); select the columns in the model's order instead",
            missing.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", "),
            schema.names.join(", ")
        )
        .into())
    }
}

/// The observations in the worksheet as a dataset shaped like the model's predictors. Short columns
/// are padded with missing values.
fn dataset_for(ws: &Worksheet, schema: &Dataset, cols: &[usize]) -> Res<(Dataset, usize)> {
    let n = cols.iter().map(|&c| ws.column(c).map_or(0, Column::len)).max().unwrap_or(0);
    if n == 0 {
        return Err("the predictor columns are empty".into());
    }
    let mut ds = Dataset::new();
    for (f, &c) in cols.iter().enumerate() {
        let name = &schema.names[f];
        match &schema.features[f] {
            Feature::Numeric(_) => {
                let mut v = numeric(ws, c).map_err(|_| {
                    format!("{} is text but the model's predictor \"{name}\" is numeric", col_title(ws, c))
                })?;
                v.resize(n, f64::NAN);
                ds = ds.with_numeric(name, v)?;
            }
            Feature::Categorical { .. } => {
                let mut v = labels(ws, c)?;
                v.resize(n, String::new());
                ds = ds.with_categorical(name, &v)?;
            }
        }
    }
    Ok((ds, n))
}

fn kv(title: &str, rows: Vec<(&str, String)>) -> Table {
    let mut t = Table::new(title, vec![RCol::new("Item", ColumnKind::Text), RCol::new("Value", ColumnKind::Text)]);
    for (k, v) in rows {
        t.push_row(vec![Cell::text(k), Cell::text(v)]).expect("two cells");
    }
    t
}

// ----------------------------------------------------------------------------- score

fn score_run(grid: &GridController, v: &Values) -> Res<Output> {
    let saved = load(grid, v.text("model"))?;
    let ws = grid.worksheet();
    let model = &saved.model;
    let cols = predictor_columns(ws, model.schema(), v.vars("pred"))?;
    let (ds, n) = dataset_for(ws, model.schema(), &cols)?;
    let pred = model.predict(&ds)?;
    let prefix = v.text("prefix").trim().to_string();
    let mut stored = Vec::new();
    let mut s = Section::new(format!("Score New Data: {}", v.text("model").trim()));
    let incomplete = (0..n).filter(|&i| (0..ds.n_features()).any(|f| ds.is_missing(f, i))).count();
    let levels = model.levels();
    let scored = if levels.is_empty() {
        pred.values.iter().filter(|x| x.is_finite()).count()
    } else {
        pred.probabilities.iter().filter(|p| p.iter().all(|x| x.is_finite())).count()
    };
    s.table(kv(
        "Scoring Summary",
        vec![
            ("Model", format!("{} ({})", v.text("model").trim(), model.algorithm())),
            ("Response", saved.response.clone()),
            ("Fitted on", format!("{} observations", saved.n_train)),
            ("Predictors", model.schema().names.join(", ")),
            ("Rows scored", format!("{scored} of {n}")),
            ("Rows with a missing predictor", incomplete.to_string()),
        ],
    ));
    if levels.is_empty() {
        let vals: Vec<f64> = pred.values.iter().copied().filter(|x| x.is_finite()).collect();
        if !vals.is_empty() {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let (mn, mx) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
            s.table(kv(
                "Predictions",
                vec![("Minimum", format!("{mn}")), ("Mean", format!("{mean}")), ("Maximum", format!("{mx}"))],
            ));
        }
        stored.push(Column::numeric(format!("{prefix}PRED"), pred.values.clone()));
    } else {
        let mut counts = vec![0usize; levels.len()];
        let mut cls = Vec::with_capacity(n);
        for (i, p) in pred.probabilities.iter().enumerate() {
            if p.iter().all(|x| x.is_finite()) {
                counts[pred.classes[i]] += 1;
                cls.push(levels[pred.classes[i]].clone());
            } else {
                cls.push(String::new());
            }
        }
        let mut t = Table::new(
            "Predicted Classes",
            vec![RCol::new("Class", ColumnKind::Text), RCol::new("Count", ColumnKind::Integer)],
        );
        for (l, c) in levels.iter().zip(&counts) {
            t.push_row(vec![Cell::text(l.clone()), Cell::Int(*c as i64)]).expect("two cells");
        }
        s.table(t);
        stored.push(Column::text(format!("{prefix}CLASS"), cls));
        if v.flag("store_prob") {
            for (k, l) in levels.iter().enumerate() {
                let p: Vec<f64> = pred.probabilities.iter().map(|q| q[k]).collect();
                stored.push(Column::numeric(format!("{prefix}PROB_{l}"), p));
            }
        }
    }
    if scored < n {
        s.text("Rows whose predictors are missing where the model needs them get a missing prediction.");
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------- partial dependence

fn feature_index(schema: &Dataset, name: &str) -> Res<usize> {
    let name = name.trim();
    schema
        .names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| format!("\"{name}\" is not a predictor of the model ({})", schema.names.join(", ")).into())
}

fn pdp_run(grid: &GridController, v: &Values) -> Res<Output> {
    let saved = load(grid, v.text("model"))?;
    let ws = grid.worksheet();
    let model: &PredictiveModel = &saved.model;
    let schema = model.schema();
    let cols = predictor_columns(ws, schema, v.vars("pred"))?;
    let (ds, _) = dataset_for(ws, schema, &cols)?;
    let class =
        {
            let t = v.text("class").trim();
            if t.is_empty() || model.levels().is_empty() {
                None
            } else {
                Some(
                    model.levels().iter().position(|l| l == t).ok_or_else(|| {
                        format!("\"{t}\" is not a class of the model ({})", model.levels().join(", "))
                    })?,
                )
            }
        };
    let (lo, hi) = (v.num("lo") / 100.0, v.num("hi") / 100.0);
    let opts = PdOptions {
        grid_points: v.count("grid"),
        range: (lo, hi),
        ice_rows: if v.flag("ice") { v.count("ice_rows") } else { 0 },
        class,
        max_rows: v.count("max_rows"),
    };
    let wanted: Vec<usize> = {
        let t = v.text("features");
        if t.trim().is_empty() {
            (0..schema.n_features()).collect()
        } else {
            t.split(';').filter(|p| !p.trim().is_empty()).map(|p| feature_index(schema, p)).collect::<Res<_>>()?
        }
    };
    let pairs: Vec<(usize, usize)> = v
        .text("pairs")
        .split(';')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (a, b) = p.split_once('*').ok_or_else(|| format!("write a pair as x*y, not \"{}\"", p.trim()))?;
            Ok((feature_index(schema, a)?, feature_index(schema, b)?))
        })
        .collect::<Res<_>>()?;
    if wanted.is_empty() && pairs.is_empty() {
        return Err("choose at least one predictor or pair".into());
    }
    let mut s = Section::new(format!("Partial Dependence: {}", v.text("model").trim()));
    s.text(format!(
        "{} fitted to {} ({} observations); averaged over {} reference observations.",
        model.algorithm(),
        saved.response,
        saved.n_train,
        ds.n_rows()
    ));
    let mut skipped = Vec::new();
    for &f in &wanted {
        match partial_dependence(model, &ds, f, &opts) {
            Ok(pd) => s.blocks.extend(pd.report_section().blocks),
            Err(e) => skipped.push(format!("{}: {e}", schema.names[f])),
        }
    }
    for &(a, b) in &pairs {
        s.blocks.extend(partial_dependence_2d(model, &ds, a, b, &opts)?.report_section().blocks);
    }
    for k in skipped {
        s.text(format!("Not plotted: {k}."));
    }
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

// ----------------------------------------------------------------------------- compare

fn compare_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let inp = gather(ws, v)?;
    let mut cands: Vec<(String, ModelSpec)> = Vec::new();
    if v.flag("baseline") {
        cands.push(("Baseline (no predictors)".into(), ModelSpec::Baseline));
    }
    if v.flag("m_cart") {
        cands.push((
            "CART".into(),
            ModelSpec::Cart(CartOptions {
                min_leaf: v.count("cart_leaf").max(1),
                cv_folds: v.count("cart_cv"),
                pruning: if v.count("cart_cv") >= 2 { Pruning::OneSe } else { Pruning::None },
                seed: seed(v),
                ..CartOptions::default()
            }),
        ));
    }
    if v.flag("m_forest") {
        cands.push((
            "Random Forests".into(),
            ModelSpec::Forest(ForestOptions {
                n_trees: v.count("forest_trees").max(1),
                seed: seed(v),
                ..ForestOptions::default()
            }),
        ));
    }
    if v.flag("m_boost") {
        cands.push((
            "Gradient Boosting".into(),
            ModelSpec::Boosting(GbdtOptions {
                n_trees: v.count("boost_trees").max(1),
                seed: seed(v),
                ..GbdtOptions::default()
            }),
        ));
    }
    let mut notes = Vec::new();
    if v.flag("m_mars") {
        let multi = matches!(&inp.resp, openstats_stats::ml::Response::Class { levels, .. } if levels.len() != 2);
        if multi {
            notes.push("MARS was left out: it handles a numeric or two-class response.".to_string());
        } else {
            cands.push((
                "MARS".into(),
                ModelSpec::Mars(MarsOptions { max_degree: v.count("mars_degree").max(1), ..MarsOptions::default() }),
            ));
        }
    }
    if cands.is_empty() {
        return Err("choose at least one model to compare".into());
    }
    let c = compare_models(&inp.ds, &inp.resp, &cands, v.count("folds"), seed(v))?;
    let mut s = c.report_section();
    s.heading = format!("Compare Models: {} versus {}", inp.response_name, inp.predictor_names.join(", "));
    for n in notes {
        s.text(n);
    }
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

// ----------------------------------------------------------------------------- manage

fn models_run(grid: &mut GridController, v: &Values) -> Res<String> {
    let name = v.text("model").trim().to_string();
    match v.choice("action") {
        0 => {
            let names = grid.model_names();
            if names.is_empty() {
                return Ok("The project has no saved models. Use \"Save model in the project as\" in CART, Random Forests, Gradient Boosting or MARS.".into());
            }
            let items: Vec<String> = names
                .iter()
                .map(|n| match grid.model_json(n).map(SavedModel::from_json) {
                    Some(Ok(m)) => format!(
                        "{n} ({}, response {}, {} observations, predictors: {})",
                        m.model.algorithm(),
                        m.response,
                        m.n_train,
                        m.model.schema().names.join(", ")
                    ),
                    _ => format!("{n} (unreadable)"),
                })
                .collect();
            Ok(format!("Saved models ({}): {}.", names.len(), items.join("; ")))
        }
        1 => {
            if grid.remove_model(&name) {
                Ok(format!("Deleted the model \"{name}\"."))
            } else {
                Err(format!("there is no saved model named \"{name}\"").into())
            }
        }
        _ => {
            let new = v.text("new_name").trim().to_string();
            let json =
                grid.model_json(&name).ok_or_else(|| format!("there is no saved model named \"{name}\""))?.to_string();
            if new.is_empty() {
                return Err("enter the new name".into());
            }
            if new != name && grid.model_json(&new).is_some() {
                return Err(format!("a model named \"{new}\" already exists").into());
            }
            grid.store_model(&new, json)?;
            if new != name {
                grid.remove_model(&name);
            }
            Ok(format!("Renamed the model \"{name}\" to \"{new}\"."))
        }
    }
}

// ----------------------------------------------------------------------------- registration

/// Register the second group of predictive-analytics dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options, Storage};
    let menu = |last: &'static str| ["Stat", "Predictive Analytics", last];
    add(
        r,
        DialogSpec::new(
            "stat.ml.score",
            "Score New Data",
            "Apply a model saved in the project (CART, Random Forests, Gradient Boosting or MARS) to the rows of a worksheet. Predictors are matched to the worksheet's columns by name, or taken in the model's order from the columns selected below. Missing predictor values are handled as in the fit (trees use surrogate splits; MARS gives a missing prediction).",
        )
        .text(Main, "model", "Saved model name", "")
        .vars("pred", "Predictor columns (blank = match by name)", VarKind::Any, 0, 100)
        .text(Storage, "prefix", "Prefix for the stored columns", "SCORE_")
        .check(Storage, "store_prob", "Class probabilities (classification models)", true),
        &menu("Score New Data..."),
        Runner::Project(score_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.ml.pdp",
            "Partial Dependence Plots",
            "Show how a saved model's prediction changes with one predictor (or two together) after averaging over the other predictors in the worksheet's rows (Friedman 2001), with an individual conditional expectation (ICE) curve for each of a sample of rows. For a classifier the target is the probability of the chosen class.",
        )
        .text(Main, "model", "Saved model name", "")
        .vars("pred", "Reference data: predictor columns (blank = match by name)", VarKind::Any, 0, 100)
        .text(Main, "features", "Predictors to plot, separated by ; (blank = all)", "")
        .text(Main, "pairs", "Two-way pairs as x*y, separated by ; (blank = none)", "")
        .text(Main, "class", "Class whose probability is plotted (blank = last level)", "")
        .integer(Options, "grid", "Grid points for a numeric predictor", 20.0, 2.0, 200.0)
        .number(Options, "lo", "Lower end of the grid (percentile of the data)", 5.0, 0.0, 49.0)
        .number(Options, "hi", "Upper end of the grid (percentile of the data)", 95.0, 51.0, 100.0)
        .integer(Options, "max_rows", "Reference rows used (0 = all)", 1000.0, 0.0, 10000000.0)
        .check(Graphs, "ice", "Show individual (ICE) curves", true)
        .integer(Graphs, "ice_rows", "Number of ICE curves", 30.0, 1.0, 500.0),
        &menu("Partial Dependence Plots..."),
        Runner::Project(pdp_run),
    );
    add(
        r,
        base(DialogSpec::new(
            "stat.ml.compare",
            "Compare Models",
            "Compare tree models, MARS and a no-predictor baseline by K-fold cross-validation: every model is refitted on each training fold and scored on the held-out fold, with the same folds for all. Numeric response: RMSE, MAE, R-squared. Classes: error rate, log-loss and (two classes) AUC.",
        ))
        .check(Options, "baseline", "Include the baseline (training mean or class proportions)", true)
        .check(Options, "m_cart", "CART", true)
        .check(Options, "m_forest", "Random Forests", true)
        .check(Options, "m_boost", "Gradient Boosting", true)
        .check(Options, "m_mars", "MARS", true)
        .integer(Options, "folds", "Cross-validation folds", 5.0, 2.0, 100.0)
        .integer(Options, "cart_leaf", "CART: minimum observations in a terminal node", 5.0, 1.0, 100000.0)
        .integer(Options, "cart_cv", "CART: inner cross-validation folds for pruning (0 = no pruning)", 0.0, 0.0, 20.0)
        .integer(Options, "forest_trees", "Random Forests: number of trees", 100.0, 1.0, 5000.0)
        .integer(Options, "boost_trees", "Gradient Boosting: number of trees", 100.0, 1.0, 5000.0)
        .integer(Options, "mars_degree", "MARS: maximum interaction degree", 1.0, 1.0, 5.0)
        .integer(Options, "seed", "Random seed", 1.0, 0.0, 4.0e9),
        &menu("Compare Models..."),
        Runner::Report(compare_run),
    );
    add(
        r,
        DialogSpec::new("stat.ml.models", "Saved Models", "List, rename or delete the models saved in the project.")
            .choice(Main, "action", "Action", &["List models", "Delete a model", "Rename a model"], 0)
            .text(Main, "model", "Model name (delete or rename)", "")
            .text(Main, "new_name", "New name (rename)", ""),
        &menu("Saved Models..."),
        Runner::Mutate(models_run),
    );
}
