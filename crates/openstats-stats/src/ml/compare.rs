//! Model comparison by K-fold cross-validation: every candidate is refitted on each training
//! fold and scored on the held-out fold, with the same folds for all candidates.

use super::cart::{fit_cart, CartOptions};
use super::data::{Dataset, Response};
use super::gbdt::{fit_gbdt, GbdtOptions};
use super::mars::{fit_mars, MarsOptions};
use super::mars_ext::{fit_mars_binary, random_folds};
use super::random_forest::{fit_forest, ForestOptions};
use super::store::{expand_indicators, MarsLink, MarsObject, PredictiveModel};
use super::{auc, class_fit, regression_fit, Predictions};
use crate::error::{invalid, Result};
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};

/// A candidate model and its settings.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// Classification or regression tree.
    Cart(CartOptions),
    /// Random forest.
    Forest(ForestOptions),
    /// Gradient boosting.
    Boosting(GbdtOptions),
    /// MARS (numeric or two-class response; rows with a missing predictor are not used to fit).
    Mars(MarsOptions),
    /// The no-information model: the training mean, or the training class proportions.
    Baseline,
}

/// Fit one candidate on `ds` / `y`.
pub fn fit_spec(spec: &ModelSpec, ds: &Dataset, y: &Response) -> Result<PredictiveModel> {
    Ok(match spec {
        ModelSpec::Cart(o) => PredictiveModel::from_cart(&fit_cart(ds, y, o)?),
        ModelSpec::Forest(o) => PredictiveModel::from_forest(&fit_forest(ds, y, o)?),
        ModelSpec::Boosting(o) => PredictiveModel::from_gbdt(&fit_gbdt(ds, y, o)?),
        ModelSpec::Mars(o) => {
            let n = ds.n_rows();
            let keep: Vec<usize> = (0..n).filter(|&i| (0..ds.n_features()).all(|f| !ds.is_missing(f, i))).collect();
            if keep.len() < 8 {
                return invalid("MARS needs at least 8 observations without missing predictors");
            }
            let d = ds.select_rows(&keep);
            let (x, names) = expand_indicators(&d);
            match y {
                Response::Numeric(v) => {
                    let yv: Vec<f64> = keep.iter().map(|&i| v[i]).collect();
                    let m = fit_mars(&x, &names, &yv, o)?;
                    PredictiveModel::Mars(MarsObject::from_model(&m, ds, MarsLink::Identity))
                }
                Response::Class { codes, levels } => {
                    if levels.len() != 2 {
                        return invalid("MARS needs a numeric or a two-class response");
                    }
                    let c: Vec<usize> = keep.iter().map(|&i| codes[i]).collect();
                    let m = fit_mars_binary(&x, &names, &c, levels, o)?;
                    PredictiveModel::Mars(MarsObject::from_binary(&m, ds))
                }
            }
        }
        ModelSpec::Baseline => {
            let one =
                Dataset::from_features(vec!["_".into()], vec![super::data::Feature::Numeric(vec![0.0; ds.n_rows()])])?;
            // A stump that never splits: min_leaf larger than the data.
            let mut o = CartOptions { cv_folds: 0, pruning: super::cart::Pruning::None, ..CartOptions::default() };
            o.min_leaf = ds.n_rows().max(2);
            o.min_split = ds.n_rows().max(2) + 1;
            let m = fit_cart(&one, y, &o)?;
            let mut pm = PredictiveModel::from_cart(&m);
            if let PredictiveModel::Tree { schema, .. } = &mut pm {
                *schema = ds.schema();
            }
            // The stump predicts without reading predictors, but the schema must match `ds`.
            return Ok(pm);
        }
    })
}

/// Held-out performance of one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonRow {
    /// Candidate name.
    pub name: String,
    /// Observations scored (those with a prediction).
    pub n_scored: usize,
    /// Regression: root mean squared error (else `NaN`).
    pub rmse: f64,
    /// Regression: mean absolute error.
    pub mae: f64,
    /// Regression: R².
    pub r2: f64,
    /// Classification: misclassification rate.
    pub error_rate: f64,
    /// Classification: mean log-loss.
    pub log_loss: f64,
    /// Two classes: area under the ROC curve (the second level is the event).
    pub auc: f64,
    /// Seconds spent fitting all folds.
    pub seconds: f64,
}

/// Result of [`compare_models`].
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// One row per candidate, in the order given.
    pub rows: Vec<ComparisonRow>,
    /// Whether the response is categorical.
    pub classification: bool,
    /// Class labels (empty for regression).
    pub levels: Vec<String>,
    /// Folds.
    pub folds: usize,
    /// Observations.
    pub n: usize,
    /// Index of the best candidate (smallest RMSE, or log-loss for classification).
    pub best: usize,
}

fn response_subset(y: &Response, rows: &[usize]) -> Response {
    match y {
        Response::Numeric(v) => Response::Numeric(rows.iter().map(|&i| v[i]).collect()),
        Response::Class { codes, levels } => {
            Response::Class { codes: rows.iter().map(|&i| codes[i]).collect(), levels: levels.clone() }
        }
    }
}

/// Compare candidates by `folds`-fold cross-validation with a shared seeded fold assignment.
pub fn compare_models(
    ds: &Dataset,
    y: &Response,
    candidates: &[(String, ModelSpec)],
    folds: usize,
    seed: u64,
) -> Result<Comparison> {
    let n = ds.n_rows();
    if y.len() != n {
        return invalid("the response and predictors have different lengths");
    }
    if candidates.is_empty() {
        return invalid("choose at least one model");
    }
    if folds < 2 || folds > n {
        return invalid("the number of folds must be between 2 and the number of observations");
    }
    let fold = random_folds(n, folds, seed);
    let mut rows = Vec::new();
    for (name, spec) in candidates {
        let start = std::time::Instant::now();
        let mut num_pred = vec![f64::NAN; n];
        let mut prob_pred: Vec<Vec<f64>> = vec![Vec::new(); n];
        // folds are independent: fit them in parallel, then merge in fold order
        let results = crate::parallel::par_map(folds, |f| -> Result<(Vec<usize>, Predictions)> {
            let train: Vec<usize> = (0..n).filter(|&i| fold[i] != f).collect();
            let held: Vec<usize> = (0..n).filter(|&i| fold[i] == f).collect();
            let model = fit_spec(spec, &ds.select_rows(&train), &response_subset(y, &train))?;
            let p = model.predict(&ds.select_rows(&held))?;
            Ok((held, p))
        });
        for r in results {
            let (held, p) = r?;
            for (k, &i) in held.iter().enumerate() {
                if y.is_class() {
                    prob_pred[i] = p.probabilities[k].clone();
                } else {
                    num_pred[i] = p.values[k];
                }
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        let mut row = ComparisonRow {
            name: name.clone(),
            n_scored: 0,
            rmse: f64::NAN,
            mae: f64::NAN,
            r2: f64::NAN,
            error_rate: f64::NAN,
            log_loss: f64::NAN,
            auc: f64::NAN,
            seconds,
        };
        match y {
            Response::Numeric(v) => {
                let ok: Vec<usize> = (0..n).filter(|&i| num_pred[i].is_finite()).collect();
                let (yy, pp): (Vec<f64>, Vec<f64>) = ok.iter().map(|&i| (v[i], num_pred[i])).unzip();
                let r = regression_fit(&yy, &pp);
                row.n_scored = ok.len();
                row.rmse = r.rmse;
                row.mae = r.mae;
                row.r2 = r.r2;
            }
            Response::Class { codes, levels } => {
                let ok: Vec<usize> = (0..n)
                    .filter(|&i| prob_pred[i].iter().all(|p| p.is_finite()) && !prob_pred[i].is_empty())
                    .collect();
                let cc: Vec<usize> = ok.iter().map(|&i| codes[i]).collect();
                let pp: Vec<Vec<f64>> = ok.iter().map(|&i| prob_pred[i].clone()).collect();
                let cf = class_fit(&cc, &pp, levels);
                row.n_scored = ok.len();
                row.error_rate = 1.0 - cf.accuracy;
                row.log_loss = cf.log_loss.unwrap_or(f64::NAN);
                row.auc = cf.auc.unwrap_or(f64::NAN);
                let _ = auc;
            }
        }
        rows.push(row);
    }
    let classification = y.is_class();
    let key = |r: &ComparisonRow| if classification { r.log_loss } else { r.rmse };
    let best = (0..rows.len()).min_by(|&a, &b| key(&rows[a]).total_cmp(&key(&rows[b]))).unwrap_or(0);
    let levels = match y {
        Response::Class { levels, .. } => levels.clone(),
        Response::Numeric(_) => Vec::new(),
    };
    Ok(Comparison { rows, classification, levels, folds, n, best })
}

impl Reportable for Comparison {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Model Comparison");
        s.text(format!(
            "{}-fold cross-validation on {} observations; every model is refitted on each training fold \
             and scored on the held-out fold, with the same folds for all models.",
            self.folds, self.n
        ));
        let mut cols = vec![Column::new("Model", ColumnKind::Text), Column::new("N scored", ColumnKind::Integer)];
        if self.classification {
            cols.push(Column::new("Error rate", ColumnKind::Significant(6)));
            cols.push(Column::new("Log-loss", ColumnKind::Significant(6)));
            if self.levels.len() == 2 {
                cols.push(Column::new("AUC", ColumnKind::Significant(6)));
            }
        } else {
            cols.push(Column::new("RMSE", ColumnKind::Significant(6)));
            cols.push(Column::new("MAE", ColumnKind::Significant(6)));
            cols.push(Column::new("R-Sq", ColumnKind::Significant(6)));
        }
        cols.push(Column::new("Seconds", ColumnKind::Fixed(2)));
        let mut t = Table::new("Cross-Validated Performance", cols);
        for r in &self.rows {
            let mut cells = vec![Cell::text(r.name.clone()), Cell::Int(r.n_scored as i64)];
            if self.classification {
                cells.push(Cell::Num(r.error_rate));
                cells.push(Cell::Num(r.log_loss));
                if self.levels.len() == 2 {
                    cells.push(Cell::Num(r.auc));
                }
            } else {
                cells.push(Cell::Num(r.rmse));
                cells.push(Cell::Num(r.mae));
                cells.push(Cell::Num(r.r2));
            }
            cells.push(Cell::Num(r.seconds));
            t.row(cells);
        }
        s.table(t);
        s.text(format!(
            "Best model by {}: {}.",
            if self.classification { "log-loss" } else { "RMSE" },
            self.rows[self.best].name
        ));
        let metric = |r: &ComparisonRow| if self.classification { r.log_loss } else { r.rmse };
        s.chart(ChartSpec {
            kind: ChartKind::Bar,
            title: format!("Cross-Validated {}", if self.classification { "Log-Loss" } else { "RMSE" }),
            x_label: "Model".into(),
            y_label: if self.classification { "Log-loss".into() } else { "RMSE".into() },
            categories: self.rows.iter().map(|r| r.name.clone()).collect(),
            series: vec![Series { name: "CV error".into(), x: Vec::new(), y: self.rows.iter().map(metric).collect() }],
            reference_lines: Vec::new(),
        });
        s
    }
}
