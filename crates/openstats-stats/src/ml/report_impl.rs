//! [`Reportable`] implementations of the tree-based models.

use super::cart::{terminal_numbers, CartModel, Pruning};
use super::data::Dataset;
use super::gbdt::GbdtModel;
use super::mars::MarsModel;
use super::random_forest::ForestModel;
use super::tree::{fmt_num, Test, Tree};
use super::{relative_importance, ClassFit, RegressionFit};
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};

const T: ColumnKind = ColumnKind::Text;
const S: ColumnKind = ColumnKind::Significant(6);
const I: ColumnKind = ColumnKind::Integer;

/// Most terminal-node rows printed.
const MAX_RULE_ROWS: usize = 200;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn kv(title: &str, rows: Vec<(&str, String)>) -> Table {
    let mut t = Table::new(title, vec![Column::new("Statistic", T), Column::new("Value", T)]);
    for (k, v) in rows {
        t.row(vec![Cell::text(k), Cell::text(v)]);
    }
    t
}

fn pct(x: f64) -> String {
    format!("{:.2}%", 100.0 * x)
}

fn regression_table(title: &str, f: &RegressionFit) -> Table {
    kv(
        title,
        vec![
            ("Observations", f.n.to_string()),
            ("R-squared", if f.r2.is_nan() { "*".into() } else { pct(f.r2) }),
            ("RMSE", fmt_num(f.rmse)),
            ("MAE", fmt_num(f.mae)),
        ],
    )
}

fn class_table(title: &str, f: &ClassFit) -> Table {
    let mut rows = vec![
        ("Observations", f.n.to_string()),
        ("Accuracy", pct(f.accuracy)),
        ("Misclassification rate", pct(1.0 - f.accuracy)),
    ];
    if let Some(a) = f.auc {
        rows.push(("Area under ROC curve", fmt_num(a)));
    }
    if let Some(l) = f.log_loss {
        rows.push(("Log-loss", fmt_num(l)));
    }
    kv(title, rows)
}

fn confusion_table(title: &str, f: &ClassFit) -> Table {
    let mut cols = vec![Column::new("Actual class", T)];
    cols.extend(f.levels.iter().map(|l| Column::new(format!("Pred {l}"), I)));
    cols.push(Column::new("Total", I));
    cols.push(Column::new("% Correct", ColumnKind::Fixed(2)));
    let mut t = Table::new(title, cols);
    for (a, row) in f.confusion.iter().enumerate() {
        let total: usize = row.iter().sum();
        let mut cells = vec![Cell::text(f.levels[a].clone())];
        cells.extend(row.iter().map(|&c| int(c)));
        cells.push(int(total));
        cells.push(if total > 0 { Cell::Num(100.0 * row[a] as f64 / total as f64) } else { Cell::Missing });
        t.row(cells);
    }
    t
}

/// A table of per-predictor columns sorted by `sort` (descending), plus a relative-importance bar chart.
fn importance_block(s: &mut Section, schema: &Dataset, cols: Vec<(&str, Vec<f64>)>, sort: usize, rel: &[usize]) {
    let p = schema.names.len();
    if p == 0 || cols.is_empty() {
        return;
    }
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| cols[sort].1[b].total_cmp(&cols[sort].1[a]).then(a.cmp(&b)));
    let mut columns = vec![Column::new("Predictor", T)];
    for (h, _) in &cols {
        columns.push(Column::new(*h, S));
    }
    let rels: Vec<Vec<f64>> = rel.iter().map(|&c| relative_importance(&cols[c].1)).collect();
    for &c in rel {
        columns.push(Column::new(format!("Relative {}", cols[c].0), ColumnKind::Fixed(1)));
    }
    let mut t = Table::new("Variable Importance", columns);
    for &j in &order {
        let mut row = vec![Cell::text(schema.names[j].clone())];
        row.extend(cols.iter().map(|(_, v)| Cell::Num(v[j])));
        row.extend(rels.iter().map(|r| Cell::Num(r[j])));
        t.row(row);
    }
    s.table(t);
    if let Some(r) = rels.first() {
        s.chart(ChartSpec {
            kind: ChartKind::Bar,
            title: "Variable Importance".into(),
            x_label: "Predictor".into(),
            y_label: "Relative importance (%)".into(),
            categories: order.iter().map(|&j| schema.names[j].clone()).collect(),
            series: vec![Series {
                name: "Relative importance".into(),
                x: (1..=p).map(|i| i as f64).collect(),
                y: order.iter().map(|&j| r[j]).collect(),
            }],
            reference_lines: Vec::new(),
        });
    }
}

fn class_probs(levels: &[String], p: &[f64]) -> String {
    levels
        .iter()
        .zip(p)
        .map(|(l, q)| format!("{l}: {}", fmt_num((q * 1000.0).round() / 1000.0)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_surrogate(tree: &Tree, schema: &Dataset, t: &Test, reversed: bool) -> String {
    // "x <= 3" sends left; reversed means the same condition sends right.
    let text = tree.describe_test(t, schema, true);
    if reversed {
        format!("{text} -> right")
    } else {
        format!("{text} -> left")
    }
}

impl Reportable for CartModel {
    fn report_section(&self) -> Section {
        let class = self.is_classification();
        let mut s = Section::new(if class { "Classification Tree" } else { "Regression Tree" });
        let o = &self.options;
        let pruning = match o.pruning {
            Pruning::None => "None (largest tree)".to_string(),
            Pruning::MinCv => format!("Minimum cross-validated error ({}-fold)", o.cv_folds),
            Pruning::OneSe => format!("Smallest tree within 1 SE of the minimum ({}-fold)", o.cv_folds),
            Pruning::Alpha(a) => format!("Complexity parameter {}", fmt_num(a)),
            Pruning::Leaves(m) => format!("At most {m} terminal nodes"),
        };
        let mut rows = vec![
            (
                "Response type",
                if class { format!("Classification ({} classes)", self.levels.len()) } else { "Regression".into() },
            ),
            ("Splitting criterion", self.criterion.label().to_string()),
            ("Observations", self.n.to_string()),
            ("Predictors", self.schema.n_features().to_string()),
            ("Missing predictor values", self.n_missing_cells.to_string()),
            ("Surrogate splits per node (up to)", o.max_surrogates.to_string()),
            ("Minimum node size to split / terminal node size", format!("{} / {}", o.min_split, o.min_leaf)),
            ("Terminal nodes, largest tree", self.full_leaves.to_string()),
            ("Pruning rule", pruning),
            ("Terminal nodes, selected tree", self.tree.n_leaves().to_string()),
            ("Depth of selected tree", self.tree.depth().to_string()),
        ];
        if let Some(r) = self.sequence.get(self.selected) {
            rows.push(("Complexity parameter of selected tree", fmt_num(r.alpha)));
        }
        s.table(kv("Model Summary", rows));

        if let Some(f) = &self.regression {
            let mut t = regression_table("Model Fit (training data)", f);
            if let Some(cv) = &self.cv {
                t.row(vec![
                    Cell::text("Cross-validated R-squared"),
                    Cell::text(format!("{} (SE {})", pct(cv.score), pct(cv.se))),
                ]);
            }
            s.table(t);
        }
        if let Some(f) = &self.classification {
            let mut t = class_table("Model Fit (training data)", f);
            if let Some(cv) = &self.cv {
                t.row(vec![
                    Cell::text("Cross-validated accuracy"),
                    Cell::text(format!("{} (SE {})", pct(cv.score), pct(cv.se))),
                ]);
            }
            s.table(t);
            s.table(confusion_table("Confusion Matrix (training data)", f));
        }

        // Cost-complexity table.
        let has_cv = self.sequence.iter().any(|r| r.cv_error.is_some());
        let mut cols = vec![Column::new("CP", S), Column::new("Terminal Nodes", I), Column::new("Relative Error", S)];
        if has_cv {
            cols.push(Column::new("CV Error", S));
            cols.push(Column::new("CV SE", S));
        }
        cols.push(Column::new("Selected", T));
        let mut t = Table::new("Cost-Complexity Pruning Sequence", cols);
        for (k, r) in self.sequence.iter().enumerate() {
            let mut row = vec![Cell::Num(r.alpha), int(r.leaves), Cell::Num(r.error)];
            if has_cv {
                row.push(Cell::opt(r.cv_error));
                row.push(Cell::opt(r.cv_se));
            }
            row.push(Cell::text(if k == self.selected { "*" } else { "" }));
            t.row(row);
        }
        t.footnotes.push(
            "Errors are relative to the root node. R_a(T) = R(T) + a * (terminal nodes); CV Error is the K-fold cross-validated error."
                .into(),
        );
        s.table(t);
        if has_cv && self.sequence.len() > 1 {
            let pts: Vec<(f64, f64)> =
                self.sequence.iter().filter_map(|r| r.cv_error.map(|e| (r.leaves as f64, e))).collect();
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: "Cross-Validated Error versus Tree Size".into(),
                x_label: "Terminal nodes".into(),
                y_label: "Relative error".into(),
                categories: Vec::new(),
                series: vec![
                    Series {
                        name: "CV error".into(),
                        x: pts.iter().map(|p| p.0).collect(),
                        y: pts.iter().map(|p| p.1).collect(),
                    },
                    Series {
                        name: "Resubstitution error".into(),
                        x: self.sequence.iter().map(|r| r.leaves as f64).collect(),
                        y: self.sequence.iter().map(|r| r.error).collect(),
                    },
                ],
                reference_lines: Vec::new(),
            });
        }

        importance_block(&mut s, &self.schema, vec![("Importance", self.importance.clone())], 0, &[0]);

        // Splits with surrogates.
        let tree = &self.tree;
        let mut t = Table::new(
            "Splits",
            vec![
                Column::new("Node", I),
                Column::new("N", I),
                Column::new("Split (left branch)", T),
                Column::new("Improvement", S),
                Column::new("Surrogate splits (agreement)", T),
            ],
        );
        for (id, node) in tree.nodes.iter().enumerate() {
            let Some(sp) = &node.split else { continue };
            let sur: Vec<String> = sp
                .surrogates
                .iter()
                .map(|x| {
                    format!(
                        "{} ({})",
                        describe_surrogate(tree, &self.schema, &x.test, x.reversed),
                        fmt_num((x.agreement * 1000.0).round() / 1000.0)
                    )
                })
                .collect();
            t.row(vec![
                int(id + 1),
                int(node.n),
                Cell::text(tree.describe_test(&sp.test, &self.schema, true)),
                Cell::Num(sp.gain),
                Cell::text(sur.join("; ")),
            ]);
        }
        if !t.rows.is_empty() {
            s.table(t);
        }

        // Terminal-node rules.
        let numbers = terminal_numbers(tree);
        let mut cols = vec![Column::new("Terminal Node", I), Column::new("Node", I), Column::new("N", I)];
        cols.push(Column::new(if class { "Predicted Class" } else { "Predicted Value" }, if class { T } else { S }));
        if class {
            cols.push(Column::new("Class Proportions", T));
        }
        cols.push(Column::new("Rule", T));
        let mut t = Table::new("Terminal Node Rules", cols);
        let mut shown = 0;
        for (id, node) in tree.nodes.iter().enumerate() {
            if !node.is_leaf() {
                continue;
            }
            if shown == MAX_RULE_ROWS {
                t.footnotes
                    .push(format!("Only the first {MAX_RULE_ROWS} of {} terminal nodes are listed.", tree.n_leaves()));
                break;
            }
            shown += 1;
            let rule = {
                let c = tree.path_conditions(id, &self.schema);
                if c.is_empty() {
                    "(all observations)".to_string()
                } else {
                    c.join(" AND ")
                }
            };
            let mut row = vec![int(numbers[id]), int(id + 1), int(node.n)];
            if class {
                row.push(Cell::text(self.levels[tree.class_of(id)].clone()));
                row.push(Cell::text(class_probs(&self.levels, &tree.proportions(id))));
            } else {
                row.push(Cell::Num(node.value));
            }
            row.push(Cell::text(rule));
            t.row(row);
        }
        s.table(t);
        if self.n_missing_cells > 0 {
            s.text("Observations with a missing predictor value follow the node's surrogate splits, in order of agreement, and the majority branch when every surrogate is also missing.");
        }
        s
    }
}

impl Reportable for ForestModel {
    fn report_section(&self) -> Section {
        let class = self.is_classification();
        let mut s = Section::new(if class { "Random Forest Classification" } else { "Random Forest Regression" });
        let o = &self.options;
        let mut rows = vec![
            (
                "Response type",
                if class { format!("Classification ({} classes)", self.levels.len()) } else { "Regression".into() },
            ),
            ("Observations", self.n.to_string()),
            ("Predictors", self.schema.n_features().to_string()),
            ("Number of trees", self.trees.len().to_string()),
            ("Predictors tried per split (mtry)", self.mtry.to_string()),
            ("Minimum terminal node size", self.min_leaf.to_string()),
            ("Splitting criterion", self.criterion.label().to_string()),
            (
                "Bootstrap sample size",
                format!("{} of {} observations", ((o.sample_fraction * self.n as f64).round() as usize).max(1), self.n),
            ),
            (
                "Average terminal nodes per tree",
                fmt_num(self.trees.iter().map(|t| t.n_leaves() as f64).sum::<f64>() / self.trees.len() as f64),
            ),
        ];
        let with_oob = self.has_oob.iter().filter(|&&b| b).count();
        rows.push(("Observations with an out-of-bag prediction", with_oob.to_string()));
        if class {
            rows.push(("Out-of-bag error rate", pct(self.oob_error)));
        } else {
            rows.push(("Out-of-bag mean squared error", fmt_num(self.oob_error)));
            if let Some(r2) = self.oob_r2 {
                rows.push(("Out-of-bag R-squared", if r2.is_nan() { "*".into() } else { pct(r2) }));
            }
        }
        s.table(kv("Model Summary", rows));
        if let Some(f) = &self.oob_class_fit {
            s.table(class_table("Out-of-Bag Performance", f));
            s.table(confusion_table("Confusion Matrix (out-of-bag)", f));
        }
        if let Some(f) = &self.oob_regression_fit {
            s.table(regression_table("Out-of-Bag Performance", f));
        }
        if self.oob_curve.len() > 1 {
            let (xs, ys): (Vec<f64>, Vec<f64>) = self
                .oob_curve
                .iter()
                .enumerate()
                .filter(|(_, e)| e.is_finite())
                .map(|(i, &e)| ((i + 1) as f64, e))
                .unzip();
            if xs.len() > 1 {
                s.chart(ChartSpec {
                    kind: ChartKind::Line,
                    title: "Out-of-Bag Error versus Number of Trees".into(),
                    x_label: "Trees".into(),
                    y_label: if class { "Error rate".into() } else { "Mean squared error".into() },
                    categories: Vec::new(),
                    series: vec![Series { name: "Out-of-bag error".into(), x: xs, y: ys }],
                    reference_lines: Vec::new(),
                });
            }
        }
        let mut cols = vec![("Mean decrease in impurity", self.impurity_importance.clone())];
        let mut rel = vec![0];
        let mut sort = 0;
        if let Some(pm) = &self.permutation {
            cols = vec![
                (
                    if class { "Permutation (increase in OOB error)" } else { "Permutation (increase in OOB MSE)" },
                    pm.mean.clone(),
                ),
                ("Permutation SE", pm.se.clone()),
                ("Mean decrease in impurity", self.impurity_importance.clone()),
            ];
            rel = vec![0, 2];
            sort = 0;
        }
        importance_block(&mut s, &self.schema, cols, sort, &rel);
        if let Some(cov) = self.interval_coverage {
            let level = o.interval_level.unwrap_or(0.95);
            s.text(format!(
                "Out-of-bag {}% quantile-forest prediction intervals contain the observed response for {} of the observations that have one (nominal {}).",
                fmt_num(level * 100.0),
                pct(cov),
                pct(level)
            ));
        }
        s.text("Out-of-bag estimates use, for each observation, only the trees whose bootstrap sample did not contain it, so they are not optimistic like the in-sample fit.");
        s
    }
}

impl Reportable for GbdtModel {
    fn report_section(&self) -> Section {
        let class = self.is_classification();
        let mut s =
            Section::new(if class { "Gradient Boosting Classification" } else { "Gradient Boosting Regression" });
        let o = &self.options;
        let mut rows = vec![
            (
                "Response type",
                if class { format!("Classification ({} classes)", self.levels.len()) } else { "Regression".into() },
            ),
            ("Loss function", self.loss.label(self.levels.len()).to_string()),
            ("Observations", self.n.to_string()),
            ("Predictors", self.schema.n_features().to_string()),
            ("Training / validation observations", format!("{} / {}", self.n_train, self.n_valid)),
            ("Learning rate", fmt_num(self.learning_rate)),
            ("Maximum tree depth", o.max_depth.to_string()),
            ("Minimum terminal node size", o.min_leaf.to_string()),
            ("Subsample fraction", fmt_num(o.subsample)),
            ("Trees requested", o.n_trees.to_string()),
            ("Trees fitted", self.train_loss.len().to_string()),
            ("Trees kept", self.best_iteration.to_string()),
        ];
        if self.loss == super::gbdt::GbdtLoss::Huber {
            rows.push(("Huber quantile", fmt_num(o.huber_alpha)));
        }
        if self.n_valid > 0 {
            rows.push(("Best iteration (lowest validation loss)", self.best_iteration.to_string()));
        }
        s.table(kv("Model Summary", rows));
        if let Some(f) = &self.train_regression {
            s.table(regression_table("Model Fit (training rows)", f));
        }
        if let Some(f) = &self.valid_regression {
            s.table(regression_table("Model Fit (validation rows)", f));
        }
        if let Some(f) = &self.train_class {
            s.table(class_table("Model Fit (training rows)", f));
            s.table(confusion_table("Confusion Matrix (training rows)", f));
        }
        if let Some(f) = &self.valid_class {
            s.table(class_table("Model Fit (validation rows)", f));
            s.table(confusion_table("Confusion Matrix (validation rows)", f));
        }
        if self.train_loss.len() > 1 {
            let xs: Vec<f64> = (1..=self.train_loss.len()).map(|i| i as f64).collect();
            let mut series = vec![Series { name: "Training".into(), x: xs.clone(), y: self.train_loss.clone() }];
            if !self.valid_loss.is_empty() {
                series.push(Series { name: "Validation".into(), x: xs, y: self.valid_loss.clone() });
            }
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: "Loss versus Number of Trees".into(),
                x_label: "Trees".into(),
                y_label: "Loss".into(),
                categories: Vec::new(),
                series,
                reference_lines: Vec::new(),
            });
        }
        importance_block(&mut s, &self.schema, vec![("Influence", self.importance.clone())], 0, &[0]);
        s
    }
}

impl Reportable for MarsModel {
    fn report_section(&self) -> Section {
        let mut s = Section::new("MARS Regression");
        let rows = vec![
            ("Observations", self.n.to_string()),
            ("Predictors", self.names.len().to_string()),
            ("Terms created by the forward pass", self.forward_terms.to_string()),
            ("Terms in the final model", self.terms.len().to_string()),
            ("Penalty per knot", fmt_num(self.penalty)),
            ("Minimum span / end span", format!("{} / {}", self.spans.0, self.spans.1)),
            ("R-squared", pct(self.r2())),
            ("Generalized R-squared (GRSq)", pct(self.grsq())),
            ("GCV", fmt_num(self.gcv)),
            ("Root mean squared error", fmt_num(self.rmse())),
        ];
        s.table(kv("Model Summary", rows));
        let mut t = Table::new(
            "Basis Functions",
            vec![Column::new("Term", T), Column::new("Definition", T), Column::new("Coefficient", S)],
        );
        for (i, term) in self.terms.iter().enumerate() {
            let name = if i == 0 { "Constant".to_string() } else { format!("BF{i}") };
            t.row(vec![Cell::text(name), Cell::text(self.term_label(term)), Cell::Num(term.coef)]);
        }
        s.table(t);
        let mut eq = String::from("Y = ");
        for (i, term) in self.terms.iter().enumerate() {
            if i == 0 {
                eq.push_str(&fmt_num(term.coef));
            } else {
                let sign = if term.coef < 0.0 { '-' } else { '+' };
                eq.push_str(&format!("\n  {sign} {} * {}", fmt_num(term.coef.abs()), self.term_label(term)));
            }
        }
        s.text(eq);
        if !self.path.is_empty() {
            let mut p = Table::new(
                "Backward Elimination",
                vec![Column::new("Terms", I), Column::new("RSS", S), Column::new("GCV", S), Column::new("Selected", T)],
            );
            for r in &self.path {
                p.row(vec![
                    int(r.terms),
                    Cell::Num(r.rss),
                    Cell::Num(r.gcv),
                    Cell::text(if r.selected { "*" } else { "" }),
                ]);
            }
            s.table(p);
            let pts: Vec<&super::mars::PruneRow> = self.path.iter().rev().filter(|r| r.gcv.is_finite()).collect();
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: "GCV versus Number of Terms".into(),
                x_label: "Terms".into(),
                y_label: "GCV".into(),
                categories: Vec::new(),
                series: vec![Series {
                    name: "GCV".into(),
                    x: pts.iter().map(|r| r.terms as f64).collect(),
                    y: pts.iter().map(|r| r.gcv).collect(),
                }],
                reference_lines: Vec::new(),
            });
        }
        if !self.importance.is_empty() {
            let mut t = Table::new(
                "Variable Importance",
                vec![
                    Column::new("Predictor", T),
                    Column::new("GCV importance", ColumnKind::Fixed(1)),
                    Column::new("RSS importance", ColumnKind::Fixed(1)),
                ],
            );
            for (v, g, r) in &self.importance {
                t.row(vec![Cell::text(self.names[*v].clone()), Cell::Num(*g), Cell::Num(*r)]);
            }
            s.table(t);
            s.chart(ChartSpec {
                kind: ChartKind::Bar,
                title: "Relative Variable Importance".into(),
                x_label: "Predictor".into(),
                y_label: "Importance".into(),
                categories: self.importance.iter().map(|(v, _, _)| self.names[*v].clone()).collect(),
                series: vec![Series {
                    name: "Importance".into(),
                    x: Vec::new(),
                    y: self.importance.iter().map(|e| e.1).collect(),
                }],
                reference_lines: Vec::new(),
            });
        }
        s.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: "Fitted versus Observed".into(),
            x_label: "Fitted".into(),
            y_label: "Observed".into(),
            categories: Vec::new(),
            series: vec![Series { name: "Observed".into(), x: self.fitted.clone(), y: self.y.clone() }],
            reference_lines: Vec::new(),
        });
        s
    }
}

impl Reportable for super::mars_ext::MarsBinaryModel {
    fn report_section(&self) -> Section {
        let mut s = Section::new(format!("MARS Binary Logistic Regression: P({})", self.levels[1]));
        s.table(kv(
            "Model Summary",
            vec![
                ("Observations", self.n.to_string()),
                ("Event", self.levels[1].clone()),
                ("Terms (including intercept)", self.terms.len().to_string()),
                ("Basis chosen by", format!("least-squares MARS (GCV penalty {})", fmt_num(self.basis_model.penalty))),
                ("Deviance", fmt_num(self.deviance)),
                ("Null deviance", fmt_num(self.null_deviance)),
                ("Deviance R-sq (McFadden)", pct(self.pseudo_r2())),
                ("AIC", fmt_num(self.aic)),
                (
                    "IRLS iterations",
                    format!("{}{}", self.iterations, if self.converged { "" } else { " (not converged)" }),
                ),
            ],
        ));
        let mut t = Table::new(
            "Coefficients",
            vec![
                Column::new("Term", T),
                Column::new("Coef", S),
                Column::new("SE Coef", S),
                Column::new("Z-Value", ColumnKind::Fixed(2)),
                Column::new("P-Value", ColumnKind::PValue),
            ],
        );
        for c in &self.coefficients {
            t.row(vec![
                Cell::text(c.label.clone()),
                Cell::Num(c.coef),
                Cell::Num(c.se),
                Cell::Num(c.z),
                Cell::Num(c.p),
            ]);
        }
        s.table(t);
        s.table(class_table("Classification of the Training Data (cutoff 0.5)", &self.fit));
        s.table(confusion_table("Confusion Matrix (training data)", &self.fit));
        if !self.converged {
            s.text("Warning: the iteratively reweighted least squares did not converge; the classes may be separated.");
        }
        s
    }
}

impl Reportable for super::mars_ext::MarsCv {
    fn report_section(&self) -> Section {
        let mut s = Section::new("MARS Cross-Validation");
        s.text(format!(
            "{}-fold cross-validation: the whole model (forward pass, pruning{}) is refitted on each training fold.",
            self.folds,
            if self.binary { ", logistic coefficients" } else { "" }
        ));
        let mut cols = vec![
            Column::new("Degree", I),
            Column::new("Penalty", ColumnKind::Fixed(1)),
            Column::new("Mean terms", ColumnKind::Fixed(1)),
        ];
        if self.binary {
            cols.extend([Column::new("Log-loss", S), Column::new("Error rate", S), Column::new("AUC", S)]);
        } else {
            cols.extend([Column::new("CV RMSE", S), Column::new("CV R-Sq", S)]);
        }
        cols.push(Column::new("Best", T));
        let mut t = Table::new("Cross-Validated Settings", cols);
        for (i, r) in self.rows.iter().enumerate() {
            let mut cells = vec![int(r.degree), Cell::Num(r.penalty), Cell::Num(r.mean_terms)];
            if self.binary {
                cells.extend([Cell::Num(r.log_loss), Cell::Num(r.error_rate), Cell::Num(r.auc)]);
            } else {
                cells.extend([Cell::Num(r.rmse), Cell::Num(r.r2)]);
            }
            cells.push(Cell::text(if i == self.best { "*" } else { "" }));
            t.row(cells);
        }
        s.table(t);
        let b = &self.rows[self.best];
        s.text(format!("Best setting: degree {}, penalty {}.", b.degree, fmt_num(b.penalty)));
        s
    }
}
