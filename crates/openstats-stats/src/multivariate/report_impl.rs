//! [`Reportable`] implementations for the multivariate results.

use super::cluster::PartitionSummary;
use super::*;
use crate::report::{
    Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Reportable, Section, Series, Table,
};
use openstats_math::Matrix;

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn chart(kind: ChartKind, title: &str, xl: &str, yl: &str, series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind,
        title: title.into(),
        x_label: xl.into(),
        y_label: yl.into(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    }
}

fn series(name: &str, x: Vec<f64>, y: Vec<f64>) -> Series {
    Series { name: name.into(), x, y }
}

/// Table with a text label column followed by numeric columns.
fn labelled(title: &str, first: &str, heads: &[String], kind: ColumnKind, rows: &[(String, Vec<f64>)]) -> Table {
    let mut cols = vec![Column::new(first, T)];
    cols.extend(heads.iter().map(|h| Column::new(h.clone(), kind)));
    let mut t = Table::new(title, cols);
    for (name, vals) in rows {
        let mut r = vec![Cell::text(name.clone())];
        r.extend(vals.iter().map(|&v| Cell::Num(v)));
        t.row(r);
    }
    t
}

fn heads(prefix: &str, k: usize) -> Vec<String> {
    (1..=k).map(|i| format!("{prefix}{i}")).collect()
}

fn matrix_table(title: &str, first: &str, names: &[String], col_heads: &[String], m: &Matrix) -> Table {
    let rows: Vec<(String, Vec<f64>)> = (0..m.rows()).map(|i| (names[i].clone(), m.row(i).to_vec())).collect();
    labelled(title, first, col_heads, S, &rows)
}

fn count_table(title: &str, groups: &[String], c: &[Vec<usize>], first: &str) -> Table {
    let g = groups.len();
    let mut cols = vec![Column::new(first, T)];
    cols.extend(groups.iter().map(|n| Column::new(format!("Put into {n}"), I)));
    cols.push(Column::new("Total N", I));
    cols.push(Column::new("N correct", I));
    cols.push(Column::new("Proportion", ColumnKind::Fixed(3)));
    let mut t = Table::new(title, cols);
    let mut tot = 0;
    let mut ok = 0;
    for a in 0..g {
        let n: usize = c[a].iter().sum();
        let mut r = vec![Cell::text(groups[a].clone())];
        r.extend(c[a].iter().map(|&v| int(v)));
        r.push(int(n));
        r.push(int(c[a][a]));
        r.push(Cell::Num(if n > 0 { c[a][a] as f64 / n as f64 } else { f64::NAN }));
        t.row(r);
        tot += n;
        ok += c[a][a];
    }
    let mut r = vec![Cell::text("All")];
    for b in 0..g {
        r.push(int((0..g).map(|a| c[a][b]).sum()));
    }
    r.push(int(tot));
    r.push(int(ok));
    r.push(Cell::Num(if tot > 0 { ok as f64 / tot as f64 } else { f64::NAN }));
    t.row(r);
    t
}

impl Reportable for Pca {
    fn report_section(&self) -> Section {
        let p = self.names.len();
        let kind = match self.matrix {
            MatrixKind::Correlation => "Correlation",
            MatrixKind::Covariance => "Covariance",
        };
        let mut s = Section::new(format!("Principal Component Analysis: {}", self.names.join(", ")));
        let mut t = Table::new(
            format!("Eigenanalysis of the {kind} Matrix"),
            vec![
                Column::new("Component", I),
                Column::new("Eigenvalue", S),
                Column::new("Proportion", ColumnKind::Fixed(3)),
                Column::new("Cumulative", ColumnKind::Fixed(3)),
            ],
        );
        for i in 0..p {
            t.row(vec![
                int(i + 1),
                Cell::Num(self.eigenvalues[i]),
                Cell::Num(self.proportion[i]),
                Cell::Num(self.cumulative[i]),
            ]);
        }
        t.footnotes.push(format!("{} observations used; {} with missing values omitted.", self.n, self.n_dropped));
        s.table(t);
        let show = p.min(12);
        s.table(matrix_table(
            "Eigenvectors",
            "Variable",
            &self.names,
            &heads("PC", show),
            &self.loadings.submatrix(0, p, 0, show),
        ));
        let comps: Vec<f64> = (1..=p).map(|i| i as f64).collect();
        let mut scree = chart(
            ChartKind::Line,
            "Scree Plot",
            "Component Number",
            "Eigenvalue",
            vec![series("Eigenvalue", comps, self.eigenvalues.clone())],
        );
        if self.matrix == MatrixKind::Correlation {
            scree.reference_lines.push(ReferenceLine { axis: Axis::Y, value: 1.0, label: "1".into() });
        }
        s.chart(scree);
        if self.scores.cols() >= 2 {
            s.chart(chart(
                ChartKind::Scatter,
                "Score Plot",
                "First Component",
                "Second Component",
                vec![series("Scores", self.scores.col(0), self.scores.col(1))],
            ));
        }
        if p >= 2 {
            s.chart(chart(
                ChartKind::Scatter,
                "Loading Plot",
                "First Component",
                "Second Component",
                vec![series("Loadings", self.loadings.col(0), self.loadings.col(1))],
            ));
        }
        s
    }
}

impl Reportable for FactorAnalysis {
    fn report_section(&self) -> Section {
        let p = self.names.len();
        let m = self.loadings.cols();
        let method = match self.method {
            FactorMethod::PrincipalComponents => "Principal Component",
            FactorMethod::MaximumLikelihood => "Maximum Likelihood",
        };
        let mut s = Section::new(format!("Factor Analysis: {}", self.names.join(", ")));
        s.text(format!("{method} extraction of {m} factors"));
        let fh = heads("Factor", m);
        let with_comm = |title: &str, l: &Matrix, with_variance: bool| {
            let mut cols = vec![Column::new("Variable", T)];
            cols.extend(fh.iter().map(|h| Column::new(h.clone(), S)));
            cols.push(Column::new("Communality", S));
            let mut t = Table::new(title, cols);
            for i in 0..p {
                let mut r = vec![Cell::text(self.names[i].clone())];
                r.extend((0..m).map(|j| Cell::Num(l[(i, j)])));
                r.push(Cell::Num(self.communalities[i]));
                t.row(r);
            }
            if with_variance {
                let mut r = vec![Cell::text("Variance")];
                r.extend(self.variance.iter().map(|&v| Cell::Num(v)));
                r.push(Cell::Num(self.communalities.iter().sum()));
                t.row(r);
                let mut r = vec![Cell::text("% Var")];
                r.extend(self.proportion.iter().map(|&v| Cell::Num(v)));
                r.push(Cell::Num(self.communalities.iter().sum::<f64>() / self.total_variance));
                t.row(r);
            }
            t
        };
        if self.rotation != Rotation::None {
            let mut un = with_comm("Unrotated Factor Loadings and Communalities", &self.unrotated, false);
            let uv: Vec<f64> = (0..m).map(|j| (0..p).map(|i| self.unrotated[(i, j)].powi(2)).sum()).collect();
            let mut r = vec![Cell::text("Variance")];
            r.extend(uv.iter().map(|&v| Cell::Num(v)));
            r.push(Cell::Num(self.communalities.iter().sum()));
            un.row(r);
            let mut r = vec![Cell::text("% Var")];
            r.extend(uv.iter().map(|&v| Cell::Num(v / self.total_variance)));
            r.push(Cell::Num(self.communalities.iter().sum::<f64>() / self.total_variance));
            un.row(r);
            s.table(un);
            let name = match self.rotation {
                Rotation::Varimax => "Varimax",
                Rotation::Quartimax => "Quartimax",
                Rotation::Equamax => "Equamax",
                Rotation::Orthomax(_) => "Orthomax",
                Rotation::None => "",
            };
            s.table(with_comm(&format!("Rotated Factor Loadings and Communalities ({name})"), &self.loadings, true));
        } else {
            s.table(with_comm("Unrotated Factor Loadings and Communalities", &self.loadings, true));
        }
        let mut un = Table::new("Uniqueness", vec![Column::new("Variable", T), Column::new("Uniqueness", S)]);
        for i in 0..p {
            un.row(vec![Cell::text(self.names[i].clone()), Cell::Num(self.uniquenesses[i])]);
        }
        s.table(un);
        if let Some(t) = &self.fit_test {
            let mut tb = Table::new(
                "Likelihood Ratio Test for the Number of Factors",
                vec![Column::new("Chi-Square", S), Column::new("DF", I), Column::new("P-Value", P)],
            );
            tb.row(vec![Cell::Num(t.chi_square), int(t.df), Cell::Num(t.p_value)]);
            s.table(tb);
        }
        if !self.converged {
            s.warn("The maximum likelihood iterations did not converge; the solution may be a local optimum.");
        }
        if self.uniquenesses.iter().any(|&u| u < 0.0051 * 1.0 && self.method == FactorMethod::MaximumLikelihood) {
            s.warn("A uniqueness reached its lower bound (a Heywood case): the solution is on the boundary.");
        }
        s.table(matrix_table("Factor Score Coefficients", "Variable", &self.names, &fh, &self.score_coefficients));
        let ev: Vec<f64> = self.eigenvalues.clone();
        s.chart(chart(
            ChartKind::Line,
            "Scree Plot",
            "Factor Number",
            "Eigenvalue",
            vec![series("Eigenvalue", (1..=ev.len()).map(|i| i as f64).collect(), ev)],
        ));
        if m >= 2 {
            s.chart(chart(
                ChartKind::Scatter,
                "Loading Plot",
                "Factor 1",
                "Factor 2",
                vec![series("Loadings", self.loadings.col(0), self.loadings.col(1))],
            ));
            if let Some(sc) = &self.scores {
                s.chart(chart(
                    ChartKind::Scatter,
                    "Score Plot",
                    "Factor 1",
                    "Factor 2",
                    vec![series("Scores", sc.col(0), sc.col(1))],
                ));
            }
        }
        s
    }
}

impl Reportable for Discriminant {
    fn report_section(&self) -> Section {
        let g = self.groups.len();
        let p = self.variables.len();
        let mut s = Section::new(format!(
            "{} Discriminant Analysis: {}",
            if self.quadratic { "Quadratic" } else { "Linear" },
            self.variables.join(", ")
        ));
        let mut t = Table::new(
            "Group Information",
            vec![Column::new("Group", T), Column::new("N", I), Column::new("Prior", ColumnKind::Fixed(5))],
        );
        for k in 0..g {
            t.row(vec![Cell::text(self.groups[k].clone()), int(self.counts[k]), Cell::Num(self.priors[k])]);
        }
        t.footnotes.push(format!("{} observations used; {} omitted for missing values.", self.n, self.n_dropped));
        s.table(t);
        s.table(count_table("Summary of Classification", &self.groups, &self.confusion, "True Group"));
        if let Some(c) = &self.loo_confusion {
            s.table(count_table(
                "Summary of Classification with Cross-Validation (leave one out)",
                &self.groups,
                c,
                "True Group",
            ));
        }
        s.table(matrix_table("Group Means", "Group", &self.groups, &self.variables, &self.means));
        if !self.quadratic {
            let mut cols = vec![Column::new("Variable", T)];
            cols.extend(self.groups.iter().map(|n| Column::new(n.clone(), S)));
            let mut tb = Table::new("Linear Discriminant Function for Groups", cols);
            let mut r = vec![Cell::text("Constant")];
            r.extend(self.constants.iter().map(|&c| Cell::Num(c)));
            tb.row(r);
            for v in 0..p {
                let mut r = vec![Cell::text(self.variables[v].clone())];
                r.extend((0..g).map(|k| Cell::Num(self.coefficients[(k, v)])));
                tb.row(r);
            }
            s.table(tb);
            s.table(matrix_table(
                "Squared Distance Between Groups",
                "Group",
                &self.groups,
                &self.groups,
                &self.squared_distances,
            ));
        }
        // Misclassified observations.
        let mut mis = Table::new(
            "Summary of Misclassified Observations",
            vec![
                Column::new("Observation", I),
                Column::new("True group", T),
                Column::new("Predicted group", T),
                Column::new("Posterior probability", ColumnKind::Fixed(3)),
            ],
        );
        for i in 0..self.n {
            if self.predicted[i] != self.true_group[i] {
                mis.row(vec![
                    int(self.rows_used[i] + 1),
                    Cell::text(self.groups[self.true_group[i]].clone()),
                    Cell::text(self.groups[self.predicted[i]].clone()),
                    Cell::Num(self.posterior[(i, self.predicted[i])]),
                ]);
            }
        }
        if !mis.rows.is_empty() {
            s.table(mis);
        }
        if let Some(c) = &self.canonical {
            let k = c.eigenvalues.len();
            let mut t = Table::new(
                "Canonical Discriminant Functions",
                vec![
                    Column::new("Function", I),
                    Column::new("Eigenvalue", S),
                    Column::new("Proportion", ColumnKind::Fixed(3)),
                    Column::new("Cumulative", ColumnKind::Fixed(3)),
                ],
            );
            for i in 0..k {
                t.row(vec![
                    int(i + 1),
                    Cell::Num(c.eigenvalues[i]),
                    Cell::Num(c.proportion[i]),
                    Cell::Num(c.cumulative[i]),
                ]);
            }
            s.table(t);
            s.table(matrix_table(
                "Canonical Coefficients",
                "Variable",
                &self.variables,
                &heads("Function", k),
                &c.coefficients,
            ));
            if k >= 2 {
                s.chart(chart(
                    ChartKind::Scatter,
                    "Canonical Scores",
                    "Function 1",
                    "Function 2",
                    (0..g)
                        .map(|gi| {
                            let idx: Vec<usize> = (0..self.n).filter(|&i| self.true_group[i] == gi).collect();
                            series(
                                &self.groups[gi],
                                idx.iter().map(|&i| c.scores[(i, 0)]).collect(),
                                idx.iter().map(|&i| c.scores[(i, 1)]).collect(),
                            )
                        })
                        .collect(),
                ));
            }
        }
        s
    }
}

impl Reportable for Linkage {
    fn report_section(&self) -> Section {
        let kind = if self.rows_used.is_empty() && self.items.iter().any(|s| s.parse::<usize>().is_err()) {
            "Variables"
        } else {
            "Observations"
        };
        let method = match self.method {
            LinkageMethod::Single => "Single",
            LinkageMethod::Complete => "Complete",
            LinkageMethod::Average => "Average",
            LinkageMethod::Centroid => "Centroid",
            LinkageMethod::Median => "Median",
            LinkageMethod::Ward => "Ward",
            LinkageMethod::McQuitty => "McQuitty",
        };
        let mut s = Section::new(format!("Cluster Analysis of {kind}: {method} Linkage"));
        let mut t = Table::new(
            "Amalgamation Steps",
            vec![
                Column::new("Step", I),
                Column::new("Number of clusters", I),
                Column::new("Similarity level", S),
                Column::new("Distance level", S),
                Column::new("Clusters joined", I),
                Column::new("Clusters joined with", I),
                Column::new("New cluster", I),
                Column::new("Number of obs. in new cluster", I),
            ],
        );
        for (i, st) in self.steps.iter().enumerate() {
            t.row(vec![
                int(i + 1),
                int(self.n - i - 1),
                Cell::Num(st.similarity),
                Cell::Num(st.distance),
                int(st.label_left),
                int(st.label_right),
                int(st.label_new),
                int(st.size),
            ]);
        }
        if self.n_dropped > 0 {
            t.footnotes.push(format!("{} observations with missing values were omitted.", self.n_dropped));
        }
        s.table(t);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (x, y) in self.dendrogram_paths() {
            xs.extend(x);
            ys.extend(y);
            xs.push(f64::NAN);
            ys.push(f64::NAN);
        }
        let mut c = chart(
            ChartKind::Line,
            "Dendrogram",
            "Observations (in dendrogram order)",
            "Distance",
            vec![series("Dendrogram", xs, ys)],
        );
        c.categories = self.order.iter().map(|&i| self.items[i].clone()).collect();
        s.chart(c);
        s
    }
}

/// Final partition tables (cluster sizes, within SS, distances, centroids).
pub fn partition_section(title: &str, names: &[String], labels: &[usize], sm: &PartitionSummary) -> Section {
    let k = sm.sizes.len();
    let mut s = Section::new(title);
    let mut t = Table::new(
        "Final Partition",
        vec![
            Column::new("Cluster", I),
            Column::new("Number of observations", I),
            Column::new("Within cluster sum of squares", S),
            Column::new("Average distance from centroid", S),
            Column::new("Maximum distance from centroid", S),
        ],
    );
    for c in 0..k {
        t.row(vec![
            int(c + 1),
            int(sm.sizes[c]),
            Cell::Num(sm.within_ss[c]),
            Cell::Num(sm.average_distance[c]),
            Cell::Num(sm.maximum_distance[c]),
        ]);
    }
    t.footnotes.push(format!("Total within-cluster sum of squares: {:.6}", sm.total_within_ss));
    s.table(t);
    let ch = heads("Cluster", k);
    s.table(matrix_table("Cluster Centroids", "Variable", names, &ch, &sm.centroids.t()));
    s.table(matrix_table("Distances Between Cluster Centroids", "Cluster", &ch, &ch, &sm.centroid_distances));
    let _ = labels;
    s
}

impl Reportable for KMeans {
    fn report_section(&self) -> Section {
        let names: Vec<String> = (1..=self.summary.centroids.cols()).map(|i| format!("C{i}")).collect();
        let mut s = partition_section(
            &format!("K-means Cluster Analysis ({} clusters)", self.k),
            &names,
            &self.labels,
            &self.summary,
        );
        if self.standardized {
            s.text("Variables were standardized; distances are in standardized units.");
        }
        if !self.converged {
            s.warn("The k-means iterations did not converge within the iteration limit.");
        }
        s
    }
}

impl Reportable for Manova {
    fn report_section(&self) -> Section {
        let mut s = Section::new(format!("General MANOVA: {}", self.responses.join(", ")));
        s.text(format!("{} observations used; error degrees of freedom {}.", self.n, self.error_df));
        for t in &self.terms {
            let mut tb = Table::new(
                format!("MANOVA for {} (s = {}, df = {})", t.label, self.responses.len().min(t.df), t.df),
                vec![
                    Column::new("Criterion", T),
                    Column::new("Test Statistic", S),
                    Column::new("F", S),
                    Column::new("Num DF", S),
                    Column::new("Denom DF", S),
                    Column::new("P", P),
                ],
            );
            for x in &t.tests {
                tb.row(vec![
                    Cell::text(x.name),
                    Cell::Num(x.statistic),
                    Cell::Num(x.f),
                    Cell::Num(x.df1),
                    Cell::Num(x.df2),
                    Cell::Num(x.p_value),
                ]);
            }
            tb.footnotes.push(
                "Roy's F is an upper bound; the other F values are exact when min(p, df) <= 2 and approximate otherwise.".into(),
            );
            s.table(tb);
            let mut ev = Table::new(
                format!("Eigenvalues of E^-1 H for {}", t.label),
                vec![Column::new("Eigenvalue", S), Column::new("Proportion", ColumnKind::Fixed(4))],
            );
            let tot: f64 = t.eigenvalues.iter().sum();
            for &e in t.eigenvalues.iter().take(self.responses.len().min(t.df)) {
                ev.row(vec![Cell::Num(e), Cell::Num(if tot > 0.0 { e / tot } else { f64::NAN })]);
            }
            s.table(ev);
        }
        s.table(matrix_table("Error SSCP Matrix", "Response", &self.responses, &self.responses, &self.error_sscp));
        s.table(matrix_table(
            "Partial Correlations from the Error SSCP Matrix",
            "Response",
            &self.responses,
            &self.responses,
            &self.error_correlation,
        ));
        s
    }
}

impl Reportable for ItemAnalysis {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Cronbach's Alpha");
        let mut t = Table::new(
            "Item Analysis",
            vec![
                Column::new("Item", T),
                Column::new("Adj. total mean", S),
                Column::new("Adj. total StDev", S),
                Column::new("Item-adj. total corr", S),
                Column::new("Squared multiple corr", S),
                Column::new("Alpha if item omitted", S),
            ],
        );
        for r in &self.items {
            t.row(vec![
                Cell::text(r.name.clone()),
                Cell::Num(r.adjusted_total_mean),
                Cell::Num(r.adjusted_total_sd),
                Cell::Num(r.item_total_correlation),
                Cell::Num(r.squared_multiple_correlation),
                Cell::Num(r.alpha_if_omitted),
            ]);
        }
        t.footnotes.push(format!("{} observations used.", self.n));
        s.table(t);
        let mut a = Table::new("Cronbach's Alpha", vec![Column::new("Alpha", S), Column::new("Standardized alpha", S)]);
        a.row(vec![Cell::Num(self.alpha), Cell::Num(self.standardized_alpha)]);
        s.table(a);
        let names: Vec<String> = self.items.iter().map(|i| i.name.clone()).collect();
        s.table(matrix_table("Correlation Matrix", "Item", &names, &names, &self.correlations));
        s
    }
}

fn coord_table(
    title: &str,
    first: &str,
    labels: &[String],
    mass: &[f64],
    coord: &Matrix,
    contrib: &Matrix,
    quality: &Matrix,
    inertia: &[f64],
) -> Table {
    let k = coord.cols();
    let mut cols = vec![Column::new(first, T), Column::new("Mass", S), Column::new("Inertia", S)];
    for a in 0..k {
        cols.push(Column::new(format!("Coord {}", a + 1), S));
        cols.push(Column::new(format!("Contr {}", a + 1), S));
        cols.push(Column::new(format!("Sq Cos {}", a + 1), S));
    }
    let mut t = Table::new(title, cols);
    for i in 0..labels.len() {
        let mut row = vec![Cell::Text(labels[i].clone()), Cell::Num(mass[i]), Cell::Num(inertia[i])];
        for a in 0..k {
            row.push(Cell::Num(coord[(i, a)]));
            row.push(Cell::Num(contrib[(i, a)]));
            row.push(Cell::Num(quality[(i, a)]));
        }
        t.row(row);
    }
    t
}

impl Reportable for CorrespondenceAnalysis {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Correspondence Analysis");
        let mut inertia = Table::new(
            "Analysis of Contingency Table",
            vec![
                Column::new("Axis", I),
                Column::new("Singular value", S),
                Column::new("Inertia", S),
                Column::new("Proportion", S),
                Column::new("Cumulative", S),
            ],
        );
        let mut cum = 0.0;
        for (a, &lam) in self.inertia.iter().enumerate() {
            let prop = if self.total_inertia > 0.0 { lam / self.total_inertia } else { 0.0 };
            cum += prop;
            inertia.row(vec![
                int(a + 1),
                Cell::Num(self.singular_values[a]),
                Cell::Num(lam),
                Cell::Num(prop),
                Cell::Num(cum),
            ]);
        }
        inertia.footnotes.push(format!(
            "Total inertia {:.6}; chi-square {:.4} on {} observations.",
            self.total_inertia, self.chi_square, self.total
        ));
        s.table(inertia);
        s.table(coord_table(
            "Row Contributions",
            "Row",
            &self.row_labels,
            &self.row_mass,
            &self.row_coord,
            &self.row_contrib,
            &self.row_quality,
            &self.row_inertia,
        ));
        s.table(coord_table(
            "Column Contributions",
            "Column",
            &self.col_labels,
            &self.col_mass,
            &self.col_coord,
            &self.col_contrib,
            &self.col_quality,
            &self.col_inertia,
        ));
        if self.row_coord.cols() >= 2 {
            let pts =
                |m: &Matrix| (0..m.rows()).map(|i| (m[(i, 0)], m[(i, 1)])).unzip::<f64, f64, Vec<f64>, Vec<f64>>();
            let (rx, ry) = pts(&self.row_coord);
            let (cx, cy) = pts(&self.col_coord);
            s.chart(chart(
                ChartKind::Scatter,
                "Symmetric Plot",
                &format!("Component 1 ({:.1}%)", 100.0 * self.inertia[0] / self.total_inertia.max(1e-300)),
                &format!("Component 2 ({:.1}%)", 100.0 * self.inertia[1] / self.total_inertia.max(1e-300)),
                vec![Series { name: "Rows".into(), x: rx, y: ry }, Series { name: "Columns".into(), x: cx, y: cy }],
            ));
        }
        s
    }
}

impl Promax {
    /// Pattern, structure and factor correlations labelled with the variable names.
    pub fn report_with_names(&self, names: &[String]) -> Section {
        let m = self.pattern.cols();
        let mat = |title: &str, a: &Matrix| {
            let mut cols = vec![Column::new("Variable", T)];
            cols.extend((0..m).map(|j| Column::new(format!("Factor{}", j + 1), S)));
            let mut t = Table::new(title, cols);
            for i in 0..a.rows() {
                let mut row = vec![Cell::Text(names.get(i).cloned().unwrap_or_else(|| format!("V{}", i + 1)))];
                row.extend((0..m).map(|j| Cell::Num(a[(i, j)])));
                t.row(row);
            }
            t
        };
        let mut s = Section::new("Promax Rotation");
        s.table(mat("Pattern Loadings", &self.pattern));
        s.table(mat("Structure Loadings", &self.structure));
        let labels: Vec<String> = (1..=m).map(|j| format!("Factor{j}")).collect();
        let mut cols = vec![Column::new("", T)];
        cols.extend(labels.iter().map(|l| Column::new(l.clone(), S)));
        let mut t = Table::new("Factor Correlations", cols);
        for i in 0..m {
            let mut row = vec![Cell::Text(labels[i].clone())];
            row.extend((0..m).map(|j| Cell::Num(self.factor_correlation[(i, j)])));
            t.row(row);
        }
        s.table(t);
        s.text(format!("Promax rotation with power {}.", self.power));
        s
    }
}

impl Reportable for Promax {
    fn report_section(&self) -> Section {
        self.report_with_names(&[])
    }
}
