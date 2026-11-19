//! Multivariate analyses (Stat ▸ Multivariate): principal components, factor analysis,
//! discriminant analysis, clustering of observations and variables, k-means, MANOVA and item analysis.
//!
//! Rows with a missing value in any of the analysed columns are left out (listwise); stored columns
//! are missing in those rows.

use super::analyses::{add, out};
use super::analyses_adv::ident;
use super::analysis::{col_title, expand, labels, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::model::Data;
use openstats_stats::multivariate::{
    cluster_variables, correspondence_analysis, cronbach_alpha, discriminant, factor_analysis, hierarchical, kmeans,
    manova, multiple_correspondence, pca, promax, DiscriminantOptions, Distance, FactorMethod, FactorOptions,
    KMeansInit, KMeansOptions, LinkageMethod, ManovaOptions, MatrixKind, PartitionSummary, PcaOptions, Priors,
    Rotation,
};
use openstats_stats::report::{Block, Cell, Column as RCol, ColumnKind, Reportable, Section, Table};

const MATRICES: [&str; 2] = ["Correlation", "Covariance"];
const LINKAGES: [&str; 7] = ["Single", "Complete", "Average", "Centroid", "Median", "Ward", "McQuitty"];
const DISTANCES: [&str; 5] = ["Euclidean", "Squared Euclidean", "Manhattan", "Pearson", "Squared Pearson"];

fn matrix_kind(v: &Values) -> MatrixKind {
    if v.choice("matrix") == 1 {
        MatrixKind::Covariance
    } else {
        MatrixKind::Correlation
    }
}

fn linkage(v: &Values) -> LinkageMethod {
    match v.choice("linkage") {
        0 => LinkageMethod::Single,
        1 => LinkageMethod::Complete,
        3 => LinkageMethod::Centroid,
        4 => LinkageMethod::Median,
        5 => LinkageMethod::Ward,
        6 => LinkageMethod::McQuitty,
        _ => LinkageMethod::Average,
    }
}

fn distance(v: &Values) -> Distance {
    match v.choice("distance") {
        1 => Distance::SquaredEuclidean,
        2 => Distance::Manhattan,
        3 => Distance::Pearson,
        4 => Distance::SquaredPearson,
        _ => Distance::Euclidean,
    }
}

/// The selected numeric columns: values, titles and the worksheet length.
fn gather(ws: &Worksheet, v: &Values, key: &str) -> Res<(Vec<Vec<f64>>, Vec<String>, usize)> {
    let sel = v.vars(key);
    if sel.is_empty() {
        return Err("select at least one variable".into());
    }
    let mut cols = Vec::new();
    for &c in sel {
        cols.push(numeric(ws, c)?);
    }
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return Err("the selected columns have different lengths".into());
    }
    Ok((cols, sel.iter().map(|&c| col_title(ws, c)).collect(), n))
}

fn keep_charts(s: &mut Section, keep: impl Fn(&str) -> bool) {
    s.blocks.retain(|b| match b {
        Block::Chart(c) => keep(&c.title),
        _ => true,
    });
}

fn optional_count(v: &Values, key: &str) -> Res<Option<usize>> {
    let x = v.num(key);
    if x.is_nan() {
        return Ok(None);
    }
    if x < 1.0 || x.fract() != 0.0 || x > 1e6 {
        return Err(format!("{key}: enter a whole number of at least 1, or leave it blank").into());
    }
    Ok(Some(x as usize))
}

// --------------------------------------------------------------------------- PCA

fn pca_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (cols, names, n) = gather(ws, v, "vars")?;
    let p = cols.len();
    if p < 2 {
        return Err("select at least two variables".into());
    }
    let k = optional_count(v, "ncomp")?.map(|k| k.min(p)).unwrap_or(p);
    let r = pca(&cols, &names, &PcaOptions { matrix: matrix_kind(v), n_components: Some(k) })?;
    let mut s = r.report_section();
    let (scree, score, load) = (v.flag("scree"), v.flag("score_plot"), v.flag("loading_plot"));
    keep_charts(&mut s, |t| match t {
        "Scree Plot" => scree,
        "Score Plot" => score,
        "Loading Plot" => load,
        _ => true,
    });
    let mut stored = Vec::new();
    if v.flag("store_scores") {
        for j in 0..k {
            stored.push(expand(&format!("PC{}", j + 1), n, &r.rows_used, &r.scores.col(j)));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// --------------------------------------------------------------------------- Factor analysis

fn factor_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (cols, names, n) = gather(ws, v, "vars")?;
    let m = v.count("factors");
    let mut o = FactorOptions::new(m);
    o.method =
        if v.choice("method") == 1 { FactorMethod::MaximumLikelihood } else { FactorMethod::PrincipalComponents };
    o.matrix = matrix_kind(v);
    o.rotation = match v.choice("rotation") {
        1 | 5 => Rotation::Varimax,
        2 => Rotation::Quartimax,
        3 => Rotation::Equamax,
        4 => Rotation::Orthomax(v.num("gamma")),
        _ => Rotation::None,
    };
    o.kaiser = v.flag("kaiser");
    o.scores = v.flag("store_scores") || v.flag("score_plot");
    let r = factor_analysis(&cols, &names, &o)?;
    let mut s = r.report_section();
    if v.choice("rotation") == 5 {
        if m < 2 {
            return Err("promax rotation needs at least two factors".into());
        }
        let pm = promax(&r.unrotated, &r.loadings, &r.rotation_matrix, v.num("power"))?;
        s.blocks.extend(pm.report_with_names(&names).blocks);
    }
    let (scree, load, score) = (v.flag("scree"), v.flag("loading_plot"), v.flag("score_plot"));
    keep_charts(&mut s, |t| match t {
        "Scree Plot" => scree,
        "Loading Plot" => load,
        "Score Plot" => score,
        _ => true,
    });
    let mut stored = Vec::new();
    if v.flag("store_scores") {
        if let Some(sc) = &r.scores {
            for j in 0..m {
                stored.push(expand(&format!("FSCORE{}", j + 1), n, &r.rows_used, &sc.col(j)));
            }
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ------------------------------------------------------------------- correspondence analysis

fn ca_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars").to_vec();
    if cols.len() < 2 {
        return Err("select at least two columns of counts".into());
    }
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let n = data[0].len();
    let row_labels = match v.var("row_names") {
        Some(c) => labels(ws, c)?,
        None => (1..=n).map(|i| format!("Row {i}")).collect(),
    };
    // Rows made only of missing cells are blank worksheet rows and are skipped.
    let keep: Vec<usize> = (0..n).filter(|&i| data.iter().any(|c| !c[i].is_nan())).collect();
    if keep.iter().any(|&i| data.iter().any(|c| c[i].is_nan())) {
        return Err("the table contains missing counts".into());
    }
    let table: Vec<Vec<f64>> = keep.iter().map(|&i| data.iter().map(|c| c[i]).collect()).collect();
    let rl: Vec<String> = keep.iter().map(|&i| row_labels[i].clone()).collect();
    let cl: Vec<String> = cols.iter().map(|&c| col_title(ws, c)).collect();
    let k = v.count("components");
    let r = correspondence_analysis(&table, &rl, &cl, (k > 0).then_some(k))?;
    let mut s = r.report_section();
    s.heading = "Simple Correspondence Analysis".to_string();
    if !v.flag("plot") {
        s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    }
    let mut stored = Vec::new();
    if v.flag("store_coord") {
        for a in 0..r.row_coord.cols() {
            let mut full = vec![f64::NAN; n];
            for (j, &i) in keep.iter().enumerate() {
                full[i] = r.row_coord[(j, a)];
            }
            stored.push(Column::numeric(format!("RowCoord{}", a + 1), full));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn mca_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars").to_vec();
    if cols.len() < 2 {
        return Err("select at least two categorical columns".into());
    }
    let data: Vec<Vec<String>> = cols.iter().map(|&c| labels(ws, c)).collect::<Res<_>>()?;
    let names: Vec<String> = cols.iter().map(|&c| col_title(ws, c)).collect();
    let k = v.count("components");
    let r = multiple_correspondence(&data, &names, (k > 0).then_some(k))?;
    let mut s = r.report_section();
    s.heading = "Multiple Correspondence Analysis".to_string();
    if !v.flag("plot") {
        s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    }
    let n = data[0].len();
    let used: Vec<usize> = (0..n).filter(|&i| data.iter().all(|c| !c[i].is_empty())).collect();
    let mut stored = Vec::new();
    if v.flag("store_coord") {
        for a in 0..r.row_coord.cols() {
            let mut full = vec![f64::NAN; n];
            for (j, &i) in used.iter().enumerate() {
                full[i] = r.row_coord[(j, a)];
            }
            stored.push(Column::numeric(format!("ObsCoord{}", a + 1), full));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// --------------------------------------------------------------------------- Discriminant

fn discriminant_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let g = v.var("groups").ok_or("select the grouping column")?;
    let (cols, names, n) = gather(ws, v, "vars")?;
    if v.vars("vars").contains(&g) {
        return Err("the grouping column cannot also be a predictor".into());
    }
    let groups = labels(ws, g)?;
    if groups.len() != n {
        return Err("the grouping column and the predictors have different lengths".into());
    }
    let priors = match v.choice("priors") {
        1 => Priors::Proportional,
        2 => {
            let nums: Result<Vec<f64>, _> = v
                .text("prior_values")
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .map(str::parse::<f64>)
                .collect();
            Priors::Given(nums.map_err(|_| "enter the prior probabilities as numbers separated by spaces or commas")?)
        }
        _ => Priors::Equal,
    };
    let o =
        DiscriminantOptions { quadratic: v.choice("method") == 1, priors, cross_validate: v.flag("cross_validate") };
    let r = discriminant(&cols, &names, &groups, &o)?;
    let mut s = r.report_section();
    let canon = v.flag("canonical_plot");
    keep_charts(&mut s, |t| canon || t != "Canonical Scores");
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        let mut col = vec![String::new(); n];
        for (pos, &row) in r.rows_used.iter().enumerate() {
            col[row] = r.groups[r.predicted[pos]].clone();
        }
        stored.push(Column::text("DAFITS", col));
    }
    if v.flag("store_probs") {
        for (k, name) in r.groups.iter().enumerate() {
            stored.push(expand(&format!("DAPROB_{name}"), n, &r.rows_used, &r.posterior.col(k)));
        }
    }
    if v.flag("store_canonical") {
        if let Some(c) = &r.canonical {
            for j in 0..c.eigenvalues.len() {
                stored.push(expand(&format!("DASCORE{}", j + 1), n, &r.rows_used, &c.scores.col(j)));
            }
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// --------------------------------------------------------------------------- Clustering

fn cluster_obs_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (cols, names, n) = gather(ws, v, "vars")?;
    let (k, sim) = (optional_count(v, "k")?, v.num("similarity"));
    if k.is_some() && !sim.is_nan() {
        return Err("give either a number of clusters or a similarity level, not both".into());
    }
    if !sim.is_nan() && !(0.0..=100.0).contains(&sim) {
        return Err("the similarity level must be between 0 and 100".into());
    }
    let std = v.flag("standardize");
    let l = hierarchical(&cols, linkage(v), distance(v), std)?;
    let mut s = l.report_section();
    if !v.flag("dendrogram") {
        keep_charts(&mut s, |t| t != "Dendrogram");
    }
    let mut stored = Vec::new();
    let labels = match (k, sim.is_nan()) {
        (Some(k), _) => Some(l.cut_clusters(k)?),
        (None, false) => Some(l.cut_similarity(sim)?),
        _ => None,
    };
    if let Some(lab) = labels {
        let sm = PartitionSummary::from_partition(&cols, &l.rows_used, &lab, std)?;
        let mut ps = openstats_stats::multivariate::partition_section("Final Partition", &names, &lab, &sm);
        s.blocks.append(&mut ps.blocks);
        if v.flag("store_membership") {
            let vals: Vec<f64> = lab.iter().map(|&x| x as f64).collect();
            stored.push(expand("CLUSTER", n, &l.rows_used, &vals));
        }
    } else if v.flag("store_membership") {
        return Err("choose a number of clusters or a similarity level to store the cluster membership".into());
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn cluster_vars_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (cols, names, _) = gather(ws, v, "vars")?;
    let (k, sim) = (optional_count(v, "k")?, v.num("similarity"));
    if k.is_some() && !sim.is_nan() {
        return Err("give either a number of clusters or a similarity level, not both".into());
    }
    let l = cluster_variables(&cols, &names, linkage(v), v.choice("measure") == 1)?;
    let mut s = l.report_section();
    if !v.flag("dendrogram") {
        keep_charts(&mut s, |t| t != "Dendrogram");
    }
    let labels = match (k, sim.is_nan()) {
        (Some(k), _) => Some(l.cut_clusters(k)?),
        (None, false) => Some(l.cut_similarity(sim)?),
        _ => None,
    };
    if let Some(lab) = labels {
        let mut t = Table::new(
            "Cluster Membership",
            vec![RCol::new("Variable", ColumnKind::Text), RCol::new("Cluster", ColumnKind::Integer)],
        );
        for (name, c) in names.iter().zip(&lab) {
            t.push_row(vec![Cell::text(name.clone()), Cell::Int(*c as i64)])?;
        }
        s.table(t);
    }
    Ok(Output { section: s, stored: Vec::new(), models: Vec::new() })
}

fn kmeans_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (cols, names, n) = gather(ws, v, "vars")?;
    let k = v.count("k");
    let mut o = KMeansOptions::new(k);
    o.standardize = v.flag("standardize");
    let seed = v.num("seed");
    let seed = if seed.is_finite() && seed >= 0.0 { seed as u64 } else { 1 };
    o.init = if v.choice("init") == 1 { KMeansInit::Random(seed) } else { KMeansInit::PlusPlus(seed) };
    if let Some(pc) = v.var("start") {
        // The starting partition lists a cluster number for every row that will be used.
        let col = numeric(ws, pc)?;
        let complete = openstats_stats::multivariate::complete_rows(&cols);
        let mut part = Vec::with_capacity(complete.len());
        for &r in &complete {
            let x = col.get(r).copied().unwrap_or(f64::NAN);
            if x.is_nan() || x < 1.0 || x > k as f64 || x.fract() != 0.0 {
                return Err(format!(
                    "row {} of the initial partition column must be a cluster number from 1 to {k}",
                    r + 1
                )
                .into());
            }
            part.push(x as usize);
        }
        o.init = KMeansInit::Partition(part);
    }
    let r = kmeans(&cols, &o)?;
    let mut s = openstats_stats::multivariate::partition_section(
        &format!("K-means Cluster Analysis: {}", names.join(", ")),
        &names,
        &r.labels,
        &r.summary,
    );
    if r.standardized {
        s.text("Variables were standardized; distances are in standardized units.");
    }
    if !r.converged {
        s.warn("The k-means iterations did not converge within the iteration limit.");
    }
    let mut stored = Vec::new();
    if v.flag("store_membership") {
        let vals: Vec<f64> = r.labels.iter().map(|&x| x as f64).collect();
        stored.push(expand("KCLUSTER", n, &r.rows_used, &vals));
    }
    if v.flag("store_distance") {
        stored.push(expand("KDIST", n, &r.rows_used, &r.distances));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// --------------------------------------------------------------------------- MANOVA, item analysis

fn manova_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (resp, rnames, n) = gather(ws, v, "responses")?;
    if resp.len() < 2 {
        return Err("select at least two responses".into());
    }
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    if factors.is_empty() && covs.is_empty() {
        return Err("select at least one factor or covariate".into());
    }
    let responses = v.vars("responses");
    if factors.iter().chain(&covs).any(|c| responses.contains(c)) {
        return Err("a response cannot also be a model term".into());
    }
    if factors.iter().any(|f| covs.contains(f)) {
        return Err("a column cannot be both a factor and a covariate".into());
    }
    let mut taken: Vec<String> = Vec::new();
    let mut data = Data::new();
    let mut atoms = Vec::new();
    for &c in &factors {
        let id = ident(ws, c, &taken);
        taken.push(id.clone());
        let l = labels(ws, c)?;
        if l.len() != n {
            return Err("the model terms and the responses have different lengths".into());
        }
        data = data.with_factor(&id, &l)?;
        atoms.push(id);
    }
    for &c in &covs {
        let id = ident(ws, c, &taken);
        taken.push(id.clone());
        let x = numeric(ws, c)?;
        if x.len() != n {
            return Err("the model terms and the responses have different lengths".into());
        }
        data = data.with_numeric(&id, x)?;
        atoms.push(id);
    }
    let order = v.count("order").max(1);
    let rhs = if order > 1 && atoms.len() > 1 {
        format!("({})^{}", atoms.join(" + "), order.min(atoms.len()))
    } else {
        atoms.join(" + ")
    };
    let r = manova(&resp, &rnames, &data, &rhs, &ManovaOptions { treatment_coding: false })?;
    let mut s = r.report_section();
    s.heading = format!(
        "General MANOVA: {} versus {}",
        rnames.join(", "),
        factors.iter().chain(&covs).map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    );
    out(s)
}

fn item_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (cols, names, _) = gather(ws, v, "vars")?;
    let r = cronbach_alpha(&cols, &names)?;
    let mut s = r.report_section();
    s.heading = format!("Cronbach's Alpha: {}", names.join(", "));
    out(s)
}

pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Options, Storage};
    let menu = |last: &'static str| ["Stat", "Multivariate", last];
    let cluster_options = |s: DialogSpec| {
        s.choice(Options, "linkage", "Linkage method", &LINKAGES, 2)
            .optional_number(Options, "k", "Number of clusters in the final partition (blank = none)")
            .optional_number(Options, "similarity", "or: similarity level of the final partition, % (blank = none)")
            .check(Graphs, "dendrogram", "Dendrogram", true)
    };
    add(
        r,
        DialogSpec::new(
            "stat.mv.pca",
            "Principal Components",
            "Eigenanalysis of the correlation or covariance matrix, with component scores.",
        )
        .vars("vars", "Variables", VarKind::Numeric, 2, 4000)
        .choice(Options, "matrix", "Matrix to use", &MATRICES, 0)
        .optional_number(Options, "ncomp", "Number of components to compute (blank = all)")
        .check(Graphs, "scree", "Scree plot", true)
        .check(Graphs, "score_plot", "Score plot (first two components)", true)
        .check(Graphs, "loading_plot", "Loading plot (first two components)", true)
        .check(Storage, "store_scores", "Component scores", false),
        &menu("Principal Components..."),
        Runner::Report(pca_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.factor",
            "Factor Analysis",
            "Principal-component or maximum-likelihood factors with orthogonal rotation and regression scores.",
        )
        .vars("vars", "Variables", VarKind::Numeric, 2, 4000)
        .integer(Options, "factors", "Number of factors", 2.0, 1.0, 100.0)
        .choice(Options, "method", "Method of extraction", &["Principal components", "Maximum likelihood"], 0)
        .choice(Options, "matrix", "Matrix to factor", &MATRICES, 0)
        .choice(
            Options,
            "rotation",
            "Type of rotation",
            &["None", "Varimax", "Quartimax", "Equamax", "Orthomax", "Promax (oblique, after varimax)"],
            1,
        )
        .number(Options, "power", "Promax power", 4.0, 1.01, 20.0)
        .number(Options, "gamma", "Orthomax gamma (0 = quartimax, 1 = varimax)", 1.0, 0.0, 100.0)
        .check(Options, "kaiser", "Kaiser row normalization in the rotation", true)
        .check(Graphs, "scree", "Scree plot", true)
        .check(Graphs, "loading_plot", "Loading plot (first two factors)", true)
        .check(Graphs, "score_plot", "Score plot (first two factors)", false)
        .check(Storage, "store_scores", "Factor scores (regression method)", false),
        &menu("Factor Analysis..."),
        Runner::Report(factor_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.ca",
            "Simple Correspondence Analysis",
            "Correspondence analysis of a table of counts (one worksheet column per table column).",
        )
        .vars("vars", "Columns of the table", VarKind::Numeric, 2, 4000)
        .vars("row_names", "Row names (optional)", VarKind::Any, 0, 1)
        .integer(Options, "components", "Number of components (0 = all)", 0.0, 0.0, 100.0)
        .check(Graphs, "plot", "Symmetric plot", true)
        .check(Storage, "store_coord", "Row coordinates", false),
        &menu("Simple Correspondence Analysis..."),
        Runner::Report(ca_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.mca",
            "Multiple Correspondence Analysis",
            "Correspondence analysis of the indicator matrix of two or more categorical variables.",
        )
        .vars("vars", "Categorical variables", VarKind::Any, 2, 4000)
        .integer(Options, "components", "Number of components (0 = all)", 0.0, 0.0, 100.0)
        .check(Graphs, "plot", "Symmetric plot", true)
        .check(Storage, "store_coord", "Observation coordinates", false),
        &menu("Multiple Correspondence Analysis..."),
        Runner::Report(mca_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.discriminant",
            "Discriminant Analysis",
            "Classify observations into groups with linear or quadratic discriminant functions.",
        )
        .vars("groups", "Groups", VarKind::Any, 1, 1)
        .vars("vars", "Predictors", VarKind::Numeric, 1, 4000)
        .choice(Options, "method", "Discriminant function", &["Linear", "Quadratic"], 0)
        .choice(
            Options,
            "priors",
            "Prior probabilities",
            &["Equal for all groups", "Proportional to group sizes", "Specified below"],
            0,
        )
        .text(Options, "prior_values", "Specified priors (one per group in level order)", "")
        .check(Options, "cross_validate", "Leave-one-out cross-validation", true)
        .check(Graphs, "canonical_plot", "Canonical score plot", true)
        .check(Storage, "store_fits", "Predicted group", false)
        .check(Storage, "store_probs", "Posterior probabilities", false)
        .check(Storage, "store_canonical", "Canonical discriminant scores", false),
        &menu("Discriminant Analysis..."),
        Runner::Report(discriminant_run),
    );
    add(
        r,
        cluster_options(
            DialogSpec::new(
                "stat.mv.cluster_obs",
                "Cluster Observations",
                "Agglomerative hierarchical clustering of the rows, with a dendrogram and an optional final partition.",
            )
            .vars("vars", "Variables", VarKind::Numeric, 1, 4000)
            .choice(Options, "distance", "Distance measure", &DISTANCES, 0)
            .check(Options, "standardize", "Standardize variables", false),
        )
        .check(Storage, "store_membership", "Cluster membership (needs a final partition)", false),
        &menu("Cluster Observations..."),
        Runner::Report(cluster_obs_run),
    );
    add(
        r,
        cluster_options(
            DialogSpec::new(
                "stat.mv.cluster_vars",
                "Cluster Variables",
                "Agglomerative clustering of the columns using their correlations.",
            )
            .vars("vars", "Variables", VarKind::Numeric, 2, 4000)
            .choice(
                Options,
                "measure",
                "Distance measure",
                &["Correlation (1 - r)", "Absolute correlation (1 - |r|)"],
                0,
            ),
        ),
        &menu("Cluster Variables..."),
        Runner::Report(cluster_vars_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.kmeans",
            "Cluster K-Means",
            "Partition the rows into k clusters minimizing the within-cluster sum of squares.",
        )
        .vars("vars", "Variables", VarKind::Numeric, 1, 4000)
        .integer(Options, "k", "Number of clusters", 3.0, 2.0, 1000.0)
        .choice(Options, "init", "Starting centres", &["k-means++", "Random observations"], 0)
        .integer(Options, "seed", "Random seed", 1.0, 0.0, 4294967295.0)
        .vars("start", "Initial partition column (optional)", VarKind::Numeric, 0, 1)
        .check(Options, "standardize", "Standardize variables", false)
        .check(Storage, "store_membership", "Cluster membership", false)
        .check(Storage, "store_distance", "Distance from the cluster centre", false),
        &menu("Cluster K-Means..."),
        Runner::Report(kmeans_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.manova",
            "General MANOVA",
            "Multivariate analysis of variance with Wilks, Lawley-Hotelling, Pillai and Roy statistics.",
        )
        .vars("responses", "Responses", VarKind::Numeric, 2, 100)
        .vars("factors", "Factors", VarKind::Any, 0, 20)
        .vars("covariates", "Covariates", VarKind::Numeric, 0, 20)
        .integer(Options, "order", "Interactions up to order", 1.0, 1.0, 6.0),
        &menu("General MANOVA..."),
        Runner::Report(manova_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.mv.item",
            "Item Analysis",
            "Cronbach's alpha, item-total statistics and the item correlation matrix.",
        )
        .vars("vars", "Items", VarKind::Numeric, 2, 4000),
        &menu("Item Analysis..."),
        Runner::Report(item_run),
    );
}
