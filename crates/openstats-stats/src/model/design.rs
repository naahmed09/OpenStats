//! Design-matrix builder: worksheet columns + a [`Formula`] → numeric matrix.
//!
//! Categorical predictors are encoded with one of the [`Contrast`] codings
//! (treatment / dummy, sum-to-zero, Helmert, orthogonal polynomial, custom).
//! The encoding is remembered in a [`ModelSpec`] so that new data (for
//! prediction) is encoded with exactly the levels and contrasts of the fit.
//!
//! Rows with a missing value (`NaN` for numbers, the empty string for text)
//! in any column the model uses are dropped (listwise); the indices of the
//! rows kept are returned.

use super::expr::Expr;
use super::formula::{AtomKind, Formula, Term};
use crate::error::{invalid, Result, StatsError};
use openstats_math::linalg::{Matrix, Qr};
use std::collections::{BTreeMap, BTreeSet};

/// A set of named worksheet columns.
#[derive(Debug, Clone, Default)]
pub struct Data {
    numeric: BTreeMap<String, Vec<f64>>,
    factors: BTreeMap<String, Vec<String>>,
    n: Option<usize>,
}

impl Data {
    /// Empty data set.
    pub fn new() -> Data {
        Data::default()
    }

    fn check(&mut self, name: &str, len: usize) -> Result<()> {
        if name.is_empty() {
            return invalid("column names must not be empty");
        }
        if self.numeric.contains_key(name) || self.factors.contains_key(name) {
            return invalid(format!("duplicate column name '{name}'"));
        }
        match self.n {
            Some(n) if n != len => invalid(format!("column '{name}' has {len} rows, expected {n}")),
            _ => {
                self.n = Some(len);
                Ok(())
            }
        }
    }

    /// Add a numeric column (`NaN` = missing).
    pub fn with_numeric(mut self, name: &str, values: Vec<f64>) -> Result<Data> {
        self.check(name, values.len())?;
        if values.iter().any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        self.numeric.insert(name.to_string(), values);
        Ok(self)
    }

    /// Add a categorical column (empty string = missing).
    pub fn with_factor<S: AsRef<str>>(mut self, name: &str, values: &[S]) -> Result<Data> {
        self.check(name, values.len())?;
        self.factors.insert(name.to_string(), values.iter().map(|s| s.as_ref().to_string()).collect());
        Ok(self)
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.n.unwrap_or(0)
    }

    /// Numeric column by name.
    pub fn numeric(&self, name: &str) -> Option<&[f64]> {
        self.numeric.get(name).map(Vec::as_slice)
    }

    /// Categorical column by name.
    pub fn factor(&self, name: &str) -> Option<&[String]> {
        self.factors.get(name).map(Vec::as_slice)
    }
}

/// Which level is the reference of a treatment coding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reference {
    /// The first level in the level order.
    First,
    /// The last level.
    Last,
    /// A named level.
    Level(String),
}

/// How the levels of a factor are encoded as numeric columns.
#[derive(Debug, Clone, PartialEq)]
pub enum Contrast {
    /// Dummy (0/1) indicators for every level but the reference.
    Treatment(Reference),
    /// Sum-to-zero (deviation) coding: levels 1..L−1 get indicators, the
    /// last level is −1 in every column.
    SumToZero,
    /// Helmert contrasts (R's `contr.helmert`): each level against the mean
    /// of the previous ones.
    Helmert,
    /// Orthogonal polynomial contrasts (linear, quadratic, …); optional
    /// numeric level scores, equally spaced by default.
    OrthogonalPolynomial(Option<Vec<f64>>),
    /// User contrast matrix: one row per level, one column per contrast.
    Custom(Matrix),
}

/// Options of the design builder.
#[derive(Debug, Clone)]
pub struct DesignOptions {
    /// Coding for factors without a specific entry.
    pub default_contrast: Contrast,
    /// Per-factor coding, by atom label (column name, or `C(x)`).
    pub contrasts: BTreeMap<String, Contrast>,
    /// Explicit level order per factor; default is sorted (numerically when
    /// every level is a number).
    pub level_order: BTreeMap<String, Vec<String>>,
}

impl Default for DesignOptions {
    fn default() -> Self {
        DesignOptions {
            default_contrast: Contrast::Treatment(Reference::First),
            contrasts: BTreeMap::new(),
            level_order: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
enum AtomSpec {
    Numeric(Expr),
    Factor { column: String, from_numeric: bool, levels: Vec<String>, contrast: Contrast },
}

/// The encoding learned from the fitting data.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The parsed formula.
    pub formula: Formula,
    atoms: BTreeMap<String, AtomSpec>,
}

/// Columns belonging to one model term.
#[derive(Debug, Clone, PartialEq)]
pub struct TermColumns {
    /// Term label (`A`, `A*B`, `x^2`, `A(B)`).
    pub label: String,
    /// Interaction order.
    pub order: u32,
    /// First column index in the design matrix.
    pub start: usize,
    /// Number of columns.
    pub len: usize,
}

/// A numeric design matrix with its bookkeeping.
#[derive(Debug, Clone)]
pub struct DesignMatrix {
    /// n × p design matrix (first column is the constant when the model has an intercept).
    pub x: Matrix,
    /// Column names.
    pub names: Vec<String>,
    /// Column ranges of the terms (the constant is not a term).
    pub terms: Vec<TermColumns>,
    /// Whether column 0 is the constant.
    pub intercept: bool,
    /// Response (when requested).
    pub y: Option<Vec<f64>>,
    /// Offset per row (zeros without offset terms); the regression is of `y − offset`.
    pub offset: Vec<f64>,
    /// Indices of the worksheet rows kept (complete cases).
    pub rows_used: Vec<usize>,
    /// Number of rows dropped for missing values.
    pub n_dropped: usize,
}

// ---------------------------------------------------------------------------
// Contrast matrices
// ---------------------------------------------------------------------------

fn contrast_matrix(c: &Contrast, levels: &[String], label: &str) -> Result<(Matrix, Vec<String>)> {
    let l = levels.len();
    match c {
        Contrast::Treatment(r) => {
            let ref_idx = match r {
                Reference::First => 0,
                Reference::Last => l - 1,
                Reference::Level(name) => match levels.iter().position(|x| x == name) {
                    Some(i) => i,
                    None => return invalid(format!("reference level '{name}' is not a level of {label}")),
                },
            };
            let keep: Vec<usize> = (0..l).filter(|&i| i != ref_idx).collect();
            let m = Matrix::from_fn(l, keep.len(), |i, j| f64::from(i == keep[j]));
            Ok((m, keep.iter().map(|&i| format!("{label}[{}]", levels[i])).collect()))
        }
        Contrast::SumToZero => {
            let m = Matrix::from_fn(l, l - 1, |i, j| {
                if i == j {
                    1.0
                } else if i == l - 1 {
                    -1.0
                } else {
                    0.0
                }
            });
            Ok((m, (0..l - 1).map(|j| format!("{label}[{}]", levels[j])).collect()))
        }
        Contrast::Helmert => {
            let m = Matrix::from_fn(l, l - 1, |i, j| {
                if i <= j {
                    -1.0
                } else if i == j + 1 {
                    (j + 1) as f64
                } else {
                    0.0
                }
            });
            Ok((m, (1..l).map(|j| format!("{label}.H{j}")).collect()))
        }
        Contrast::OrthogonalPolynomial(scores) => {
            let s: Vec<f64> = match scores {
                Some(v) => {
                    if v.len() != l || v.iter().any(|x| !x.is_finite()) {
                        return invalid(format!("{label}: polynomial scores must be {l} finite numbers"));
                    }
                    let mut sorted = v.clone();
                    sorted.sort_by(f64::total_cmp);
                    if sorted.windows(2).any(|w| w[0] == w[1]) {
                        return invalid(format!("{label}: polynomial scores must be distinct"));
                    }
                    v.clone()
                }
                None => (1..=l).map(|i| i as f64).collect(),
            };
            let mean = s.iter().sum::<f64>() / l as f64;
            let spread = s.iter().fold(0.0f64, |m, v| m.max((v - mean).abs()));
            let z: Vec<f64> = s.iter().map(|v| (v - mean) / spread).collect();
            let v = Matrix::from_fn(l, l, |i, j| z[i].powi(j as i32));
            let q = Qr::new(&v)?.q();
            let mut m = Matrix::zeros(l, l - 1);
            for j in 1..l {
                let mut col = q.col(j);
                // Sign: the value at the largest score is positive (R's contr.poly).
                let imax = (0..l).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap_or(l - 1);
                if col[imax] < 0.0 {
                    col.iter_mut().for_each(|x| *x = -*x);
                }
                m.set_col(j - 1, &col);
            }
            let names = (1..l)
                .map(|j| match j {
                    1 => format!("{label}.L"),
                    2 => format!("{label}.Q"),
                    3 => format!("{label}.C"),
                    k => format!("{label}^{k}"),
                })
                .collect();
            Ok((m, names))
        }
        Contrast::Custom(m) => {
            if m.rows() != l || m.cols() == 0 || m.cols() > l - 1 || !m.is_finite() {
                return invalid(format!("{label}: custom contrast must have {l} rows and 1 to {} columns", l - 1));
            }
            Ok((m.clone(), (1..=m.cols()).map(|j| format!("{label}[C{j}]")).collect()))
        }
    }
}

pub(crate) fn level_sort(levels: &mut [String]) {
    let nums: Option<Vec<f64>> = levels.iter().map(|s| s.trim().parse::<f64>().ok()).collect();
    if nums.is_some() {
        levels.sort_by(|a, b| {
            let (x, y) = (a.trim().parse::<f64>().unwrap_or(0.0), b.trim().parse::<f64>().unwrap_or(0.0));
            x.total_cmp(&y).then_with(|| a.cmp(b))
        });
    } else {
        levels.sort();
    }
}

// ---------------------------------------------------------------------------
// Spec construction
// ---------------------------------------------------------------------------

fn numeric_vars_ok(e: &Expr, data: &Data) -> Result<()> {
    for v in e.variables() {
        if data.numeric.contains_key(&v) {
            continue;
        }
        if data.factors.contains_key(&v) {
            return invalid(format!("'{v}' is categorical and cannot be used in an arithmetic expression"));
        }
        return invalid(format!("unknown column '{v}'"));
    }
    Ok(())
}

impl ModelSpec {
    /// Parse `formula`, classify its variables from `data` and learn the
    /// factor levels from the complete cases.
    pub fn new(formula: &str, data: &Data, opts: &DesignOptions) -> Result<ModelSpec> {
        let f = Formula::parse(formula)?;
        // 1. Atom specs without levels.
        let mut atoms: BTreeMap<String, AtomSpec> = BTreeMap::new();
        let all_atoms =
            f.terms.iter().flat_map(|t| t.factors.iter().map(|(a, _)| a).chain(t.nested_in.iter())).collect::<Vec<_>>();
        for a in all_atoms {
            if atoms.contains_key(&a.label) {
                continue;
            }
            let spec = match &a.kind {
                AtomKind::Var(name) => {
                    if data.numeric.contains_key(name) {
                        AtomSpec::Numeric(Expr::Var(name.clone()))
                    } else if data.factors.contains_key(name) {
                        AtomSpec::Factor {
                            column: name.clone(),
                            from_numeric: false,
                            levels: Vec::new(),
                            contrast: Contrast::SumToZero,
                        }
                    } else {
                        return invalid(format!("unknown column '{name}'"));
                    }
                }
                AtomKind::Expr(e) => {
                    numeric_vars_ok(e, data)?;
                    AtomSpec::Numeric(e.clone())
                }
                AtomKind::AsFactor(name) => {
                    if !data.numeric.contains_key(name) {
                        return invalid(format!("C({name}) needs a numeric column named '{name}'"));
                    }
                    AtomSpec::Factor {
                        column: name.clone(),
                        from_numeric: true,
                        levels: Vec::new(),
                        contrast: Contrast::SumToZero,
                    }
                }
            };
            atoms.insert(a.label.clone(), spec);
        }
        for o in &f.offsets {
            numeric_vars_ok(o, data)?;
        }
        if let Some(r) = &f.response {
            numeric_vars_ok(r, data)?;
        }
        let mut spec = ModelSpec { formula: f, atoms };
        // 2. Complete cases, then levels.
        let rows = spec.complete_rows(data, true)?;
        if rows.is_empty() {
            return invalid("no complete cases: every row has a missing value in a column the model uses");
        }
        let labels: Vec<String> = spec.atoms.keys().cloned().collect();
        for label in labels {
            let values = match spec.atoms.get(&label) {
                Some(AtomSpec::Factor { column, from_numeric, .. }) => {
                    spec.factor_values(data, column, *from_numeric, &rows)
                }
                _ => continue,
            };
            let mut set: Vec<String> = values.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
            level_sort(&mut set);
            let levels = match opts.level_order.get(&label) {
                Some(order) => {
                    if let Some(missing) = set.iter().find(|l| !order.contains(l)) {
                        return invalid(format!("level order for {label} lacks the observed level '{missing}'"));
                    }
                    order.iter().filter(|l| set.contains(l)).cloned().collect()
                }
                None => set,
            };
            if levels.len() < 2 {
                return invalid(format!("factor {label} has fewer than two levels among the complete cases"));
            }
            let contrast = opts.contrasts.get(&label).cloned().unwrap_or_else(|| opts.default_contrast.clone());
            // Validate the contrast now.
            contrast_matrix(&contrast, &levels, &label)?;
            if let Some(AtomSpec::Factor { levels: lv, contrast: c, .. }) = spec.atoms.get_mut(&label) {
                *lv = levels;
                *c = contrast;
            }
        }
        Ok(spec)
    }

    fn factor_values(&self, data: &Data, column: &str, from_numeric: bool, rows: &[usize]) -> Vec<String> {
        if from_numeric {
            let col = data.numeric.get(column).map(Vec::as_slice).unwrap_or(&[]);
            rows.iter().map(|&r| format!("{}", col[r])).collect()
        } else {
            let col = data.factors.get(column).map(Vec::as_slice).unwrap_or(&[]);
            rows.iter().map(|&r| col[r].clone()).collect()
        }
    }

    /// Rows where every column used by the model is present.
    fn complete_rows(&self, data: &Data, with_response: bool) -> Result<Vec<usize>> {
        let n = data.n_rows();
        let mut num_cols: Vec<&str> = Vec::new();
        let mut fac_cols: Vec<&str> = Vec::new();
        let mut exprs: Vec<&Expr> = self.formula.offsets.iter().collect();
        if with_response {
            match &self.formula.response {
                Some(r) => exprs.push(r),
                None => return invalid("the formula has no response (write `y ~ terms`)"),
            }
        }
        for a in self.atoms.values() {
            match a {
                AtomSpec::Numeric(e) => exprs.push(e),
                AtomSpec::Factor { column, from_numeric, .. } => {
                    if *from_numeric {
                        num_cols.push(column);
                    } else {
                        fac_cols.push(column);
                    }
                }
            }
        }
        let mut vars: Vec<String> = exprs.iter().flat_map(|e| e.variables()).collect();
        vars.extend(num_cols.iter().map(|s| s.to_string()));
        vars.sort();
        vars.dedup();
        let mut keep = Vec::new();
        'rows: for r in 0..n {
            for v in &vars {
                match data.numeric.get(v) {
                    Some(col) => {
                        if col.get(r).is_none_or(|x| x.is_nan()) {
                            continue 'rows;
                        }
                    }
                    None => return invalid(format!("unknown numeric column '{v}'")),
                }
            }
            for c in &fac_cols {
                match data.factors.get(*c) {
                    Some(col) => {
                        if col.get(r).is_none_or(|s| s.is_empty()) {
                            continue 'rows;
                        }
                    }
                    None => return invalid(format!("unknown categorical column '{c}'")),
                }
            }
            keep.push(r);
        }
        Ok(keep)
    }

    fn eval(&self, e: &Expr, data: &Data, rows: &[usize], label: &str) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(rows.len());
        for &r in rows {
            let v = e.eval(&|name| data.numeric.get(name).map_or(f64::NAN, |c| c[r]));
            if !v.is_finite() {
                return invalid(format!(
                    "{label} is not finite at worksheet row {} (check log/sqrt/division inputs)",
                    r + 1
                ));
            }
            out.push(v);
        }
        Ok(out)
    }

    /// Encode `data` with this specification.
    pub fn encode(&self, data: &Data, with_response: bool) -> Result<DesignMatrix> {
        let rows = self.complete_rows(data, with_response)?;
        let n_dropped = data.n_rows() - rows.len();
        let n = rows.len();
        if n == 0 {
            return invalid("no complete cases to encode");
        }
        // Atom values: numeric columns or level indices.
        let mut num_vals: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut idx_vals: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (label, a) in &self.atoms {
            match a {
                AtomSpec::Numeric(e) => {
                    num_vals.insert(label, self.eval(e, data, &rows, label)?);
                }
                AtomSpec::Factor { column, from_numeric, levels, .. } => {
                    let vals = self.factor_values(data, column, *from_numeric, &rows);
                    let mut idx = Vec::with_capacity(n);
                    for v in vals {
                        match levels.iter().position(|l| *l == v) {
                            Some(i) => idx.push(i),
                            None => {
                                return invalid(format!(
                                    "level '{v}' of {label} was not present when the model was fitted"
                                ))
                            }
                        }
                    }
                    idx_vals.insert(label, idx);
                }
            }
        }

        let mut cols: Vec<Vec<f64>> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        if self.formula.intercept {
            cols.push(vec![1.0; n]);
            names.push("Constant".into());
        }
        let mut terms = Vec::new();
        let mut full_dummy_used = self.formula.intercept;
        for t in &self.formula.terms {
            let (tc, tn) = self.term_columns(t, &num_vals, &idx_vals, n, !full_dummy_used)?;
            if !full_dummy_used && t.factors.len() == 1 && t.nested_in.is_empty() && t.factors[0].1 == 1 {
                if let Some(AtomSpec::Factor { .. }) = self.atoms.get(&t.factors[0].0.label) {
                    full_dummy_used = true;
                }
            }
            terms.push(TermColumns { label: t.label(), order: t.order(), start: cols.len(), len: tc.len() });
            cols.extend(tc);
            names.extend(tn);
        }
        if cols.is_empty() {
            return invalid("the model has no columns");
        }
        let x = Matrix::from_columns(&cols)?;
        let y = if with_response {
            let r = self.formula.response.as_ref().ok_or_else(|| StatsError::InvalidArgument("no response".into()))?;
            Some(self.eval(r, data, &rows, &r.text())?)
        } else {
            None
        };
        let mut offset = vec![0.0; n];
        for o in &self.formula.offsets {
            let v = self.eval(o, data, &rows, &format!("offset({})", o.text()))?;
            offset.iter_mut().zip(v).for_each(|(a, b)| *a += b);
        }
        Ok(DesignMatrix { x, names, terms, intercept: self.formula.intercept, y, offset, rows_used: rows, n_dropped })
    }

    fn atom_columns(
        &self,
        label: &str,
        power: u32,
        num_vals: &BTreeMap<&str, Vec<f64>>,
        idx_vals: &BTreeMap<&str, Vec<usize>>,
        full_dummy: bool,
    ) -> Result<(Vec<Vec<f64>>, Vec<String>)> {
        match self.atoms.get(label) {
            Some(AtomSpec::Numeric(_)) => {
                let v = &num_vals[label];
                if power == 1 {
                    Ok((vec![v.clone()], vec![label.to_string()]))
                } else {
                    Ok((vec![v.iter().map(|x| x.powi(power as i32)).collect()], vec![format!("{label}^{power}")]))
                }
            }
            Some(AtomSpec::Factor { levels, contrast, .. }) => {
                let idx = &idx_vals[label];
                let (m, nm) = if full_dummy && matches!(contrast, Contrast::Treatment(_)) {
                    let l = levels.len();
                    (
                        Matrix::from_fn(l, l, |i, j| f64::from(i == j)),
                        levels.iter().map(|lv| format!("{label}[{lv}]")).collect(),
                    )
                } else {
                    contrast_matrix(contrast, levels, label)?
                };
                let cols = (0..m.cols()).map(|c| idx.iter().map(|&i| m[(i, c)]).collect()).collect();
                Ok((cols, nm))
            }
            None => invalid(format!("internal: atom {label} not in the model specification")),
        }
    }

    fn term_columns(
        &self,
        t: &Term,
        num_vals: &BTreeMap<&str, Vec<f64>>,
        idx_vals: &BTreeMap<&str, Vec<usize>>,
        n: usize,
        allow_full_dummy: bool,
    ) -> Result<(Vec<Vec<f64>>, Vec<String>)> {
        let pure_factor_main = t.factors.len() == 1
            && t.nested_in.is_empty()
            && t.factors[0].1 == 1
            && matches!(self.atoms.get(&t.factors[0].0.label), Some(AtomSpec::Factor { .. }));
        // Kronecker product of the atoms' columns, the first atom varying fastest.
        let mut cols: Vec<Vec<f64>> = vec![vec![1.0; n]];
        let mut names: Vec<String> = vec![String::new()];
        for (a, p) in &t.factors {
            let (ac, an) = self.atom_columns(&a.label, *p, num_vals, idx_vals, allow_full_dummy && pure_factor_main)?;
            let mut nc = Vec::with_capacity(cols.len() * ac.len());
            let mut nn = Vec::with_capacity(nc.capacity());
            for (j, aj) in ac.iter().enumerate() {
                for (i, ci) in cols.iter().enumerate() {
                    nc.push(ci.iter().zip(aj).map(|(x, y)| x * y).collect());
                    nn.push(if names[i].is_empty() { an[j].clone() } else { format!("{}:{}", names[i], an[j]) });
                }
            }
            cols = nc;
            names = nn;
        }
        if t.nested_in.is_empty() {
            return Ok((cols, names));
        }
        // Nested: replicate the columns within every observed cell of the conditioning factors.
        let mut keys: Vec<(&str, &Vec<usize>, &Vec<String>)> = Vec::new();
        for a in &t.nested_in {
            match self.atoms.get(&a.label) {
                Some(AtomSpec::Factor { levels, .. }) => keys.push((&a.label, &idx_vals[a.label.as_str()], levels)),
                _ => return invalid(format!("{} is nested within {}, which must be categorical", t.label(), a.label)),
            }
        }
        let mut cells: BTreeSet<Vec<usize>> = BTreeSet::new();
        for r in 0..n {
            cells.insert(keys.iter().map(|(_, idx, _)| idx[r]).collect());
        }
        let mut out_c = Vec::new();
        let mut out_n = Vec::new();
        for cell in &cells {
            let ind: Vec<f64> =
                (0..n).map(|r| f64::from(keys.iter().zip(cell).all(|((_, idx, _), &c)| idx[r] == c))).collect();
            let tag =
                keys.iter().zip(cell).map(|((l, _, lv), &c)| format!("{l}={}", lv[c])).collect::<Vec<_>>().join(",");
            for (c, nm) in cols.iter().zip(&names) {
                out_c.push(c.iter().zip(&ind).map(|(x, y)| x * y).collect());
                out_n.push(format!("{nm}({tag})"));
            }
        }
        Ok((out_c, out_n))
    }
}

/// Parse `formula`, learn the encoding from `data` and build the design matrix.
pub fn build_design(formula: &str, data: &Data, opts: &DesignOptions) -> Result<(ModelSpec, DesignMatrix)> {
    let spec = ModelSpec::new(formula, data, opts)?;
    let dm = spec.encode(data, true)?;
    Ok((spec, dm))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Data {
        Data::new()
            .with_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_numeric("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_factor("g", &["a", "b", "c", "a", "b", "c"])
            .unwrap()
            .with_factor("h", &["u", "u", "u", "v", "v", "v"])
            .unwrap()
    }

    fn col(dm: &DesignMatrix, j: usize) -> Vec<f64> {
        dm.x.col(j)
    }

    #[test]
    fn treatment_coding_and_names() {
        let (_, dm) = build_design("y ~ x + g", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "x", "g[b]", "g[c]"]);
        assert_eq!(col(&dm, 2), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(col(&dm, 3), [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(dm.terms[1], TermColumns { label: "g".into(), order: 1, start: 2, len: 2 });
        assert_eq!(dm.y.as_ref().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Reference choices.
        let mut o = DesignOptions::default();
        o.contrasts.insert("g".into(), Contrast::Treatment(Reference::Last));
        let (_, d2) = build_design("y ~ g", &data(), &o).unwrap();
        assert_eq!(d2.names, ["Constant", "g[a]", "g[b]"]);
        o.contrasts.insert("g".into(), Contrast::Treatment(Reference::Level("b".into())));
        let (_, d3) = build_design("y ~ g", &data(), &o).unwrap();
        assert_eq!(d3.names, ["Constant", "g[a]", "g[c]"]);
        o.contrasts.insert("g".into(), Contrast::Treatment(Reference::Level("zz".into())));
        assert!(build_design("y ~ g", &data(), &o).is_err());
    }

    #[test]
    fn sum_to_zero_coding() {
        let o = DesignOptions { default_contrast: Contrast::SumToZero, ..DesignOptions::default() };
        let (_, dm) = build_design("y ~ g", &data(), &o).unwrap();
        assert_eq!(col(&dm, 1), [1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
        assert_eq!(col(&dm, 2), [0.0, 1.0, -1.0, 0.0, 1.0, -1.0]);
        // Each contrast column sums to zero over a balanced design.
        assert!(col(&dm, 1).iter().sum::<f64>().abs() < 1e-15);
    }

    #[test]
    fn helmert_and_polynomial_contrasts_match_r() {
        let lv: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let (h, names) = contrast_matrix(&Contrast::Helmert, &lv, "f").unwrap();
        assert_eq!(names, ["f.H1", "f.H2", "f.H3"]);
        // contr.helmert(4)
        let want = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 3.0]];
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(h[(i, j)], want[i][j]);
            }
        }
        // contr.poly(4)
        let (p, names) = contrast_matrix(&Contrast::OrthogonalPolynomial(None), &lv, "f").unwrap();
        assert_eq!(names, ["f.L", "f.Q", "f.C"]);
        let want = [
            [-0.6708203932499369, 0.5, -0.2236067977499790],
            [-0.2236067977499790, -0.5, 0.6708203932499369],
            [0.2236067977499790, -0.5, -0.6708203932499369],
            [0.6708203932499369, 0.5, 0.2236067977499790],
        ];
        for i in 0..4 {
            for j in 0..3 {
                assert!((p[(i, j)] - want[i][j]).abs() < 1e-13, "({i},{j}) {} vs {}", p[(i, j)], want[i][j]);
            }
        }
        // Orthonormal columns; unequal spacing still orthogonal.
        let (q, _) =
            contrast_matrix(&Contrast::OrthogonalPolynomial(Some(vec![1.0, 2.0, 4.0, 8.0])), &lv, "f").unwrap();
        let g = q.t_matmul(&q).unwrap();
        assert!(g.approx_eq(&Matrix::identity(3), 1e-12));
        assert!(contrast_matrix(&Contrast::OrthogonalPolynomial(Some(vec![1.0, 1.0, 2.0, 3.0])), &lv, "f").is_err());
        assert!(contrast_matrix(&Contrast::Custom(Matrix::zeros(3, 1)), &lv, "f").is_err());
    }

    #[test]
    fn interactions_polynomials_and_nesting() {
        let (_, dm) = build_design("y ~ x*g", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "x", "g[b]", "g[c]", "x:g[b]", "x:g[c]"]);
        assert_eq!(col(&dm, 4), [0.0, 2.0, 0.0, 0.0, 5.0, 0.0]);
        let (_, dm) = build_design("y ~ x|x", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "x", "x^2"]);
        assert_eq!(col(&dm, 2), [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
        let (_, dm) = build_design("y ~ g + I(x^2) + log(x)", &data(), &DesignOptions::default()).unwrap();
        assert!((col(&dm, 4)[1] - 2.0f64.ln()).abs() < 1e-15 && col(&dm, 3)[2] == 9.0);
        // Two-factor interaction: first factor varies fastest.
        let (_, dm) = build_design("y ~ g:h", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "g[b]:h[v]", "g[c]:h[v]"]);
        // Nested: g within h.
        let (_, dm) = build_design("y ~ h + g(h)", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "h[v]", "g[b](h=u)", "g[c](h=u)", "g[b](h=v)", "g[c](h=v)"]);
        assert_eq!(col(&dm, 2), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(col(&dm, 5), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        // Slopes of x within each level of h.
        let (_, dm) = build_design("y ~ h + x(h)", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "h[v]", "x(h=u)", "x(h=v)"]);
        assert_eq!(col(&dm, 3), [0.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn intercept_control_and_offset() {
        let (_, dm) = build_design("y ~ 0 + g + x", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["g[a]", "g[b]", "g[c]", "x"]); // full dummy coding without an intercept
        assert!(!dm.intercept);
        let (_, dm) = build_design("y ~ x - 1", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["x"]);
        let (_, dm) = build_design("y ~ x + offset(2*x)", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.offset, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        let (_, dm) = build_design("y ~ 1", &data(), &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant"]);
    }

    #[test]
    fn factor_from_numeric_and_level_order() {
        let d = Data::new()
            .with_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_numeric("dose", vec![10.0, 5.0, 10.0, 5.0, 20.0, 20.0])
            .unwrap();
        let (_, dm) = build_design("y ~ C(dose)", &d, &DesignOptions::default()).unwrap();
        assert_eq!(dm.names, ["Constant", "C(dose)[10]", "C(dose)[20]"]); // numeric level order: 5, 10, 20
        let mut o = DesignOptions::default();
        o.level_order.insert("C(dose)".into(), vec!["20".into(), "10".into(), "5".into()]);
        let (_, dm) = build_design("y ~ C(dose)", &d, &o).unwrap();
        assert_eq!(dm.names, ["Constant", "C(dose)[10]", "C(dose)[5]"]);
        o.level_order.insert("C(dose)".into(), vec!["20".into(), "10".into()]);
        assert!(build_design("y ~ C(dose)", &d, &o).is_err());
    }

    #[test]
    fn missing_rows_dropped_and_prediction_encoding() {
        let d = Data::new()
            .with_numeric("y", vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0])
            .unwrap()
            .with_numeric("x", vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0])
            .unwrap()
            .with_factor("g", &["a", "b", "a", "", "b", "a"])
            .unwrap();
        let (spec, dm) = build_design("y ~ x + g", &d, &DesignOptions::default()).unwrap();
        assert_eq!(dm.rows_used, [0, 1, 5]);
        assert_eq!(dm.n_dropped, 3);
        // New data encoded with the fitted levels.
        let nd = Data::new().with_numeric("x", vec![7.0, 8.0]).unwrap().with_factor("g", &["b", "a"]).unwrap();
        let e = spec.encode(&nd, false).unwrap();
        assert_eq!(e.x.col(2), [1.0, 0.0]);
        assert!(e.y.is_none());
        let bad = Data::new().with_numeric("x", vec![7.0]).unwrap().with_factor("g", &["zzz"]).unwrap();
        assert!(spec.encode(&bad, false).is_err());
    }

    #[test]
    fn data_and_formula_errors() {
        assert!(Data::new().with_numeric("a", vec![1.0]).unwrap().with_numeric("a", vec![2.0]).is_err());
        assert!(Data::new().with_numeric("a", vec![1.0]).unwrap().with_numeric("b", vec![1.0, 2.0]).is_err());
        assert!(Data::new().with_numeric("a", vec![f64::INFINITY]).is_err());
        let d = data();
        for bad in ["y ~ nosuch", "y ~ log(g)", "y ~ g(x)", "x ~ I(log(x - 7))", "~ x"] {
            assert!(build_design(bad, &d, &DesignOptions::default()).is_err(), "{bad}");
        }
        let one = Data::new().with_numeric("y", vec![1.0, 2.0]).unwrap().with_factor("g", &["a", "a"]).unwrap();
        assert!(build_design("y ~ g", &one, &DesignOptions::default()).is_err());
    }
}
