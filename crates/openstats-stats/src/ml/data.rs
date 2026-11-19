//! Predictor data, responses and the pre-sorted index shared by every tree-based model.

use crate::error::{invalid, Result, StatsError};
use std::collections::BTreeSet;

/// Most levels a categorical predictor or a class response may have.
pub const MAX_LEVELS: usize = 256;

/// One predictor column.
#[derive(Debug, Clone, PartialEq)]
pub enum Feature {
    /// A numeric predictor; missing values are `NaN`.
    Numeric(Vec<f64>),
    /// A categorical predictor: `codes[i]` indexes `levels`, `-1` means missing.
    Categorical {
        /// Level of every observation (`-1` = missing).
        codes: Vec<i32>,
        /// The distinct labels, in their sorted order.
        levels: Vec<String>,
    },
}

impl Feature {
    fn len(&self) -> usize {
        match self {
            Feature::Numeric(v) => v.len(),
            Feature::Categorical { codes, .. } => codes.len(),
        }
    }
}

/// A set of predictors with names, all of the same length.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dataset {
    /// Predictor names.
    pub names: Vec<String>,
    /// Predictor columns.
    pub features: Vec<Feature>,
    n: usize,
}

/// Distinct labels sorted numerically when every label is a number, alphabetically otherwise.
pub(crate) fn sorted_levels<'a>(labels: impl Iterator<Item = &'a str>) -> Vec<String> {
    let set: BTreeSet<&str> = labels.filter(|s| !s.is_empty()).collect();
    let mut v: Vec<String> = set.into_iter().map(str::to_string).collect();
    let nums: Option<Vec<f64>> = v.iter().map(|s| s.trim().parse::<f64>().ok().filter(|x| x.is_finite())).collect();
    if let Some(nums) = nums {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| nums[a].total_cmp(&nums[b]).then_with(|| v[a].cmp(&v[b])));
        v = idx.into_iter().map(|i| v[i].clone()).collect();
    }
    v
}

impl Dataset {
    /// An empty dataset.
    pub fn new() -> Dataset {
        Dataset::default()
    }

    fn check_len(&mut self, len: usize) -> Result<()> {
        if self.features.is_empty() {
            self.n = len;
            Ok(())
        } else if self.n == len {
            Ok(())
        } else {
            invalid(format!("predictor columns differ in length ({} and {len})", self.n))
        }
    }

    /// Add a numeric predictor. Missing values are `NaN`; infinite values are an error.
    pub fn with_numeric(mut self, name: &str, values: Vec<f64>) -> Result<Dataset> {
        if values.iter().any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        self.check_len(values.len())?;
        self.names.push(name.to_string());
        self.features.push(Feature::Numeric(values));
        Ok(self)
    }

    /// Add a categorical predictor from labels; an empty label is a missing value.
    pub fn with_categorical(mut self, name: &str, labels: &[String]) -> Result<Dataset> {
        let levels = sorted_levels(labels.iter().map(String::as_str));
        if levels.len() > MAX_LEVELS {
            return invalid(format!("{name} has {} levels; at most {MAX_LEVELS} are supported", levels.len()));
        }
        self.check_len(labels.len())?;
        let codes = labels
            .iter()
            .map(|l| levels.iter().position(|x| x == l).map_or(-1, |p| i32::try_from(p).unwrap_or(-1)))
            .collect();
        self.names.push(name.to_string());
        self.features.push(Feature::Categorical { codes, levels });
        Ok(self)
    }

    /// Build a dataset from finished columns (all of the same length).
    pub fn from_features(names: Vec<String>, features: Vec<Feature>) -> Result<Dataset> {
        if names.len() != features.len() {
            return invalid("the number of names differs from the number of predictors");
        }
        let n = features.first().map_or(0, Feature::len);
        if features.iter().any(|f| f.len() != n) {
            return invalid("predictor columns differ in length");
        }
        for f in &features {
            if let Feature::Categorical { codes, levels } = f {
                if codes.iter().any(|&c| c >= 0 && usize::try_from(c).map_or(true, |c| c >= levels.len())) {
                    return invalid("a categorical code is outside its levels");
                }
            }
        }
        Ok(Dataset { names, features, n })
    }

    /// The observations `rows` (in that order, repeats allowed) as a new dataset.
    pub fn select_rows(&self, rows: &[usize]) -> Dataset {
        let features = self
            .features
            .iter()
            .map(|f| match f {
                Feature::Numeric(v) => Feature::Numeric(rows.iter().map(|&r| v[r]).collect()),
                Feature::Categorical { codes, levels } => {
                    Feature::Categorical { codes: rows.iter().map(|&r| codes[r]).collect(), levels: levels.clone() }
                }
            })
            .collect();
        Dataset { names: self.names.clone(), features, n: rows.len() }
    }

    /// A copy in which predictor `f` is the same for every observation: the number `value` for a
    /// numeric predictor, the level with index `value as usize` for a categorical one.
    pub fn with_constant(&self, f: usize, value: f64) -> Result<Dataset> {
        let mut d = self.clone();
        match d.features.get_mut(f) {
            Some(Feature::Numeric(v)) => v.iter_mut().for_each(|x| *x = value),
            Some(Feature::Categorical { codes, levels }) => {
                if !(value >= 0.0 && (value as usize) < levels.len()) {
                    return invalid("level index out of range");
                }
                codes.iter_mut().for_each(|c| *c = value as i32);
            }
            None => return invalid("no such predictor"),
        }
        Ok(d)
    }

    /// Number of observations.
    pub fn n_rows(&self) -> usize {
        self.n
    }

    /// Number of predictors.
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Levels of categorical predictor `f` (`None` for a numeric one).
    pub fn levels(&self, f: usize) -> Option<&[String]> {
        match self.features.get(f) {
            Some(Feature::Categorical { levels, .. }) => Some(levels),
            _ => None,
        }
    }

    /// The same predictors with no observations: the schema a fitted model remembers.
    pub fn schema(&self) -> Dataset {
        let features = self
            .features
            .iter()
            .map(|f| match f {
                Feature::Numeric(_) => Feature::Numeric(Vec::new()),
                Feature::Categorical { levels, .. } => {
                    Feature::Categorical { codes: Vec::new(), levels: levels.clone() }
                }
            })
            .collect();
        Dataset { names: self.names.clone(), features, n: 0 }
    }

    /// Re-express these observations in the level coding of `template`: categorical labels are
    /// matched by name (an unseen label becomes missing), numeric columns are copied. The
    /// predictors must correspond one to one in kind and order.
    pub fn remap_to(&self, template: &Dataset) -> Result<Dataset> {
        if self.features.len() != template.features.len() {
            return invalid(format!(
                "the model uses {} predictors but {} were supplied",
                template.features.len(),
                self.features.len()
            ));
        }
        let mut out = Dataset { names: template.names.clone(), features: Vec::new(), n: self.n };
        for (j, (mine, theirs)) in self.features.iter().zip(&template.features).enumerate() {
            let f = match (mine, theirs) {
                (Feature::Numeric(v), Feature::Numeric(_)) => Feature::Numeric(v.clone()),
                (Feature::Categorical { codes, levels }, Feature::Categorical { levels: tl, .. }) => {
                    let map: Vec<i32> = levels
                        .iter()
                        .map(|l| tl.iter().position(|x| x == l).map_or(-1, |p| i32::try_from(p).unwrap_or(-1)))
                        .collect();
                    let codes = codes
                        .iter()
                        .map(|&c| usize::try_from(c).ok().and_then(|c| map.get(c).copied()).unwrap_or(-1))
                        .collect();
                    Feature::Categorical { codes, levels: tl.clone() }
                }
                _ => return invalid(format!("predictor {} differs in type from the model's", j + 1)),
            };
            out.features.push(f);
        }
        Ok(out)
    }

    /// Whether predictor `f` is missing for observation `row`.
    pub fn is_missing(&self, f: usize, row: usize) -> bool {
        match self.features.get(f) {
            Some(Feature::Numeric(v)) => v.get(row).is_none_or(|x| x.is_nan()),
            Some(Feature::Categorical { codes, .. }) => codes.get(row).is_none_or(|&c| c < 0),
            None => true,
        }
    }

    /// Number of missing predictor cells.
    pub fn missing_cells(&self) -> usize {
        (0..self.n_features()).map(|f| (0..self.n).filter(|&r| self.is_missing(f, r)).count()).sum()
    }

    /// Check the dataset can be modelled.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.features.is_empty() {
            return invalid("select at least one predictor");
        }
        if self.n < 2 {
            return Err(StatsError::InsufficientData { what: "a tree model", needed: 2, got: self.n });
        }
        if u32::try_from(self.n).is_err() {
            return Err(StatsError::TooLarge("more than 4 billion observations".into()));
        }
        if self.features.iter().any(|f| f.len() != self.n) {
            return invalid("predictor columns differ in length");
        }
        Ok(())
    }
}

/// The response of a model.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    /// A numeric response (regression).
    Numeric(Vec<f64>),
    /// A categorical response (classification).
    Class {
        /// Class of every observation, an index into `levels`.
        codes: Vec<usize>,
        /// The class labels in sorted order.
        levels: Vec<String>,
    },
}

impl Response {
    /// A numeric response; missing or infinite values are errors.
    pub fn numeric(y: Vec<f64>) -> Result<Response> {
        if y.iter().any(|v| v.is_nan()) {
            return invalid("the response has missing values");
        }
        if y.iter().any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        Ok(Response::Numeric(y))
    }

    /// A categorical response from labels; an empty label is an error.
    pub fn from_labels(labels: &[String]) -> Result<Response> {
        if labels.iter().any(String::is_empty) {
            return invalid("the response has missing values");
        }
        let levels = sorted_levels(labels.iter().map(String::as_str));
        if levels.len() > MAX_LEVELS {
            return invalid(format!("the response has {} classes; at most {MAX_LEVELS} are supported", levels.len()));
        }
        let codes = labels.iter().map(|l| levels.iter().position(|x| x == l).unwrap_or(0)).collect();
        Ok(Response::Class { codes, levels })
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        match self {
            Response::Numeric(y) => y.len(),
            Response::Class { codes, .. } => codes.len(),
        }
    }

    /// Whether there are no observations.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this is a classification response.
    pub fn is_class(&self) -> bool {
        matches!(self, Response::Class { .. })
    }
}

/// For every numeric predictor, the observations with a value, sorted by that value. Built once
/// per dataset and shared by all trees grown on it (bootstrap samples, boosting rounds, folds).
#[derive(Debug, Clone)]
pub(crate) struct Presort {
    pub order: Vec<Vec<u32>>,
}

impl Presort {
    pub(crate) fn new(ds: &Dataset) -> Presort {
        let order = ds
            .features
            .iter()
            .map(|f| match f {
                Feature::Numeric(v) => {
                    let mut idx: Vec<u32> =
                        (0..v.len()).filter(|&i| !v[i].is_nan()).map(|i| u32::try_from(i).unwrap_or(0)).collect();
                    idx.sort_by(|&a, &b| v[a as usize].total_cmp(&v[b as usize]).then(a.cmp(&b)));
                    idx
                }
                Feature::Categorical { .. } => Vec::new(),
            })
            .collect();
        Presort { order }
    }
}
