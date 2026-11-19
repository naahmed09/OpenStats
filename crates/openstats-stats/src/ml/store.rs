//! Portable prediction-only models and their JSON form.
//!
//! A fitted [`CartModel`], [`ForestModel`], [`GbdtModel`] or MARS model carries a lot that only
//! matters while it is being reported (pruning tables, out-of-bag results, training fits). What a
//! saved model needs is smaller: the schema of the predictors (names, kinds and levels) and the
//! structure that turns predictors into a prediction. [`PredictiveModel`] is exactly that, it
//! predicts like the model it came from, and it round-trips through text ([`SavedModel::to_json`]
//! and [`SavedModel::from_json`]) so a project file can hold it.

use super::data::{Dataset, Feature};
use super::gbdt::{GbdtLoss, GbdtModel};
use super::mars::{Factor, MarsModel, Term};
use super::mars_ext::MarsBinaryModel;
use super::random_forest::ForestModel;
use super::tree::{Criterion, Node, Split, Surrogate, Test, Tree};
use super::{argmax, CartModel, Predictions};
use crate::error::{invalid, Result};
use std::fmt::Write as _;

/// How a MARS model turns its linear predictor into a prediction.
#[derive(Debug, Clone, PartialEq)]
pub enum MarsLink {
    /// The prediction is the linear predictor (numeric response).
    Identity,
    /// Binary response: probability of the second class, `1 / (1 + exp(−η))`.
    Logit {
        /// The two class labels; the probability is that of `levels[1]`.
        levels: Vec<String>,
    },
}

/// A MARS model ready to predict from the original predictors.
#[derive(Debug, Clone, PartialEq)]
pub struct MarsObject {
    /// The original predictors (numeric, or categorical with their levels).
    pub schema: Dataset,
    /// Names of the expanded columns (a categorical predictor contributes one 0/1 indicator for
    /// every level except the first).
    pub names: Vec<String>,
    /// Basis functions with their final coefficients (the first is the intercept).
    pub terms: Vec<Term>,
    /// The link.
    pub link: MarsLink,
}

/// A fitted model reduced to what prediction needs.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictiveModel {
    /// A single classification or regression tree.
    Tree {
        /// The tree.
        tree: Tree,
        /// Predictors the tree was grown on.
        schema: Dataset,
        /// Class labels (empty for regression).
        levels: Vec<String>,
    },
    /// A random forest.
    Forest {
        /// The trees.
        trees: Vec<Tree>,
        /// Predictors.
        schema: Dataset,
        /// Class labels (empty for regression).
        levels: Vec<String>,
    },
    /// Gradient boosting.
    Boosting {
        /// Initial value(s).
        init: Vec<f64>,
        /// `trees[stage][dimension]`.
        trees: Vec<Vec<Tree>>,
        /// Shrinkage.
        learning_rate: f64,
        /// Predictors.
        schema: Dataset,
        /// Class labels (empty for regression).
        levels: Vec<String>,
        /// Loss (decides the link).
        loss: GbdtLoss,
    },
    /// Multivariate adaptive regression splines.
    Mars(MarsObject),
}

/// Expand predictors into the numeric columns MARS works on: numeric predictors as they are, a
/// categorical predictor as 0/1 indicators for every level but the first (missing stays `NaN`).
pub fn expand_indicators(ds: &Dataset) -> (Vec<Vec<f64>>, Vec<String>) {
    let mut cols = Vec::new();
    let mut names = Vec::new();
    for (f, feat) in ds.features.iter().enumerate() {
        match feat {
            Feature::Numeric(v) => {
                cols.push(v.clone());
                names.push(ds.names[f].clone());
            }
            Feature::Categorical { codes, levels } => {
                for (l, label) in levels.iter().enumerate().skip(1) {
                    cols.push(
                        codes
                            .iter()
                            .map(|&c| match c {
                                c if c < 0 => f64::NAN,
                                c if c as usize == l => 1.0,
                                _ => 0.0,
                            })
                            .collect(),
                    );
                    names.push(format!("{}={}", ds.names[f], label));
                }
            }
        }
    }
    (cols, names)
}

fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
}

fn softmax(f: &[f64]) -> Vec<f64> {
    let m = f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = f.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

impl MarsObject {
    /// Wrap a fitted MARS model (and the original predictors it was expanded from).
    pub fn from_model(m: &MarsModel, schema: &Dataset, link: MarsLink) -> MarsObject {
        MarsObject { schema: schema.schema(), names: expand_indicators(schema).1, terms: m.terms.clone(), link }
    }

    /// Wrap a fitted binary MARS model (logistic coefficients) and the original predictors.
    pub fn from_binary(m: &MarsBinaryModel, schema: &Dataset) -> MarsObject {
        MarsObject {
            schema: schema.schema(),
            names: expand_indicators(schema).1,
            terms: m.terms.clone(),
            link: MarsLink::Logit { levels: m.levels.clone() },
        }
    }

    /// Linear predictor of every observation (`NaN` where a predictor the model uses is missing).
    pub fn linear_predictor(&self, ds: &Dataset) -> Result<Vec<f64>> {
        let d = ds.remap_to(&self.schema)?;
        let (cols, _) = expand_indicators(&d);
        if cols.len() != self.names.len() {
            return invalid("the predictors do not match the model");
        }
        let mut row = vec![0.0; cols.len()];
        Ok((0..d.n_rows())
            .map(|i| {
                for (v, c) in cols.iter().enumerate() {
                    row[v] = c[i];
                }
                self.terms.iter().map(|t| t.coef * t.eval(&row)).sum()
            })
            .collect())
    }
}

fn predict_forest(trees: &[Tree], levels: &[String], d: &Dataset) -> Predictions {
    let t = trees.len() as f64;
    let mut out = Predictions::default();
    for i in 0..d.n_rows() {
        if levels.is_empty() {
            out.values.push(trees.iter().map(|tr| tr.predict_value(d, i)).sum::<f64>() / t);
        } else {
            let mut probs = vec![0.0; levels.len()];
            for tr in trees {
                for (a, q) in probs.iter_mut().zip(tr.proportions(tr.leaf(d, i))) {
                    *a += q;
                }
            }
            probs.iter_mut().for_each(|a| *a /= t);
            out.classes.push(argmax(&probs));
            out.probabilities.push(probs);
        }
    }
    out
}

impl PredictiveModel {
    /// From a fitted tree.
    pub fn from_cart(m: &CartModel) -> PredictiveModel {
        PredictiveModel::Tree { tree: m.tree.clone(), schema: m.schema.clone(), levels: m.levels.clone() }
    }

    /// From a fitted forest.
    pub fn from_forest(m: &ForestModel) -> PredictiveModel {
        PredictiveModel::Forest { trees: m.trees.clone(), schema: m.schema.clone(), levels: m.levels.clone() }
    }

    /// From a fitted boosting model (only the stages kept).
    pub fn from_gbdt(m: &GbdtModel) -> PredictiveModel {
        PredictiveModel::Boosting {
            init: m.init.clone(),
            trees: m.trees.clone(),
            learning_rate: m.learning_rate,
            schema: m.schema.clone(),
            levels: m.levels.clone(),
            loss: m.loss,
        }
    }

    /// The predictors the model expects.
    pub fn schema(&self) -> &Dataset {
        match self {
            PredictiveModel::Tree { schema, .. }
            | PredictiveModel::Forest { schema, .. }
            | PredictiveModel::Boosting { schema, .. } => schema,
            PredictiveModel::Mars(m) => &m.schema,
        }
    }

    /// Class labels (empty for a regression model).
    pub fn levels(&self) -> &[String] {
        match self {
            PredictiveModel::Tree { levels, .. }
            | PredictiveModel::Forest { levels, .. }
            | PredictiveModel::Boosting { levels, .. } => levels,
            PredictiveModel::Mars(m) => match &m.link {
                MarsLink::Identity => &[],
                MarsLink::Logit { levels } => levels,
            },
        }
    }

    /// Whether the model classifies.
    pub fn is_classification(&self) -> bool {
        !self.levels().is_empty()
    }

    /// Short name of the method.
    pub fn algorithm(&self) -> &'static str {
        match self {
            PredictiveModel::Tree { .. } => "CART",
            PredictiveModel::Forest { .. } => "Random Forests",
            PredictiveModel::Boosting { .. } => "Gradient Boosting",
            PredictiveModel::Mars(m) => match m.link {
                MarsLink::Identity => "MARS",
                MarsLink::Logit { .. } => "MARS (binary)",
            },
        }
    }

    /// Predict observations described by `ds`: the same predictors, in the same order, as the fit
    /// (categorical labels are matched by name, an unseen label counts as missing).
    pub fn predict(&self, ds: &Dataset) -> Result<Predictions> {
        let d = ds.remap_to(self.schema())?;
        Ok(match self {
            PredictiveModel::Tree { tree, levels, .. } => {
                let mut out = Predictions::default();
                for i in 0..d.n_rows() {
                    let leaf = tree.leaf(&d, i);
                    if levels.is_empty() {
                        out.values.push(tree.nodes[leaf].value);
                    } else {
                        out.classes.push(tree.class_of(leaf));
                        out.probabilities.push(tree.proportions(leaf));
                    }
                }
                out
            }
            PredictiveModel::Forest { trees, levels, .. } => predict_forest(trees, levels, &d),
            PredictiveModel::Boosting { init, trees, learning_rate, levels, .. } => {
                let mut out = Predictions::default();
                for i in 0..d.n_rows() {
                    let mut f = init.clone();
                    for stage in trees {
                        for (dim, tree) in stage.iter().enumerate() {
                            f[dim] += learning_rate * tree.predict_value(&d, i);
                        }
                    }
                    if levels.is_empty() {
                        out.values.push(f[0]);
                    } else {
                        let probs = if levels.len() == 2 {
                            let p = sigmoid(f[0]);
                            vec![1.0 - p, p]
                        } else {
                            softmax(&f)
                        };
                        out.classes.push(argmax(&probs));
                        out.probabilities.push(probs);
                    }
                }
                out
            }
            PredictiveModel::Mars(m) => {
                let eta = m.linear_predictor(&d)?;
                match &m.link {
                    MarsLink::Identity => Predictions { values: eta, ..Predictions::default() },
                    MarsLink::Logit { .. } => {
                        let mut out = Predictions::default();
                        for e in eta {
                            let p = sigmoid(e);
                            out.classes.push(usize::from(p > 0.5));
                            out.probabilities.push(vec![1.0 - p, p]);
                        }
                        out
                    }
                }
            }
        })
    }

    /// One number per observation: the predicted value (regression) or the probability of class
    /// `class` (classification).
    pub fn score(&self, ds: &Dataset, class: usize) -> Result<Vec<f64>> {
        let p = self.predict(ds)?;
        if self.is_classification() {
            if class >= self.levels().len() {
                return invalid("class index out of range");
            }
            Ok(p.probabilities.iter().map(|q| q[class]).collect())
        } else {
            Ok(p.values)
        }
    }
}

fn check_tree(t: &Tree, schema: &Dataset, classes: usize) -> Result<()> {
    if t.n_classes != classes {
        return invalid("saved model: a tree's class count disagrees with the model");
    }
    let check_test = |test: &Test| -> Result<()> {
        match (test, schema.features.get(test.feature())) {
            (Test::LessEq { .. }, Some(Feature::Numeric(_))) => Ok(()),
            (Test::InSet { left, .. }, Some(Feature::Categorical { levels, .. })) if left.len() == levels.len() => {
                Ok(())
            }
            _ => invalid("saved model: a split does not match its predictor"),
        }
    };
    for nd in &t.nodes {
        if classes > 0 && nd.counts.len() != classes {
            return invalid("saved model: a node's class counts have the wrong length");
        }
        if let Some(sp) = &nd.split {
            check_test(&sp.test)?;
            for g in &sp.surrogates {
                check_test(&g.test)?;
            }
        }
    }
    Ok(())
}

impl PredictiveModel {
    /// Check that the parts of a model agree with each other (used when reading a saved model, so a
    /// damaged file is refused instead of failing later inside `predict`).
    fn validate(&self) -> Result<()> {
        match self {
            PredictiveModel::Tree { tree, schema, levels } => check_tree(tree, schema, levels.len()),
            PredictiveModel::Forest { trees, schema, levels } => {
                if trees.is_empty() {
                    return invalid("saved model: a forest has no trees");
                }
                trees.iter().try_for_each(|t| check_tree(t, schema, levels.len()))
            }
            PredictiveModel::Boosting { init, trees, schema, levels, .. } => {
                let dims = if levels.len() > 2 { levels.len() } else { 1 };
                if init.len() != dims {
                    return invalid("saved model: the initial values do not match the classes");
                }
                for st in trees {
                    if st.len() != dims {
                        return invalid("saved model: a boosting stage has the wrong number of trees");
                    }
                    st.iter().try_for_each(|t| check_tree(t, schema, 0))?;
                }
                Ok(())
            }
            PredictiveModel::Mars(m) => {
                let (cols, names) = expand_indicators(&m.schema);
                if cols.len() != m.names.len() || names != m.names {
                    return invalid("saved model: the basis names do not match the predictors");
                }
                if m.terms.is_empty() {
                    return invalid("saved model: a MARS model has no terms");
                }
                if m.terms.iter().flat_map(|t| &t.factors).any(|f| f.var >= m.names.len()) {
                    return invalid("saved model: a basis function refers to a missing predictor");
                }
                if let MarsLink::Logit { levels } = &m.link {
                    if levels.len() != 2 {
                        return invalid("saved model: a binary MARS model needs two classes");
                    }
                }
                Ok(())
            }
        }
    }
}

/// A model plus what a project needs to list and apply it.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedModel {
    /// Name of the response column the model was fitted to.
    pub response: String,
    /// Number of observations the model was fitted on.
    pub n_train: usize,
    /// The model.
    pub model: PredictiveModel,
}

// ------------------------------------------------------------------ JSON

#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn esc(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn num(x: f64, out: &mut String) {
    if x.is_nan() {
        out.push_str("\"NaN\"");
    } else if x == f64::INFINITY {
        out.push_str("\"Infinity\"");
    } else if x == f64::NEG_INFINITY {
        out.push_str("\"-Infinity\"");
    } else {
        let _ = write!(out, "{x:?}");
    }
}

impl Json {
    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(x) => num(*x, out),
            Json::Str(s) => esc(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    esc(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }

    fn get(&self, key: &str) -> Result<&Json> {
        match self {
            Json::Obj(o) => o
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or_else(|| crate::error::StatsError::InvalidArgument(format!("saved model: missing \"{key}\""))),
            _ => invalid("saved model: an object was expected"),
        }
    }

    fn f64(&self) -> Result<f64> {
        match self {
            Json::Num(x) => Ok(*x),
            Json::Str(s) => match s.as_str() {
                "NaN" => Ok(f64::NAN),
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                _ => invalid("saved model: a number was expected"),
            },
            _ => invalid("saved model: a number was expected"),
        }
    }

    fn usize(&self) -> Result<usize> {
        let x = self.f64()?;
        if x >= 0.0 && x.fract() == 0.0 && x < 9.0e15 {
            Ok(x as usize)
        } else {
            invalid("saved model: a count was expected")
        }
    }

    fn opt_index(&self) -> Result<Option<usize>> {
        let x = self.f64()?;
        if x < 0.0 {
            Ok(None)
        } else {
            self.usize().map(Some)
        }
    }

    fn bool(&self) -> Result<bool> {
        match self {
            Json::Bool(b) => Ok(*b),
            _ => invalid("saved model: a boolean was expected"),
        }
    }

    fn str(&self) -> Result<&str> {
        match self {
            Json::Str(s) => Ok(s),
            _ => invalid("saved model: text was expected"),
        }
    }

    fn arr(&self) -> Result<&[Json]> {
        match self {
            Json::Arr(a) => Ok(a),
            _ => invalid("saved model: a list was expected"),
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<()> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            invalid(format!("saved model: expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            invalid(format!("saved model: bad literal at byte {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > 64 {
            return invalid("saved model: nested too deeply");
        }
        self.ws();
        match self.s.get(self.i) {
            None => invalid("saved model: unexpected end"),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    v.push(self.value(depth + 1)?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(v));
                        }
                        _ => return invalid(format!("saved model: bad list at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(v));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    v.push((k, self.value(depth + 1)?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(v));
                        }
                        _ => return invalid(format!("saved model: bad object at byte {}", self.i)),
                    }
                }
            }
            Some(_) => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[st..self.i])
                    .ok()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(Json::Num)
                    .ok_or_else(|| {
                        crate::error::StatsError::InvalidArgument(format!("saved model: bad number at byte {st}"))
                    })
            }
        }
    }

    fn string(&mut self) -> Result<String> {
        if self.s.get(self.i) != Some(&b'"') {
            return invalid(format!("saved model: expected a string at byte {}", self.i));
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.i) else { return invalid("saved model: unterminated string") };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else { return invalid("saved model: bad escape") };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let hex = |p: &mut Parser<'_>| -> Result<u32> {
                                let h = p
                                    .s
                                    .get(p.i..p.i + 4)
                                    .and_then(|b| std::str::from_utf8(b).ok())
                                    .and_then(|t| u32::from_str_radix(t, 16).ok())
                                    .ok_or_else(|| {
                                        crate::error::StatsError::InvalidArgument("saved model: bad \\u escape".into())
                                    })?;
                                p.i += 4;
                                Ok(h)
                            };
                            let mut cp = hex(self)?;
                            if (0xD800..0xDC00).contains(&cp) && self.s[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = hex(self)?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                            }
                            char::from_u32(cp).unwrap_or('\u{FFFD}')
                        }
                        _ => return invalid("saved model: bad escape"),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| crate::error::StatsError::InvalidArgument("saved model: bad UTF-8".into()))
    }
}

fn parse(text: &str) -> Result<Json> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return invalid("saved model: trailing characters");
    }
    Ok(v)
}

// ---- encoders

fn n(x: f64) -> Json {
    Json::Num(x)
}

fn u(x: usize) -> Json {
    Json::Num(x as f64)
}

fn opt(x: Option<usize>) -> Json {
    Json::Num(x.map_or(-1.0, |v| v as f64))
}

fn strs(v: &[String]) -> Json {
    Json::Arr(v.iter().map(|s| Json::Str(s.clone())).collect())
}

fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn enc_schema(d: &Dataset) -> Json {
    obj(vec![
        ("names", strs(&d.names)),
        (
            "features",
            Json::Arr(
                d.features
                    .iter()
                    .map(|f| match f {
                        Feature::Numeric(_) => obj(vec![("kind", Json::Str("numeric".into()))]),
                        Feature::Categorical { levels, .. } => {
                            obj(vec![("kind", Json::Str("categorical".into())), ("levels", strs(levels))])
                        }
                    })
                    .collect(),
            ),
        ),
    ])
}

fn enc_test(t: &Test) -> Json {
    match t {
        Test::LessEq { feature, threshold } => {
            obj(vec![("t", Json::Str("le".into())), ("f", u(*feature)), ("x", n(*threshold))])
        }
        Test::InSet { feature, left } => obj(vec![
            ("t", Json::Str("in".into())),
            ("f", u(*feature)),
            ("left", Json::Arr(left.iter().map(|&b| Json::Bool(b)).collect())),
        ]),
    }
}

fn enc_tree(t: &Tree) -> Json {
    let criterion = match t.criterion {
        Criterion::Variance => "variance",
        Criterion::Gini => "gini",
        Criterion::Entropy => "entropy",
    };
    let nodes = t
        .nodes
        .iter()
        .map(|nd| {
            let split = match &nd.split {
                None => Json::Null,
                Some(s) => obj(vec![
                    ("test", enc_test(&s.test)),
                    ("gain", n(s.gain)),
                    ("dl", Json::Bool(s.default_left)),
                    ("nl", u(s.n_left)),
                    ("nr", u(s.n_right)),
                    (
                        "sur",
                        Json::Arr(
                            s.surrogates
                                .iter()
                                .map(|g| {
                                    obj(vec![
                                        ("test", enc_test(&g.test)),
                                        ("rev", Json::Bool(g.reversed)),
                                        ("agree", n(g.agreement)),
                                        ("adj", n(g.adjusted)),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ]),
            };
            let (l, r) = nd.children.map_or((-1.0, -1.0), |(l, r)| (l as f64, r as f64));
            Json::Arr(vec![
                opt(nd.parent),
                n(l),
                n(r),
                u(nd.depth),
                u(nd.n),
                n(nd.value),
                Json::Arr(nd.counts.iter().map(|&c| n(c)).collect()),
                n(nd.deviance),
                n(nd.error),
                split,
            ])
        })
        .collect();
    obj(vec![("criterion", Json::Str(criterion.into())), ("classes", u(t.n_classes)), ("nodes", Json::Arr(nodes))])
}

fn enc_term(t: &Term) -> Json {
    obj(vec![
        ("coef", n(t.coef)),
        (
            "factors",
            Json::Arr(t.factors.iter().map(|f| Json::Arr(vec![u(f.var), n(f.knot), n(f64::from(f.sign))])).collect()),
        ),
    ])
}

impl SavedModel {
    /// The model as JSON text.
    pub fn to_json(&self) -> String {
        let m = &self.model;
        let mut pairs: Vec<(&str, Json)> = vec![
            ("format", Json::Str("openstats-model".into())),
            ("version", u(1)),
            ("response", Json::Str(self.response.clone())),
            ("n_train", u(self.n_train)),
        ];
        match m {
            PredictiveModel::Tree { tree, schema, levels } => {
                pairs.push(("kind", Json::Str("tree".into())));
                pairs.push(("schema", enc_schema(schema)));
                pairs.push(("levels", strs(levels)));
                pairs.push(("tree", enc_tree(tree)));
            }
            PredictiveModel::Forest { trees, schema, levels } => {
                pairs.push(("kind", Json::Str("forest".into())));
                pairs.push(("schema", enc_schema(schema)));
                pairs.push(("levels", strs(levels)));
                pairs.push(("trees", Json::Arr(trees.iter().map(enc_tree).collect())));
            }
            PredictiveModel::Boosting { init, trees, learning_rate, schema, levels, loss } => {
                pairs.push(("kind", Json::Str("boosting".into())));
                pairs.push(("schema", enc_schema(schema)));
                pairs.push(("levels", strs(levels)));
                pairs.push(("init", Json::Arr(init.iter().map(|&x| n(x)).collect())));
                pairs.push(("rate", n(*learning_rate)));
                pairs.push((
                    "loss",
                    Json::Str(
                        match loss {
                            GbdtLoss::SquaredError => "squared",
                            GbdtLoss::Huber => "huber",
                            GbdtLoss::Deviance => "deviance",
                        }
                        .into(),
                    ),
                ));
                pairs.push((
                    "stages",
                    Json::Arr(trees.iter().map(|st| Json::Arr(st.iter().map(enc_tree).collect())).collect()),
                ));
            }
            PredictiveModel::Mars(o) => {
                pairs.push(("kind", Json::Str("mars".into())));
                pairs.push(("schema", enc_schema(&o.schema)));
                pairs.push(("names", strs(&o.names)));
                pairs.push(("terms", Json::Arr(o.terms.iter().map(enc_term).collect())));
                match &o.link {
                    MarsLink::Identity => pairs.push(("link", Json::Str("identity".into()))),
                    MarsLink::Logit { levels } => {
                        pairs.push(("link", Json::Str("logit".into())));
                        pairs.push(("levels", strs(levels)));
                    }
                }
            }
        }
        let mut s = String::new();
        obj(pairs).write(&mut s);
        s
    }

    /// Read a model written by [`to_json`](Self::to_json).
    pub fn from_json(text: &str) -> Result<SavedModel> {
        let j = parse(text)?;
        if j.get("format")?.str()? != "openstats-model" {
            return invalid("this is not an OpenStats model");
        }
        if j.get("version")?.usize()? != 1 {
            return invalid("this model was saved by a newer version of OpenStats");
        }
        let response = j.get("response")?.str()?.to_string();
        let n_train = j.get("n_train")?.usize()?;
        let kind = j.get("kind")?.str()?;
        let schema = || dec_schema(j.get("schema")?);
        let levels = || -> Result<Vec<String>> { dec_strs(j.get("levels")?) };
        let model = match kind {
            "tree" => PredictiveModel::Tree { tree: dec_tree(j.get("tree")?)?, schema: schema()?, levels: levels()? },
            "forest" => PredictiveModel::Forest {
                trees: j.get("trees")?.arr()?.iter().map(dec_tree).collect::<Result<_>>()?,
                schema: schema()?,
                levels: levels()?,
            },
            "boosting" => PredictiveModel::Boosting {
                init: j.get("init")?.arr()?.iter().map(Json::f64).collect::<Result<_>>()?,
                trees: j
                    .get("stages")?
                    .arr()?
                    .iter()
                    .map(|st| st.arr()?.iter().map(dec_tree).collect::<Result<Vec<_>>>())
                    .collect::<Result<_>>()?,
                learning_rate: j.get("rate")?.f64()?,
                schema: schema()?,
                levels: levels()?,
                loss: match j.get("loss")?.str()? {
                    "squared" => GbdtLoss::SquaredError,
                    "huber" => GbdtLoss::Huber,
                    "deviance" => GbdtLoss::Deviance,
                    _ => return invalid("saved model: unknown loss"),
                },
            },
            "mars" => {
                let terms = j
                    .get("terms")?
                    .arr()?
                    .iter()
                    .map(|t| {
                        let factors = t
                            .get("factors")?
                            .arr()?
                            .iter()
                            .map(|f| {
                                let f = f.arr()?;
                                if f.len() != 3 {
                                    return invalid("saved model: a factor has three fields");
                                }
                                Ok(Factor { var: f[0].usize()?, knot: f[1].f64()?, sign: f[2].f64()? as i8 })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(Term { factors, coef: t.get("coef")?.f64()? })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let link = match j.get("link")?.str()? {
                    "identity" => MarsLink::Identity,
                    "logit" => MarsLink::Logit { levels: levels()? },
                    _ => return invalid("saved model: unknown link"),
                };
                PredictiveModel::Mars(MarsObject { schema: schema()?, names: dec_strs(j.get("names")?)?, terms, link })
            }
            _ => return invalid("saved model: unknown kind"),
        };
        model.validate()?;
        Ok(SavedModel { response, n_train, model })
    }
}

fn dec_strs(j: &Json) -> Result<Vec<String>> {
    j.arr()?.iter().map(|s| s.str().map(str::to_string)).collect()
}

fn dec_schema(j: &Json) -> Result<Dataset> {
    let names = dec_strs(j.get("names")?)?;
    let features = j
        .get("features")?
        .arr()?
        .iter()
        .map(|f| match f.get("kind")?.str()? {
            "numeric" => Ok(Feature::Numeric(Vec::new())),
            "categorical" => Ok(Feature::Categorical { codes: Vec::new(), levels: dec_strs(f.get("levels")?)? }),
            _ => invalid("saved model: unknown predictor kind"),
        })
        .collect::<Result<Vec<_>>>()?;
    Dataset::from_features(names, features)
}

fn dec_test(j: &Json) -> Result<Test> {
    match j.get("t")?.str()? {
        "le" => Ok(Test::LessEq { feature: j.get("f")?.usize()?, threshold: j.get("x")?.f64()? }),
        "in" => Ok(Test::InSet {
            feature: j.get("f")?.usize()?,
            left: j.get("left")?.arr()?.iter().map(Json::bool).collect::<Result<_>>()?,
        }),
        _ => invalid("saved model: unknown test"),
    }
}

fn dec_tree(j: &Json) -> Result<Tree> {
    let criterion = match j.get("criterion")?.str()? {
        "variance" => Criterion::Variance,
        "gini" => Criterion::Gini,
        "entropy" => Criterion::Entropy,
        _ => return invalid("saved model: unknown criterion"),
    };
    let rows = j.get("nodes")?.arr()?;
    let mut nodes = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let r = r.arr()?;
        if r.len() != 10 {
            return invalid("saved model: a tree node has ten fields");
        }
        let children = match (r[1].opt_index()?, r[2].opt_index()?) {
            (Some(l), Some(rt)) => {
                if l <= i || rt <= i || l >= rows.len() || rt >= rows.len() {
                    return invalid("saved model: a child index is out of range");
                }
                Some((l, rt))
            }
            (None, None) => None,
            _ => return invalid("saved model: a node has one child"),
        };
        let split = match &r[9] {
            Json::Null => None,
            s => Some(Split {
                test: dec_test(s.get("test")?)?,
                gain: s.get("gain")?.f64()?,
                default_left: s.get("dl")?.bool()?,
                n_left: s.get("nl")?.usize()?,
                n_right: s.get("nr")?.usize()?,
                surrogates: s
                    .get("sur")?
                    .arr()?
                    .iter()
                    .map(|g| {
                        Ok(Surrogate {
                            test: dec_test(g.get("test")?)?,
                            reversed: g.get("rev")?.bool()?,
                            agreement: g.get("agree")?.f64()?,
                            adjusted: g.get("adj")?.f64()?,
                        })
                    })
                    .collect::<Result<_>>()?,
            }),
        };
        if children.is_some() != split.is_some() {
            return invalid("saved model: a node's split and children disagree");
        }
        nodes.push(Node {
            parent: r[0].opt_index()?,
            children,
            depth: r[3].usize()?,
            n: r[4].usize()?,
            value: r[5].f64()?,
            counts: r[6].arr()?.iter().map(Json::f64).collect::<Result<_>>()?,
            deviance: r[7].f64()?,
            error: r[8].f64()?,
            split,
        });
    }
    if nodes.is_empty() {
        return invalid("saved model: a tree has no nodes");
    }
    Ok(Tree { nodes, criterion, n_classes: j.get("classes")?.usize()? })
}
