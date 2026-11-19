//! Declarative dialog descriptions and the values a user enters into them.

use openstats_core::column::ColumnKind;
use openstats_core::Worksheet;
use std::collections::BTreeMap;

/// Which columns a variable box accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    /// Numeric (and date/time) columns.
    Numeric,
    /// Text columns.
    Text,
    /// Any column.
    Any,
}

/// The page of the dialog a field is on (Minitab's main window and its sub-dialogs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Page {
    /// The main dialog.
    Main,
    /// "Options…".
    Options,
    /// "Graphs…".
    Graphs,
    /// "Storage…" (results stored in the worksheet).
    Storage,
}

impl Page {
    /// Button label.
    pub fn label(self) -> &'static str {
        match self {
            Page::Main => "Main",
            Page::Options => "Options...",
            Page::Graphs => "Graphs...",
            Page::Storage => "Storage...",
        }
    }

    /// All pages in display order.
    pub const ALL: [Page; 4] = [Page::Main, Page::Options, Page::Graphs, Page::Storage];
}

/// What a field edits.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    /// Columns picked from the worksheet.
    Variables {
        /// Accepted column kind.
        kind: VarKind,
        /// Fewest columns (0 = optional).
        min: usize,
        /// Most columns.
        max: usize,
    },
    /// A number.
    Number {
        /// Initial value.
        default: f64,
        /// Smallest accepted value.
        min: f64,
        /// Largest accepted value.
        max: f64,
        /// Only whole numbers are accepted.
        integer: bool,
        /// May be left empty (then the value is `NaN`).
        optional: bool,
    },
    /// Free text.
    Text {
        /// Initial value.
        default: String,
    },
    /// One of several options.
    Choice {
        /// Option labels.
        options: Vec<String>,
        /// Index of the initial option.
        default: usize,
    },
    /// A checkbox.
    Check {
        /// Initial state.
        default: bool,
    },
}

/// One input of a dialog.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Key under which the value is stored.
    pub name: String,
    /// Label shown to the user.
    pub label: String,
    /// Kind of input.
    pub kind: FieldKind,
    /// Page the field belongs to.
    pub page: Page,
}

/// A complete dialog description.
#[derive(Debug, Clone, PartialEq)]
pub struct DialogSpec {
    /// Stable identifier, e.g. `stat.basic.ttest1`.
    pub id: String,
    /// Window title.
    pub title: String,
    /// One-paragraph help text.
    pub help: String,
    /// Inputs in display order.
    pub fields: Vec<Field>,
    /// `(variable field, choice field, choice value)`: the variable box is not required while the choice field has
    /// that value (for example, the column boxes of a t-test are not needed in "Summarized data" mode).
    pub optional_when: Vec<(String, String, usize)>,
}

impl DialogSpec {
    /// A spec with no fields.
    pub fn new(id: &str, title: &str, help: &str) -> DialogSpec {
        DialogSpec {
            id: id.into(),
            title: title.into(),
            help: help.into(),
            fields: Vec::new(),
            optional_when: Vec::new(),
        }
    }

    /// The variable box `var` is not required while choice field `choice` equals option `value`.
    pub fn vars_optional_when(mut self, var: &str, choice: &str, value: usize) -> DialogSpec {
        self.optional_when.push((var.into(), choice.into(), value));
        self
    }

    fn push(mut self, name: &str, label: &str, page: Page, kind: FieldKind) -> DialogSpec {
        self.fields.push(Field { name: name.into(), label: label.into(), kind, page });
        self
    }

    /// Add a variable box (main page).
    pub fn vars(self, name: &str, label: &str, kind: VarKind, min: usize, max: usize) -> DialogSpec {
        self.push(name, label, Page::Main, FieldKind::Variables { kind, min, max })
    }

    /// Add a number field on `page`.
    pub fn number(self, page: Page, name: &str, label: &str, default: f64, min: f64, max: f64) -> DialogSpec {
        self.push(name, label, page, FieldKind::Number { default, min, max, integer: false, optional: false })
    }

    /// Add a whole-number field on `page`.
    pub fn integer(self, page: Page, name: &str, label: &str, default: f64, min: f64, max: f64) -> DialogSpec {
        self.push(name, label, page, FieldKind::Number { default, min, max, integer: true, optional: false })
    }

    /// Add a number field that may be left empty.
    pub fn optional_number(self, page: Page, name: &str, label: &str) -> DialogSpec {
        self.push(
            name,
            label,
            page,
            FieldKind::Number {
                default: f64::NAN,
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
                integer: false,
                optional: true,
            },
        )
    }

    /// Add a text field.
    pub fn text(self, page: Page, name: &str, label: &str, default: &str) -> DialogSpec {
        self.push(name, label, page, FieldKind::Text { default: default.into() })
    }

    /// Add a choice field.
    pub fn choice(self, page: Page, name: &str, label: &str, options: &[&str], default: usize) -> DialogSpec {
        self.push(
            name,
            label,
            page,
            FieldKind::Choice { options: options.iter().map(|s| (*s).to_string()).collect(), default },
        )
    }

    /// Add a checkbox.
    pub fn check(self, page: Page, name: &str, label: &str, default: bool) -> DialogSpec {
        self.push(name, label, page, FieldKind::Check { default })
    }

    /// The field called `name`.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Pages that have at least one field (Main first).
    pub fn pages(&self) -> Vec<Page> {
        Page::ALL.iter().copied().filter(|p| *p == Page::Main || self.fields.iter().any(|f| f.page == *p)).collect()
    }

    /// The initial values.
    pub fn defaults(&self) -> Values {
        let mut v = Values::default();
        for f in &self.fields {
            let val = match &f.kind {
                FieldKind::Variables { .. } => Value::Vars(Vec::new()),
                FieldKind::Number { default, .. } => Value::Num(*default),
                FieldKind::Text { default } => Value::Text(default.clone()),
                FieldKind::Choice { default, .. } => Value::Choice(*default),
                FieldKind::Check { default } => Value::Bool(*default),
            };
            v.0.insert(f.name.clone(), val);
        }
        v
    }
}

/// A value entered into a field.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Selected column indices.
    Vars(Vec<usize>),
    /// A number (`NaN` for an empty optional field).
    Num(f64),
    /// Text.
    Text(String),
    /// Index of the chosen option.
    Choice(usize),
    /// A checkbox.
    Bool(bool),
}

/// The values of every field of a dialog.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Values(pub BTreeMap<String, Value>);

impl Values {
    /// Selected columns of a variable field (empty if unknown).
    pub fn vars(&self, name: &str) -> &[usize] {
        match self.0.get(name) {
            Some(Value::Vars(v)) => v,
            _ => &[],
        }
    }

    /// The first selected column of a variable field.
    pub fn var(&self, name: &str) -> Option<usize> {
        self.vars(name).first().copied()
    }

    /// A number (`NaN` if unknown or empty).
    pub fn num(&self, name: &str) -> f64 {
        match self.0.get(name) {
            Some(Value::Num(x)) => *x,
            _ => f64::NAN,
        }
    }

    /// A number as a count (0 if unknown, empty or negative).
    pub fn count(&self, name: &str) -> usize {
        let x = self.num(name);
        if x.is_finite() && x > 0.0 {
            x as usize
        } else {
            0
        }
    }

    /// Text (empty if unknown).
    pub fn text(&self, name: &str) -> &str {
        match self.0.get(name) {
            Some(Value::Text(s)) => s,
            _ => "",
        }
    }

    /// Chosen option index (0 if unknown).
    pub fn choice(&self, name: &str) -> usize {
        match self.0.get(name) {
            Some(Value::Choice(i)) => *i,
            _ => 0,
        }
    }

    /// A checkbox (false if unknown).
    pub fn flag(&self, name: &str) -> bool {
        matches!(self.0.get(name), Some(Value::Bool(true)))
    }

    /// Set a value.
    pub fn set(&mut self, name: &str, v: Value) {
        self.0.insert(name.to_string(), v);
    }
}

/// A problem found while validating dialog input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// Field name.
    pub field: String,
    /// Message for the user.
    pub message: String,
}

/// What validation needs to know about a worksheet column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColInfo {
    /// Display label (`C1`, `C2-T`, …).
    pub label: String,
    /// Column name (may be empty).
    pub name: String,
    /// Kind of the column.
    pub kind: ColumnKind,
}

impl ColInfo {
    /// Describe every column of `ws`.
    pub fn of(ws: &Worksheet) -> Vec<ColInfo> {
        (0..ws.n_cols())
            .map(|c| ColInfo {
                label: ws.column_label(c),
                name: ws.columns[c].name().to_string(),
                kind: ws.columns[c].kind(),
            })
            .collect()
    }

    /// `C1 name` as shown in the column list.
    pub fn list_text(&self) -> String {
        if self.name.is_empty() {
            self.label.clone()
        } else {
            format!("{} {}", self.label, self.name)
        }
    }
}

/// Check `values` against `spec` for worksheet `ws`.
pub fn validate(spec: &DialogSpec, values: &Values, ws: &Worksheet) -> Vec<Problem> {
    validate_info(spec, values, &ColInfo::of(ws))
}

/// Check `values` against `spec` for a worksheet described by `cols`.
pub fn validate_info(spec: &DialogSpec, values: &Values, cols: &[ColInfo]) -> Vec<Problem> {
    let mut out = Vec::new();
    let mut bad = |f: &Field, m: String| out.push(Problem { field: f.name.clone(), message: m });
    for f in &spec.fields {
        match &f.kind {
            FieldKind::Variables { kind, min, max } => {
                let v = values.vars(&f.name);
                let waived =
                    spec.optional_when.iter().any(|(var, ch, val)| *var == f.name && values.choice(ch) == *val);
                if v.len() < *min && !waived {
                    bad(
                        f,
                        if *min == 1 {
                            format!("{}: select a column.", f.label)
                        } else {
                            format!("{}: select at least {min} columns.", f.label)
                        },
                    );
                }
                if v.len() > *max {
                    bad(f, format!("{}: select at most {max} column{}.", f.label, if *max == 1 { "" } else { "s" }));
                }
                for &c in v {
                    match cols.get(c) {
                        None => bad(f, format!("{}: column C{} is empty.", f.label, c + 1)),
                        Some(col) => {
                            let ok = match kind {
                                VarKind::Any => true,
                                VarKind::Text => col.kind == ColumnKind::Text,
                                VarKind::Numeric => col.kind != ColumnKind::Text,
                            };
                            if !ok {
                                let want = if *kind == VarKind::Text { "text" } else { "numeric" };
                                bad(f, format!("{}: {} is not a {want} column.", f.label, col.label));
                            }
                        }
                    }
                }
            }
            FieldKind::Number { min, max, integer, optional, .. } => {
                let x = values.num(&f.name);
                if x.is_nan() {
                    if !optional {
                        bad(f, format!("{}: enter a number.", f.label));
                    }
                } else if x < *min || x > *max {
                    bad(
                        f,
                        format!(
                            "{}: must be between {} and {}.",
                            f.label,
                            openstats_core::value::format_number(*min),
                            openstats_core::value::format_number(*max)
                        ),
                    );
                } else if *integer && x.fract() != 0.0 {
                    bad(f, format!("{}: must be a whole number.", f.label));
                }
            }
            FieldKind::Choice { options, .. } => {
                if values.choice(&f.name) >= options.len() {
                    bad(f, format!("{}: choose an option.", f.label));
                }
            }
            FieldKind::Text { .. } | FieldKind::Check { .. } => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_core::column::Column;

    fn ws() -> Worksheet {
        let mut w = Worksheet::new("W");
        w.add_column(Column::numeric("x", vec![1.0, 2.0])).unwrap();
        w.add_column(Column::text("g", vec!["a", "b"])).unwrap();
        w
    }

    fn spec() -> DialogSpec {
        DialogSpec::new("t", "T", "help")
            .vars("y", "Response", VarKind::Numeric, 1, 1)
            .vars("f", "Factor", VarKind::Text, 0, 1)
            .number(Page::Options, "level", "Confidence level", 95.0, 0.0, 100.0)
            .integer(Page::Options, "n", "Size", 5.0, 2.0, 10.0)
            .choice(Page::Options, "alt", "Alternative", &["not equal", "less"], 0)
            .check(Page::Graphs, "hist", "Histogram", true)
            .optional_number(Page::Options, "target", "Target")
    }

    #[test]
    fn defaults_and_pages() {
        let s = spec();
        let v = s.defaults();
        assert_eq!(v.num("level"), 95.0);
        assert!(v.flag("hist") && v.vars("y").is_empty() && v.num("target").is_nan());
        assert_eq!(s.pages(), vec![Page::Main, Page::Options, Page::Graphs]);
        assert_eq!(v.count("n"), 5);
        assert_eq!(v.count("target"), 0);
    }

    #[test]
    fn validation() {
        let s = spec();
        let w = ws();
        let mut v = s.defaults();
        let p = validate(&s, &v, &w);
        assert_eq!(p.len(), 1);
        assert!(p[0].message.contains("select a column"));
        v.set("y", Value::Vars(vec![1]));
        assert!(validate(&s, &v, &w)[0].message.contains("not a numeric column"));
        v.set("y", Value::Vars(vec![0]));
        v.set("f", Value::Vars(vec![0]));
        assert!(validate(&s, &v, &w)[0].message.contains("not a text column"));
        v.set("f", Value::Vars(vec![1]));
        assert!(validate(&s, &v, &w).is_empty());
        v.set("y", Value::Vars(vec![0, 0]));
        assert!(validate(&s, &v, &w)[0].message.contains("at most 1 column."));
        v.set("y", Value::Vars(vec![7]));
        assert!(validate(&s, &v, &w)[0].message.contains("is empty"));
        v.set("y", Value::Vars(vec![0]));
        v.set("level", Value::Num(120.0));
        assert!(validate(&s, &v, &w)[0].message.contains("between 0 and 100"));
        v.set("level", Value::Num(f64::NAN));
        assert!(validate(&s, &v, &w)[0].message.contains("enter a number"));
        v.set("level", Value::Num(95.0));
        v.set("n", Value::Num(2.5));
        assert!(validate(&s, &v, &w)[0].message.contains("whole"));
        v.set("n", Value::Num(3.0));
        v.set("alt", Value::Choice(9));
        assert!(validate(&s, &v, &w)[0].message.contains("choose"));
    }
}
