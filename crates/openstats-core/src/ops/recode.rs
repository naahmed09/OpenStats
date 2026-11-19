//! Data > Recode (to numeric / text / date-time) and Calc > Make Indicator
//! Variables.

use crate::column::{level_key, Column, ColumnData, ColumnKind};
use crate::error::{CoreError, Result};
use crate::value::Value;

/// A single recoding rule. Rules are tried in order; the first match wins.
#[derive(Debug, Clone, PartialEq)]
pub enum RecodeRule {
    /// Exact value match (text comparison honours `case_sensitive`).
    Value { from: Value, to: Value },
    /// Numeric (or date-as-serial) interval match.
    Range { lo: f64, hi: f64, lo_inclusive: bool, hi_inclusive: bool, to: Value },
    /// Missing input values.
    Missing { to: Value },
}

/// What to do with values that match no rule.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Unmatched {
    /// Keep the original value (converted to the output type when possible,
    /// missing otherwise).
    #[default]
    KeepOriginal,
    /// Set to missing.
    Missing,
    /// Set to a fixed value.
    Value(Value),
}

/// Recode a column's values into a new column of kind `out`.
pub fn recode(
    src: &ColumnData,
    rules: &[RecodeRule],
    out: ColumnKind,
    unmatched: &Unmatched,
    case_sensitive: bool,
) -> Result<ColumnData> {
    let mut result = ColumnData::missing(out, src.len());
    let norm = |v: &Value| -> String {
        match v {
            Value::Text(s) if !case_sensitive => format!("t{}", s.to_lowercase()),
            other => level_key(other),
        }
    };
    // Pre-normalise exact-match keys, coercing rule values to the source type
    // so that a text rule "3" matches numeric 3.
    let keys: Vec<Option<String>> = rules
        .iter()
        .map(|r| match r {
            RecodeRule::Value { from, .. } => {
                let mut probe = ColumnData::empty(src.kind());
                Some(match probe.push(from) {
                    Ok(()) => norm(&probe.get(0)),
                    Err(_) => norm(from),
                })
            }
            _ => None,
        })
        .collect();
    for i in 0..src.len() {
        let v = src.get(i);
        let missing = v.is_missing();
        let vk = if missing { None } else { Some(norm(&v)) };
        let x = v.as_f64();
        let mut target: Option<&Value> = None;
        for (r, k) in rules.iter().zip(keys.iter()) {
            let hit = match r {
                RecodeRule::Value { .. } => !missing && k.as_ref() == vk.as_ref(),
                RecodeRule::Range { lo, hi, lo_inclusive, hi_inclusive, .. } => {
                    !missing
                        && !x.is_nan()
                        && (if *lo_inclusive { x >= *lo } else { x > *lo })
                        && (if *hi_inclusive { x <= *hi } else { x < *hi })
                }
                RecodeRule::Missing { .. } => missing,
            };
            if hit {
                target = Some(match r {
                    RecodeRule::Value { to, .. } | RecodeRule::Range { to, .. } | RecodeRule::Missing { to } => to,
                });
                break;
            }
        }
        let value = match target {
            Some(t) => t.clone(),
            None => match unmatched {
                Unmatched::KeepOriginal => {
                    if missing {
                        Value::Missing
                    } else {
                        let converted = src.take(&[i]).convert(out, false)?;
                        converted.get(0)
                    }
                }
                Unmatched::Missing => Value::Missing,
                Unmatched::Value(v) => v.clone(),
            },
        };
        result.set(i, &value).map_err(|e| {
            CoreError::invalid(format!("row {}: recoded value does not fit {}: {e}", i + 1, out.name()))
        })?;
    }
    Ok(result)
}

/// One 0/1 indicator column per level (value order), named `Name_Level`.
/// Missing input rows are missing in every indicator.
pub fn indicator_columns(col: &Column) -> Vec<Column> {
    let (levels, codes) = col.level_codes();
    let base = if col.name().is_empty() { "C".to_string() } else { col.name().to_string() };
    levels
        .iter()
        .enumerate()
        .map(|(li, lv)| {
            let data = codes
                .iter()
                .map(|c| match c {
                    None => f64::NAN,
                    Some(k) if *k == li => 1.0,
                    Some(_) => 0.0,
                })
                .collect();
            Column::numeric(format!("{base}_{}", lv.to_text()), data)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recode_ranges_and_values() {
        let src = ColumnData::Numeric(vec![1.0, 5.0, 10.0, f64::NAN, 42.0]);
        let rules = vec![
            RecodeRule::Range { lo: 0.0, hi: 5.0, lo_inclusive: true, hi_inclusive: false, to: "Low".into() },
            RecodeRule::Range { lo: 5.0, hi: 10.0, lo_inclusive: true, hi_inclusive: true, to: "High".into() },
            RecodeRule::Missing { to: "None".into() },
        ];
        let out = recode(&src, &rules, ColumnKind::Text, &Unmatched::KeepOriginal, true).unwrap();
        assert_eq!(out.as_text().unwrap(), ["Low", "High", "High", "None", "42"]);
        let out = recode(&src, &rules[..1], ColumnKind::Text, &Unmatched::Value("Other".into()), true).unwrap();
        assert_eq!(out.as_text().unwrap(), ["Low", "Other", "Other", "Other", "Other"]);
    }

    #[test]
    fn recode_text_case_insensitive() {
        let src = ColumnData::Text(vec!["Pass".into(), "FAIL".into(), "pass".into(), "".into()]);
        let rules = vec![
            RecodeRule::Value { from: "pass".into(), to: Value::Numeric(1.0) },
            RecodeRule::Value { from: "fail".into(), to: Value::Numeric(0.0) },
        ];
        let out = recode(&src, &rules, ColumnKind::Numeric, &Unmatched::Missing, false).unwrap();
        assert_eq!(&out.as_numeric().unwrap()[..3], &[1.0, 0.0, 1.0]);
        assert!(out.is_missing(3));
        let strict = recode(&src, &rules, ColumnKind::Numeric, &Unmatched::Missing, true).unwrap();
        assert!(strict.is_missing(0) && strict.is_missing(1));
        let num = ColumnData::Numeric(vec![3.0]);
        let r = recode(
            &num,
            &[RecodeRule::Value { from: "3".into(), to: "three".into() }],
            ColumnKind::Text,
            &Unmatched::Missing,
            true,
        )
        .unwrap();
        assert_eq!(r.get(0), Value::Text("three".into()));
        let bad = recode(
            &src,
            &[RecodeRule::Value { from: "pass".into(), to: "x".into() }],
            ColumnKind::Numeric,
            &Unmatched::Missing,
            false,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn indicators() {
        let c = Column::text("G", vec!["a", "b", "", "a"]);
        let ind = indicator_columns(&c);
        assert_eq!(ind.len(), 2);
        assert_eq!(ind[0].name(), "G_a");
        assert_eq!(ind[0].data.as_numeric().unwrap()[..2], [1.0, 0.0]);
        assert!(ind[1].data.is_missing(2));
    }
}
