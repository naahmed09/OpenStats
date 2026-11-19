//! Data > Change Data Type.

use crate::column::{ColumnData, ColumnKind, NumericFormat};
use crate::datetime::{self, ParseOptions};
use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;
use crate::value::parse_number_with;

/// Options controlling type conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeTypeOptions {
    /// Fail on the first unconvertible value instead of making it missing.
    pub strict: bool,
    /// Decimal separator for text → numeric.
    pub decimal: char,
    /// Date parsing options for text → date/time.
    pub date: ParseOptions,
    /// Number format for numeric → text.
    pub numeric_format: NumericFormat,
    /// Pattern for date/time → text (`None` = default display).
    pub datetime_format: Option<String>,
}

impl Default for ChangeTypeOptions {
    fn default() -> Self {
        ChangeTypeOptions {
            strict: false,
            decimal: '.',
            date: ParseOptions::default(),
            numeric_format: NumericFormat::Automatic,
            datetime_format: None,
        }
    }
}

/// Convert a column to another storage type.
pub fn change_type(src: &ColumnData, to: ColumnKind, opts: &ChangeTypeOptions) -> Result<ColumnData> {
    let fail = |row: usize, what: &str| {
        CoreError::invalid(format!("row {}: cannot convert {what:?} to {}", row + 1, to.name()))
    };
    match (src, to) {
        (_, k) if k == src.kind() => Ok(src.clone()),
        (ColumnData::Text(v), ColumnKind::Numeric) => {
            let mut out = Vec::with_capacity(v.len());
            for (i, s) in v.iter().enumerate() {
                match parse_number_with(s, opts.decimal) {
                    Ok(x) => out.push(x),
                    Err(_) if opts.strict => return Err(fail(i, s)),
                    Err(_) => out.push(f64::NAN),
                }
            }
            Ok(ColumnData::Numeric(out))
        }
        (ColumnData::Text(v), ColumnKind::DateTime) => {
            let mut out = Vec::with_capacity(v.len());
            for (i, s) in v.iter().enumerate() {
                match datetime::parse_with(s, opts.date) {
                    Ok(t) => out.push(t),
                    Err(_) if opts.strict => return Err(fail(i, s)),
                    Err(_) => out.push(DATETIME_MISSING),
                }
            }
            Ok(ColumnData::DateTime(out))
        }
        (ColumnData::Numeric(v), ColumnKind::Text) => Ok(ColumnData::Text(
            v.iter().map(|&x| if x.is_nan() { String::new() } else { opts.numeric_format.format(x) }).collect(),
        )),
        (ColumnData::DateTime(v), ColumnKind::Text) => Ok(ColumnData::Text(
            v.iter()
                .map(|&t| {
                    if t == DATETIME_MISSING {
                        String::new()
                    } else {
                        match &opts.datetime_format {
                            Some(p) => datetime::format(t, p),
                            None => datetime::format_default(t),
                        }
                    }
                })
                .collect(),
        )),
        _ => src.convert(to, opts.strict),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        let t = ColumnData::Text(vec!["1,5".into(), "x".into()]);
        let o = ChangeTypeOptions { decimal: ',', ..Default::default() };
        let n = change_type(&t, ColumnKind::Numeric, &o).unwrap();
        assert_eq!(n.get(0).as_f64(), 1.5);
        assert!(n.is_missing(1));
        assert!(change_type(
            &t,
            ColumnKind::Numeric,
            &ChangeTypeOptions { strict: true, decimal: ',', ..Default::default() }
        )
        .is_err());
        let d = change_type(
            &ColumnData::Text(vec!["16/11/2026".into()]),
            ColumnKind::DateTime,
            &ChangeTypeOptions {
                date: ParseOptions { order: crate::datetime::DateOrder::Dmy, ..Default::default() },
                ..Default::default()
            },
        )
        .unwrap();
        let txt = change_type(
            &d,
            ColumnKind::Text,
            &ChangeTypeOptions { datetime_format: Some("yyyy-MM-dd".into()), ..Default::default() },
        )
        .unwrap();
        assert_eq!(txt.as_text().unwrap(), ["2026-11-16"]);
        let ft = change_type(
            &ColumnData::Numeric(vec![2.0 / 3.0, f64::NAN]),
            ColumnKind::Text,
            &ChangeTypeOptions { numeric_format: NumericFormat::Fixed(3), ..Default::default() },
        )
        .unwrap();
        assert_eq!(ft.as_text().unwrap(), ["0.667", ""]);
    }
}
