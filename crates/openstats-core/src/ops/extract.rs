//! Data > Date/Time > Extract and Data > Text functions.

use crate::column::ColumnData;
use crate::datetime::DateTime;
use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;

/// Parts that can be extracted from a date/time column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePart {
    Year,
    Quarter,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    /// Seconds including the fractional part.
    FractionalSecond,
    /// 1 = Sunday … 7 = Saturday (Minitab convention).
    Weekday,
    DayOfYear,
    /// ISO-8601 week number.
    IsoWeek,
    MonthName,
    MonthAbbrev,
    WeekdayName,
    WeekdayAbbrev,
    /// Date part only (time set to midnight), as date/time.
    DateOnly,
    /// Time part only (on 1970-01-01), as date/time.
    TimeOnly,
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

/// Extract a part of every value of a date/time column.
pub fn extract_date_part(src: &ColumnData, part: DatePart) -> Result<ColumnData> {
    let v = src.as_datetime().ok_or(CoreError::TypeMismatch { expected: "date/time", found: src.kind().name() })?;
    use DatePart::*;
    Ok(match part {
        MonthName | MonthAbbrev | WeekdayName | WeekdayAbbrev => ColumnData::Text(
            v.iter()
                .map(|&t| {
                    if t == DATETIME_MISSING {
                        return String::new();
                    }
                    let dt = DateTime::from_timestamp(t);
                    let full = match part {
                        MonthName | MonthAbbrev => MONTHS[dt.month as usize - 1],
                        _ => DAYS[dt.weekday() as usize],
                    };
                    if matches!(part, MonthAbbrev | WeekdayAbbrev) {
                        full[..3].to_string()
                    } else {
                        full.to_string()
                    }
                })
                .collect(),
        ),
        DateOnly | TimeOnly => ColumnData::DateTime(
            v.iter()
                .map(|&t| {
                    if t == DATETIME_MISSING {
                        return t;
                    }
                    let tod = t.rem_euclid(crate::datetime::MICROS_PER_DAY);
                    if part == DateOnly {
                        t - tod
                    } else {
                        tod
                    }
                })
                .collect(),
        ),
        _ => ColumnData::Numeric(
            v.iter()
                .map(|&t| {
                    if t == DATETIME_MISSING {
                        return f64::NAN;
                    }
                    let dt = DateTime::from_timestamp(t);
                    match part {
                        Year => dt.year as f64,
                        Quarter => dt.quarter() as f64,
                        Month => dt.month as f64,
                        Day => dt.day as f64,
                        Hour => dt.hour as f64,
                        Minute => dt.minute as f64,
                        Second => dt.second as f64,
                        FractionalSecond => dt.second as f64 + dt.micros as f64 / 1e6,
                        Weekday => (dt.weekday() + 1) as f64,
                        DayOfYear => dt.day_of_year() as f64,
                        IsoWeek => dt.iso_week().1 as f64,
                        _ => unreachable!("handled above"),
                    }
                })
                .collect(),
        ),
    })
}

/// Text transformations (Calc > Calculator text functions as column ops).
#[derive(Debug, Clone, PartialEq)]
pub enum TextOp {
    /// First `n` characters.
    Left(usize),
    /// Last `n` characters.
    Right(usize),
    /// `len` characters starting at 1-based `start`.
    Mid {
        start: usize,
        len: usize,
    },
    Trim,
    Upper,
    Lower,
    /// Capitalise the first letter of each word.
    Proper,
    /// Replace every occurrence of `from` with `to`.
    Substitute {
        from: String,
        to: String,
    },
}

/// Apply a text operation to every value (text column in → text column out).
pub fn text_op(src: &ColumnData, op: &TextOp) -> Result<ColumnData> {
    let v = src.as_text().ok_or(CoreError::TypeMismatch { expected: "text", found: src.kind().name() })?;
    if let TextOp::Substitute { from, .. } = op {
        if from.is_empty() {
            return Err(CoreError::invalid("substitute: search text is empty"));
        }
    }
    Ok(ColumnData::Text(
        v.iter()
            .map(|s| {
                let chars: Vec<char> = s.chars().collect();
                match op {
                    TextOp::Left(n) => chars.iter().take(*n).collect(),
                    TextOp::Right(n) => chars[chars.len().saturating_sub(*n)..].iter().collect(),
                    TextOp::Mid { start, len } => chars.iter().skip(start.saturating_sub(1)).take(*len).collect(),
                    TextOp::Trim => s.trim().to_string(),
                    TextOp::Upper => s.to_uppercase(),
                    TextOp::Lower => s.to_lowercase(),
                    TextOp::Proper => {
                        let mut out = String::with_capacity(s.len());
                        let mut start = true;
                        for c in chars {
                            if c.is_alphabetic() {
                                if start {
                                    out.extend(c.to_uppercase());
                                } else {
                                    out.extend(c.to_lowercase());
                                }
                                start = false;
                            } else {
                                out.push(c);
                                start = !c.is_alphanumeric() && c != '\'';
                            }
                        }
                        out
                    }
                    TextOp::Substitute { from, to } => s.replace(from.as_str(), to),
                }
            })
            .collect(),
    ))
}

/// Character length of each text value (missing text → 0, as in Minitab).
pub fn text_length(src: &ColumnData) -> Result<ColumnData> {
    let v = src.as_text().ok_or(CoreError::TypeMismatch { expected: "text", found: src.kind().name() })?;
    Ok(ColumnData::Numeric(v.iter().map(|s| s.chars().count() as f64).collect()))
}

/// 1-based position of `needle` in each value (0 if absent).
pub fn text_find(src: &ColumnData, needle: &str) -> Result<ColumnData> {
    let v = src.as_text().ok_or(CoreError::TypeMismatch { expected: "text", found: src.kind().name() })?;
    Ok(ColumnData::Numeric(
        v.iter()
            .map(|s| match s.find(needle) {
                Some(b) => (s[..b].chars().count() + 1) as f64,
                None => 0.0,
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datetime::parse;

    #[test]
    fn date_parts() {
        let d = ColumnData::DateTime(vec![parse("2026-11-16 13:45:30.5").unwrap(), DATETIME_MISSING]);
        let get = |p| extract_date_part(&d, p).unwrap();
        assert_eq!(get(DatePart::Year).get(0).as_f64(), 2026.0);
        assert_eq!(get(DatePart::Weekday).get(0).as_f64(), 2.0);
        assert_eq!(get(DatePart::FractionalSecond).get(0).as_f64(), 30.5);
        assert_eq!(get(DatePart::IsoWeek).get(0).as_f64(), 47.0);
        assert_eq!(get(DatePart::MonthAbbrev).as_text().unwrap(), ["Nov", ""]);
        assert_eq!(get(DatePart::WeekdayName).as_text().unwrap()[0], "Monday");
        assert_eq!(
            crate::datetime::format(get(DatePart::DateOnly).as_datetime().unwrap()[0], "yyyy-MM-dd HH:mm"),
            "2026-11-16 00:00"
        );
        assert!(get(DatePart::Year).is_missing(1));
        assert!(extract_date_part(&ColumnData::Numeric(vec![]), DatePart::Year).is_err());
    }

    #[test]
    fn text_ops() {
        let t = ColumnData::Text(vec!["  héllo wORLD ".into(), "o'neil-smith".into()]);
        assert_eq!(text_op(&t, &TextOp::Trim).unwrap().get(0).to_text(), "héllo wORLD");
        assert_eq!(text_op(&t, &TextOp::Proper).unwrap().as_text().unwrap(), ["  Héllo World ", "O'neil-Smith"]);
        assert_eq!(text_op(&t, &TextOp::Left(4)).unwrap().get(0).to_text(), "  hé");
        assert_eq!(text_op(&t, &TextOp::Right(3)).unwrap().get(1).to_text(), "ith");
        assert_eq!(text_op(&t, &TextOp::Mid { start: 3, len: 5 }).unwrap().get(0).to_text(), "héllo");
        assert_eq!(
            text_op(&t, &TextOp::Substitute { from: "-".into(), to: " ".into() }).unwrap().get(1).to_text(),
            "o'neil smith"
        );
        assert_eq!(text_length(&t).unwrap().get(0).as_f64(), 14.0);
        assert_eq!(text_find(&t, "llo").unwrap().get(0).as_f64(), 5.0);
        assert_eq!(text_find(&t, "zzz").unwrap().get(0).as_f64(), 0.0);
    }
}
