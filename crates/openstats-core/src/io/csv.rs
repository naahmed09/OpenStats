//! RFC-4180 delimited text (CSV/TSV) reader and writer with type inference.

use crate::column::{Column, ColumnData, ColumnKind};
use crate::datetime::{self, ParseOptions};
use crate::error::{CoreError, Result};
use crate::missing::{is_missing_token, DATETIME_MISSING};
use crate::value::{format_number, parse_number_with};
use crate::worksheet::{Worksheet, MAX_NAME_LEN};
use std::collections::HashSet;

/// Reader options.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvReadOptions {
    /// Field delimiter; `None` auto-detects among `,` `;` tab `|`.
    pub delimiter: Option<char>,
    pub quote: char,
    /// First (non-skipped) record holds column names.
    pub has_header: bool,
    /// Records to skip before the header/data.
    pub skip_rows: usize,
    /// Decimal separator for numbers (`.` or `,`).
    pub decimal: char,
    pub date: ParseOptions,
    /// Infer numeric/date-time columns; otherwise everything is text.
    pub infer_types: bool,
    /// Trim whitespace around unquoted fields.
    pub trim: bool,
    pub sheet_name: String,
    /// Stop after this many data rows (`None` = all).
    pub max_rows: Option<usize>,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        CsvReadOptions {
            delimiter: None,
            quote: '"',
            has_header: true,
            skip_rows: 0,
            decimal: '.',
            date: ParseOptions::default(),
            infer_types: true,
            trim: true,
            sheet_name: "Worksheet 1".into(),
            max_rows: None,
        }
    }
}

/// Writer options.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvWriteOptions {
    pub delimiter: char,
    pub quote: char,
    pub header: bool,
    pub decimal: char,
    /// Token written for numeric/date-time missing values.
    pub missing: String,
    /// Write date/times as ISO-8601 instead of the column display format.
    pub iso_dates: bool,
    /// Use each numeric column's display format instead of full precision.
    pub use_display_format: bool,
    pub line_ending: &'static str,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        CsvWriteOptions {
            delimiter: ',',
            quote: '"',
            header: true,
            decimal: '.',
            missing: "*".into(),
            iso_dates: true,
            use_display_format: false,
            line_ending: "\r\n",
        }
    }
}

/// Split text into records of fields (RFC 4180: quoted fields may contain
/// delimiters, doubled quotes and line breaks; CRLF, LF and CR line endings).
/// Each field is returned with a flag telling whether it was quoted.
pub fn parse_records(text: &str, delim: char, quote: char, trim: bool) -> Result<Vec<Vec<(String, bool)>>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut records = Vec::new();
    let mut record: Vec<(String, bool)> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut in_quotes = false;
    let mut after_quote = false; // closing quote seen; only delimiter/EOL may follow (whitespace tolerated)
    let mut chars = text.char_indices().peekable();
    let mut line = 1usize;
    let finish_field = |field: &mut String, quoted: bool, record: &mut Vec<(String, bool)>| {
        let f = std::mem::take(field);
        let f = if trim && !quoted { f.trim().to_string() } else { f };
        record.push((f, quoted));
    };
    while let Some((pos, c)) = chars.next() {
        if in_quotes {
            if c == quote {
                if chars.peek().map(|&(_, n)| n) == Some(quote) {
                    field.push(quote);
                    chars.next();
                } else {
                    in_quotes = false;
                    after_quote = true;
                }
            } else {
                if c == '\n' {
                    line += 1;
                }
                field.push(c);
            }
            continue;
        }
        if c == delim {
            finish_field(&mut field, quoted, &mut record);
            quoted = false;
            after_quote = false;
        } else if c == '\n' || c == '\r' {
            if c == '\r' && chars.peek().map(|&(_, n)| n) == Some('\n') {
                chars.next();
            }
            finish_field(&mut field, quoted, &mut record);
            quoted = false;
            after_quote = false;
            // Skip completely blank lines.
            if !(record.len() == 1 && record[0].0.is_empty() && !record[0].1) {
                records.push(std::mem::take(&mut record));
            } else {
                record.clear();
            }
            line += 1;
        } else if after_quote {
            if !c.is_whitespace() {
                return Err(CoreError::parse(
                    "CSV",
                    format!("unexpected {c:?} after closing quote on line {line}"),
                    Some(pos),
                ));
            }
        } else if c == quote && field.trim().is_empty() {
            field.clear();
            in_quotes = true;
            quoted = true;
        } else {
            field.push(c);
        }
    }
    if in_quotes {
        return Err(CoreError::parse("CSV", format!("unterminated quoted field starting before line {line}"), None));
    }
    if !field.is_empty() || quoted || !record.is_empty() {
        finish_field(&mut field, quoted, &mut record);
        if !(record.len() == 1 && record[0].0.is_empty() && !record[0].1) {
            records.push(record);
        }
    }
    Ok(records)
}

/// Pick the delimiter whose per-line count (outside quotes) is most
/// consistent and non-zero over the first lines.
pub fn detect_delimiter(text: &str, quote: char) -> char {
    let candidates = [',', '\t', ';', '|'];
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(30).collect();
    let mut best = (',', 0usize, 0usize); // (delim, lines agreeing with mode, mode count)
    for &d in &candidates {
        let counts: Vec<usize> = lines
            .iter()
            .map(|l| {
                let mut inq = false;
                l.chars()
                    .filter(|&c| {
                        if c == quote {
                            inq = !inq;
                        }
                        !inq && c == d
                    })
                    .count()
            })
            .collect();
        let mut freq: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &c in counts.iter().filter(|&&c| c > 0) {
            *freq.entry(c).or_default() += 1;
        }
        if let Some((&mode, &agree)) = freq.iter().max_by_key(|(&k, &v)| (v, k)) {
            if (agree, mode) > (best.1, best.2) {
                best = (d, agree, mode);
            }
        }
    }
    best.0
}

/// Turn header text into a valid, unique column name.
pub(crate) fn sanitize_name(raw: &str, taken: &HashSet<String>) -> String {
    let mut s: String = raw.trim().replace('#', "_").trim_start_matches(['\'', '*']).to_string();
    s = s.chars().take(MAX_NAME_LEN).collect::<String>().trim().to_string();
    if matches!(
        crate::refs::parse_ref(&s),
        Ok(crate::refs::Ref::Column(_)
            | crate::refs::Ref::ColumnRange(..)
            | crate::refs::Ref::Constant(_)
            | crate::refs::Ref::Matrix(_))
    ) {
        s = format!("{s}_");
    }
    let mut name = crate::ops::unique_name(&s, taken);
    while name.chars().count() > MAX_NAME_LEN {
        let shorter: String = s.chars().take(s.chars().count().saturating_sub(1)).collect();
        s = shorter;
        name = crate::ops::unique_name(&s, taken);
    }
    name
}

/// Infer the column kind from raw field strings.
fn infer_kind(values: &[(String, bool)], opts: &CsvReadOptions) -> ColumnKind {
    if !opts.infer_types {
        return ColumnKind::Text;
    }
    // A quoted, non-empty token such as "*" is deliberate text, not missing.
    let present: Vec<&str> = values
        .iter()
        .filter(|(s, quoted)| !(is_missing_token(s) && (!*quoted || s.is_empty())))
        .map(|(s, _)| s.as_str())
        .collect();
    if present.is_empty() {
        return ColumnKind::Numeric;
    }
    // Any surviving missing token was quoted on purpose, so the column is text.
    if present.iter().any(|s| is_missing_token(s)) {
        return ColumnKind::Text;
    }
    if present.iter().all(|s| parse_number_with(s, opts.decimal).is_ok()) {
        return ColumnKind::Numeric;
    }
    // A date column must contain at least one separator or letter to avoid
    // misreading identifiers; every present value must parse.
    if present.iter().all(|s| s.chars().any(|c| matches!(c, '/' | '-' | ':' | '.') || c.is_alphabetic()))
        && present.iter().all(|s| datetime::parse_with(s, opts.date).is_ok())
    {
        return ColumnKind::DateTime;
    }
    ColumnKind::Text
}

/// Parse delimited text into a worksheet.
pub fn read_csv(text: &str, opts: &CsvReadOptions) -> Result<Worksheet> {
    let delim = opts.delimiter.unwrap_or_else(|| detect_delimiter(text, opts.quote));
    if delim == opts.decimal && opts.infer_types {
        return Err(CoreError::invalid("delimiter and decimal separator must differ"));
    }
    let mut records = parse_records(text, delim, opts.quote, opts.trim)?;
    records.drain(..opts.skip_rows.min(records.len()));
    let header: Option<Vec<(String, bool)>> =
        if opts.has_header && !records.is_empty() { Some(records.remove(0)) } else { None };
    if let Some(m) = opts.max_rows {
        records.truncate(m);
    }
    let ncols = records.iter().map(|r| r.len()).chain(header.iter().map(|h| h.len())).max().unwrap_or(0);
    let mut ws = Worksheet::new(opts.sheet_name.clone());
    let mut taken = HashSet::new();
    for j in 0..ncols {
        let raw: Vec<(String, bool)> = records.iter().map(|r| r.get(j).cloned().unwrap_or_default()).collect();
        let kind = infer_kind(&raw, opts);
        let data = match kind {
            ColumnKind::Numeric => ColumnData::Numeric(
                raw.iter().map(|(s, _)| parse_number_with(s, opts.decimal).unwrap_or(f64::NAN)).collect(),
            ),
            ColumnKind::DateTime => ColumnData::DateTime(
                raw.iter().map(|(s, _)| datetime::parse_with(s, opts.date).unwrap_or(DATETIME_MISSING)).collect(),
            ),
            ColumnKind::Text => ColumnData::Text(
                raw.into_iter().map(|(s, quoted)| if !quoted && s == "*" { String::new() } else { s }).collect(),
            ),
        };
        let name = match &header {
            Some(h) => sanitize_name(h.get(j).map(|(s, _)| s.as_str()).unwrap_or(""), &taken),
            None => String::new(),
        };
        if !name.is_empty() {
            taken.insert(name.to_lowercase());
        }
        let mut col = Column::new(name, data);
        col.data.trim_trailing_missing();
        ws.columns.push(col);
    }
    Ok(ws)
}

fn quote_field(s: &str, opts: &CsvWriteOptions) -> String {
    let needs = s.contains(opts.delimiter)
        || s.contains(opts.quote)
        || s.contains('\n')
        || s.contains('\r')
        || s.starts_with(' ')
        || s.ends_with(' ')
        // A text value equal to the missing token must stay distinguishable.
        || s == opts.missing;
    if needs {
        let q = opts.quote.to_string();
        format!("{q}{}{q}", s.replace(&q, &format!("{q}{q}")))
    } else {
        s.to_string()
    }
}

/// Serialise a worksheet as delimited text. Rows span the longest column.
pub fn write_csv(ws: &Worksheet, opts: &CsvWriteOptions) -> String {
    let mut out = String::new();
    let n = ws.n_rows();
    let sep = opts.delimiter.to_string();
    if opts.header {
        let names: Vec<String> = (0..ws.n_cols()).map(|j| quote_field(&ws.display_name(j), opts)).collect();
        out.push_str(&names.join(&sep));
        out.push_str(opts.line_ending);
    }
    for r in 0..n {
        let fields: Vec<String> = ws
            .columns
            .iter()
            .map(|c| match &c.data {
                ColumnData::Numeric(v) => {
                    let x = v.get(r).copied().unwrap_or(f64::NAN);
                    if x.is_nan() {
                        return opts.missing.clone();
                    }
                    let s = if opts.use_display_format { c.props.numeric_format.format(x) } else { format_number(x) };
                    let s = if opts.decimal != '.' { s.replace('.', &opts.decimal.to_string()) } else { s };
                    quote_field(&s, opts)
                }
                ColumnData::DateTime(v) => {
                    let t = v.get(r).copied().unwrap_or(DATETIME_MISSING);
                    if t == DATETIME_MISSING {
                        return opts.missing.clone();
                    }
                    let s = if opts.iso_dates {
                        let dt = datetime::DateTime::from_timestamp(t);
                        if dt.micros != 0 {
                            datetime::format(t, datetime::ISO_FORMAT)
                        } else if dt.is_midnight() {
                            datetime::format(t, "yyyy-MM-dd")
                        } else {
                            datetime::format(t, "yyyy-MM-dd'T'HH:mm:ss")
                        }
                    } else {
                        c.display(r)
                    };
                    quote_field(&s, opts)
                }
                ColumnData::Text(v) => quote_field(v.get(r).map(String::as_str).unwrap_or(""), opts),
            })
            .collect();
        out.push_str(&fields.join(&sep));
        out.push_str(opts.line_ending);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    #[test]
    fn rfc4180_records() {
        let t = "a,\"b,1\",\"say \"\"hi\"\"\"\r\n\"multi\nline\", x ,\n\n3,4,5";
        let r = parse_records(t, ',', '"', true).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0][1].0, "b,1");
        assert_eq!(r[0][2].0, "say \"hi\"");
        assert_eq!(r[1][0].0, "multi\nline");
        assert_eq!(r[1][1].0, "x");
        assert_eq!(r[1][2].0, "");
        assert!(parse_records("\"open", ',', '"', true).is_err());
        assert!(parse_records("\"a\"b,c", ',', '"', true).is_err());
    }

    #[test]
    fn delimiter_detection() {
        assert_eq!(detect_delimiter("a;b;c\n1;2,5;3\n", '"'), ';');
        assert_eq!(detect_delimiter("a\tb\n1\t2\n", '"'), '\t');
        assert_eq!(detect_delimiter("a,b\n\"1,2\",3\n", '"'), ',');
    }

    #[test]
    fn type_inference() {
        let t = "Weight,Lot,When,Empty,C1,Notes\n1.5,A,2026-11-16,,x,*\n*,B,11/17/2026 13:00,,y,\"*\"\n2,7,,,,\n";
        let ws = read_csv(t, &CsvReadOptions::default()).unwrap();
        assert_eq!(ws.columns[0].kind(), ColumnKind::Numeric);
        assert!(ws.columns[0].data.is_missing(1));
        assert_eq!(ws.columns[1].kind(), ColumnKind::Text);
        assert_eq!(ws.columns[2].kind(), ColumnKind::DateTime);
        assert_eq!(ws.columns[2].len(), 2);
        assert_eq!(ws.columns[3].kind(), ColumnKind::Numeric);
        assert_eq!(ws.columns[4].name(), "C1_");
        assert_eq!(ws.columns[5].get(0), Value::Text(String::new()));
        assert_eq!(ws.columns[5].get(1), Value::Text("*".into()));
    }

    #[test]
    fn european_decimal() {
        let o = CsvReadOptions { decimal: ',', ..Default::default() };
        let ws = read_csv("x;y\n1,5;2\n3;4,25\n", &o).unwrap();
        assert_eq!(ws.columns[0].data, ColumnData::Numeric(vec![1.5, 3.0]));
        let w =
            write_csv(&ws, &CsvWriteOptions { delimiter: ';', decimal: ',', line_ending: "\n", ..Default::default() });
        assert_eq!(w, "x;y\n1,5;2\n3;4,25\n");
    }

    #[test]
    fn roundtrip() {
        let ws = Worksheet::with_columns(
            "s",
            vec![
                Column::numeric("x", vec![0.1 + 0.2, f64::NAN, 1e300, -2.5e-12]),
                Column::text("t", vec!["a,b", "say \"x\"", " pad", "*"]),
                Column::datetime(
                    "d",
                    vec![datetime::parse("2026-11-16").unwrap(), datetime::parse("2026-11-16 08:30:00.25").unwrap()],
                ),
            ],
        )
        .unwrap();
        let text = write_csv(&ws, &CsvWriteOptions::default());
        let back =
            read_csv(&text, &CsvReadOptions { sheet_name: "s".into(), trim: false, ..Default::default() }).unwrap();
        assert_eq!(back.columns[0].data, ws.columns[0].data);
        assert_eq!(back.columns[1].data, ws.columns[1].data);
        assert_eq!(back.columns[2].data, ws.columns[2].data);
    }

    #[test]
    fn no_header_and_ragged() {
        let o = CsvReadOptions { has_header: false, ..Default::default() };
        let ws = read_csv("1,2,3\n4\n", &o).unwrap();
        assert_eq!(ws.n_cols(), 3);
        assert_eq!(ws.columns[2].len(), 1);
        assert_eq!(ws.columns[0].name(), "");
    }
}
