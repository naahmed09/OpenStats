//! Hand-written JSON (RFC 8259) parser/serializer and the workbook JSON schema.
//!
//! Workbook schema (version 1):
//!
//! ```json
//! { "format": "openstats", "version": 1, "title": "…", "active": 0,
//!   "metadata": { "key": "value" },
//!   "worksheets": [ { "name": "…", "description": "…",
//!       "columns": [ { "name": "…", "type": "numeric" | "text" | "datetime",
//!                      "description": "…", "format": "auto" | "fixed:2" | "sci:3" | "pct:1",
//!                      "datetime_format": "…" | null, "value_order": ["…"] | null, "value_labels": {"1": "Low"} (optional),
//!                      "formula": "…" | null,
//!                      "data": [ … ] } ],
//!       "constants": { "1": 3.5, "2": "text", "3": {"datetime": "…"} },
//!       "matrices": { "1": { "rows": 2, "cols": 2, "data": [ … ] } } } ] }
//! ```
//!
//! Numeric missing values are `null`; infinities are the strings `"inf"` /
//! `"-inf"`. Date/times are ISO-8601 strings in UTC (`…Z`), or `"us:<int>"`
//! (microseconds since epoch) for years outside 0–9999.

use crate::column::{Column, ColumnData, ColumnKind, ColumnProps, NumericFormat};
use crate::datetime;
use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;
use crate::value::Value;
use crate::workbook::Workbook;
use crate::worksheet::{StoredMatrix, Worksheet};
use std::fmt::Write as _;

/// Maximum nesting depth accepted by the parser.
pub const MAX_DEPTH: usize = 256;

/// A JSON value. Objects keep key order.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Json::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Number(x) = self {
            Some(*x)
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().filter(|x| *x >= 0.0 && x.fract() == 0.0 && *x <= 9.007e15).map(|x| x as usize)
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    b: &'a [u8],
    s: &'a str,
    p: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, m: impl Into<String>) -> CoreError {
        CoreError::parse("JSON", m, Some(self.p))
    }
    fn ws(&mut self) {
        while self.p < self.b.len() && matches!(self.b[self.p], b' ' | b'\t' | b'\n' | b'\r') {
            self.p += 1;
        }
    }
    fn expect(&mut self, c: u8) -> Result<()> {
        if self.b.get(self.p) == Some(&c) {
            self.p += 1;
            Ok(())
        } else {
            Err(self.err(format!("expected '{}'", c as char)))
        }
    }
    fn literal(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.s[self.p..].starts_with(word) {
            self.p += word.len();
            Ok(v)
        } else {
            Err(self.err("invalid literal"))
        }
    }
    fn value(&mut self) -> Result<Json> {
        self.ws();
        match self.b.get(self.p) {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(&c) => Err(self.err(format!("unexpected character {:?}", c as char))),
        }
    }
    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        Ok(())
    }
    fn object(&mut self) -> Result<Json> {
        self.enter()?;
        self.p += 1;
        let mut kv = Vec::new();
        self.ws();
        if self.b.get(self.p) == Some(&b'}') {
            self.p += 1;
            self.depth -= 1;
            return Ok(Json::Object(kv));
        }
        loop {
            self.ws();
            if self.b.get(self.p) != Some(&b'"') {
                return Err(self.err("expected string key"));
            }
            let k = self.string()?;
            self.ws();
            self.expect(b':')?;
            let v = self.value()?;
            kv.push((k, v));
            self.ws();
            match self.b.get(self.p) {
                Some(b',') => self.p += 1,
                Some(b'}') => {
                    self.p += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        self.depth -= 1;
        Ok(Json::Object(kv))
    }
    fn array(&mut self) -> Result<Json> {
        self.enter()?;
        self.p += 1;
        let mut v = Vec::new();
        self.ws();
        if self.b.get(self.p) == Some(&b']') {
            self.p += 1;
            self.depth -= 1;
            return Ok(Json::Array(v));
        }
        loop {
            v.push(self.value()?);
            self.ws();
            match self.b.get(self.p) {
                Some(b',') => self.p += 1,
                Some(b']') => {
                    self.p += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        self.depth -= 1;
        Ok(Json::Array(v))
    }
    fn hex4(&mut self) -> Result<u32> {
        let h = self.s.get(self.p..self.p + 4).ok_or_else(|| self.err("truncated \\u escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| self.err("invalid \\u escape"))?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(self.err("invalid \\u escape"));
        }
        self.p += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String> {
        self.p += 1; // opening quote
        let mut out = String::new();
        loop {
            let start = self.p;
            while self.p < self.b.len() && self.b[self.p] != b'"' && self.b[self.p] != b'\\' {
                if self.b[self.p] < 0x20 {
                    return Err(self.err("control character in string"));
                }
                self.p += 1;
            }
            out.push_str(&self.s[start..self.p]);
            match self.b.get(self.p) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.p += 1;
                    return Ok(out);
                }
                Some(_) => {
                    self.p += 1;
                    let e = *self.b.get(self.p).ok_or_else(|| self.err("unterminated escape"))?;
                    self.p += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.s.get(self.p..self.p + 2) != Some("\\u") {
                                    return Err(self.err("unpaired high surrogate"));
                                }
                                self.p += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.err("unpaired low surrogate"));
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or_else(|| self.err("invalid code point"))?);
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
            }
        }
    }
    fn number(&mut self) -> Result<Json> {
        let start = self.p;
        let b = self.b;
        if b.get(self.p) == Some(&b'-') {
            self.p += 1;
        }
        match b.get(self.p) {
            Some(b'0') => self.p += 1,
            Some(b'1'..=b'9') => {
                while self.p < b.len() && b[self.p].is_ascii_digit() {
                    self.p += 1;
                }
            }
            _ => return Err(self.err("invalid number")),
        }
        if b.get(self.p) == Some(&b'.') {
            self.p += 1;
            let d = self.p;
            while self.p < b.len() && b[self.p].is_ascii_digit() {
                self.p += 1;
            }
            if d == self.p {
                return Err(self.err("digit expected after '.'"));
            }
        }
        if matches!(b.get(self.p), Some(b'e' | b'E')) {
            self.p += 1;
            if matches!(b.get(self.p), Some(b'+' | b'-')) {
                self.p += 1;
            }
            let d = self.p;
            while self.p < b.len() && b[self.p].is_ascii_digit() {
                self.p += 1;
            }
            if d == self.p {
                return Err(self.err("digit expected in exponent"));
            }
        }
        self.s[start..self.p].parse::<f64>().map(Json::Number).map_err(|_| self.err("invalid number"))
    }
}

/// Parse a JSON document.
pub fn parse(text: &str) -> Result<Json> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut p = Parser { b: text.as_bytes(), s: text, p: 0, depth: 0 };
    let v = p.value()?;
    p.ws();
    if p.p != p.b.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Serializer
// ---------------------------------------------------------------------------

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
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

fn write_value(out: &mut String, v: &Json, pretty: bool, indent: usize) {
    let nl = |out: &mut String, n: usize| {
        if pretty {
            out.push('\n');
            for _ in 0..n {
                out.push_str("  ");
            }
        }
    };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(x) => {
            if x.is_finite() {
                let _ = write!(out, "{x}");
            } else {
                out.push_str("null");
            }
        }
        Json::String(s) => write_string(out, s),
        Json::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            // Arrays of scalars stay on one line for compact data columns.
            let scalar = a.iter().all(|x| !matches!(x, Json::Array(_) | Json::Object(_)));
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if pretty && scalar {
                        out.push(' ');
                    }
                }
                if !scalar {
                    nl(out, indent + 1);
                }
                write_value(out, x, pretty, indent + 1);
            }
            if !scalar {
                nl(out, indent);
            }
            out.push(']');
        }
        Json::Object(kv) => {
            if kv.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                nl(out, indent + 1);
                write_string(out, k);
                out.push(':');
                if pretty {
                    out.push(' ');
                }
                write_value(out, x, pretty, indent + 1);
            }
            nl(out, indent);
            out.push('}');
        }
    }
}

/// Serialise a JSON value (non-finite numbers become `null`).
pub fn to_string(v: &Json, pretty: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, v, pretty, 0);
    out
}

// ---------------------------------------------------------------------------
// Workbook schema
// ---------------------------------------------------------------------------

fn s(x: &str) -> Json {
    Json::String(x.to_string())
}

fn num_json(x: f64) -> Json {
    if x.is_nan() {
        Json::Null
    } else if x == f64::INFINITY {
        s("inf")
    } else if x == f64::NEG_INFINITY {
        s("-inf")
    } else {
        Json::Number(x)
    }
}

fn num_from_json(j: &Json) -> Result<f64> {
    match j {
        Json::Null => Ok(f64::NAN),
        Json::Number(x) => Ok(*x),
        Json::String(t) if t == "inf" => Ok(f64::INFINITY),
        Json::String(t) if t == "-inf" => Ok(f64::NEG_INFINITY),
        _ => Err(CoreError::Format(format!("expected number, found {}", to_string(j, false)))),
    }
}

fn dt_json(t: i64) -> Json {
    if t == DATETIME_MISSING {
        return Json::Null;
    }
    let dt = datetime::DateTime::from_timestamp(t);
    if (0..=9999).contains(&dt.year) {
        s(&format!("{}Z", datetime::format(t, datetime::ISO_FORMAT)))
    } else {
        s(&format!("us:{t}"))
    }
}

fn dt_from_json(j: &Json) -> Result<i64> {
    match j {
        Json::Null => Ok(DATETIME_MISSING),
        Json::String(t) => {
            if let Some(us) = t.strip_prefix("us:") {
                us.parse().map_err(|_| CoreError::Format(format!("bad timestamp {t:?}")))
            } else {
                datetime::parse(t).map_err(|e| CoreError::Format(e.to_string()))
            }
        }
        _ => Err(CoreError::Format("expected date/time string".into())),
    }
}

fn format_to_str(f: NumericFormat) -> String {
    match f {
        NumericFormat::Automatic => "auto".into(),
        NumericFormat::Fixed(d) => format!("fixed:{d}"),
        NumericFormat::Scientific(d) => format!("sci:{d}"),
        NumericFormat::Percent(d) => format!("pct:{d}"),
    }
}

fn format_from_str(t: &str) -> Result<NumericFormat> {
    let bad = || CoreError::Format(format!("bad numeric format {t:?}"));
    if t == "auto" {
        return Ok(NumericFormat::Automatic);
    }
    let (k, d) = t.split_once(':').ok_or_else(bad)?;
    let d: u8 = d.parse().map_err(|_| bad())?;
    match k {
        "fixed" => Ok(NumericFormat::Fixed(d)),
        "sci" => Ok(NumericFormat::Scientific(d)),
        "pct" => Ok(NumericFormat::Percent(d)),
        _ => Err(bad()),
    }
}

fn value_json(v: &Value) -> Json {
    match v {
        Value::Numeric(x) => num_json(*x),
        Value::Text(t) => s(t),
        Value::DateTime(t) => Json::Object(vec![("datetime".into(), dt_json(*t))]),
        Value::Missing => Json::Null,
    }
}

fn value_from_json(j: &Json) -> Result<Value> {
    Ok(match j {
        Json::Null => Value::Missing,
        Json::Number(x) => Value::Numeric(*x),
        Json::String(t) if t == "inf" || t == "-inf" => Value::Numeric(num_from_json(j)?),
        Json::String(t) => Value::Text(t.clone()),
        Json::Object(_) => match j.get("datetime") {
            Some(d) => Value::DateTime(dt_from_json(d)?),
            None => return Err(CoreError::Format("unknown constant object".into())),
        },
        _ => return Err(CoreError::Format("unsupported constant value".into())),
    })
}

/// Encode a workbook as JSON text.
pub fn workbook_to_json(wb: &Workbook, pretty: bool) -> String {
    let sheets = wb
        .worksheets
        .iter()
        .map(|ws| {
            let cols = ws
                .columns
                .iter()
                .map(|c| {
                    let (ty, data) = match &c.data {
                        ColumnData::Numeric(v) => ("numeric", v.iter().map(|&x| num_json(x)).collect()),
                        ColumnData::Text(v) => ("text", v.iter().map(|t| s(t)).collect()),
                        ColumnData::DateTime(v) => ("datetime", v.iter().map(|&t| dt_json(t)).collect()),
                    };
                    let mut cols = vec![
                        ("name".into(), s(c.name())),
                        ("type".into(), s(ty)),
                        ("description".into(), s(&c.props.description)),
                        ("format".into(), s(&format_to_str(c.props.numeric_format))),
                        ("datetime_format".into(), c.props.datetime_format.as_deref().map_or(Json::Null, s)),
                        (
                            "value_order".into(),
                            c.props
                                .value_order
                                .as_ref()
                                .map_or(Json::Null, |o| Json::Array(o.iter().map(|x| s(x)).collect())),
                        ),
                        ("formula".into(), c.props.formula.as_deref().map_or(Json::Null, s)),
                        ("data".into(), Json::Array(data)),
                    ];
                    if !c.props.value_labels.is_empty() {
                        cols.push((
                            "value_labels".into(),
                            Json::Object(c.props.value_labels.iter().map(|(k, l)| (k.clone(), s(l))).collect()),
                        ));
                    }
                    Json::Object(cols)
                })
                .collect();
            let consts = ws.constants.iter().map(|(k, v)| (k.to_string(), value_json(v))).collect();
            let mats = ws
                .matrices
                .iter()
                .map(|(k, m)| {
                    (
                        k.to_string(),
                        Json::Object(vec![
                            ("rows".into(), Json::Number(m.rows as f64)),
                            ("cols".into(), Json::Number(m.cols as f64)),
                            ("data".into(), Json::Array(m.data.iter().map(|&x| num_json(x)).collect())),
                        ]),
                    )
                })
                .collect();
            Json::Object(vec![
                ("name".into(), s(&ws.name)),
                ("description".into(), s(&ws.description)),
                ("columns".into(), Json::Array(cols)),
                ("constants".into(), Json::Object(consts)),
                ("matrices".into(), Json::Object(mats)),
            ])
        })
        .collect();
    let doc = Json::Object(vec![
        ("format".into(), s("openstats")),
        ("version".into(), Json::Number(1.0)),
        ("title".into(), s(&wb.title)),
        ("active".into(), Json::Number(wb.active as f64)),
        ("metadata".into(), Json::Object(wb.metadata.iter().map(|(k, v)| (k.clone(), s(v))).collect())),
        ("worksheets".into(), Json::Array(sheets)),
    ]);
    to_string(&doc, pretty)
}

fn req<'a>(j: &'a Json, key: &str) -> Result<&'a Json> {
    j.get(key).ok_or_else(|| CoreError::Format(format!("missing field {key:?}")))
}

fn opt_str(j: &Json, key: &str) -> Result<String> {
    match j.get(key) {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::String(t)) => Ok(t.clone()),
        Some(_) => Err(CoreError::Format(format!("field {key:?} must be a string"))),
    }
}

/// Decode a workbook from JSON text.
pub fn workbook_from_json(text: &str) -> Result<Workbook> {
    let doc = parse(text)?;
    if req(&doc, "format")?.as_str() != Some("openstats") {
        return Err(CoreError::Format("not an OpenStats JSON document".into()));
    }
    let version = req(&doc, "version")?.as_usize().ok_or_else(|| CoreError::Format("bad version".into()))?;
    if version != 1 {
        return Err(CoreError::Format(format!("unsupported JSON version {version}")));
    }
    let mut wb = Workbook::empty();
    wb.title = opt_str(&doc, "title")?;
    if let Some(Json::Object(kv)) = doc.get("metadata") {
        for (k, v) in kv {
            wb.metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
        }
    }
    let sheets =
        req(&doc, "worksheets")?.as_array().ok_or_else(|| CoreError::Format("worksheets must be an array".into()))?;
    for sj in sheets {
        let mut ws = Worksheet::new(opt_str(sj, "name")?);
        ws.description = opt_str(sj, "description")?;
        for cj in req(sj, "columns")?.as_array().ok_or_else(|| CoreError::Format("columns must be an array".into()))? {
            let data_j =
                req(cj, "data")?.as_array().ok_or_else(|| CoreError::Format("data must be an array".into()))?;
            let kind = match req(cj, "type")?.as_str() {
                Some("numeric") => ColumnKind::Numeric,
                Some("text") => ColumnKind::Text,
                Some("datetime") => ColumnKind::DateTime,
                other => return Err(CoreError::Format(format!("unknown column type {other:?}"))),
            };
            let data = match kind {
                ColumnKind::Numeric => ColumnData::Numeric(data_j.iter().map(num_from_json).collect::<Result<_>>()?),
                ColumnKind::Text => ColumnData::Text(
                    data_j
                        .iter()
                        .map(|x| match x {
                            Json::String(t) => Ok(t.clone()),
                            Json::Null => Ok(String::new()),
                            _ => Err(CoreError::Format("text column holds a non-string".into())),
                        })
                        .collect::<Result<_>>()?,
                ),
                ColumnKind::DateTime => ColumnData::DateTime(data_j.iter().map(dt_from_json).collect::<Result<_>>()?),
            };
            let props = ColumnProps {
                name: opt_str(cj, "name")?,
                description: opt_str(cj, "description")?,
                numeric_format: match cj.get("format").and_then(Json::as_str) {
                    Some(f) => format_from_str(f)?,
                    None => NumericFormat::Automatic,
                },
                datetime_format: cj.get("datetime_format").and_then(Json::as_str).map(String::from),
                value_order: match cj.get("value_order") {
                    Some(Json::Array(a)) => {
                        Some(a.iter().map(|x| x.as_str().unwrap_or_default().to_string()).collect())
                    }
                    _ => None,
                },
                formula: cj.get("formula").and_then(Json::as_str).map(String::from),
                value_labels: match cj.get("value_labels") {
                    Some(Json::Object(kv)) => {
                        kv.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string())).collect()
                    }
                    _ => Vec::new(),
                },
            };
            ws.validate_name(&props.name, None).map_err(|e| CoreError::Format(e.to_string()))?;
            ws.columns.push(Column { props, data });
        }
        if let Some(Json::Object(kv)) = sj.get("constants") {
            for (k, v) in kv {
                let idx: usize = k.parse().map_err(|_| CoreError::Format(format!("bad constant index {k:?}")))?;
                ws.constants.insert(idx, value_from_json(v)?);
            }
        }
        if let Some(Json::Object(kv)) = sj.get("matrices") {
            for (k, v) in kv {
                let idx: usize = k.parse().map_err(|_| CoreError::Format(format!("bad matrix index {k:?}")))?;
                let rows = req(v, "rows")?.as_usize().ok_or_else(|| CoreError::Format("bad rows".into()))?;
                let cols = req(v, "cols")?.as_usize().ok_or_else(|| CoreError::Format("bad cols".into()))?;
                let data = req(v, "data")?
                    .as_array()
                    .ok_or_else(|| CoreError::Format("bad matrix data".into()))?
                    .iter()
                    .map(num_from_json)
                    .collect::<Result<Vec<_>>>()?;
                ws.matrices.insert(idx, StoredMatrix::new(rows, cols, data)?);
            }
        }
        wb.worksheets.push(ws);
    }
    let active = doc.get("active").and_then(Json::as_usize).unwrap_or(0);
    wb.active = if active < wb.worksheets.len() { active } else { 0 };
    Ok(wb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spec_cases() {
        let j =
            parse(r#" {"a": [1, -2.5e3, true, false, null], "b": "x\"y\\\u00e9\ud83d\ude00\n", "c": {}} "#).unwrap();
        assert_eq!(j.get("a").unwrap().as_array().unwrap()[1], Json::Number(-2500.0));
        assert_eq!(j.get("b").unwrap().as_str().unwrap(), "x\"y\\é😀\n");
        for bad in
            ["", "{", "[1,]", "{\"a\" 1}", "01", "1.", "-", "\"\\x\"", "\"\\ud800\"", "nul", "[1] x", "\"a\nb\"", "1e"]
        {
            assert!(parse(bad).is_err(), "should reject {bad:?}");
        }
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(parse(&deep).is_err());
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn serialize_roundtrip() {
        let j = Json::Object(vec![
            (
                "k\t".into(),
                Json::Array(vec![Json::Number(0.1), Json::Number(1e300), Json::Null, Json::String("\u{1}".into())]),
            ),
            ("o".into(), Json::Object(vec![("n".into(), Json::Array(vec![Json::Array(vec![])]))])),
        ]);
        for pretty in [false, true] {
            assert_eq!(parse(&to_string(&j, pretty)).unwrap(), j);
        }
        assert_eq!(to_string(&Json::Number(f64::NAN), false), "null");
        assert_eq!(to_string(&Json::Number(3.0), false), "3");
    }

    #[test]
    fn workbook_roundtrip() {
        let mut wb = Workbook::new();
        wb.title = "Demo".into();
        wb.metadata.insert("author".into(), "QA".into());
        {
            let ws = wb.sheet_mut(0).unwrap();
            let mut c = Column::numeric("x", vec![1.5, f64::NAN, f64::INFINITY, -0.0]);
            c.props.numeric_format = NumericFormat::Fixed(2);
            c.props.description = "len\n\"mm\"".into();
            ws.add_column(c).unwrap();
            let mut t = Column::text("g", vec!["a", "", "ü"]);
            t.props.value_order = Some(vec!["ü".into(), "a".into()]);
            ws.add_column(t).unwrap();
            let mut d = Column::datetime(
                "d",
                vec![
                    datetime::parse("2026-11-16 01:02:03.000004").unwrap(),
                    DATETIME_MISSING,
                    datetime::DateTime::date(12000, 1, 1).unwrap().to_timestamp(),
                ],
            );
            d.props.datetime_format = Some("yyyy".into());
            ws.add_column(d).unwrap();
            ws.constants.insert(1, Value::Numeric(2.0));
            ws.constants.insert(2, Value::Text("k".into()));
            ws.constants.insert(3, Value::DateTime(0));
            ws.matrices.insert(1, StoredMatrix::new(1, 2, vec![1.0, f64::NAN]).unwrap());
        }
        wb.worksheets.push(Worksheet::new("Second"));
        wb.active = 1;
        for pretty in [false, true] {
            let text = workbook_to_json(&wb, pretty);
            assert_eq!(workbook_from_json(&text).unwrap(), wb);
        }
        assert!(workbook_from_json("{\"format\":\"other\"}").is_err());
        assert!(workbook_from_json("{\"format\":\"openstats\",\"version\":9,\"worksheets\":[]}").is_err());
    }
}
