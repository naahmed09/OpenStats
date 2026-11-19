//! Native binary project format (`.otb`).
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! offset  size  field
//! 0       4     magic  "OTAB"
//! 4       2     format major version (1)
//! 6       2     format minor version (1)  — readers accept any minor of a known major
//! 8       8     payload length N
//! 16      N     payload
//! 16+N    4     CRC-32 (IEEE) of the payload
//! ```
//!
//! Payload:
//!
//! ```text
//! str title · u32 active · u32 n_meta · (str key, str value)×n_meta · u32 n_sheets · sheet×n_sheets
//! sheet  = str name · str description · u32 n_cols · column×n_cols
//!          · u32 n_const · (u32 k, value)×n_const · u32 n_mat · (u32 m, u32 rows, u32 cols, f64×rows·cols)×n_mat
//! column = str name · str description · u8 kind (0 num, 1 text, 2 datetime)
//!          · u8 fmt_tag (0 auto, 1 fixed, 2 sci, 3 pct) · u8 fmt_decimals
//!          · opt_str datetime_format · u8 has_order [· u32 n · str×n]
//!          · opt_str formula (minor ≥ 1 only) · u32 n_labels · (str value, str label)×n (minor ≥ 2 only)
//!          · u64 len · data
//! data   = f64×len (numeric, IEEE bits) | i64×len (datetime) | str×len (text)
//! value  = u8 tag (0 missing, 1 numeric f64, 2 text str, 3 datetime i64) · payload
//! str    = u32 byte length · UTF-8 bytes;  opt_str = u8 present · [str]
//! ```

use super::crc32::crc32;
use crate::column::{Column, ColumnData, ColumnProps, NumericFormat};
use crate::error::{CoreError, Result};
use crate::value::Value;
use crate::workbook::Workbook;
use crate::worksheet::{StoredMatrix, Worksheet};

pub const MAGIC: &[u8; 4] = b"OTAB";
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 2;
const HEADER_LEN: usize = 16;

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: usize) {
        let v = u32::try_from(v).expect("count exceeds u32 range");
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn opt_str(&mut self, s: Option<&str>) {
        match s {
            Some(x) => {
                self.u8(1);
                self.str(x);
            }
            None => self.u8(0),
        }
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::Missing => self.u8(0),
            Value::Numeric(x) => {
                self.u8(1);
                self.f64(*x)
            }
            Value::Text(s) => {
                self.u8(2);
                self.str(s)
            }
            Value::DateTime(t) => {
                self.u8(3);
                self.i64(*t)
            }
        }
    }
}

fn encode_column(w: &mut W, c: &Column) {
    w.str(&c.props.name);
    w.str(&c.props.description);
    w.u8(match c.data {
        ColumnData::Numeric(_) => 0,
        ColumnData::Text(_) => 1,
        ColumnData::DateTime(_) => 2,
    });
    let (tag, dec) = match c.props.numeric_format {
        NumericFormat::Automatic => (0, 0),
        NumericFormat::Fixed(d) => (1, d),
        NumericFormat::Scientific(d) => (2, d),
        NumericFormat::Percent(d) => (3, d),
    };
    w.u8(tag);
    w.u8(dec);
    w.opt_str(c.props.datetime_format.as_deref());
    match &c.props.value_order {
        Some(o) => {
            w.u8(1);
            w.u32(o.len());
            for s in o {
                w.str(s);
            }
        }
        None => w.u8(0),
    }
    w.opt_str(c.props.formula.as_deref());
    w.u32(c.props.value_labels.len());
    for (k, l) in &c.props.value_labels {
        w.str(k);
        w.str(l);
    }
    w.u64(c.data.len() as u64);
    match &c.data {
        ColumnData::Numeric(v) => v.iter().for_each(|&x| w.f64(x)),
        ColumnData::DateTime(v) => v.iter().for_each(|&t| w.i64(t)),
        ColumnData::Text(v) => v.iter().for_each(|s| w.str(s)),
    }
}

/// Serialise a workbook to `.otb` bytes.
pub fn encode(wb: &Workbook) -> Vec<u8> {
    let mut p = W(Vec::new());
    p.str(&wb.title);
    p.u32(wb.active);
    p.u32(wb.metadata.len());
    for (k, v) in &wb.metadata {
        p.str(k);
        p.str(v);
    }
    p.u32(wb.worksheets.len());
    for ws in &wb.worksheets {
        p.str(&ws.name);
        p.str(&ws.description);
        p.u32(ws.columns.len());
        for c in &ws.columns {
            encode_column(&mut p, c);
        }
        p.u32(ws.constants.len());
        for (k, v) in &ws.constants {
            p.u32(*k);
            p.value(v);
        }
        p.u32(ws.matrices.len());
        for (k, m) in &ws.matrices {
            p.u32(*k);
            p.u32(m.rows);
            p.u32(m.cols);
            m.data.iter().for_each(|&x| p.f64(x));
        }
    }
    let payload = p.0;
    let mut out = W(Vec::with_capacity(payload.len() + HEADER_LEN + 4));
    out.0.extend_from_slice(MAGIC);
    out.u16(VERSION_MAJOR);
    out.u16(VERSION_MINOR);
    out.u64(payload.len() as u64);
    out.0.extend_from_slice(&payload);
    out.0.extend_from_slice(&crc32(&payload).to_le_bytes());
    out.0
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    fn err(&self, m: &str) -> CoreError {
        CoreError::Format(format!("corrupt .otb payload at byte {}: {m}", self.p))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.b.len() - self.p < n {
            return Err(self.err("unexpected end of data"));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    /// A count whose elements each need at least `min_size` bytes: rejects
    /// counts that could not possibly fit (prevents huge allocations).
    fn count(&mut self, min_size: usize) -> Result<usize> {
        let n = self.u32()?;
        self.check_fits(n as u64, min_size)?;
        Ok(n)
    }
    fn check_fits(&self, n: u64, min_size: usize) -> Result<()> {
        let need = n.checked_mul(min_size as u64).ok_or_else(|| self.err("count overflow"))?;
        if need > (self.b.len() - self.p) as u64 {
            return Err(self.err("count exceeds remaining data"));
        }
        Ok(())
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()?;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| self.err("invalid UTF-8"))
    }
    fn opt_str(&mut self) -> Result<Option<String>> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.str()?)),
            _ => Err(self.err("bad option tag")),
        }
    }
    fn value(&mut self) -> Result<Value> {
        Ok(match self.u8()? {
            0 => Value::Missing,
            1 => Value::Numeric(self.f64()?),
            2 => Value::Text(self.str()?),
            3 => Value::DateTime(self.i64()?),
            _ => return Err(self.err("bad value tag")),
        })
    }
}

fn decode_column(r: &mut R, minor: u16) -> Result<Column> {
    let name = r.str()?;
    let description = r.str()?;
    let kind = r.u8()?;
    let tag = r.u8()?;
    let dec = r.u8()?;
    let numeric_format = match tag {
        0 => NumericFormat::Automatic,
        1 => NumericFormat::Fixed(dec),
        2 => NumericFormat::Scientific(dec),
        3 => NumericFormat::Percent(dec),
        _ => return Err(r.err("bad numeric format tag")),
    };
    let datetime_format = r.opt_str()?;
    let value_order = match r.u8()? {
        0 => None,
        1 => {
            let n = r.count(4)?;
            Some((0..n).map(|_| r.str()).collect::<Result<Vec<_>>>()?)
        }
        _ => return Err(r.err("bad value-order tag")),
    };
    let formula = if minor >= 1 { r.opt_str()? } else { None };
    let value_labels = if minor >= 2 {
        let n = r.count(8)?;
        (0..n).map(|_| Ok((r.str()?, r.str()?))).collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let len = r.u64()?;
    let data = match kind {
        0 => {
            r.check_fits(len, 8)?;
            ColumnData::Numeric((0..len).map(|_| r.f64()).collect::<Result<_>>()?)
        }
        1 => {
            r.check_fits(len, 4)?;
            ColumnData::Text((0..len).map(|_| r.str()).collect::<Result<_>>()?)
        }
        2 => {
            r.check_fits(len, 8)?;
            ColumnData::DateTime((0..len).map(|_| r.i64()).collect::<Result<_>>()?)
        }
        _ => return Err(r.err("bad column kind")),
    };
    Ok(Column {
        props: ColumnProps { name, description, numeric_format, datetime_format, value_order, formula, value_labels },
        data,
    })
}

/// Parse `.otb` bytes into a workbook, verifying magic, version, length and CRC.
pub fn decode(bytes: &[u8]) -> Result<Workbook> {
    if bytes.len() < HEADER_LEN + 4 || &bytes[..4] != MAGIC {
        return Err(CoreError::Format("not an OpenStats .otb file".into()));
    }
    let major = u16::from_le_bytes([bytes[4], bytes[5]]);
    let minor = u16::from_le_bytes([bytes[6], bytes[7]]);
    if major != VERSION_MAJOR {
        return Err(CoreError::Format(format!(
            "unsupported .otb version {major} (this build reads version {VERSION_MAJOR})"
        )));
    }
    let len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
    if len != (bytes.len() - HEADER_LEN - 4) as u64 {
        return Err(CoreError::Format("truncated or padded .otb file".into()));
    }
    let payload = &bytes[HEADER_LEN..HEADER_LEN + len as usize];
    let stored = u32::from_le_bytes(bytes[HEADER_LEN + len as usize..].try_into().expect("4 bytes"));
    if crc32(payload) != stored {
        return Err(CoreError::Format(".otb checksum mismatch (file is corrupt)".into()));
    }
    let mut r = R { b: payload, p: 0 };
    let mut wb = Workbook::empty();
    wb.title = r.str()?;
    let active = r.u32()?;
    let n_meta = r.count(8)?;
    for _ in 0..n_meta {
        let k = r.str()?;
        let v = r.str()?;
        wb.metadata.insert(k, v);
    }
    let n_sheets = r.count(16)?;
    for _ in 0..n_sheets {
        let mut ws = Worksheet::new(r.str()?);
        ws.description = r.str()?;
        let n_cols = r.count(24)?;
        for _ in 0..n_cols {
            let c = decode_column(&mut r, minor)?;
            ws.validate_name(c.name(), None).map_err(|e| CoreError::Format(e.to_string()))?;
            ws.columns.push(c);
        }
        let n_const = r.count(5)?;
        for _ in 0..n_const {
            let k = r.u32()?;
            let v = r.value()?;
            ws.constants.insert(k, v);
        }
        let n_mat = r.count(12)?;
        for _ in 0..n_mat {
            let k = r.u32()?;
            let rows = r.u32()?;
            let cols = r.u32()?;
            let n = (rows as u64).checked_mul(cols as u64).ok_or_else(|| r.err("matrix too large"))?;
            r.check_fits(n, 8)?;
            let data = (0..n).map(|_| r.f64()).collect::<Result<Vec<_>>>()?;
            ws.matrices.insert(k, StoredMatrix::new(rows, cols, data)?);
        }
        wb.worksheets.push(ws);
    }
    if r.p != payload.len() {
        return Err(r.err("trailing bytes"));
    }
    wb.active = if active < wb.worksheets.len() { active } else { 0 };
    Ok(wb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datetime;

    fn sample() -> Workbook {
        let mut wb = Workbook::new();
        wb.title = "Ünïcode title".into();
        wb.metadata.insert("app".into(), "OpenStats".into());
        let ws = wb.sheet_mut(0).unwrap();
        let mut c = Column::numeric("x", vec![1.0, f64::NAN, -0.0, f64::INFINITY, 1e-310]);
        c.props.numeric_format = NumericFormat::Scientific(4);
        ws.add_column(c).unwrap();
        let mut t = Column::text("t", vec!["", "a\nb", "日本"]);
        t.props.value_order = Some(vec!["日本".into()]);
        ws.add_column(t).unwrap();
        let mut d =
            Column::datetime("d", vec![datetime::parse("2026-11-16").unwrap(), crate::missing::DATETIME_MISSING]);
        d.props.datetime_format = Some("d MMM".into());
        ws.add_column(d).unwrap();
        ws.constants.insert(4, Value::Text("k".into()));
        ws.constants.insert(5, Value::Missing);
        ws.matrices.insert(2, StoredMatrix::new(2, 1, vec![3.0, 4.0]).unwrap());
        wb.worksheets.push(Worksheet::new("Empty"));
        wb
    }

    #[test]
    fn roundtrip_exact_bits() {
        let wb = sample();
        let bytes = encode(&wb);
        assert_eq!(&bytes[..4], MAGIC);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, wb);
        let x = back.worksheets[0].columns[0].data.as_numeric().unwrap();
        assert!(x[2].is_sign_negative());
        assert_eq!(x[4], 1e-310);
    }

    #[test]
    fn reads_minor_version_0_files() {
        // Version 1.0 layout: no formula field in column records.
        let mut p = W(Vec::new());
        p.str("old");
        p.u32(0);
        p.u32(0);
        p.u32(1);
        p.str("Sheet");
        p.str("");
        p.u32(1);
        p.str("x");
        p.str("");
        p.u8(0);
        p.u8(0);
        p.u8(0);
        p.opt_str(None);
        p.u8(0);
        p.u64(2);
        p.f64(1.5);
        p.f64(f64::NAN);
        p.u32(0);
        p.u32(0);
        let payload = p.0;
        let mut f = W(Vec::new());
        f.0.extend_from_slice(MAGIC);
        f.u16(1);
        f.u16(0);
        f.u64(payload.len() as u64);
        f.0.extend_from_slice(&payload);
        f.0.extend_from_slice(&crc32(&payload).to_le_bytes());
        let wb = decode(&f.0).unwrap();
        let c = &wb.worksheets[0].columns[0];
        assert_eq!(c.name(), "x");
        assert_eq!(c.props.formula, None);
        assert_eq!(c.data, ColumnData::Numeric(vec![1.5, f64::NAN]));
        // And formulas survive the current version.
        let mut wb2 = wb.clone();
        wb2.worksheets[0].columns[0].props.formula = Some("C2 * 2".into());
        assert_eq!(decode(&encode(&wb2)).unwrap(), wb2);
    }

    #[test]
    fn corruption_detected() {
        let bytes = encode(&sample());
        let mut flipped = bytes.clone();
        flipped[HEADER_LEN + 3] ^= 0x40;
        assert!(decode(&flipped).unwrap_err().to_string().contains("checksum"));
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let mut v2 = bytes.clone();
        v2[4] = 2;
        assert!(decode(&v2).unwrap_err().to_string().contains("version"));
        assert!(decode(b"nope").is_err());
        // A payload with a forged huge count and a valid CRC must fail cleanly.
        let mut p = W(Vec::new());
        p.str("t");
        p.u32(0);
        p.u32(0);
        p.0.extend_from_slice(&u32::MAX.to_le_bytes());
        let payload = p.0;
        let mut f = W(Vec::new());
        f.0.extend_from_slice(MAGIC);
        f.u16(1);
        f.u16(0);
        f.u64(payload.len() as u64);
        f.0.extend_from_slice(&payload);
        f.0.extend_from_slice(&crc32(&payload).to_le_bytes());
        assert!(decode(&f.0).unwrap_err().to_string().contains("count"));
    }
}
