//! How cells and headers are displayed.

use openstats_charts::core::font::text_width;
use openstats_core::column::{Column, ColumnData};
use openstats_core::locale::Locale;
use openstats_core::value::format_number;
use openstats_core::worksheet::Worksheet;
use openstats_core::NumericFormat;

/// Horizontal alignment of cell text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    /// Left (text).
    Left,
    /// Right (numbers, dates).
    Right,
}

/// A cell as shown in the grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellText {
    /// The text to draw.
    pub text: String,
    /// Alignment.
    pub align: Align,
    /// Whether the cell holds a missing value (drawn as `*`, dimmed).
    pub missing: bool,
}

/// Display text of row `row` of `col`; `None` for rows beyond the end of the column (blank
/// cells). Numeric and date/time missing values and empty text show `*`.
pub fn cell_text(col: &Column, row: usize) -> Option<CellText> {
    cell_text_with(col, row, &Locale::default())
}

/// [`cell_text`] with a locale's decimal mark and date order; labelled values show their label, left aligned.
pub fn cell_text_with(col: &Column, row: usize, loc: &Locale) -> Option<CellText> {
    if row >= col.len() {
        return None;
    }
    let missing = col.data.is_missing(row);
    let labelled = col.label_for(row).is_some();
    let text = if missing { "*".to_string() } else { col.display_with(row, loc) };
    let align = if matches!(col.data, ColumnData::Text(_)) || labelled { Align::Left } else { Align::Right };
    Some(CellText { text, align, missing })
}

/// Text of a cell for editing and the formula bar: numbers at full precision (not the display
/// format), missing values empty, date/times in the column's pattern.
pub fn raw_text(col: &Column, row: usize) -> String {
    if row >= col.len() || col.data.is_missing(row) {
        return String::new();
    }
    match &col.data {
        ColumnData::Numeric(v) => format_number(v[row]),
        _ => col.display(row),
    }
}

/// Header label and name of column `c`: `("C1", "Height")`, `("C2-T", "")`, `("C3-D", "Day")`.
/// Columns past the data show the plain numeric label and no name.
pub fn header_parts(ws: &Worksheet, c: usize) -> (String, String) {
    match ws.columns.get(c) {
        Some(col) => (ws.column_label(c), col.name().to_string()),
        None => (format!("C{}", c + 1), String::new()),
    }
}

/// Shorten `text` with an ellipsis so that it fits in `width` pixels.
pub fn fit_text(text: &str, width: f64, size: f64, bold: bool) -> String {
    if text_width(text, size, bold) <= width {
        return text.to_string();
    }
    let ell = text_width("…", size, bold);
    let mut out = String::new();
    let mut w = 0.0;
    for ch in text.chars() {
        let cw = text_width(&ch.to_string(), size, bold);
        if w + cw + ell > width {
            break;
        }
        w += cw;
        out.push(ch);
    }
    out.push('…');
    out
}

/// Text for a number column that does not fit: scientific notation, then `#` filling.
pub fn fit_number(col: &Column, row: usize, text: &str, width: f64, size: f64) -> String {
    if text_width(text, size, false) <= width {
        return text.to_string();
    }
    if let ColumnData::Numeric(v) = &col.data {
        let sci = NumericFormat::Scientific(2).format(v[row]);
        if text_width(&sci, size, false) <= width {
            return sci;
        }
    }
    let hash = text_width("#", size, false).max(1.0);
    "#".repeat(((width / hash).floor() as usize).max(1))
}

/// Column width that fits the header and the first `sample` rows.
pub fn autosize(ws: &Worksheet, c: usize, size: f64, sample: usize, min: f64, max: f64) -> f64 {
    let Some(col) = ws.columns.get(c) else { return min.max(80.0).min(max) };
    let (label, name) = header_parts(ws, c);
    let mut w = text_width(&label, size, true).max(text_width(&name, size, true));
    for r in 0..col.len().min(sample) {
        if let Some(t) = cell_text(col, r) {
            w = w.max(text_width(&t.text, size, false));
        }
    }
    (w + 16.0).clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_core::column::ColumnKind;

    fn ws() -> Worksheet {
        let mut n = Column::numeric("Height", vec![1.5, f64::NAN, 1234.5678]);
        n.props.numeric_format = NumericFormat::Fixed(2);
        Worksheet::with_columns(
            "w",
            vec![
                n,
                Column::text("Name", vec!["ann", "", "bob"]),
                Column::datetime("Day", vec![0, i64::MIN, 86_400_000_000]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn missing_is_star_and_formats_apply() {
        let w = ws();
        let n = &w.columns[0];
        assert_eq!(cell_text(n, 0).unwrap().text, "1.50");
        let m = cell_text(n, 1).unwrap();
        assert_eq!((m.text.as_str(), m.missing, m.align), ("*", true, Align::Right));
        assert_eq!(cell_text(n, 2).unwrap().text, "1234.57");
        assert!(cell_text(n, 3).is_none());
        let t = &w.columns[1];
        assert_eq!(cell_text(t, 0).unwrap().align, Align::Left);
        assert_eq!(cell_text(t, 1).unwrap().text, "*");
        let d = &w.columns[2];
        assert_eq!(cell_text(d, 1).unwrap().text, "*");
        assert_ne!(cell_text(d, 0).unwrap().text, "*");
    }

    #[test]
    fn raw_text_keeps_full_precision() {
        let w = ws();
        assert_eq!(raw_text(&w.columns[0], 2), "1234.5678");
        assert_eq!(raw_text(&w.columns[0], 1), "");
        assert_eq!(raw_text(&w.columns[1], 0), "ann");
        assert_eq!(raw_text(&w.columns[0], 99), "");
    }

    #[test]
    fn minitab_headers() {
        let w = ws();
        assert_eq!(header_parts(&w, 0), ("C1".into(), "Height".into()));
        assert_eq!(header_parts(&w, 1), ("C2-T".into(), "Name".into()));
        assert_eq!(header_parts(&w, 2), ("C3-D".into(), "Day".into()));
        assert_eq!(header_parts(&w, 7), ("C8".into(), String::new()));
        assert_eq!(ColumnKind::Text.tag(), "T");
    }

    #[test]
    fn fitting() {
        assert_eq!(fit_text("abc", 100.0, 12.0, false), "abc");
        let f = fit_text("a very long cell value that cannot fit", 60.0, 12.0, false);
        assert!(f.ends_with('…') && text_width(&f, 12.0, false) <= 60.0);
        assert_eq!(fit_text("abc", 1.0, 12.0, false), "…");
        let w = ws();
        let big = Column::numeric("x", vec![123_456_789.123]);
        assert_eq!(fit_number(&big, 0, "123456789.123", 200.0, 12.0), "123456789.123");
        let sci = fit_number(&big, 0, "123456789.123", 64.0, 12.0);
        assert!(sci.contains('E'));
        assert!(fit_number(&big, 0, "123456789.123", 12.0, 12.0).chars().all(|c| c == '#'));
        assert!(autosize(&w, 1, 12.0, 100, 28.0, 600.0) >= 28.0);
        assert_eq!(autosize(&w, 0, 12.0, 100, 28.0, 30.0), 30.0);
    }
}
