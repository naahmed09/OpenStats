//! Office Open XML spreadsheet (`.xlsx` / `.xlsm`) import.
//!
//! Reads worksheet cell values (shared strings, inline strings, numbers,
//! booleans, ISO dates, error values and cached formula results), resolves
//! parts through the package relationships, detects date/time cells from
//! their number formats (built-in and custom), honours the 1904 date system,
//! and turns each sheet into typed columns: the first row supplies names
//! (optional) and each column becomes numeric, date/time or text.

use super::xml::{self, local_name, Event, Reader};
use super::zip::ZipArchive;
use crate::column::{Column, ColumnData, ColumnKind};
use crate::datetime::{self, MICROS_PER_DAY};
use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;
use crate::value::format_number;
use crate::workbook::Workbook;
use crate::worksheet::Worksheet;
use std::collections::{HashMap, HashSet};

fn err(m: impl Into<String>) -> CoreError {
    CoreError::Format(format!("xlsx: {}", m.into()))
}

/// Which sheets to import.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SheetSelect {
    /// Every sheet, including hidden ones.
    All,
    /// Every visible sheet (default).
    #[default]
    Visible,
    /// Sheets with these names (case-insensitive).
    Names(Vec<String>),
    /// Sheets at these 0-based positions.
    Indices(Vec<usize>),
}

/// Import options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XlsxOptions {
    pub sheets: SheetSelect,
    /// First used row holds column names.
    pub header: bool,
    /// Infer numeric/date columns; otherwise every column is text.
    pub infer_types: bool,
    /// Uncompressed size limit per part (defuses zip bombs). Default 1 GiB.
    pub max_part_bytes: u64,
}

impl Default for XlsxOptions {
    fn default() -> Self {
        XlsxOptions { sheets: SheetSelect::Visible, header: true, infer_types: true, max_part_bytes: 1 << 30 }
    }
}

/// Information about one sheet in a workbook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SheetInfo {
    pub name: String,
    pub hidden: bool,
    part: String,
}

/// A sheet as sparse rows of (column, cell).
type Row = Vec<(usize, Cell)>;

/// A parsed cell value.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Num(f64, bool), // value, has date/time format
    Text(String),
    Bool(bool),
    Missing,
}

struct Package<'a> {
    zip: ZipArchive<'a>,
    limit: u64,
}

impl<'a> Package<'a> {
    fn part(&self, path: &str) -> Result<Option<String>> {
        match self.zip.find(path).or_else(|| self.zip.find_ci(path)) {
            Some(i) => Ok(Some(xml::decode(&self.zip.read_index(i, Some(self.limit))?)?)),
            None => Ok(None),
        }
    }
    fn required(&self, path: &str) -> Result<String> {
        self.part(path)?.ok_or_else(|| err(format!("missing part {path}")))
    }
}

/// Resolve a relationship target relative to the directory of `source`.
fn resolve(source: &str, target: &str) -> String {
    if let Some(abs) = target.strip_prefix('/') {
        return abs.to_string();
    }
    let mut parts: Vec<&str> = source.split('/').collect();
    parts.pop(); // file name
    for seg in target.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

fn rels_path(part: &str) -> String {
    match part.rsplit_once('/') {
        Some((dir, file)) => format!("{dir}/_rels/{file}.rels"),
        None => format!("_rels/{part}.rels"),
    }
}

/// Relationships of a part: Id → (Type, resolved target).
fn read_rels(pkg: &Package, part: &str) -> Result<HashMap<String, (String, String)>> {
    let mut out = HashMap::new();
    let Some(text) = pkg.part(&rels_path(part))? else { return Ok(out) };
    let mut r = Reader::new(&text);
    while let Some(e) = r.next()? {
        if let Event::Start { name, .. } = &e {
            if local_name(name) == "Relationship" && e.attr("TargetMode") != Some("External") {
                if let (Some(id), Some(t), Some(ty)) = (e.attr("Id"), e.attr("Target"), e.attr("Type")) {
                    out.insert(id.to_string(), (ty.to_string(), resolve(part, t)));
                }
            }
        }
    }
    Ok(out)
}

fn workbook_part(pkg: &Package) -> Result<String> {
    for (ty, target) in read_rels(pkg, "")?.into_values() {
        if ty.ends_with("/officeDocument") {
            return Ok(target);
        }
    }
    Ok("xl/workbook.xml".into())
}

struct WorkbookInfo {
    sheets: Vec<SheetInfo>,
    date1904: bool,
    shared_strings: Vec<String>,
    date_styles: Vec<bool>,
}

fn read_workbook(pkg: &Package) -> Result<WorkbookInfo> {
    let wb_part = workbook_part(pkg)?;
    let rels = read_rels(pkg, &wb_part)?;
    let text = pkg.required(&wb_part)?;
    let mut r = Reader::new(&text);
    let mut sheets = Vec::new();
    let mut date1904 = false;
    while let Some(e) = r.next()? {
        if let Event::Start { name, .. } = &e {
            match local_name(name) {
                "workbookPr" => date1904 = matches!(e.attr("date1904"), Some("1" | "true")),
                "sheet" => {
                    let sname = e.attr("name").ok_or_else(|| err("sheet without a name"))?.to_string();
                    let rid = e.attr("r:id").or_else(|| e.attr("id")).ok_or_else(|| err("sheet without r:id"))?;
                    let (_, target) =
                        rels.get(rid).ok_or_else(|| err(format!("sheet {sname:?}: unknown relationship {rid}")))?;
                    let hidden = matches!(e.attr("state"), Some("hidden" | "veryHidden"));
                    sheets.push(SheetInfo { name: sname, hidden, part: target.clone() });
                }
                _ => {}
            }
        }
    }
    let find_rel = |suffix: &str, default: &str| -> String {
        rels.values()
            .find(|(ty, _)| ty.ends_with(suffix))
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| resolve(&wb_part, default))
    };
    let shared_strings = match pkg.part(&find_rel("/sharedStrings", "sharedStrings.xml"))? {
        Some(t) => read_shared_strings(&t)?,
        None => Vec::new(),
    };
    let date_styles = match pkg.part(&find_rel("/styles", "styles.xml"))? {
        Some(t) => read_date_styles(&t)?,
        None => Vec::new(),
    };
    Ok(WorkbookInfo { sheets, date1904, shared_strings, date_styles })
}

/// Text of an `<si>` / `<is>` element: `<t>` directly or in `<r>` runs,
/// excluding phonetic runs (`<rPh>`).
fn rich_text(r: &mut Reader) -> Result<String> {
    let start_depth = r.depth();
    let mut out = String::new();
    loop {
        match r.next()? {
            Some(Event::Start { name, empty, .. }) => match local_name(&name) {
                "t" if !empty => out.push_str(&r.read_text()?),
                "rPh" | "phoneticPr" if !empty => r.skip_element()?,
                _ => {}
            },
            Some(Event::End { .. }) if r.depth() < start_depth => return Ok(out),
            Some(_) => {}
            None => return Err(err("unterminated rich text")),
        }
    }
}

fn read_shared_strings(text: &str) -> Result<Vec<String>> {
    let mut r = Reader::new(text);
    let mut out = Vec::new();
    while let Some(e) = r.next()? {
        if let Event::Start { name, empty, .. } = &e {
            if local_name(name) == "si" {
                out.push(if *empty { String::new() } else { rich_text(&mut r)? });
            }
        }
    }
    Ok(out)
}

/// `true` if a number format code displays a date or time.
pub fn is_date_format(code: &str) -> bool {
    let lower = code.to_ascii_lowercase();
    if lower == "general" {
        return false;
    }
    // Only the first section (positive numbers) decides.
    let mut chars = lower.chars().peekable();
    let mut in_quote = false;
    while let Some(c) = chars.next() {
        if in_quote {
            if c == '"' {
                in_quote = false;
            }
            continue;
        }
        match c {
            '"' => in_quote = true,
            '\\' | '_' | '*' => {
                chars.next();
            }
            '[' => {
                // [h], [mm], [ss] are elapsed durations (numeric); colours and
                // locales ([Red], [$-409]) are ignored.
                let mut inner = String::new();
                for d in chars.by_ref() {
                    if d == ']' {
                        break;
                    }
                    inner.push(d);
                }
                if !inner.is_empty() && inner.chars().all(|x| matches!(x, 'h' | 'm' | 's')) {
                    return false;
                }
            }
            ';' => break,
            'd' | 'm' | 'y' | 'h' | 's' | 'e' | 'g' | 'b' => {
                // 'e'/'g'/'b' are era/Buddhist-year codes; exclude scientific "E+".
                if c == 'e' && matches!(chars.peek(), Some('+' | '-')) {
                    return false;
                }
                if matches!(c, 'd' | 'm' | 'y' | 'h' | 's') {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Built-in number formats that are dates/times (ECMA-376 §18.8.30 plus the
/// East-Asian date formats).
fn builtin_is_date(id: u32) -> bool {
    matches!(id, 14..=22 | 27..=36 | 45..=47 | 50..=58 | 71..=81)
}

/// For each cell style (`cellXfs` index): does it format dates?
fn read_date_styles(text: &str) -> Result<Vec<bool>> {
    let mut r = Reader::new(text);
    let mut custom: HashMap<u32, bool> = HashMap::new();
    let mut out = Vec::new();
    let mut in_cell_xfs = false;
    while let Some(e) = r.next()? {
        match &e {
            Event::Start { name, empty, .. } => match local_name(name) {
                "numFmt" => {
                    if let (Some(id), Some(code)) = (e.attr("numFmtId"), e.attr("formatCode")) {
                        if let Ok(id) = id.parse() {
                            custom.insert(id, is_date_format(code));
                        }
                    }
                }
                "cellXfs" => in_cell_xfs = !empty,
                "xf" if in_cell_xfs => {
                    let id: u32 = e.attr("numFmtId").and_then(|v| v.parse().ok()).unwrap_or(0);
                    out.push(custom.get(&id).copied().unwrap_or_else(|| builtin_is_date(id)));
                    if !empty {
                        r.skip_element()?;
                    }
                }
                _ => {}
            },
            Event::End { name } if local_name(name) == "cellXfs" => in_cell_xfs = false,
            _ => {}
        }
    }
    Ok(out)
}

/// `A1`-style reference → (row, col), both 0-based.
fn parse_cell_ref(s: &str) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut col: usize = 0;
    while i < b.len() && b[i].is_ascii_alphabetic() {
        col = col.checked_mul(26)?.checked_add((b[i].to_ascii_uppercase() - b'A' + 1) as usize)?;
        i += 1;
    }
    if i == 0 || i == b.len() || col > 16_384 {
        return None;
    }
    let row: usize = s[i..].parse().ok()?;
    if row == 0 {
        return None;
    }
    Some((row - 1, col - 1))
}

/// Parse a worksheet part into a sparse grid of rows.
fn read_sheet_cells(text: &str, info: &WorkbookInfo) -> Result<Vec<Vec<(usize, Cell)>>> {
    let mut r = Reader::new(text);
    let mut rows: Vec<Vec<(usize, Cell)>> = Vec::new();
    let mut next_row = 0usize;
    while let Some(e) = r.next()? {
        let Event::Start { name, empty, .. } = &e else { continue };
        if local_name(name) != "row" {
            continue;
        }
        let row = e.attr("r").and_then(|v| v.parse::<usize>().ok()).map(|v| v.saturating_sub(1)).unwrap_or(next_row);
        next_row = row + 1;
        if rows.len() <= row {
            if row > 1_048_576 {
                return Err(err("row index beyond the spreadsheet limit"));
            }
            rows.resize_with(row + 1, Vec::new);
        }
        if *empty {
            continue;
        }
        let row_depth = r.depth();
        let mut next_col = 0usize;
        loop {
            let ev = r.next()?.ok_or_else(|| err("unterminated row"))?;
            match &ev {
                Event::End { .. } if r.depth() < row_depth => break,
                Event::Start { name, empty, .. } if local_name(name) == "c" => {
                    let col = ev.attr("r").and_then(parse_cell_ref).map(|(_, c)| c).unwrap_or(next_col);
                    next_col = col + 1;
                    let ty = ev.attr("t").unwrap_or("n").to_string();
                    let style: usize = ev.attr("s").and_then(|v| v.parse().ok()).unwrap_or(0);
                    let is_date = info.date_styles.get(style).copied().unwrap_or(false);
                    let mut v: Option<String> = None;
                    let mut inline: Option<String> = None;
                    if !empty {
                        let cd = r.depth();
                        loop {
                            match r.next()?.ok_or_else(|| err("unterminated cell"))? {
                                Event::Start { name, empty, .. } => match local_name(&name) {
                                    "v" if !empty => v = Some(r.read_text()?),
                                    "is" if !empty => inline = Some(rich_text(&mut r)?),
                                    _ if !empty => r.skip_element()?,
                                    _ => {}
                                },
                                Event::End { .. } if r.depth() < cd => break,
                                _ => {}
                            }
                        }
                    }
                    let cell = match ty.as_str() {
                        "s" => {
                            let idx: usize = v
                                .as_deref()
                                .and_then(|x| x.trim().parse().ok())
                                .ok_or_else(|| err("bad shared string index"))?;
                            Cell::Text(
                                info.shared_strings
                                    .get(idx)
                                    .cloned()
                                    .ok_or_else(|| err("shared string index out of range"))?,
                            )
                        }
                        "inlineStr" => Cell::Text(inline.unwrap_or_default()),
                        "str" => Cell::Text(v.unwrap_or_default()),
                        "b" => match v.as_deref().map(str::trim) {
                            Some("1" | "true") => Cell::Bool(true),
                            Some("0" | "false") => Cell::Bool(false),
                            _ => Cell::Missing,
                        },
                        "e" => Cell::Missing,
                        "d" => match v.as_deref().map(|s| datetime::parse(s.trim())) {
                            Some(Ok(t)) => Cell::Num(ts_to_serial(t, info.date1904), true),
                            _ => Cell::Missing,
                        },
                        _ => match v.as_deref().map(str::trim) {
                            Some(s) if !s.is_empty() => {
                                Cell::Num(s.parse::<f64>().map_err(|_| err(format!("bad number {s:?}")))?, is_date)
                            }
                            _ => Cell::Missing,
                        },
                    };
                    if cell != Cell::Missing {
                        rows[row].push((col, cell));
                    }
                }
                Event::Start { empty: false, .. } => r.skip_element()?,
                _ => {}
            }
        }
    }
    Ok(rows)
}

fn serial_to_ts(serial: f64, date1904: bool) -> i64 {
    if date1904 {
        // 1904 system: day 0 = 1904-01-01.
        let base = datetime::days_from_civil(1904, 1, 1) * MICROS_PER_DAY;
        base + (serial * MICROS_PER_DAY as f64).round() as i64
    } else {
        datetime::from_excel_serial(serial)
    }
}

/// Inverse of [`serial_to_ts`] in the workbook's own date system.
fn ts_to_serial(ts: i64, date1904: bool) -> f64 {
    if date1904 {
        let base = datetime::days_from_civil(1904, 1, 1) * MICROS_PER_DAY;
        (ts - base) as f64 / MICROS_PER_DAY as f64
    } else {
        datetime::to_excel_serial(ts)
    }
}

fn cells_to_worksheet(name: &str, rows: Vec<Vec<(usize, Cell)>>, info: &WorkbookInfo, opts: &XlsxOptions) -> Worksheet {
    let mut ws = Worksheet::new(name);
    let first = rows.iter().position(|r| !r.is_empty());
    let Some(first) = first else { return ws };
    let ncols = rows.iter().flat_map(|r| r.iter().map(|(c, _)| c + 1)).max().unwrap_or(0);
    let (header, data_rows): (Option<&Row>, &[Row]) =
        if opts.header { (Some(&rows[first]), &rows[first + 1..]) } else { (None, &rows[first..]) };
    let n = data_rows.len();
    let mut grid: Vec<Vec<Option<&Cell>>> = vec![vec![None; n]; ncols];
    for (ri, r) in data_rows.iter().enumerate() {
        for (c, cell) in r {
            grid[*c][ri] = Some(cell);
        }
    }
    let mut taken = HashSet::new();
    for (j, col) in grid.iter().enumerate() {
        let present: Vec<&Cell> = col.iter().flatten().copied().collect();
        let kind = if !opts.infer_types {
            ColumnKind::Text
        } else if present.is_empty() {
            ColumnKind::Numeric
        } else if present.iter().all(|c| matches!(c, Cell::Num(_, true))) {
            ColumnKind::DateTime
        } else if present.iter().all(|c| matches!(c, Cell::Num(..) | Cell::Bool(_))) {
            ColumnKind::Numeric
        } else {
            ColumnKind::Text
        };
        let data = match kind {
            ColumnKind::Numeric => ColumnData::Numeric(
                col.iter()
                    .map(|c| match c {
                        Some(Cell::Num(x, _)) => *x,
                        Some(Cell::Bool(b)) => f64::from(u8::from(*b)),
                        _ => f64::NAN,
                    })
                    .collect(),
            ),
            ColumnKind::DateTime => ColumnData::DateTime(
                col.iter()
                    .map(|c| match c {
                        Some(Cell::Num(x, _)) => serial_to_ts(*x, info.date1904),
                        _ => DATETIME_MISSING,
                    })
                    .collect(),
            ),
            ColumnKind::Text => ColumnData::Text(
                col.iter()
                    .map(|c| match c {
                        Some(Cell::Text(s)) => s.clone(),
                        Some(Cell::Num(x, true)) => datetime::format_default(serial_to_ts(*x, info.date1904)),
                        Some(Cell::Num(x, false)) => format_number(*x),
                        Some(Cell::Bool(b)) => {
                            if *b {
                                "TRUE".into()
                            } else {
                                "FALSE".into()
                            }
                        }
                        Some(Cell::Missing) | None => String::new(),
                    })
                    .collect(),
            ),
        };
        let raw_name = header
            .and_then(|h| h.iter().find(|(c, _)| *c == j))
            .map(|(_, cell)| match cell {
                Cell::Text(s) => s.clone(),
                Cell::Num(x, _) => format_number(*x),
                Cell::Bool(b) => b.to_string().to_uppercase(),
                Cell::Missing => String::new(),
            })
            .unwrap_or_default();
        let cname = super::csv::sanitize_name(&raw_name, &taken);
        if !cname.is_empty() {
            taken.insert(cname.to_lowercase());
        }
        let mut column = Column::new(cname, data);
        column.data.trim_trailing_missing();
        ws.columns.push(column);
    }
    ws
}

/// List the sheets of a workbook.
pub fn sheet_names(bytes: &[u8]) -> Result<Vec<SheetInfo>> {
    let pkg = Package { zip: ZipArchive::new(bytes)?, limit: XlsxOptions::default().max_part_bytes };
    Ok(read_workbook(&pkg)?.sheets)
}

/// Import an `.xlsx` file as a workbook (one worksheet per selected sheet).
pub fn read_xlsx(bytes: &[u8], opts: &XlsxOptions) -> Result<Workbook> {
    let pkg = Package { zip: ZipArchive::new(bytes)?, limit: opts.max_part_bytes };
    let info = read_workbook(&pkg)?;
    let chosen: Vec<usize> = match &opts.sheets {
        SheetSelect::All => (0..info.sheets.len()).collect(),
        SheetSelect::Visible => (0..info.sheets.len()).filter(|&i| !info.sheets[i].hidden).collect(),
        SheetSelect::Names(names) => names
            .iter()
            .map(|n| {
                info.sheets
                    .iter()
                    .position(|s| s.name.eq_ignore_ascii_case(n))
                    .ok_or_else(|| CoreError::NotFound(format!("sheet {n:?}")))
            })
            .collect::<Result<_>>()?,
        SheetSelect::Indices(ix) => {
            if let Some(&bad) = ix.iter().find(|&&i| i >= info.sheets.len()) {
                return Err(CoreError::IndexOutOfRange { what: "sheet", index: bad, len: info.sheets.len() });
            }
            ix.clone()
        }
    };
    let mut wb = Workbook::empty();
    for i in chosen {
        let s = &info.sheets[i];
        let text = pkg.required(&s.part)?;
        let rows = read_sheet_cells(&text, &info).map_err(|e| err(format!("sheet {:?}: {e}", s.name)))?;
        let name = wb.unique_sheet_name(&s.name);
        wb.worksheets.push(cells_to_worksheet(&name, rows, &info, opts));
    }
    if wb.worksheets.is_empty() {
        wb.worksheets.push(Worksheet::new("Worksheet 1"));
    }
    Ok(wb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::zip::ZipWriter;
    use crate::value::Value;

    #[test]
    fn date_format_detection() {
        for d in ["yyyy-mm-dd", "m/d/yy h:mm", "[$-409]mmmm d, yyyy", "dd\\.mm\\.yyyy", "h:mm AM/PM", "[Red]d-mmm"] {
            assert!(is_date_format(d), "{d}");
        }
        for n in ["General", "0.00", "#,##0", "0.00E+00", "[h]:mm:ss", "\"day\" 0", "0%", "[mm]:ss", "_(* #,##0_)"] {
            assert!(!is_date_format(n), "{n}");
        }
        assert!(builtin_is_date(14) && builtin_is_date(22) && !builtin_is_date(2));
    }

    #[test]
    fn helpers() {
        assert_eq!(parse_cell_ref("A1"), Some((0, 0)));
        assert_eq!(parse_cell_ref("AB12"), Some((11, 27)));
        assert_eq!(parse_cell_ref("XFD1048576"), Some((1_048_575, 16_383)));
        assert_eq!(parse_cell_ref("12"), None);
        assert_eq!(resolve("xl/workbook.xml", "worksheets/sheet1.xml"), "xl/worksheets/sheet1.xml");
        assert_eq!(resolve("xl/worksheets/sheet1.xml", "../sharedStrings.xml"), "xl/sharedStrings.xml");
        assert_eq!(resolve("xl/workbook.xml", "/xl/x.xml"), "xl/x.xml");
        assert_eq!(rels_path("xl/workbook.xml"), "xl/_rels/workbook.xml.rels");
        assert_eq!(rels_path(""), "_rels/.rels");
    }

    /// Hand-built package exercising shared strings with rich text and
    /// phonetic runs, `str`/`e`/`d` cell types, a custom date format,
    /// the 1904 date system, sparse cells and an absolute relationship target.
    fn handmade() -> Vec<u8> {
        let mut z = ZipWriter::new();
        z.add(
            "[Content_Types].xml",
            b"<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>",
            6,
        )
        .unwrap();
        z.add("_rels/.rels", br#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#, 6).unwrap();
        z.add("xl/workbook.xml", br#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr date1904="1"/><sheets><sheet name="Data &amp; More" sheetId="1" r:id="rId1"/></sheets></workbook>"#, 6).unwrap();
        z.add("xl/_rels/workbook.xml.rels", br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="/xl/worksheets/s1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#, 6).unwrap();
        z.add("xl/sharedStrings.xml", "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><si><t>Name</t></si><si><r><rPr><b/></rPr><t>Bold</t></r><r><t xml:space=\"preserve\"> run</t></r><rPh><t>ふり</t></rPh></si><si><t>When</t></si><si/><si><t>Score</t></si></sst>".as_bytes(), 6).unwrap();
        z.add("xl/styles.xml", br#"<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="170" formatCode="dd/mm/yyyy\ hh:mm"/></numFmts><cellXfs count="3"><xf numFmtId="0"/><xf numFmtId="170"/><xf numFmtId="2"><alignment/></xf></cellXfs></styleSheet>"#, 6).unwrap();
        z.add("xl/worksheets/s1.xml", br#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="2"><c r="A2" t="s"><v>0</v></c><c r="B2" t="s"><v>2</v></c><c r="D2" t="s"><v>4</v></c></row>
<row r="3"><c r="A3" t="s"><v>1</v></c><c r="B3" s="1"><v>0.5</v></c><c r="C3" t="e"><v>#N/A</v></c><c r="D3" s="2"><f>1+1</f><v>2</v></c></row>
<row r="4"><c r="A4" t="str"><f>"x"</f><v>formula text</v></c><c r="B4" t="d"><v>2026-11-16T12:00:00</v></c><c r="D4" t="b"><v>1</v></c></row>
<row r="6"><c r="A6" t="s"><v>3</v></c><c r="D6"><v>1.25e1</v></c></row>
</sheetData></worksheet>"#, 6).unwrap();
        z.finish()
    }

    #[test]
    fn handmade_package() {
        let wb = read_xlsx(&handmade(), &XlsxOptions::default()).unwrap();
        let ws = &wb.worksheets[0];
        assert_eq!(ws.name, "Data & More");
        let names: Vec<&str> = ws.columns.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["Name", "When", "", "Score"]);
        assert_eq!(ws.columns[0].data.as_text().unwrap(), ["Bold run", "formula text"]);
        assert_eq!(ws.columns[1].kind(), ColumnKind::DateTime);
        // 1904 system: serial 0.5 = 1904-01-01 12:00.
        assert_eq!(
            crate::datetime::format(ws.columns[1].data.as_datetime().unwrap()[0], "yyyy-MM-dd HH:mm"),
            "1904-01-01 12:00"
        );
        assert_eq!(
            crate::datetime::format(ws.columns[1].data.as_datetime().unwrap()[1], "yyyy-MM-dd HH:mm"),
            "2026-11-16 12:00"
        );
        assert_eq!(ws.columns[2].len(), 0, "error cell is missing");
        assert_eq!(ws.columns[3].data, ColumnData::Numeric(vec![2.0, 1.0, f64::NAN, 12.5]));
        let no_header = read_xlsx(&handmade(), &XlsxOptions { header: false, ..Default::default() }).unwrap();
        assert_eq!(no_header.worksheets[0].columns[0].get(0), Value::Text("Name".into()));
    }

    #[test]
    fn openpyxl_fixture() {
        let bytes = include_bytes!("../../tests/fixtures/openpyxl_sample.xlsx");
        let sheets = sheet_names(bytes).unwrap();
        assert_eq!(
            sheets.iter().map(|s| (s.name.as_str(), s.hidden)).collect::<Vec<_>>(),
            [("Measurements", false), ("Lookup", false), ("Hidden", true)]
        );
        let wb = read_xlsx(bytes, &XlsxOptions::default()).unwrap();
        assert_eq!(wb.worksheets.len(), 2, "hidden sheet skipped by default");
        let ws = &wb.worksheets[0];
        let names: Vec<&str> = ws.columns.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["Part", "Diameter", "Passed", "Inspected", "Timestamp", "Notes", "Count", "Double"]);
        assert_eq!(ws.columns[0].data.as_text().unwrap(), ["P-001", "P-002", "P-003"]);
        assert_eq!(ws.columns[1].data, ColumnData::Numeric(vec![10.02, 9.98, 10.11]));
        assert_eq!(ws.columns[2].data, ColumnData::Numeric(vec![1.0, 0.0, 1.0]));
        assert_eq!(ws.columns[3].kind(), ColumnKind::DateTime);
        assert_eq!(crate::datetime::format(ws.columns[3].data.as_datetime().unwrap()[2], "yyyy-MM-dd"), "2026-11-04");
        assert_eq!(
            crate::datetime::format(ws.columns[4].data.as_datetime().unwrap()[0], "yyyy-MM-dd HH:mm:ss"),
            "2026-11-02 08:30:15"
        );
        assert_eq!(ws.columns[5].data.as_text().unwrap(), ["ok", "42", "réjected & retested"]);
        assert_eq!(ws.columns[6].data, ColumnData::Numeric(vec![3.0, f64::NAN, 7.0]));
        assert_eq!(ws.columns[7].len(), 0, "formulas without cached values are missing");
        let all = read_xlsx(bytes, &XlsxOptions { sheets: SheetSelect::All, ..Default::default() }).unwrap();
        assert_eq!(all.worksheets[2].columns[0].data, ColumnData::Numeric(vec![1.5]));
        let one =
            read_xlsx(bytes, &XlsxOptions { sheets: SheetSelect::Names(vec!["lookup".into()]), ..Default::default() })
                .unwrap();
        assert_eq!(one.worksheets[0].columns[1].data.as_text().unwrap(), ["Alpha", "Beta"]);
        assert!(read_xlsx(bytes, &XlsxOptions { sheets: SheetSelect::Indices(vec![9]), ..Default::default() }).is_err());
    }
}
