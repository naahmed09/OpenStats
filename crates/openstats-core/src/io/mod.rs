//! File formats: CSV/TSV, JSON, the native `.otb` binary project format,
//! `.xlsx` import, gzip-compressed text, and the from-scratch building blocks
//! behind them (DEFLATE/zlib/gzip, ZIP, XML, OLE2 compound files, CRC-32).
//! Minitab files are detected and inspected (see [`minitab`]).
//!
//! Every format has a pure in-memory API (`&str`/`&[u8]` in, `String`/
//! `Vec<u8>` out) so it works identically on native targets and in
//! WebAssembly, where file access goes through the browser or Tauri.

pub mod cfb;
pub mod crc32;
pub mod csv;
pub mod deflate;
pub mod json;
pub mod minitab;
pub mod otb;
pub mod xlsx;
pub mod xml;
pub mod zip;

use crate::error::{CoreError, Result};
use crate::workbook::Workbook;

/// Formats recognised by [`load_workbook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Otb,
    Json,
    Csv,
    Tsv,
    Xlsx,
    /// gzip-compressed delimited text (`.csv.gz`, `.tsv.gz`).
    GzipText,
    Minitab,
}

impl FileFormat {
    /// Guess from a file name's extension.
    pub fn from_path(path: &str) -> Option<FileFormat> {
        let ext = path.rsplit('.').next()?.to_ascii_lowercase();
        match ext.as_str() {
            "otb" => Some(FileFormat::Otb),
            "xlsx" | "xlsm" => Some(FileFormat::Xlsx),
            "gz" => Some(FileFormat::GzipText),
            "mpx" | "mwx" | "mpj" | "mtw" => Some(FileFormat::Minitab),
            "json" => Some(FileFormat::Json),
            "csv" => Some(FileFormat::Csv),
            "tsv" | "tab" => Some(FileFormat::Tsv),
            "txt" | "dat" => Some(FileFormat::Csv),
            _ => None,
        }
    }

    /// Guess from content (magic number / first non-space byte).
    pub fn sniff(bytes: &[u8]) -> FileFormat {
        if bytes.starts_with(otb::MAGIC) {
            return FileFormat::Otb;
        }
        if bytes.starts_with(&[0x1F, 0x8B]) {
            return FileFormat::GzipText;
        }
        if bytes.starts_with(b"PK\x03\x04") {
            if let Ok(z) = zip::ZipArchive::new(bytes) {
                if z.find_ci("xl/workbook.xml").is_some() {
                    return FileFormat::Xlsx;
                }
            }
            return FileFormat::Minitab;
        }
        if cfb::Cfb::is_cfb(bytes) {
            return FileFormat::Minitab;
        }
        let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
        match body.iter().find(|b| !b.is_ascii_whitespace()) {
            Some(b'{') => FileFormat::Json,
            _ => FileFormat::Csv,
        }
    }
}

/// Load a workbook from bytes, using the file name (if given) and the content
/// to choose the format. Delimited text becomes a one-sheet workbook named
/// after the file.
pub fn load_workbook(name: Option<&str>, bytes: &[u8]) -> Result<Workbook> {
    load_workbook_in(name, bytes, &crate::locale::Locale::default())
}

/// [`load_workbook`] reading delimited text with a locale's decimal mark and date order.
pub fn load_workbook_in(name: Option<&str>, bytes: &[u8], locale: &crate::locale::Locale) -> Result<Workbook> {
    let sniffed = FileFormat::sniff(bytes);
    let fmt = match (name.and_then(FileFormat::from_path), sniffed) {
        (_, FileFormat::Otb) => FileFormat::Otb,
        (_, FileFormat::Xlsx) => FileFormat::Xlsx,
        (_, FileFormat::GzipText) => FileFormat::GzipText,
        (Some(f), _) => f,
        (None, s) => s,
    };
    match fmt {
        FileFormat::Xlsx => xlsx::read_xlsx(bytes, &xlsx::XlsxOptions::default()),
        FileFormat::Minitab => Err(minitab::unsupported_error(name, bytes)),
        FileFormat::GzipText => {
            let inner = deflate::gzip_decompress(bytes, Some(4 << 30))?;
            let inner_name =
                name.map(|n| n.strip_suffix(".gz").or_else(|| n.strip_suffix(".GZ")).unwrap_or(n).to_string());
            if FileFormat::sniff(&inner) == FileFormat::GzipText {
                return Err(CoreError::Format("nested gzip data".into()));
            }
            load_workbook_in(inner_name.as_deref().or(Some("data.csv")), &inner, locale)
        }
        FileFormat::Otb => otb::decode(bytes),
        FileFormat::Json => json::workbook_from_json(utf8(bytes)?),
        FileFormat::Csv | FileFormat::Tsv => {
            let mut opts = csv::CsvReadOptions {
                decimal: locale.decimal,
                date: crate::datetime::ParseOptions { order: locale.date_order, ..Default::default() },
                ..Default::default()
            };
            if fmt == FileFormat::Tsv {
                opts.delimiter = Some('\t');
            }
            if let Some(n) = name {
                let stem = n.rsplit(['/', '\\']).next().unwrap_or(n);
                let stem = stem.rsplit_once('.').map_or(stem, |(s, _)| s);
                opts.sheet_name = stem.to_string();
            }
            let ws = csv::read_csv(utf8(bytes)?, &opts)?;
            let mut wb = Workbook::empty();
            wb.title = ws.name.clone();
            wb.worksheets.push(ws);
            Ok(wb)
        }
    }
}

/// Decode UTF-8 (stripping a BOM). Invalid UTF-8 is decoded as Windows-1252,
/// the most common encoding of legacy CSV exports.
pub fn decode_text(bytes: &[u8]) -> String {
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    match std::str::from_utf8(body) {
        Ok(s) => s.to_string(),
        Err(_) => body.iter().map(|&b| cp1252(b)).collect(),
    }
}

fn utf8(bytes: &[u8]) -> Result<&str> {
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    std::str::from_utf8(body).map_err(|e| CoreError::Format(format!("file is not valid UTF-8: {e}")))
}

/// Windows-1252 → Unicode.
fn cp1252(b: u8) -> char {
    const HIGH: [u16; 32] = [
        0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D,
        0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A,
        0x0153, 0x009D, 0x017E, 0x0178,
    ];
    match b {
        0x80..=0x9F => char::from_u32(HIGH[(b - 0x80) as usize] as u32).unwrap_or('\u{FFFD}'),
        _ => b as char,
    }
}

/// Read a file from disk (native targets) and load it.
pub fn load_workbook_file(path: &str) -> Result<Workbook> {
    let bytes = std::fs::read(path)?;
    load_workbook(Some(path), &bytes)
}

/// Save a workbook to disk in the format implied by the extension
/// (`.otb` or `.json`; CSV export is per worksheet via [`csv::write_csv`]).
pub fn save_workbook_file(wb: &Workbook, path: &str) -> Result<()> {
    let bytes = match FileFormat::from_path(path) {
        Some(FileFormat::Otb) | None => otb::encode(wb),
        Some(FileFormat::Json) => json::workbook_to_json(wb, true).into_bytes(),
        Some(FileFormat::Csv) => csv::write_csv(wb.active_sheet()?, &csv::CsvWriteOptions::default()).into_bytes(),
        Some(FileFormat::Tsv) => {
            let o = csv::CsvWriteOptions { delimiter: '\t', ..Default::default() };
            csv::write_csv(wb.active_sheet()?, &o).into_bytes()
        }
        Some(FileFormat::GzipText) => {
            deflate::gzip_compress(csv::write_csv(wb.active_sheet()?, &csv::CsvWriteOptions::default()).as_bytes(), 6)
        }
        Some(FileFormat::Xlsx | FileFormat::Minitab) => {
            return Err(CoreError::invalid("saving in this format is not supported; use .otb, .json, .csv or .tsv"))
        }
    };
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffing_and_loading() {
        assert_eq!(FileFormat::from_path("a/b.CSV"), Some(FileFormat::Csv));
        assert_eq!(FileFormat::sniff(b"  {\"x\":1}"), FileFormat::Json);
        let wb = load_workbook(Some("C:\\data\\parts.csv"), b"\xEF\xBB\xBFa,b\n1,x\n").unwrap();
        assert_eq!(wb.worksheets[0].name, "parts");
        assert_eq!(wb.worksheets[0].n_cols(), 2);
        assert_eq!(decode_text(b"caf\xe9 \x80"), "café €");
        let gz = deflate::gzip_compress(b"a,b\n1,2\n", 6);
        let wb = load_workbook(Some("x.csv.gz"), &gz).unwrap();
        assert_eq!(wb.worksheets[0].n_cols(), 2);
        assert_eq!(FileFormat::sniff(include_bytes!("../../tests/fixtures/openpyxl_sample.xlsx")), FileFormat::Xlsx);
        let wb = load_workbook(None, include_bytes!("../../tests/fixtures/openpyxl_sample.xlsx")).unwrap();
        assert_eq!(wb.worksheets[0].name, "Measurements");
        let ole = cfb::write(&[("Data", b"x")], 3).unwrap();
        assert!(load_workbook(Some("old.mpj"), &ole).unwrap_err().to_string().contains("unpublished"));
    }

    #[test]
    fn file_roundtrip() {
        let dir = std::env::temp_dir();
        let mut wb = Workbook::new();
        wb.sheet_mut(0).unwrap().add_column(crate::Column::numeric("x", vec![1.0, f64::NAN])).unwrap();
        for ext in ["otb", "json"] {
            let p = dir.join(format!("openstats_io_test_{}.{ext}", std::process::id()));
            let ps = p.to_str().unwrap();
            save_workbook_file(&wb, ps).unwrap();
            assert_eq!(load_workbook_file(ps).unwrap(), wb);
            std::fs::remove_file(ps).unwrap();
        }
    }
}
