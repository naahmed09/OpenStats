//! Minitab file detection and container inspection.
//!
//! | Extension | Minitab version | Container |
//! |---|---|---|
//! | `.mpx` (project), `.mwx` (worksheet) | 19 and later | ZIP |
//! | `.mpj` (project), `.mtw` (worksheet) | 18 and earlier | OLE2 compound file (or older proprietary binary) |
//!
//! Minitab publishes no specification for the parts *inside* these
//! containers. This module therefore opens the container with the
//! from-scratch [`zip`](super::zip) and [`cfb`](super::cfb) readers and
//! reports its structure (part names, sizes and, for text parts, a short
//! preview). That makes it possible to write a worksheet decoder against real
//! sample files. Loading a Minitab file through
//! [`load_workbook`](super::load_workbook) returns an error that includes this
//! inspection report, so data is never imported silently or guessed.

use super::cfb::Cfb;
use super::zip::ZipArchive;
use crate::error::{CoreError, Result};
use std::fmt;

/// Which Minitab file type the name/content indicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinitabFormat {
    /// Project, Minitab 19+.
    Mpx,
    /// Worksheet, Minitab 19+.
    Mwx,
    /// Project, Minitab ≤ 18.
    Mpj,
    /// Worksheet, Minitab ≤ 18.
    Mtw,
}

impl MinitabFormat {
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = path.rsplit('.').next()?.to_ascii_lowercase();
        match ext.as_str() {
            "mpx" => Some(Self::Mpx),
            "mwx" => Some(Self::Mwx),
            "mpj" => Some(Self::Mpj),
            "mtw" => Some(Self::Mtw),
            _ => None,
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Self::Mpx => "Minitab project (.mpx, Minitab 19+)",
            Self::Mwx => "Minitab worksheet (.mwx, Minitab 19+)",
            Self::Mpj => "Minitab project (.mpj, Minitab 18 and earlier)",
            Self::Mtw => "Minitab worksheet (.mtw, Minitab 18 and earlier)",
        }
    }
}

/// Physical container of the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Zip,
    CompoundFile,
    /// Neither ZIP nor OLE2 (e.g. very old Minitab binary worksheets).
    Unknown,
}

/// One part (ZIP entry or compound-file stream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub path: String,
    pub size: u64,
    /// First characters of the part when it decodes as text (XML, JSON, …).
    pub text_preview: Option<String>,
}

/// Result of [`inspect`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inspection {
    pub format: Option<MinitabFormat>,
    pub container: Container,
    pub parts: Vec<Part>,
}

impl fmt::Display for Inspection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = self.format.map_or("file", MinitabFormat::description);
        writeln!(f, "{what}; container: {:?}; {} part(s)", self.container, self.parts.len())?;
        for p in self.parts.iter().take(50) {
            write!(f, "  {} ({} bytes)", p.path, p.size)?;
            if let Some(t) = &p.text_preview {
                write!(f, ": {t}")?;
            }
            writeln!(f)?;
        }
        if self.parts.len() > 50 {
            writeln!(f, "  … {} more", self.parts.len() - 50)?;
        }
        Ok(())
    }
}

const PREVIEW_BYTES: usize = 120;

fn preview(bytes: &[u8]) -> Option<String> {
    let head = &bytes[..bytes.len().min(PREVIEW_BYTES * 4)];
    let text = match std::str::from_utf8(head) {
        Ok(t) => t.to_string(),
        // Cut inside a multi-byte sequence at the end: retry on the valid prefix.
        Err(e) if e.valid_up_to() > 0 && head.len() - e.valid_up_to() < 4 => {
            std::str::from_utf8(&head[..e.valid_up_to()]).ok()?.to_string()
        }
        Err(_) => return None,
    };
    if text.chars().any(|c| c.is_control() && !c.is_whitespace()) {
        return None;
    }
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(one_line.chars().take(PREVIEW_BYTES).collect())
}

/// Identify the container and list its parts. Part contents are read only to
/// build short text previews (at most 64 MiB per part).
pub fn inspect(name: Option<&str>, bytes: &[u8]) -> Result<Inspection> {
    let format = name.and_then(MinitabFormat::from_path);
    if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
        let z = ZipArchive::new(bytes)?;
        let parts = z
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.is_dir())
            .map(|(i, e)| Part {
                path: e.name.clone(),
                size: e.size,
                text_preview: z.read_index(i, Some(64 << 20)).ok().and_then(|b| preview(&b)),
            })
            .collect();
        return Ok(Inspection { format, container: Container::Zip, parts });
    }
    if Cfb::is_cfb(bytes) {
        let c = Cfb::new(bytes)?;
        let parts = c
            .entries()
            .into_iter()
            .filter(|e| e.kind == super::cfb::EntryKind::Stream)
            .map(|e| Part {
                text_preview: if e.size <= 64 << 20 { c.read(&e.path).ok().and_then(|b| preview(&b)) } else { None },
                path: e.path,
                size: e.size,
            })
            .collect();
        return Ok(Inspection { format, container: Container::CompoundFile, parts });
    }
    Ok(Inspection { format, container: Container::Unknown, parts: Vec::new() })
}

/// The error returned when a Minitab file is opened: explains the limitation
/// and embeds the inspection report.
pub fn unsupported_error(name: Option<&str>, bytes: &[u8]) -> CoreError {
    let report = match inspect(name, bytes) {
        Ok(i) => i.to_string(),
        Err(e) => format!("container could not be read: {e}"),
    };
    CoreError::Format(format!(
        "Minitab's internal worksheet layout is unpublished, so OpenStats cannot import this file's data yet. \
         Export it from Minitab as .xlsx or .csv, or share a sample so a decoder can be built for it.\n{report}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{cfb, zip::ZipWriter};

    #[test]
    fn inspects_zip_and_cfb_containers() {
        let mut z = ZipWriter::new();
        z.add("worksheet.xml", b"<ws><col name=\"C1\"/></ws>", 6).unwrap();
        z.add("blob.bin", &[0u8, 1, 2, 3, 255], 0).unwrap();
        let zip = z.finish();
        let i = inspect(Some("project.MPX"), &zip).unwrap();
        assert_eq!(i.format, Some(MinitabFormat::Mpx));
        assert_eq!(i.container, Container::Zip);
        assert_eq!(i.parts[0].text_preview.as_deref(), Some("<ws><col name=\"C1\"/></ws>"));
        assert_eq!(i.parts[1].text_preview, None);
        assert!(i.to_string().contains("worksheet.xml (25 bytes)"));

        let ole = cfb::write(&[("Worksheet", b"\x01\x02binary"), ("Info", b"Minitab 17")], 3).unwrap();
        let i = inspect(Some("old.mtw"), &ole).unwrap();
        assert_eq!(i.container, Container::CompoundFile);
        assert_eq!(i.parts.len(), 2);
        assert_eq!(i.parts[0].path, "Info");
        assert_eq!(i.parts[0].text_preview.as_deref(), Some("Minitab 17"));

        let raw = inspect(Some("ancient.mtw"), b"\x00\x00MTW binary").unwrap();
        assert_eq!(raw.container, Container::Unknown);
        let e = unsupported_error(Some("x.mpx"), &zip).to_string();
        assert!(e.contains("unpublished") && e.contains("worksheet.xml"));
    }
}
