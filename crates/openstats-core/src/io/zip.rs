//! ZIP archive reader (with ZIP64) and writer (APPNOTE 6.3.x subset).
//!
//! Reading supports stored (0) and deflated (8) entries, ZIP64 end records
//! and per-entry ZIP64 extra fields, UTF-8 (flag bit 11) and CP437 names,
//! data descriptors, and CRC-32 verification. Encrypted entries are reported
//! as unsupported. Writing produces standard archives (stored or deflated per
//! entry, whichever is smaller) readable by every common tool.

use super::crc32::crc32;
use super::deflate;
use crate::error::{CoreError, Result};

fn err(m: impl Into<String>) -> CoreError {
    CoreError::Format(format!("zip: {}", m.into()))
}

const SIG_LOCAL: u32 = 0x0403_4b50;
const SIG_CENTRAL: u32 = 0x0201_4b50;
const SIG_EOCD: u32 = 0x0605_4b50;
const SIG_EOCD64: u32 = 0x0606_4b50;
const SIG_LOC64: u32 = 0x0706_4b50;

/// CP437 upper half (bytes 0x80–0xFF), used for names without the UTF-8 flag.
const CP437_HIGH: &str = "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{a0}";

fn cp437(bytes: &[u8]) -> String {
    let high: Vec<char> = CP437_HIGH.chars().collect();
    bytes.iter().map(|&b| if b < 0x80 { b as char } else { high[(b - 0x80) as usize] }).collect()
}

fn u16_at(b: &[u8], p: usize) -> Result<u16> {
    b.get(p..p + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| err("truncated archive"))
}
fn u32_at(b: &[u8], p: usize) -> Result<u32> {
    b.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().expect("4"))).ok_or_else(|| err("truncated archive"))
}
fn u64_at(b: &[u8], p: usize) -> Result<u64> {
    b.get(p..p + 8).map(|s| u64::from_le_bytes(s.try_into().expect("8"))).ok_or_else(|| err("truncated archive"))
}

/// Central-directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub flags: u16,
    pub crc32: u32,
    pub compressed_size: u64,
    pub size: u64,
    pub local_offset: u64,
}

impl ZipEntry {
    pub fn is_dir(&self) -> bool {
        self.name.ends_with('/')
    }
    pub fn is_encrypted(&self) -> bool {
        self.flags & 1 != 0
    }
}

/// A parsed archive borrowing its bytes.
#[derive(Debug, Clone)]
pub struct ZipArchive<'a> {
    data: &'a [u8],
    pub entries: Vec<ZipEntry>,
    pub comment: String,
}

impl<'a> ZipArchive<'a> {
    /// Parse the central directory.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 22 {
            return Err(err("too small to be a zip archive"));
        }
        // Locate the end-of-central-directory record (comment ≤ 65535 bytes).
        let min = data.len().saturating_sub(22 + 65_535);
        let mut eocd = None;
        let mut p = data.len() - 22;
        loop {
            if u32_at(data, p)? == SIG_EOCD && p + 22 + u16_at(data, p + 20)? as usize == data.len() {
                eocd = Some(p);
                break;
            }
            if p == min {
                break;
            }
            p -= 1;
        }
        let e = eocd.ok_or_else(|| err("end of central directory not found"))?;
        let comment_len = u16_at(data, e + 20)? as usize;
        let comment = String::from_utf8_lossy(&data[e + 22..e + 22 + comment_len]).into_owned();
        let mut count = u16_at(data, e + 10)? as u64;
        let mut cd_size = u32_at(data, e + 12)? as u64;
        let mut cd_off = u32_at(data, e + 16)? as u64;
        if (count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF)
            && e >= 20
            && u32_at(data, e - 20)? == SIG_LOC64
        {
            let z = u64_at(data, e - 20 + 8)? as usize;
            if u32_at(data, z)? != SIG_EOCD64 {
                return Err(err("bad ZIP64 end record"));
            }
            count = u64_at(data, z + 32)?;
            cd_size = u64_at(data, z + 40)?;
            cd_off = u64_at(data, z + 48)?;
        }
        if cd_off.checked_add(cd_size).is_none_or(|end| end > data.len() as u64) {
            return Err(err("central directory outside the file"));
        }
        if count > cd_size / 46 + 1 {
            return Err(err("implausible entry count"));
        }
        let mut entries = Vec::with_capacity(count as usize);
        let mut p = cd_off as usize;
        for _ in 0..count {
            if u32_at(data, p)? != SIG_CENTRAL {
                return Err(err("bad central directory entry"));
            }
            let flags = u16_at(data, p + 8)?;
            let method = u16_at(data, p + 10)?;
            let crc = u32_at(data, p + 16)?;
            let mut csize = u32_at(data, p + 20)? as u64;
            let mut size = u32_at(data, p + 24)? as u64;
            let nlen = u16_at(data, p + 28)? as usize;
            let xlen = u16_at(data, p + 30)? as usize;
            let clen = u16_at(data, p + 32)? as usize;
            let mut off = u32_at(data, p + 42)? as u64;
            let name_b = data.get(p + 46..p + 46 + nlen).ok_or_else(|| err("truncated name"))?;
            let name = if flags & (1 << 11) != 0 {
                String::from_utf8(name_b.to_vec()).map_err(|_| err("invalid UTF-8 name"))?
            } else {
                cp437(name_b)
            };
            // ZIP64 extended information: only fields saturated in the header appear, in order.
            let extra = data.get(p + 46 + nlen..p + 46 + nlen + xlen).ok_or_else(|| err("truncated extra field"))?;
            let mut q = 0;
            while q + 4 <= extra.len() {
                let id = u16_at(extra, q)?;
                let len = u16_at(extra, q + 2)? as usize;
                if id == 0x0001 {
                    let mut r = q + 4;
                    if size == 0xFFFF_FFFF {
                        size = u64_at(extra, r)?;
                        r += 8;
                    }
                    if csize == 0xFFFF_FFFF {
                        csize = u64_at(extra, r)?;
                        r += 8;
                    }
                    if off == 0xFFFF_FFFF {
                        off = u64_at(extra, r)?;
                    }
                }
                q += 4 + len;
            }
            entries.push(ZipEntry { name, method, flags, crc32: crc, compressed_size: csize, size, local_offset: off });
            p += 46 + nlen + xlen + clen;
        }
        Ok(ZipArchive { data, entries, comment })
    }

    /// Index of the entry with this exact name (also tries `\` ↔ `/` and a
    /// leading `/`, which some writers produce).
    pub fn find(&self, name: &str) -> Option<usize> {
        let norm = |s: &str| s.trim_start_matches('/').replace('\\', "/");
        let want = norm(name);
        self.entries.iter().position(|e| norm(&e.name) == want)
    }

    /// Case-insensitive lookup (OOXML part names are case-insensitive).
    pub fn find_ci(&self, name: &str) -> Option<usize> {
        let want = name.trim_start_matches('/').replace('\\', "/").to_lowercase();
        self.entries.iter().position(|e| e.name.trim_start_matches('/').replace('\\', "/").to_lowercase() == want)
    }

    /// Decompressed contents of entry `i` (CRC-checked). `limit` caps the
    /// output size; the declared size is always enforced.
    pub fn read_index(&self, i: usize, limit: Option<u64>) -> Result<Vec<u8>> {
        let e = self.entries.get(i).ok_or_else(|| err("no such entry"))?;
        if e.is_encrypted() {
            return Err(err(format!("entry {:?} is encrypted", e.name)));
        }
        if let Some(l) = limit {
            if e.size > l {
                return Err(err(format!("entry {:?} is larger than the {l}-byte limit", e.name)));
            }
        }
        let p = e.local_offset as usize;
        if u32_at(self.data, p)? != SIG_LOCAL {
            return Err(err("bad local header"));
        }
        let start = p + 30 + u16_at(self.data, p + 26)? as usize + u16_at(self.data, p + 28)? as usize;
        let end = start.checked_add(e.compressed_size as usize).ok_or_else(|| err("size overflow"))?;
        let raw = self.data.get(start..end).ok_or_else(|| err("entry data outside the file"))?;
        let out = match e.method {
            0 => raw.to_vec(),
            8 => deflate::inflate_raw(raw, Some(e.size as usize))?.0,
            m => return Err(err(format!("compression method {m} is not supported"))),
        };
        if out.len() as u64 != e.size {
            return Err(err(format!("entry {:?} has wrong size", e.name)));
        }
        if crc32(&out) != e.crc32 {
            return Err(err(format!("CRC mismatch in {:?}", e.name)));
        }
        Ok(out)
    }

    /// Decompressed contents by name.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let i = self
            .find(name)
            .or_else(|| self.find_ci(name))
            .ok_or_else(|| CoreError::NotFound(format!("zip entry {name:?}")))?;
        self.read_index(i, None)
    }
}

/// Builds an archive in memory.
#[derive(Debug, Default)]
pub struct ZipWriter {
    out: Vec<u8>,
    central: Vec<u8>,
    count: usize,
    names: std::collections::HashSet<String>,
}

impl ZipWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file. `level` 0 stores; otherwise deflate is used when smaller.
    pub fn add(&mut self, name: &str, data: &[u8], level: u8) -> Result<()> {
        if name.is_empty() || name.len() > 0xFFFF {
            return Err(err("invalid entry name"));
        }
        if !self.names.insert(name.to_string()) {
            return Err(CoreError::Duplicate(name.to_string()));
        }
        if data.len() as u64 >= 0xFFFF_FFFF || self.out.len() as u64 >= 0xFFFF_FFFF || self.count >= 0xFFFF {
            return Err(err("archive would need ZIP64, which the writer does not produce"));
        }
        let crc = crc32(data);
        let compressed = if level > 0 && !data.is_empty() { Some(deflate::deflate(data, level)) } else { None };
        let (method, payload): (u16, &[u8]) = match &compressed {
            Some(c) if c.len() < data.len() => (8, c),
            _ => (0, data),
        };
        let flags: u16 = if name.is_ascii() { 0 } else { 1 << 11 };
        let offset = self.out.len() as u32;
        // DOS time/date: 1980-01-01 00:00 (deterministic output).
        let (time, date): (u16, u16) = (0, (1 << 5) | 1);
        let mut lh = Vec::with_capacity(30 + name.len());
        lh.extend_from_slice(&SIG_LOCAL.to_le_bytes());
        lh.extend_from_slice(&20u16.to_le_bytes());
        lh.extend_from_slice(&flags.to_le_bytes());
        lh.extend_from_slice(&method.to_le_bytes());
        lh.extend_from_slice(&time.to_le_bytes());
        lh.extend_from_slice(&date.to_le_bytes());
        lh.extend_from_slice(&crc.to_le_bytes());
        lh.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        lh.extend_from_slice(&(data.len() as u32).to_le_bytes());
        lh.extend_from_slice(&(name.len() as u16).to_le_bytes());
        lh.extend_from_slice(&0u16.to_le_bytes());
        lh.extend_from_slice(name.as_bytes());
        self.out.extend_from_slice(&lh);
        self.out.extend_from_slice(payload);

        let c = &mut self.central;
        c.extend_from_slice(&SIG_CENTRAL.to_le_bytes());
        c.extend_from_slice(&20u16.to_le_bytes()); // made by
        c.extend_from_slice(&20u16.to_le_bytes()); // needed
        c.extend_from_slice(&flags.to_le_bytes());
        c.extend_from_slice(&method.to_le_bytes());
        c.extend_from_slice(&time.to_le_bytes());
        c.extend_from_slice(&date.to_le_bytes());
        c.extend_from_slice(&crc.to_le_bytes());
        c.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        c.extend_from_slice(&(data.len() as u32).to_le_bytes());
        c.extend_from_slice(&(name.len() as u16).to_le_bytes());
        c.extend_from_slice(&[0; 8]); // extra len, comment len, disk, internal attrs
        c.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        c.extend_from_slice(&offset.to_le_bytes());
        c.extend_from_slice(name.as_bytes());
        self.count += 1;
        Ok(())
    }

    /// Write the central directory and return the archive bytes.
    pub fn finish(mut self) -> Vec<u8> {
        let cd_off = self.out.len() as u32;
        let cd_len = self.central.len() as u32;
        self.out.extend_from_slice(&self.central);
        self.out.extend_from_slice(&SIG_EOCD.to_le_bytes());
        self.out.extend_from_slice(&[0; 4]);
        self.out.extend_from_slice(&(self.count as u16).to_le_bytes());
        self.out.extend_from_slice(&(self.count as u16).to_le_bytes());
        self.out.extend_from_slice(&cd_len.to_le_bytes());
        self.out.extend_from_slice(&cd_off.to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes());
        self.out
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod vectors {
    include!("zip_vectors.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_python_archive() {
        let z = ZipArchive::new(vectors::BASIC).unwrap();
        let names: Vec<&str> = z.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["docs/", "stored.txt", "deflated.csv", "ünïcode/名前.txt"]);
        assert!(z.entries[0].is_dir());
        assert_eq!(z.comment, "archive comment");
        assert_eq!(z.read("stored.txt").unwrap(), b"plain stored bytes");
        assert_eq!(z.entries[2].method, 8);
        let csv = String::from_utf8(z.read("deflated.csv").unwrap()).unwrap();
        assert!(csv.starts_with("x,y\n0,0\n1,1\n") && csv.ends_with("299,89401\n"));
        assert_eq!(z.read("ünïcode/名前.txt").unwrap(), b"utf-8 name");
        assert_eq!(z.find_ci("DEFLATED.CSV"), Some(2));
        assert!(z.read("missing").is_err());
    }

    #[test]
    fn reads_zip64_entry() {
        let z = ZipArchive::new(vectors::ZIP64).unwrap();
        assert_eq!(z.entries.len(), 1);
        assert_eq!(z.read("big.txt").unwrap(), b"zip64 sized entry ".repeat(50));
    }

    #[test]
    fn writer_roundtrip_and_validation() {
        let mut w = ZipWriter::new();
        let big = b"0123456789".repeat(1000);
        w.add("a.txt", b"short", 6).unwrap();
        w.add("dir/b.bin", &big, 9).unwrap();
        w.add("é.txt", b"", 6).unwrap();
        assert!(matches!(w.add("a.txt", b"x", 0), Err(CoreError::Duplicate(_))));
        let bytes = w.finish();
        let z = ZipArchive::new(&bytes).unwrap();
        assert_eq!(z.entries[0].method, 0, "short data is stored");
        assert_eq!(z.entries[1].method, 8);
        assert_eq!(z.read("dir/b.bin").unwrap(), big);
        assert_eq!(z.read("é.txt").unwrap(), b"");
        // Corruption: flip a byte in the deflated payload → CRC or inflate error.
        let mut bad = bytes.clone();
        let off = z.entries[1].local_offset as usize + 30 + "dir/b.bin".len() + 5;
        bad[off] ^= 0xFF;
        assert!(ZipArchive::new(&bad).unwrap().read("dir/b.bin").is_err());
        assert!(ZipArchive::new(b"not a zip file at all, definitely").is_err());
        assert!(z.read_index(1, Some(100)).is_err(), "size limit");
    }

    #[test]
    fn cp437_names() {
        assert_eq!(CP437_HIGH.chars().count(), 128);
        assert_eq!(cp437(&[b'a', 0x81, 0xE1]), "aüß");
    }
}
