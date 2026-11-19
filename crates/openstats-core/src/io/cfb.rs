//! Microsoft Compound File Binary format ([MS-CFB]), a.k.a. OLE2 structured
//! storage: the container used by legacy Office files and older Minitab
//! project/worksheet files.
//!
//! The reader supports versions 3 (512-byte sectors) and 4 (4096-byte
//! sectors), DIFAT chains, the mini stream, nested storages, and validates
//! every chain against the file size with loop detection. The writer
//! produces spec-conformant files with a flat set of streams (small streams
//! go to the mini stream), used for round-trip tests and export.

use crate::error::{CoreError, Result};
use std::cmp::Ordering;

pub const SIGNATURE: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
const FREESECT: u32 = 0xFFFF_FFFF;
const ENDOFCHAIN: u32 = 0xFFFF_FFFE;
const FATSECT: u32 = 0xFFFF_FFFD;
const NOSTREAM: u32 = 0xFFFF_FFFF;
const MINI_SECTOR: usize = 64;
const MINI_CUTOFF: u64 = 4096;

fn err(m: impl Into<String>) -> CoreError {
    CoreError::Format(format!("compound file: {}", m.into()))
}

fn rd16(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}
fn rd32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().expect("4"))
}
fn rd64(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().expect("8"))
}

/// Directory entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Storage,
    Stream,
    Root,
}

#[derive(Debug, Clone)]
struct Dir {
    name: String,
    kind: Option<EntryKind>,
    left: u32,
    right: u32,
    child: u32,
    start: u32,
    size: u64,
}

/// A stream or storage found in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfbEntry {
    /// Path from the root, `/`-separated (`Storage/Stream`).
    pub path: String,
    pub kind: EntryKind,
    pub size: u64,
}

/// A parsed compound file.
#[derive(Debug, Clone)]
pub struct Cfb<'a> {
    data: &'a [u8],
    sector_size: usize,
    fat: Vec<u32>,
    minifat: Vec<u32>,
    dir: Vec<Dir>,
    ministream: Vec<u8>,
    entries: Vec<(CfbEntry, usize)>,
}

/// CFB name ordering: shorter names first, then case-insensitive code units.
fn cfb_cmp(a: &str, b: &str) -> Ordering {
    let (ua, ub): (Vec<u16>, Vec<u16>) = (a.encode_utf16().collect(), b.encode_utf16().collect());
    ua.len().cmp(&ub.len()).then_with(|| a.to_uppercase().cmp(&b.to_uppercase()))
}

impl<'a> Cfb<'a> {
    /// `true` if the bytes start with the compound-file signature.
    pub fn is_cfb(bytes: &[u8]) -> bool {
        bytes.starts_with(&SIGNATURE)
    }

    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 512 || !Self::is_cfb(data) {
            return Err(err("not a compound file"));
        }
        let major = rd16(data, 26);
        let shift = rd16(data, 30);
        let sector_size = match (major, shift) {
            (3, 9) => 512,
            (4, 12) => 4096,
            _ => return Err(err(format!("unsupported version {major} / sector shift {shift}"))),
        };
        if rd16(data, 28) != 0xFFFE || rd16(data, 32) != 6 {
            return Err(err("bad byte order or mini sector shift"));
        }
        let n_sectors = (data.len().saturating_sub(sector_size)).div_ceil(sector_size);
        let sector = |n: u32| -> Result<&'a [u8]> {
            let off = (n as usize)
                .checked_add(1)
                .and_then(|k| k.checked_mul(sector_size))
                .ok_or_else(|| err("sector index overflow"))?;
            if n as usize >= n_sectors || off >= data.len() {
                return Err(err(format!("sector {n} outside the file")));
            }
            // The last sector may be truncated in files written by some tools.
            Ok(&data[off..(off + sector_size).min(data.len())])
        };
        let per = sector_size / 4;

        // DIFAT: 109 header entries plus chained DIFAT sectors.
        let n_fat = rd32(data, 44) as usize;
        let mut difat: Vec<u32> = (0..109).map(|i| rd32(data, 76 + 4 * i)).collect();
        let mut next = rd32(data, 68);
        let mut guard = 0;
        while next != ENDOFCHAIN && next != FREESECT {
            let s = sector(next)?;
            if s.len() < sector_size {
                return Err(err("truncated DIFAT sector"));
            }
            difat.extend((0..per - 1).map(|i| rd32(s, 4 * i)));
            next = rd32(s, 4 * (per - 1));
            guard += 1;
            if guard > n_sectors {
                return Err(err("DIFAT chain loops"));
            }
        }
        if n_fat > difat.len() {
            return Err(err("header declares more FAT sectors than the DIFAT lists"));
        }
        let mut fat = Vec::with_capacity(n_fat * per);
        for &fs in difat.iter().take(n_fat) {
            let s = sector(fs)?;
            fat.extend((0..s.len() / 4).map(|i| rd32(s, 4 * i)));
        }
        let read_chain = |start: u32, fat: &[u32], max_len: Option<u64>| -> Result<Vec<u8>> {
            let mut out = Vec::new();
            let mut cur = start;
            let mut seen = vec![false; fat.len()];
            while cur != ENDOFCHAIN {
                if cur == FREESECT || cur as usize >= fat.len() {
                    return Err(err(format!("broken sector chain at {cur:#x}")));
                }
                if std::mem::replace(&mut seen[cur as usize], true) {
                    return Err(err("sector chain loops"));
                }
                out.extend_from_slice(sector(cur)?);
                cur = fat[cur as usize];
                if let Some(m) = max_len {
                    if out.len() as u64 >= m {
                        break;
                    }
                }
            }
            Ok(out)
        };

        // Directory.
        let dir_bytes = read_chain(rd32(data, 48), &fat, None)?;
        let mut dir = Vec::with_capacity(dir_bytes.len() / 128);
        for e in dir_bytes.chunks_exact(128) {
            let name_len = (rd16(e, 64) as usize).min(64);
            let units: Vec<u16> = (0..name_len.saturating_sub(2) / 2).map(|i| rd16(e, 2 * i)).collect();
            let kind = match e[66] {
                1 => Some(EntryKind::Storage),
                2 => Some(EntryKind::Stream),
                5 => Some(EntryKind::Root),
                _ => None,
            };
            let mut size = rd64(e, 120);
            if sector_size == 512 {
                size &= 0xFFFF_FFFF; // v3 writers may leave garbage in the high half
            }
            dir.push(Dir {
                name: String::from_utf16_lossy(&units),
                kind,
                left: rd32(e, 68),
                right: rd32(e, 72),
                child: rd32(e, 76),
                start: rd32(e, 116),
                size,
            });
        }
        if dir.first().and_then(|d| d.kind) != Some(EntryKind::Root) {
            return Err(err("first directory entry is not the root"));
        }

        // Mini FAT and mini stream.
        let mut minifat = Vec::new();
        if rd32(data, 64) > 0 {
            let mf = read_chain(rd32(data, 60), &fat, None)?;
            minifat = (0..mf.len() / 4).map(|i| rd32(&mf, 4 * i)).collect();
        }
        let root = &dir[0];
        let ministream = if root.start != ENDOFCHAIN && root.size > 0 {
            let mut m = read_chain(root.start, &fat, Some(root.size))?;
            m.truncate(root.size as usize);
            m
        } else {
            Vec::new()
        };

        let mut cfb = Cfb { data, sector_size, fat, minifat, dir, ministream, entries: Vec::new() };
        cfb.entries = cfb.walk()?;
        Ok(cfb)
    }

    /// In-order traversal of the red-black sibling trees.
    fn walk(&self) -> Result<Vec<(CfbEntry, usize)>> {
        let mut out = Vec::new();
        let mut visited = vec![false; self.dir.len()];
        // Explicit stack: (node, prefix, phase) to avoid deep recursion.
        fn visit(
            cfb: &Cfb,
            node: u32,
            prefix: &str,
            visited: &mut [bool],
            out: &mut Vec<(CfbEntry, usize)>,
            depth: usize,
        ) -> Result<()> {
            if node == NOSTREAM {
                return Ok(());
            }
            let i = node as usize;
            if i >= cfb.dir.len() || visited[i] || depth > 512 {
                return Err(err("corrupt directory tree"));
            }
            visited[i] = true;
            let d = &cfb.dir[i];
            visit(cfb, d.left, prefix, visited, out, depth + 1)?;
            let path = if prefix.is_empty() { d.name.clone() } else { format!("{prefix}/{}", d.name) };
            match d.kind {
                Some(EntryKind::Stream) => out.push((CfbEntry { path, kind: EntryKind::Stream, size: d.size }, i)),
                Some(EntryKind::Storage) => {
                    out.push((CfbEntry { path: path.clone(), kind: EntryKind::Storage, size: 0 }, i));
                    visit(cfb, d.child, &path, visited, out, depth + 1)?;
                }
                _ => {}
            }
            visit(cfb, d.right, prefix, visited, out, depth + 1)
        }
        visited[0] = true;
        visit(self, self.dir[0].child, "", &mut visited, &mut out, 0)?;
        Ok(out)
    }

    /// All storages and streams, depth-first in directory order.
    pub fn entries(&self) -> Vec<CfbEntry> {
        self.entries.iter().map(|(e, _)| e.clone()).collect()
    }

    /// Read a stream by path (case-insensitive per component).
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        let (_, idx) = self
            .entries
            .iter()
            .find(|(e, _)| e.kind == EntryKind::Stream && e.path.to_uppercase() == path.to_uppercase())
            .ok_or_else(|| CoreError::NotFound(format!("stream {path:?}")))?;
        let d = &self.dir[*idx];
        let size = d.size as usize;
        let mut out = Vec::with_capacity(size);
        if d.size < MINI_CUTOFF {
            let mut cur = d.start;
            let mut seen = vec![false; self.minifat.len()];
            while out.len() < size {
                if cur == ENDOFCHAIN || cur as usize >= self.minifat.len() {
                    return Err(err("broken mini stream chain"));
                }
                if std::mem::replace(&mut seen[cur as usize], true) {
                    return Err(err("mini chain loops"));
                }
                let off = cur as usize * MINI_SECTOR;
                let chunk = self
                    .ministream
                    .get(off..off + MINI_SECTOR)
                    .ok_or_else(|| err("mini sector outside the mini stream"))?;
                out.extend_from_slice(chunk);
                cur = self.minifat[cur as usize];
            }
        } else {
            let mut cur = d.start;
            let mut seen = vec![false; self.fat.len()];
            while out.len() < size {
                if cur == ENDOFCHAIN || cur as usize >= self.fat.len() {
                    return Err(err("stream chain ends early"));
                }
                if std::mem::replace(&mut seen[cur as usize], true) {
                    return Err(err("stream chain loops"));
                }
                let off = (cur as usize + 1) * self.sector_size;
                let s = self
                    .data
                    .get(off..(off + self.sector_size).min(self.data.len()))
                    .ok_or_else(|| err("sector outside the file"))?;
                out.extend_from_slice(s);
                cur = self.fat[cur as usize];
            }
        }
        out.truncate(size);
        Ok(out)
    }
}

/// Write a compound file containing the given (name, data) streams in the
/// root storage. `version` is 3 (512-byte sectors) or 4 (4096-byte sectors).
pub fn write(streams: &[(&str, &[u8])], version: u16) -> Result<Vec<u8>> {
    let ss: usize = match version {
        3 => 512,
        4 => 4096,
        _ => return Err(err("version must be 3 or 4")),
    };
    for (i, (name, _)) in streams.iter().enumerate() {
        if name.is_empty() || name.encode_utf16().count() > 31 || name.contains(['/', '\\', ':', '!']) {
            return Err(err(format!("invalid stream name {name:?}")));
        }
        if streams[..i].iter().any(|(n, _)| cfb_cmp(n, name) == Ordering::Equal) {
            return Err(CoreError::Duplicate((*name).to_string()));
        }
    }
    let mut sectors: Vec<Vec<u8>> = Vec::new();
    let mut fat: Vec<u32> = Vec::new();
    let alloc = |bytes: &[u8], sectors: &mut Vec<Vec<u8>>, fat: &mut Vec<u32>| -> u32 {
        if bytes.is_empty() {
            return ENDOFCHAIN;
        }
        let start = sectors.len() as u32;
        for (k, chunk) in bytes.chunks(ss).enumerate() {
            let mut s = chunk.to_vec();
            s.resize(ss, 0);
            sectors.push(s);
            fat.push(if (k + 1) * ss < bytes.len() { start + k as u32 + 1 } else { ENDOFCHAIN });
        }
        start
    };

    // Mini stream for small streams.
    let mut mini = Vec::new();
    let mut minifat: Vec<u32> = Vec::new();
    let mut starts = vec![ENDOFCHAIN; streams.len()];
    for (i, (_, data)) in streams.iter().enumerate() {
        if (data.len() as u64) < MINI_CUTOFF {
            if data.is_empty() {
                continue;
            }
            let first = (mini.len() / MINI_SECTOR) as u32;
            let n = data.len().div_ceil(MINI_SECTOR);
            for k in 0..n {
                minifat.push(if k + 1 < n { first + k as u32 + 1 } else { ENDOFCHAIN });
            }
            mini.extend_from_slice(data);
            mini.resize(mini.len().div_ceil(MINI_SECTOR) * MINI_SECTOR, 0);
            starts[i] = first;
        } else {
            starts[i] = alloc(data, &mut sectors, &mut fat);
        }
    }
    let mini_start = alloc(&mini, &mut sectors, &mut fat);
    let mut minifat_bytes: Vec<u8> = minifat.iter().flat_map(|v| v.to_le_bytes()).collect();
    if !minifat_bytes.is_empty() {
        minifat_bytes.resize(minifat_bytes.len().div_ceil(ss) * ss, 0xFF);
    }
    let minifat_start = alloc(&minifat_bytes, &mut sectors, &mut fat);
    let n_minifat = minifat_bytes.len() / ss;

    // Directory: balanced binary tree over the sorted names (all black).
    let mut order: Vec<usize> = (0..streams.len()).collect();
    order.sort_by(|&a, &b| cfb_cmp(streams[a].0, streams[b].0));
    let n = streams.len();
    let mut left = vec![NOSTREAM; n + 1];
    let mut right = vec![NOSTREAM; n + 1];
    fn build(lo: usize, hi: usize, order: &[usize], left: &mut [u32], right: &mut [u32]) -> u32 {
        if lo >= hi {
            return NOSTREAM;
        }
        let mid = (lo + hi) / 2;
        let id = order[mid] + 1; // directory id (0 is the root)
        left[id] = build(lo, mid, order, left, right);
        right[id] = build(mid + 1, hi, order, left, right);
        id as u32
    }
    let root_child = build(0, n, &order, &mut left, &mut right);
    let entry = |name: &str, kind: u8, l: u32, r: u32, c: u32, start: u32, size: u64| -> Vec<u8> {
        let mut e = vec![0u8; 128];
        let units: Vec<u16> = name.encode_utf16().collect();
        for (k, u) in units.iter().enumerate() {
            e[2 * k..2 * k + 2].copy_from_slice(&u.to_le_bytes());
        }
        e[64..66].copy_from_slice(&(((units.len() + 1) * 2) as u16).to_le_bytes());
        e[66] = kind;
        e[67] = 1; // black
        e[68..72].copy_from_slice(&l.to_le_bytes());
        e[72..76].copy_from_slice(&r.to_le_bytes());
        e[76..80].copy_from_slice(&c.to_le_bytes());
        e[116..120].copy_from_slice(&start.to_le_bytes());
        e[120..128].copy_from_slice(&size.to_le_bytes());
        e
    };
    let mut dir = entry("Root Entry", 5, NOSTREAM, NOSTREAM, root_child, mini_start, mini.len() as u64);
    for (i, (name, data)) in streams.iter().enumerate() {
        dir.extend(entry(name, 2, left[i + 1], right[i + 1], NOSTREAM, starts[i], data.len() as u64));
    }
    while dir.len() % ss != 0 {
        let mut empty = vec![0u8; 128];
        empty[68..80].fill(0xFF);
        dir.extend(empty);
    }
    let dir_start = alloc(&dir, &mut sectors, &mut fat);
    let n_dir = dir.len() / ss;

    // FAT sectors go last; each holds ss/4 entries including its own.
    let per = ss / 4;
    let data_sectors = sectors.len();
    let mut n_fat = 1;
    while n_fat * per < data_sectors + n_fat {
        n_fat += 1;
    }
    if n_fat > 109 {
        return Err(err("file too large for the writer (more than 109 FAT sectors)"));
    }
    fat.resize(data_sectors + n_fat, FATSECT);
    fat.resize(n_fat * per, FREESECT);

    let mut out = vec![0u8; ss];
    out[..8].copy_from_slice(&SIGNATURE);
    out[24..26].copy_from_slice(&0x003Eu16.to_le_bytes());
    out[26..28].copy_from_slice(&version.to_le_bytes());
    out[28..30].copy_from_slice(&0xFFFEu16.to_le_bytes());
    out[30..32].copy_from_slice(&(if ss == 512 { 9u16 } else { 12 }).to_le_bytes());
    out[32..34].copy_from_slice(&6u16.to_le_bytes());
    out[40..44].copy_from_slice(&(if version == 4 { n_dir as u32 } else { 0 }).to_le_bytes());
    out[44..48].copy_from_slice(&(n_fat as u32).to_le_bytes());
    out[48..52].copy_from_slice(&dir_start.to_le_bytes());
    out[56..60].copy_from_slice(&(MINI_CUTOFF as u32).to_le_bytes());
    out[60..64].copy_from_slice(&minifat_start.to_le_bytes());
    out[64..68].copy_from_slice(&(n_minifat as u32).to_le_bytes());
    out[68..72].copy_from_slice(&ENDOFCHAIN.to_le_bytes());
    for k in 0..109 {
        let v = if k < n_fat { (data_sectors + k) as u32 } else { FREESECT };
        out[76 + 4 * k..80 + 4 * k].copy_from_slice(&v.to_le_bytes());
    }
    for s in &sectors {
        out.extend_from_slice(s);
    }
    for chunk in fat.chunks(per) {
        for v in chunk {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
    }

    #[test]
    fn roundtrip_v3_and_v4() {
        let big = payload(70_000, 3); // spans many regular sectors
        let small = payload(100, 7); // mini stream, 2 mini sectors
        let edge = payload(4096, 9); // exactly the cutoff → regular
        let names = ["Worksheet", "Big Data", "Ünïcode", "edge", "empty", "a"];
        let datas: [&[u8]; 6] = [&small, &big, b"text", &edge, b"", b"z"];
        let streams: Vec<(&str, &[u8])> = names.iter().copied().zip(datas.iter().copied()).collect();
        for version in [3u16, 4] {
            let bytes = write(&streams, version).unwrap();
            assert!(Cfb::is_cfb(&bytes));
            let c = Cfb::new(&bytes).unwrap();
            let listed: Vec<String> = c.entries().into_iter().map(|e| e.path).collect();
            // Directory order is the CFB name order (length first).
            assert_eq!(listed, ["a", "edge", "empty", "Ünïcode", "Big Data", "Worksheet"]);
            for (n, d) in &streams {
                assert_eq!(&c.read(n).unwrap(), d, "v{version} stream {n}");
            }
            assert_eq!(c.read("WORKSHEET").unwrap(), small);
            assert!(c.read("nope").is_err());
        }
    }

    #[test]
    fn many_streams_and_validation() {
        let names: Vec<String> = (0..200).map(|i| format!("S{i:03}")).collect();
        let datas: Vec<Vec<u8>> = (0..200).map(|i| payload(i * 37, i as u8)).collect();
        let streams: Vec<(&str, &[u8])> =
            names.iter().map(String::as_str).zip(datas.iter().map(Vec::as_slice)).collect();
        let bytes = write(&streams, 3).unwrap();
        let c = Cfb::new(&bytes).unwrap();
        assert_eq!(c.entries().len(), 200);
        assert_eq!(c.read("S150").unwrap(), datas[150]);
        assert!(write(&[("a/b", b"x")], 3).is_err());
        assert!(write(&[("A", b"x"), ("a", b"y")], 3).is_err());
        assert!(write(&[("x", b"")], 5).is_err());
    }

    #[test]
    fn corruption_detected() {
        let bytes = write(&[("s", &payload(9000, 1)[..])], 3).unwrap();
        assert!(Cfb::new(&bytes[..300]).is_err());
        let mut bad = bytes.clone();
        bad[0] = 0;
        assert!(Cfb::new(&bad).is_err());
        // Make the FAT entry of sector 0 point to itself → loop.
        let mut looped = bytes.clone();
        let fat_sector = rd32(&bytes, 76) as usize;
        let off = (fat_sector + 1) * 512;
        looped[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let r = Cfb::new(&looped).and_then(|c| c.read("s"));
        assert!(r.is_err());
        // Random garbage after a valid signature must not panic.
        let mut junk = vec![0x5Au8; 4096];
        junk[..8].copy_from_slice(&SIGNATURE);
        junk[26] = 3;
        junk[28] = 0xFE;
        junk[29] = 0xFF;
        junk[30] = 9;
        junk[32] = 6;
        let _ = Cfb::new(&junk);
    }
}
