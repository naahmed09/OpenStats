//! DEFLATE (RFC 1951) decompressor and compressor, plus the zlib (RFC 1950)
//! and gzip (RFC 1952) wrappers.
//!
//! * **Inflate** handles stored, fixed-Huffman and dynamic-Huffman blocks with
//!   table-driven decoding, validates every code table (over-subscribed or
//!   invalid codes are errors), rejects distances reaching before the start of
//!   output, and supports an output-size limit to defuse decompression bombs.
//! * **Deflate** performs LZ77 matching with hash chains (zlib-style level
//!   parameters, lazy matching from level 4), builds length-limited canonical
//!   Huffman codes, and for every block emits whichever of stored, fixed or
//!   dynamic encoding is smallest.

use super::crc32::crc32;
use crate::error::{CoreError, Result};

fn err(m: impl Into<String>) -> CoreError {
    CoreError::Format(format!("deflate: {}", m.into()))
}

// ---------------------------------------------------------------------------
// Shared tables
// ---------------------------------------------------------------------------

const LEN_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn fixed_litlen_lengths() -> [u8; 288] {
    let mut l = [0u8; 288];
    for (i, v) in l.iter_mut().enumerate() {
        *v = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    l
}

fn reverse_bits(mut code: u32, len: u32) -> u32 {
    let mut r = 0;
    for _ in 0..len {
        r = (r << 1) | (code & 1);
        code >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Inflate
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
    /// Zero bytes appended past the end while peeking (must never be consumed).
    overrun: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, buf: 0, cnt: 0, overrun: 0 }
    }
    #[inline]
    fn refill(&mut self) {
        while self.cnt <= 56 {
            let byte = if self.pos < self.data.len() {
                let b = self.data[self.pos];
                self.pos += 1;
                b
            } else {
                self.overrun += 8;
                0
            };
            self.buf |= (byte as u64) << self.cnt;
            self.cnt += 8;
        }
    }
    #[inline]
    fn peek(&mut self, n: u32) -> u32 {
        if self.cnt < n {
            self.refill();
        }
        (self.buf & ((1u64 << n) - 1)) as u32
    }
    #[inline]
    fn consume(&mut self, n: u32) -> Result<()> {
        if n > self.cnt {
            return Err(err("internal bit-buffer underflow"));
        }
        self.buf >>= n;
        self.cnt -= n;
        if self.overrun > self.cnt {
            return Err(err("unexpected end of compressed data"));
        }
        Ok(())
    }
    #[inline]
    fn bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.peek(n);
        self.consume(n)?;
        Ok(v)
    }
    fn align(&mut self) -> Result<()> {
        let r = self.cnt % 8;
        self.consume(r)
    }
    /// Bytes of input fully consumed so far.
    fn consumed(&self) -> usize {
        (self.pos + self.overrun as usize / 8) - self.cnt as usize / 8
    }
}

/// Table-driven canonical Huffman decoder.
struct Huffman {
    /// Indexed by the next `max_len` input bits (LSB first); entry =
    /// `symbol << 4 | length`, 0 = invalid code.
    table: Vec<u32>,
    max_len: u32,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self> {
        let max_len = lengths.iter().copied().max().unwrap_or(0) as u32;
        if max_len > 15 {
            return Err(err("code length exceeds 15"));
        }
        let mut count = [0u32; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        // Kraft inequality: reject over-subscribed codes.
        let mut left: i64 = 1;
        for &c in &count[1..] {
            left = left * 2 - c as i64;
            if left < 0 {
                return Err(err("over-subscribed Huffman code"));
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..16 {
            code = (code + count[bits - 1]) << 1;
            next[bits] = code;
        }
        let max_len = max_len.max(1);
        let size = 1usize << max_len;
        let mut table = vec![0u32; size];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = l as u32;
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = reverse_bits(c, l) as usize;
            let mut i = rev;
            while i < size {
                table[i] = ((sym as u32) << 4) | l;
                i += 1 << l;
            }
        }
        Ok(Huffman { table, max_len })
    }

    #[inline]
    fn decode(&self, br: &mut BitReader) -> Result<u32> {
        let e = self.table[br.peek(self.max_len) as usize];
        if e == 0 {
            return Err(err("invalid Huffman code"));
        }
        br.consume(e & 15)?;
        Ok(e >> 4)
    }
}

/// Decompress a raw DEFLATE stream. Returns the output and the number of
/// input bytes consumed (trailing data such as a zlib checksum is left).
pub fn inflate_raw(data: &[u8], limit: Option<usize>) -> Result<(Vec<u8>, usize)> {
    let mut br = BitReader::new(data);
    let mut out: Vec<u8> = Vec::with_capacity(data.len().saturating_mul(3).min(1 << 26));
    let limit = limit.unwrap_or(usize::MAX);
    let fixed = (Huffman::new(&fixed_litlen_lengths())?, Huffman::new(&[5u8; 30])?);
    loop {
        let last = br.bits(1)? == 1;
        match br.bits(2)? {
            0 => {
                br.align()?;
                let len = br.bits(16)? as usize;
                let nlen = br.bits(16)? as usize;
                if len != (!nlen & 0xFFFF) {
                    return Err(err("stored block length check failed"));
                }
                // Drain whole bytes still in the bit buffer first.
                let mut remaining = len;
                while remaining > 0 && br.cnt >= 8 && br.cnt > br.overrun {
                    out.push(br.bits(8)? as u8);
                    remaining -= 1;
                }
                if remaining > 0 {
                    let start = br.consumed();
                    if start + remaining > data.len() {
                        return Err(err("stored block truncated"));
                    }
                    out.extend_from_slice(&data[start..start + remaining]);
                    // Reset the reader just past the stored bytes.
                    br = BitReader { data, pos: start + remaining, buf: 0, cnt: 0, overrun: 0 };
                }
                if out.len() > limit {
                    return Err(err("output exceeds the size limit"));
                }
            }
            1 => inflate_block(&mut br, &mut out, &fixed.0, &fixed.1, limit)?,
            2 => {
                let (lit, dist) = read_dynamic_tables(&mut br)?;
                inflate_block(&mut br, &mut out, &lit, &dist, limit)?;
            }
            _ => return Err(err("invalid block type 3")),
        }
        if last {
            break;
        }
    }
    let used = br.consumed();
    Ok((out, used))
}

fn read_dynamic_tables(br: &mut BitReader) -> Result<(Huffman, Huffman)> {
    let hlit = br.bits(5)? as usize + 257;
    let hdist = br.bits(5)? as usize + 1;
    let hclen = br.bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(err("too many length or distance codes"));
    }
    let mut cl = [0u8; 19];
    for &i in CL_ORDER.iter().take(hclen) {
        cl[i] = br.bits(3)? as u8;
    }
    let clh = Huffman::new(&cl)?;
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        let sym = clh.decode(br)?;
        let (val, rep) = match sym {
            0..=15 => (sym as u8, 1),
            16 => {
                if i == 0 {
                    return Err(err("repeat code with no previous length"));
                }
                (lengths[i - 1], 3 + br.bits(2)? as usize)
            }
            17 => (0, 3 + br.bits(3)? as usize),
            18 => (0, 11 + br.bits(7)? as usize),
            _ => return Err(err("invalid code-length symbol")),
        };
        if i + rep > lengths.len() {
            return Err(err("code lengths overflow"));
        }
        lengths[i..i + rep].fill(val);
        i += rep;
    }
    if lengths[256] == 0 {
        return Err(err("missing end-of-block code"));
    }
    Ok((Huffman::new(&lengths[..hlit])?, Huffman::new(&lengths[hlit..])?))
}

fn inflate_block(br: &mut BitReader, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman, limit: usize) -> Result<()> {
    loop {
        let sym = lit.decode(br)?;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let li = (sym - 257) as usize;
            if li >= 29 {
                return Err(err("invalid length symbol"));
            }
            let len = LEN_BASE[li] as usize + br.bits(LEN_EXTRA[li] as u32)? as usize;
            let ds = dist.decode(br)? as usize;
            if ds >= 30 {
                return Err(err("invalid distance symbol"));
            }
            let d = DIST_BASE[ds] as usize + br.bits(DIST_EXTRA[ds] as u32)? as usize;
            if d > out.len() {
                return Err(err("distance reaches before start of output"));
            }
            let start = out.len() - d;
            if d >= len {
                out.extend_from_within(start..start + len);
            } else {
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
        }
        if out.len() > limit {
            return Err(err("output exceeds the size limit"));
        }
    }
}

/// Decompress a raw DEFLATE stream (convenience wrapper).
pub fn inflate(data: &[u8]) -> Result<Vec<u8>> {
    inflate_raw(data, None).map(|(v, _)| v)
}

// ---------------------------------------------------------------------------
// Checksums and wrappers
// ---------------------------------------------------------------------------

/// Adler-32 checksum (zlib).
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// Decompress a zlib stream, verifying header and Adler-32.
pub fn zlib_decompress(data: &[u8], limit: Option<usize>) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(err("zlib stream too short"));
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0F != 8 || cmf >> 4 > 7 || ((cmf as u16) << 8 | flg as u16) % 31 != 0 {
        return Err(err("invalid zlib header"));
    }
    if flg & 0x20 != 0 {
        return Err(err("zlib preset dictionaries are not supported"));
    }
    let (out, used) = inflate_raw(&data[2..], limit)?;
    let end = 2 + used;
    let sum = data.get(end..end + 4).ok_or_else(|| err("missing Adler-32 checksum"))?;
    if u32::from_be_bytes(sum.try_into().expect("4 bytes")) != adler32(&out) {
        return Err(err("Adler-32 checksum mismatch"));
    }
    Ok(out)
}

/// Compress into a zlib stream.
pub fn zlib_compress(data: &[u8], level: u8) -> Vec<u8> {
    let level = level.min(9);
    let flevel = match level {
        0 | 1 => 0,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let cmf = 0x78u8;
    let mut flg = flevel << 6;
    flg += (31 - ((cmf as u16) << 8 | flg as u16) % 31) as u8 % 31;
    let mut out = vec![cmf, flg];
    out.extend(deflate(data, level));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Decompress a gzip file (all members concatenated), verifying CRC-32 and size.
pub fn gzip_decompress(data: &[u8], limit: Option<usize>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < data.len() {
        let d = &data[p..];
        if d.len() < 18 || d[0] != 0x1F || d[1] != 0x8B || d[2] != 8 {
            return Err(err("invalid gzip header"));
        }
        let flags = d[3];
        let mut q = 10;
        if flags & 4 != 0 {
            let xlen = u16::from_le_bytes([
                *d.get(q).ok_or_else(|| err("truncated"))?,
                *d.get(q + 1).ok_or_else(|| err("truncated"))?,
            ]) as usize;
            q += 2 + xlen;
        }
        for bit in [8u8, 16] {
            if flags & bit != 0 {
                let z = d
                    .get(q..)
                    .and_then(|s| s.iter().position(|&c| c == 0))
                    .ok_or_else(|| err("unterminated gzip field"))?;
                q += z + 1;
            }
        }
        if flags & 2 != 0 {
            q += 2;
        }
        let body = d.get(q..).ok_or_else(|| err("truncated gzip header"))?;
        let remaining = limit.map(|l| l.saturating_sub(out.len()));
        let (member, used) = inflate_raw(body, remaining)?;
        let t = q + used;
        let tr = d.get(t..t + 8).ok_or_else(|| err("missing gzip trailer"))?;
        if u32::from_le_bytes(tr[..4].try_into().expect("4")) != crc32(&member) {
            return Err(err("gzip CRC-32 mismatch"));
        }
        if u32::from_le_bytes(tr[4..].try_into().expect("4")) != member.len() as u32 {
            return Err(err("gzip size mismatch"));
        }
        out.extend(member);
        p += t + 8;
    }
    Ok(out)
}

/// Compress into a single-member gzip file.
pub fn gzip_compress(data: &[u8], level: u8) -> Vec<u8> {
    let mut out = vec![0x1F, 0x8B, 8, 0, 0, 0, 0, 0, if level >= 9 { 2 } else { 0 }, 255];
    out.extend(deflate(data, level));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

// ---------------------------------------------------------------------------
// Deflate
// ---------------------------------------------------------------------------

struct BitWriter {
    out: Vec<u8>,
    buf: u64,
    cnt: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), buf: 0, cnt: 0 }
    }
    #[inline]
    fn put(&mut self, value: u32, n: u32) {
        self.buf |= (value as u64) << self.cnt;
        self.cnt += n;
        while self.cnt >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.cnt -= 8;
        }
    }
    fn align(&mut self) {
        if self.cnt > 0 {
            self.out.push(self.buf as u8);
            self.buf = 0;
            self.cnt = 0;
        }
    }
    fn bit_len(&self) -> usize {
        self.out.len() * 8 + self.cnt as usize
    }
}

/// LZ77 token: literal byte or (length, distance) back-reference.
#[derive(Clone, Copy)]
enum Token {
    Lit(u8),
    Match(u16, u16),
}

/// zlib's per-level parameters: (good_length, max_lazy, nice_length, max_chain).
const LEVELS: [(usize, usize, usize, usize); 10] = [
    (0, 0, 0, 0),
    (4, 4, 8, 4),
    (4, 5, 16, 8),
    (4, 6, 32, 32),
    (4, 4, 16, 16),
    (8, 16, 32, 32),
    (8, 16, 128, 128),
    (8, 32, 128, 256),
    (32, 128, 258, 1024),
    (32, 258, 258, 4096),
];

const WINDOW: usize = 32_768;
const HASH_BITS: u32 = 15;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const BLOCK_TOKENS: usize = 16_384;

struct Matcher<'a> {
    data: &'a [u8],
    head: Vec<i32>,
    prev: Vec<i32>,
    max_chain: usize,
    nice: usize,
    good: usize,
}

impl<'a> Matcher<'a> {
    #[inline]
    fn hash(&self, i: usize) -> usize {
        let d = self.data;
        let v = (d[i] as u32) << 16 | (d[i + 1] as u32) << 8 | d[i + 2] as u32;
        (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
    }
    #[inline]
    fn insert(&mut self, i: usize) {
        if i + MIN_MATCH <= self.data.len() {
            let h = self.hash(i);
            self.prev[i % WINDOW] = self.head[h];
            self.head[h] = i as i32;
        }
    }
    /// Longest match at `i` (better than `prev_len`), as (len, dist).
    fn longest(&self, i: usize, prev_len: usize) -> (usize, usize) {
        let d = self.data;
        if i + MIN_MATCH > d.len() {
            return (0, 0);
        }
        let max = (d.len() - i).min(MAX_MATCH);
        let mut chain = if prev_len >= self.good { self.max_chain / 4 } else { self.max_chain };
        let mut best = (prev_len.max(MIN_MATCH - 1), 0usize);
        let mut cand = self.head[self.hash(i)];
        let lowest = i.saturating_sub(WINDOW - 1);
        while cand >= 0 && (cand as usize) >= lowest && chain > 0 {
            let c = cand as usize;
            if c >= i {
                break;
            }
            if d[c + best.0.min(max - 1)] == d[i + best.0.min(max - 1)] && d[c] == d[i] {
                let mut l = 0;
                while l < max && d[c + l] == d[i + l] {
                    l += 1;
                }
                if l > best.0 {
                    best = (l, i - c);
                    if l >= self.nice || l == max {
                        break;
                    }
                }
            }
            let next = self.prev[c % WINDOW];
            if next >= cand {
                break; // stale entry from a wrapped window slot
            }
            cand = next;
            chain -= 1;
        }
        if best.1 == 0 {
            (0, 0)
        } else {
            best
        }
    }
}

fn tokenize(data: &[u8], level: u8) -> Vec<Token> {
    let (good, max_lazy, nice, max_chain) = LEVELS[level as usize];
    let mut m = Matcher { data, head: vec![-1; 1 << HASH_BITS], prev: vec![-1; WINDOW], max_chain, nice, good };
    let lazy = level >= 4;
    let mut toks = Vec::with_capacity(data.len() / 3 + 16);
    let mut i = 0;
    while i < data.len() {
        let (len, dist) = m.longest(i, 0);
        if len >= MIN_MATCH {
            if lazy && len < max_lazy && i + 1 < data.len() {
                m.insert(i);
                let (l2, d2) = m.longest(i + 1, len);
                if l2 > len {
                    // Defer: emit a literal and take the better match next.
                    toks.push(Token::Lit(data[i]));
                    i += 1;
                    toks.push(Token::Match(l2 as u16, d2 as u16));
                    for k in i..i + l2 {
                        m.insert(k);
                    }
                    i += l2;
                    continue;
                }
                toks.push(Token::Match(len as u16, dist as u16));
                for k in i + 1..i + len {
                    m.insert(k);
                }
                i += len;
                continue;
            }
            toks.push(Token::Match(len as u16, dist as u16));
            if len <= max_lazy.max(4) || level >= 4 {
                for k in i..i + len {
                    m.insert(k);
                }
            } else {
                m.insert(i);
            }
            i += len;
        } else {
            toks.push(Token::Lit(data[i]));
            m.insert(i);
            i += 1;
        }
    }
    toks
}

fn len_symbol(len: usize) -> (usize, u32, u32) {
    let mut i = 28;
    while LEN_BASE[i] as usize > len {
        i -= 1;
    }
    (257 + i, LEN_EXTRA[i] as u32, (len - LEN_BASE[i] as usize) as u32)
}

fn dist_symbol(d: usize) -> (usize, u32, u32) {
    let mut i = 29;
    while DIST_BASE[i] as usize > d {
        i -= 1;
    }
    (i, DIST_EXTRA[i] as u32, (d - DIST_BASE[i] as usize) as u32)
}

/// Optimal Huffman code lengths limited to `limit` bits (zlib's overflow
/// redistribution on top of a standard Huffman tree). Guarantees at least two
/// non-zero lengths so the code is complete.
pub(crate) fn huffman_lengths(freq: &[u32], limit: u32) -> Vec<u8> {
    let n = freq.len();
    let mut lens = vec![0u8; n];
    let mut syms: Vec<usize> = (0..n).filter(|&i| freq[i] > 0).collect();
    if syms.len() < 2 {
        // Pad with the lowest unused symbols to keep the code complete.
        for s in 0..n {
            if syms.len() >= 2 {
                break;
            }
            if !syms.contains(&s) {
                syms.push(s);
            }
        }
        syms.sort_unstable();
        for &s in &syms {
            lens[s] = 1;
        }
        return lens;
    }
    // Build the tree with a simple two-queue method on sorted leaves.
    syms.sort_by_key(|&s| (freq[s], s));
    let m = syms.len();
    let mut weight: Vec<u64> = syms.iter().map(|&s| freq[s].max(1) as u64).collect();
    let mut parent = vec![usize::MAX; 2 * m - 1];
    weight.resize(2 * m - 1, 0);
    let (mut leaf, mut node, mut next) = (0usize, m, m);
    let pick = |leaf: &mut usize, node: &mut usize, next: usize, weight: &Vec<u64>| -> usize {
        if *leaf < m && (*node >= next || weight[*leaf] <= weight[*node]) {
            *leaf += 1;
            *leaf - 1
        } else {
            *node += 1;
            *node - 1
        }
    };
    while next < 2 * m - 1 {
        let a = pick(&mut leaf, &mut node, next, &weight);
        let b = pick(&mut leaf, &mut node, next, &weight);
        weight[next] = weight[a] + weight[b];
        parent[a] = next;
        parent[b] = next;
        next += 1;
    }
    // Depths.
    let mut depth = vec![0u32; 2 * m - 1];
    for i in (0..2 * m - 2).rev() {
        depth[i] = depth[parent[i]] + 1;
    }
    // Clip depths to `limit`, then rebalance the Kraft sum exactly. Work in
    // integer units of 2^-limit: a code of length b weighs 2^(limit-b) and a
    // complete code sums to 2^limit.
    let mut bl_count = vec![0u64; (limit + 1) as usize];
    for &d in depth.iter().take(m) {
        bl_count[d.min(limit) as usize] += 1;
    }
    let target: u64 = 1 << limit;
    let weight_of = |b: usize| -> u64 { 1u64 << (limit as usize - b) };
    let mut kraft: u64 = (1..=limit as usize).map(|b| bl_count[b] * weight_of(b)).sum();
    // Too heavy: lengthen the longest codes shorter than `limit` (smallest
    // possible decrement, so the code stays close to optimal).
    while kraft > target {
        let b = (1..limit as usize).rev().find(|&b| bl_count[b] > 0).expect("a code shorter than the limit exists");
        bl_count[b] -= 1;
        bl_count[b + 1] += 1;
        kraft -= weight_of(b + 1);
    }
    // Too light (possible after the step above): shorten the longest codes.
    while kraft < target {
        let b = (2..=limit as usize).rev().find(|&b| bl_count[b] > 0 && kraft + weight_of(b) <= target);
        match b {
            Some(b) => {
                bl_count[b] -= 1;
                bl_count[b - 1] += 1;
                kraft += weight_of(b);
            }
            None => break,
        }
    }
    // Assign lengths: most frequent symbols get the shortest codes.
    let mut k = m; // syms sorted ascending by freq: assign from the end
    for (bits, &count) in bl_count.iter().enumerate().skip(1) {
        let count = count as usize;
        for _ in 0..count {
            k -= 1;
            lens[syms[k]] = bits as u8;
        }
    }
    lens
}

/// Canonical codes (already bit-reversed for LSB-first output).
fn canonical_codes(lens: &[u8]) -> Vec<u32> {
    let mut count = [0u32; 16];
    for &l in lens {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; 16];
    let mut code = 0;
    for b in 1..16 {
        code = (code + count[b - 1]) << 1;
        next[b] = code;
    }
    lens.iter()
        .map(|&l| {
            if l == 0 {
                0
            } else {
                let c = next[l as usize];
                next[l as usize] += 1;
                reverse_bits(c, l as u32)
            }
        })
        .collect()
}

/// Run-length encode code lengths into (symbol, extra-bits value) pairs.
fn rle_lengths(lens: &[u8]) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lens.len() {
        let v = lens[i];
        let mut run = 1;
        while i + run < lens.len() && lens[i + run] == v {
            run += 1;
        }
        let mut r = run;
        if v == 0 {
            while r >= 11 {
                let k = r.min(138);
                out.push((18, (k - 11) as u8));
                r -= k;
            }
            if r >= 3 {
                out.push((17, (r - 3) as u8));
                r = 0;
            }
        } else {
            out.push((v, 0));
            r -= 1;
            while r >= 3 {
                let k = r.min(6);
                out.push((16, (k - 3) as u8));
                r -= k;
            }
        }
        for _ in 0..r {
            out.push((v, 0));
        }
        i += run;
    }
    out
}

struct DynHeader {
    lit_lens: Vec<u8>,
    dist_lens: Vec<u8>,
    cl_lens: Vec<u8>,
    rle: Vec<(u8, u8)>,
    hlit: usize,
    hdist: usize,
    hclen: usize,
}

fn dyn_header(lit_freq: &[u32], dist_freq: &[u32]) -> DynHeader {
    let lit_lens = huffman_lengths(lit_freq, 15);
    let dist_lens = huffman_lengths(dist_freq, 15);
    let mut hlit = 286;
    while hlit > 257 && lit_lens[hlit - 1] == 0 {
        hlit -= 1;
    }
    let mut hdist = 30;
    while hdist > 1 && dist_lens[hdist - 1] == 0 {
        hdist -= 1;
    }
    let mut all = lit_lens[..hlit].to_vec();
    all.extend_from_slice(&dist_lens[..hdist]);
    let rle = rle_lengths(&all);
    let mut cl_freq = [0u32; 19];
    for &(s, _) in &rle {
        cl_freq[s as usize] += 1;
    }
    let cl_lens = huffman_lengths(&cl_freq, 7);
    let mut hclen = 19;
    while hclen > 4 && cl_lens[CL_ORDER[hclen - 1]] == 0 {
        hclen -= 1;
    }
    DynHeader { lit_lens, dist_lens, cl_lens, rle, hlit, hdist, hclen }
}

fn extra_bits(s: u8) -> u32 {
    match s {
        16 => 2,
        17 => 3,
        18 => 7,
        _ => 0,
    }
}

fn data_cost(toks: &[Token], lit_lens: &[u8], dist_lens: &[u8]) -> usize {
    let mut bits = lit_lens[256] as usize;
    for t in toks {
        match *t {
            Token::Lit(b) => bits += lit_lens[b as usize] as usize,
            Token::Match(l, d) => {
                let (ls, le, _) = len_symbol(l as usize);
                let (ds, de, _) = dist_symbol(d as usize);
                bits += lit_lens[ls] as usize + le as usize + dist_lens[ds] as usize + de as usize;
            }
        }
    }
    bits
}

fn write_tokens(w: &mut BitWriter, toks: &[Token], lit_lens: &[u8], dist_lens: &[u8]) {
    let lc = canonical_codes(lit_lens);
    let dc = canonical_codes(dist_lens);
    for t in toks {
        match *t {
            Token::Lit(b) => w.put(lc[b as usize], lit_lens[b as usize] as u32),
            Token::Match(l, d) => {
                let (ls, le, lv) = len_symbol(l as usize);
                w.put(lc[ls], lit_lens[ls] as u32);
                w.put(lv, le);
                let (ds, de, dv) = dist_symbol(d as usize);
                w.put(dc[ds], dist_lens[ds] as u32);
                w.put(dv, de);
            }
        }
    }
    w.put(lc[256], lit_lens[256] as u32);
}

fn write_stored(w: &mut BitWriter, bytes: &[u8], last: bool) {
    let chunks: Vec<&[u8]> = if bytes.is_empty() { vec![&[][..]] } else { bytes.chunks(65_535).collect() };
    let n = chunks.len();
    for (k, c) in chunks.into_iter().enumerate() {
        w.put(u32::from(last && k + 1 == n), 1);
        w.put(0, 2);
        w.align();
        let len = c.len() as u16;
        w.out.extend_from_slice(&len.to_le_bytes());
        w.out.extend_from_slice(&(!len).to_le_bytes());
        w.out.extend_from_slice(c);
    }
}

/// Compress `data` into a raw DEFLATE stream. `level` 0 = stored only,
/// 1 = fastest … 9 = best compression (6 is zlib's default).
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    let level = level.min(9);
    let mut w = BitWriter::new();
    if level == 0 {
        write_stored(&mut w, data, true);
        return w.out;
    }
    let toks = tokenize(data, level);
    let fixed_lit = fixed_litlen_lengths();
    let fixed_dist = [5u8; 30];
    let mut pos = 0usize; // byte offset of the current block in `data`
    let blocks: Vec<&[Token]> = if toks.is_empty() { vec![&toks[..]] } else { toks.chunks(BLOCK_TOKENS).collect() };
    let nblocks = blocks.len();
    for (bi, block) in blocks.into_iter().enumerate() {
        let last = bi + 1 == nblocks;
        let span: usize = block.iter().map(|t| if let Token::Match(l, _) = t { *l as usize } else { 1 }).sum();
        let raw = &data[pos..pos + span];
        pos += span;

        let mut lf = vec![0u32; 286];
        let mut df = vec![0u32; 30];
        lf[256] = 1;
        for t in block {
            match *t {
                Token::Lit(b) => lf[b as usize] += 1,
                Token::Match(l, d) => {
                    lf[len_symbol(l as usize).0] += 1;
                    df[dist_symbol(d as usize).0] += 1;
                }
            }
        }
        let h = dyn_header(&lf, &df);
        let header_bits = 5
            + 5
            + 4
            + 3 * h.hclen
            + h.rle.iter().map(|&(s, _)| h.cl_lens[s as usize] as usize + extra_bits(s) as usize).sum::<usize>();
        let dyn_bits = 3 + header_bits + data_cost(block, &h.lit_lens, &h.dist_lens);
        let fixed_bits = 3 + data_cost(block, &fixed_lit, &fixed_dist);
        let pad = (8 - (w.bit_len() + 3) % 8) % 8;
        let stored_bits = raw.len().div_ceil(65_535).max(1) * (3 + 32) + pad + raw.len() * 8;

        if stored_bits <= fixed_bits && stored_bits <= dyn_bits {
            write_stored(&mut w, raw, last);
        } else if fixed_bits <= dyn_bits {
            w.put(u32::from(last), 1);
            w.put(1, 2);
            write_tokens(&mut w, block, &fixed_lit, &fixed_dist);
        } else {
            w.put(u32::from(last), 1);
            w.put(2, 2);
            w.put((h.hlit - 257) as u32, 5);
            w.put((h.hdist - 1) as u32, 5);
            w.put((h.hclen - 4) as u32, 4);
            for &i in CL_ORDER.iter().take(h.hclen) {
                w.put(h.cl_lens[i] as u32, 3);
            }
            let cc = canonical_codes(&h.cl_lens);
            for &(s, x) in &h.rle {
                w.put(cc[s as usize], h.cl_lens[s as usize] as u32);
                w.put(x as u32, extra_bits(s));
            }
            write_tokens(&mut w, block, &h.lit_lens, &h.dist_lens);
        }
    }
    w.align();
    w.out
}

#[cfg(test)]
#[rustfmt::skip]
mod vectors {
    include!("deflate_vectors.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes (xorshift64*).
    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
            })
            .collect()
    }

    fn first_block_type(stream: &[u8]) -> u8 {
        (stream[0] >> 1) & 3
    }

    #[test]
    fn reference_zlib_streams_decode() {
        use vectors::*;
        let fixed_src: Vec<u8> =
            // The compressed vectors below were produced from this text (the product's earlier name); keep it.
            b"OpenTab is an open-source statistical suite. ".repeat(20).into_iter().chain(0u8..=255).collect();
        assert_eq!(first_block_type(FIXED), 1);
        assert_eq!(inflate(FIXED).unwrap(), fixed_src);
        assert_eq!(first_block_type(DYNAMIC), 2);
        let dynamic = inflate(DYNAMIC).unwrap();
        assert_eq!(dynamic.len(), DYNAMIC_TEXT_LEN);
        assert_eq!(dynamic, WORDS_SEED_TEXT);
        assert_eq!(first_block_type(STORED), 0);
        assert_eq!(inflate(STORED).unwrap(), b"hello stored block");
        let huff_src: Vec<u8> = (0u8..=255).cycle().take(1024).chain(b"aaaa".repeat(100)).collect();
        assert_eq!(inflate(HUFFMAN_ONLY).unwrap(), huff_src);
        let rle_src: Vec<u8> = [vec![b'z'; 5000], vec![b'y'; 3000]].concat();
        assert_eq!(inflate(RLE).unwrap(), rle_src);
        assert_eq!(inflate(MULTI).unwrap(), WORDS_SEED_TEXT);
        assert_eq!(zlib_decompress(ZLIB, None).unwrap(), b"zlib wrapped payload ".repeat(10));
    }

    #[test]
    fn roundtrip_all_levels_and_shapes() {
        let mut inputs: Vec<Vec<u8>> = vec![
            vec![],
            b"a".to_vec(),
            b"abcabcabcabcabcabcabcabc".to_vec(),
            vec![0u8; 100_000],
            noise(70_000, 3),
            vectors::WORDS_SEED_TEXT.repeat(30),
            (0..200_000u32).map(|i| (i % 251) as u8 ^ (i / 1000) as u8).collect(),
        ];
        let mut mixed = noise(5_000, 9);
        mixed.extend(vectors::WORDS_SEED_TEXT);
        mixed.extend(vec![7u8; 40_000]);
        mixed.extend(noise(1_000, 11));
        inputs.push(mixed);
        for data in &inputs {
            for level in 0..=9u8 {
                let c = deflate(data, level);
                let d = inflate(&c).unwrap_or_else(|e| panic!("level {level}, len {}: {e}", data.len()));
                assert_eq!(&d, data, "level {level} len {}", data.len());
            }
        }
    }

    #[test]
    fn compressor_chooses_block_types() {
        // Incompressible data → stored; tiny repetitive data → fixed; text → dynamic.
        assert_eq!(first_block_type(&deflate(&noise(10_000, 5), 6)), 0);
        assert_eq!(first_block_type(&deflate(b"aaaaaaaaaaaaaaaaaaaa", 6)), 1);
        assert_eq!(first_block_type(&deflate(&vectors::WORDS_SEED_TEXT.repeat(3), 6)), 2);
        let text = vectors::WORDS_SEED_TEXT.repeat(20);
        let c9 = deflate(&text, 9).len();
        let c1 = deflate(&text, 1).len();
        assert!(c9 <= c1, "level 9 ({c9}) should not be larger than level 1 ({c1})");
        assert!(c9 < text.len() / 10, "text should compress well: {c9}");
    }

    #[test]
    fn huffman_lengths_are_valid_and_limited() {
        // Fibonacci frequencies force very deep trees.
        let mut fib = vec![1u32, 1];
        while fib.len() < 40 {
            let n = fib.len();
            fib.push(fib[n - 1].saturating_add(fib[n - 2]));
        }
        for (freqs, limit) in [(fib.clone(), 15u32), (fib[..25].to_vec(), 7), (vec![5, 0, 0, 0], 15), (vec![0; 10], 15)]
        {
            let l = huffman_lengths(&freqs, limit);
            assert!(l.iter().all(|&x| (x as u32) <= limit));
            let kraft: f64 = l.iter().filter(|&&x| x > 0).map(|&x| 0.5f64.powi(x as i32)).sum();
            assert!((kraft - 1.0).abs() < 1e-12, "kraft {kraft} for {freqs:?}");
            assert!(Huffman::new(&l).is_ok());
        }
    }

    #[test]
    fn huffman_lengths_random_distributions() {
        for seed in 1..300u64 {
            let bytes = noise(286 * 4, seed);
            let n = if seed % 3 == 0 { 19 } else { 286 };
            let limit = if n == 19 { 7 } else { 15 };
            // Skewed frequencies (some zero) spanning several orders of magnitude.
            let freqs: Vec<u32> = (0..n)
                .map(|i| {
                    let b = bytes[i * 4] as u32;
                    if b < 60 {
                        0
                    } else {
                        1 + (bytes[i * 4 + 1] as u32) * (1 << (bytes[i * 4 + 2] % 20))
                    }
                })
                .collect();
            let l = huffman_lengths(&freqs, limit);
            let units: u64 = l.iter().filter(|&&x| x > 0).map(|&x| 1u64 << (limit - x as u32)).sum();
            assert_eq!(units, 1 << limit, "seed {seed}: incomplete or over-full code");
            for (i, &f) in freqs.iter().enumerate() {
                assert!(f == 0 || l[i] > 0, "used symbol without a code");
            }
        }
    }

    #[test]
    fn wrappers_roundtrip() {
        let data = vectors::WORDS_SEED_TEXT.repeat(4);
        for level in [0u8, 1, 6, 9] {
            assert_eq!(zlib_decompress(&zlib_compress(&data, level), None).unwrap(), data);
            assert_eq!(gzip_decompress(&gzip_compress(&data, level), None).unwrap(), data);
        }
        let two = [gzip_compress(b"one ", 6), gzip_compress(b"two", 1)].concat();
        assert_eq!(gzip_decompress(&two, None).unwrap(), b"one two");
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn corrupt_streams_rejected() {
        use vectors::*;
        assert!(inflate(&DYNAMIC[..DYNAMIC.len() / 2]).is_err(), "truncated");
        assert!(inflate(&[0x07]).is_err(), "block type 3");
        assert!(inflate(&[0x01, 0x05, 0x00, 0x00, 0x00]).is_err(), "LEN/NLEN mismatch");
        // Fixed block with a distance before the start: lit-free match.
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(1, 2);
        let lc = canonical_codes(&fixed_litlen_lengths());
        w.put(lc[257], 7); // length 3
        w.put(reverse_bits(0, 5), 5); // distance 1 with empty output
        w.put(lc[256], 7);
        w.align();
        assert!(inflate(&w.out).unwrap_err().to_string().contains("before start"));
        let mut z = zlib_compress(b"payload", 6);
        let n = z.len();
        z[n - 1] ^= 1;
        assert!(zlib_decompress(&z, None).unwrap_err().to_string().contains("Adler"));
        // Decompression bomb guard.
        let bomb = deflate(&vec![0u8; 1_000_000], 9);
        assert!(bomb.len() < 2_000);
        assert!(inflate_raw(&bomb, Some(10_000)).is_err());
        // Random garbage must never panic.
        for seed in 1..200 {
            let _ = inflate(&noise(64, seed));
        }
    }
}
