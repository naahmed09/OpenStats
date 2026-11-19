//! PNG encoding (via the Phase 1b zlib encoder) and a decoder used for round-trip checks.

use openstats_core::io::crc32::crc32;
use openstats_core::io::deflate::{zlib_compress, zlib_decompress};

/// An 8-bit RGBA image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major RGBA bytes (`4 * width * height`).
    pub rgba: Vec<u8>,
}

impl Image {
    /// Pixel at `(x, y)` as `[r, g, b, a]`.
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 4] {
        let i = 4 * (y * self.width + x);
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// FNV-1a 64-bit hash of the pixel data (for visual-regression tests).
    pub fn hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in self.width.to_le_bytes().iter().chain(self.height.to_le_bytes().iter()).chain(self.rgba.iter()) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut body = kind.to_vec();
    body.extend_from_slice(data);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i32::from(a), i32::from(b), i32::from(c));
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Encode an RGBA image as PNG (8-bit, colour type 6, adaptive row filters).
pub fn encode(img: &Image) -> Vec<u8> {
    let (w, h) = (img.width, img.height);
    let stride = w * 4;
    let mut raw = Vec::with_capacity((stride + 1) * h);
    let zero = vec![0u8; stride];
    for y in 0..h {
        let row = &img.rgba[y * stride..(y + 1) * stride];
        let prev: &[u8] = if y == 0 { &zero } else { &img.rgba[(y - 1) * stride..y * stride] };
        let mut best: (u64, u8, Vec<u8>) = (u64::MAX, 0, Vec::new());
        for f in 0..5u8 {
            let mut out = Vec::with_capacity(stride);
            for i in 0..stride {
                let a = if i >= 4 { row[i - 4] } else { 0 };
                let b = prev[i];
                let c = if i >= 4 { prev[i - 4] } else { 0 };
                out.push(match f {
                    0 => row[i],
                    1 => row[i].wrapping_sub(a),
                    2 => row[i].wrapping_sub(b),
                    3 => row[i].wrapping_sub(((u16::from(a) + u16::from(b)) / 2) as u8),
                    _ => row[i].wrapping_sub(paeth(a, b, c)),
                });
            }
            let cost: u64 = out.iter().map(|&v| u64::from((v as i8).unsigned_abs())).sum();
            if cost < best.0 {
                best = (cost, f, out);
            }
        }
        raw.push(best.1);
        raw.extend_from_slice(&best.2);
    }
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_compress(&raw, 6));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Decode an 8-bit RGBA or RGB, non-interlaced PNG (CRCs verified).
pub fn decode(b: &[u8]) -> Result<Image, String> {
    if b.len() < 8 || b[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Err("not a PNG (bad signature)".into());
    }
    let mut i = 8;
    let (mut w, mut h, mut ct) = (0usize, 0usize, 0u8);
    let mut idat = Vec::new();
    let mut seen_end = false;
    while i + 12 <= b.len() {
        let len = u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        let kind = &b[i + 4..i + 8];
        let data = b.get(i + 8..i + 8 + len).ok_or("truncated chunk")?;
        let crc = b.get(i + 8 + len..i + 12 + len).ok_or("truncated crc")?;
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        if crc32(&body).to_be_bytes() != crc {
            return Err(format!("bad CRC in {}", String::from_utf8_lossy(kind)));
        }
        match kind {
            b"IHDR" => {
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                if data[8] != 8 || data[12] != 0 || !(data[9] == 6 || data[9] == 2) {
                    return Err("unsupported PNG format".into());
                }
                ct = data[9];
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => {
                seen_end = true;
                break;
            }
            _ => {}
        }
        i += 12 + len;
    }
    if !seen_end || w == 0 || h == 0 {
        return Err("missing IHDR or IEND".into());
    }
    let bpp = if ct == 6 { 4 } else { 3 };
    let raw = zlib_decompress(&idat, Some((w * bpp + 1) * h + 16)).map_err(|e| e.to_string())?;
    let stride = w * bpp;
    if raw.len() != (stride + 1) * h {
        return Err("wrong decompressed size".into());
    }
    let mut img = vec![0u8; stride * h];
    for y in 0..h {
        let f = raw[y * (stride + 1)];
        for x in 0..stride {
            let v = raw[y * (stride + 1) + 1 + x];
            let a = if x >= bpp { img[y * stride + x - bpp] } else { 0 };
            let up = if y > 0 { img[(y - 1) * stride + x] } else { 0 };
            let c = if x >= bpp && y > 0 { img[(y - 1) * stride + x - bpp] } else { 0 };
            img[y * stride + x] = match f {
                0 => v,
                1 => v.wrapping_add(a),
                2 => v.wrapping_add(up),
                3 => v.wrapping_add(((u16::from(a) + u16::from(up)) / 2) as u8),
                4 => v.wrapping_add(paeth(a, up, c)),
                _ => return Err("bad filter type".into()),
            };
        }
    }
    let rgba = if bpp == 4 { img } else { img.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect() };
    Ok(Image { width: w, height: h, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_corruption() {
        let (w, h) = (37, 23);
        let mut rgba = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&[(x * 7) as u8, (y * 11) as u8, ((x * y) % 256) as u8, 255]);
            }
        }
        let img = Image { width: w, height: h, rgba };
        let png = encode(&img);
        assert_eq!(&png[1..4], b"PNG");
        assert_eq!(decode(&png).unwrap(), img);
        let mut bad = png.clone();
        let n = bad.len();
        bad[n / 2] ^= 0xFF;
        assert!(decode(&bad).is_err());
        assert!(decode(&png[..20]).is_err());
        assert!(decode(b"nope").is_err());
    }
}
