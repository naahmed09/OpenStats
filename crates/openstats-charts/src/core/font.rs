//! Font metrics and bitmap glyphs without any font library (see the `tables` module for the data).
use super::tables::{ASCII, ASCII_WIDTH, EXTRA};

/// Whether `c` is a zero-width combining mark that draws a bar over the previous character.
pub fn is_overbar(c: char) -> bool {
    c == '\u{0304}' || c == '\u{203E}'
}

/// Advance of `c` in 1/1000 em (Helvetica regular).
pub fn char_units(c: char) -> u32 {
    if is_overbar(c) {
        return 0;
    }
    let u = c as u32;
    if (0x20..=0x7E).contains(&u) {
        return u32::from(ASCII_WIDTH[(u - 0x20) as usize]);
    }
    let c = if c == 'µ' { 'μ' } else { c };
    if c == '\u{00A0}' {
        return 278;
    }
    EXTRA.iter().find(|e| e.0 == c).map_or(600, |e| u32::from(e.2))
}

/// Width in pixels of `s` set at `size` pixels (`bold` widens by 6 %).
pub fn text_width(s: &str, size: f64, bold: bool) -> f64 {
    let units: u32 = s.chars().map(char_units).sum();
    f64::from(units) / 1000.0 * size * if bold { 1.06 } else { 1.0 }
}

/// Font metrics of the built-in face at one size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FontMetrics {
    /// Font size in pixels.
    pub size: f64,
}

impl FontMetrics {
    /// Metrics at `size` pixels.
    pub fn new(size: f64) -> FontMetrics {
        FontMetrics { size }
    }

    /// Text width.
    pub fn width(&self, s: &str) -> f64 {
        text_width(s, self.size, false)
    }

    /// Distance from the top of capitals to the baseline.
    pub fn ascent(&self) -> f64 {
        0.72 * self.size
    }

    /// Descender depth.
    pub fn descent(&self) -> f64 {
        0.21 * self.size
    }

    /// Line height.
    pub fn line_height(&self) -> f64 {
        1.2 * self.size
    }
}

/// Bitmap of a glyph: 8 rows × 5 columns, `true` = ink. Unknown characters give a box.
pub fn glyph_bitmap(c: char) -> [[bool; 5]; 8] {
    let u = c as u32;
    let mut g = [[false; 5]; 8];
    if (0x20..=0x7E).contains(&u) {
        let cols = ASCII[(u - 0x20) as usize];
        for (x, col) in cols.iter().enumerate() {
            for (y, row) in g.iter_mut().enumerate() {
                row[x] = col >> y & 1 == 1;
            }
        }
        return g;
    }
    let c = if c == 'µ' { 'μ' } else { c };
    if c == '\u{00A0}' {
        return g;
    }
    if let Some(e) = EXTRA.iter().find(|e| e.0 == c) {
        for (y, row) in e.1.iter().enumerate() {
            for (x, ch) in row.bytes().enumerate() {
                g[y][x] = ch == b'#';
            }
        }
        return g;
    }
    for i in 0..5 {
        g[0][i] = true;
        g[6][i] = true;
    }
    for r in g.iter_mut().take(7) {
        r[0] = true;
        r[4] = true;
    }
    g
}

/// Whether the built-in face has a drawn glyph (or metrics) for `c`.
pub fn has_glyph(c: char) -> bool {
    let u = c as u32;
    (0x20..=0x7E).contains(&u) || is_overbar(c) || c == 'µ' || c == '\u{00A0}' || EXTRA.iter().any(|e| e.0 == c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_are_helvetica() {
        assert_eq!(char_units(' '), 278);
        assert_eq!(char_units('W'), 944);
        assert_eq!(char_units('i'), 222);
        assert!((text_width("Hello", 10.0, false) - 22.78).abs() < 0.01);
        assert_eq!(char_units('\u{0304}'), 0);
        assert_eq!(text_width("X\u{0304}", 10.0, false), text_width("X", 10.0, false));
    }

    #[test]
    fn every_glyph_has_ink() {
        for u in 0x21u32..0x7F {
            let g = glyph_bitmap(char::from_u32(u).unwrap());
            assert!(g.iter().flatten().any(|&b| b), "U+{u:X}");
        }
        for e in EXTRA {
            assert!(glyph_bitmap(e.0).iter().flatten().any(|&b| b), "{}", e.0);
            assert!(has_glyph(e.0));
        }
        assert!(!has_glyph('€'));
    }
}
