//! System fonts for the text the rasteriser draws (fontdue; the font files come from the OS).

use std::collections::HashMap;

use fontdue::{Font, FontSettings};
use openstats_charts::render::raster::{TextPainter, TextRun};

/// Regular and bold faces to try, in order, per platform.
#[cfg(target_os = "macos")]
const REGULAR: &[&str] = &["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"];
#[cfg(target_os = "macos")]
const BOLD: &[&str] = &["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"];

#[cfg(target_os = "windows")]
const REGULAR: &[&str] = &["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\segoeui.ttf"];
#[cfg(target_os = "windows")]
const BOLD: &[&str] = &["C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\segoeuib.ttf"];

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const REGULAR: &[&str] = &[
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const BOLD: &[&str] = &[
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
];

fn load(paths: &[&str]) -> Option<Font> {
    paths.iter().find_map(|p| {
        let bytes = std::fs::read(p).ok()?;
        Font::from_bytes(bytes, FontSettings::default()).ok()
    })
}

type Glyph = (fontdue::Metrics, Vec<u8>);

/// The operating system's sans-serif face, rasterised on demand and cached per glyph.
pub struct SystemFont {
    regular: Font,
    bold: Option<Font>,
    cache: HashMap<(char, u32, bool), Glyph>,
}

impl SystemFont {
    /// Load the fonts, or `None` when the machine has none of the expected files (the app then
    /// keeps the built-in bitmap face).
    pub fn load() -> Option<SystemFont> {
        Some(SystemFont { regular: load(REGULAR)?, bold: load(BOLD), cache: HashMap::new() })
    }
}

impl TextPainter for SystemFont {
    fn render(&mut self, text: &str, size_px: f32, bold: bool) -> Option<TextRun> {
        let font = if bold { self.bold.as_ref().unwrap_or(&self.regular) } else { &self.regular };
        // A missing character would draw as a box: let the rasteriser use its own glyphs instead.
        if text.chars().any(|c| !c.is_whitespace() && !c.is_control() && font.lookup_glyph_index(c) == 0) {
            return None;
        }
        let key_size = size_px.to_bits();
        let mut pen = 0.0_f32;
        let mut placed: Vec<(char, i32, i32)> = Vec::new(); // glyph, x of bitmap, top above baseline
        let (mut min_x, mut max_x, mut max_top, mut min_bottom) = (i32::MAX, i32::MIN, i32::MIN, i32::MAX);
        for c in text.chars() {
            let g = self.cache.entry((c, key_size, bold)).or_insert_with(|| font.rasterize(c, size_px));
            let (m, _) = (&g.0, &g.1);
            if m.width > 0 && m.height > 0 {
                let x = (pen + m.xmin as f32).round() as i32;
                let top = m.ymin + m.height as i32;
                min_x = min_x.min(x);
                max_x = max_x.max(x + m.width as i32);
                max_top = max_top.max(top);
                min_bottom = min_bottom.min(m.ymin);
                placed.push((c, x, top));
            }
            pen += m.advance_width;
        }
        if placed.is_empty() {
            return Some(TextRun { advance: pen, left: 0, top: 0, width: 0, height: 0, coverage: Vec::new() });
        }
        let width = (max_x - min_x) as usize;
        let height = (max_top - min_bottom) as usize;
        let mut coverage = vec![0u8; width * height];
        for (c, x, top) in placed {
            let (m, bitmap) = &self.cache[&(c, key_size, bold)];
            let (ox, oy) = ((x - min_x) as usize, (max_top - top) as usize);
            for row in 0..m.height {
                for col in 0..m.width {
                    let dst = (oy + row) * width + ox + col;
                    // Overlapping glyphs (tight kerning) add up instead of overwriting.
                    coverage[dst] = coverage[dst].saturating_add(bitmap[row * m.width + col]);
                }
            }
        }
        Some(TextRun { advance: pen, left: min_x, top: max_top, width, height, coverage })
    }
}
