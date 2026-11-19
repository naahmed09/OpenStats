//! PDF output: one page, vector graphics, the standard Helvetica fonts.
//!
//! The page is the scene size in points. Text uses WinAnsi-encoded Helvetica / Helvetica-Bold;
//! characters outside Latin-1 (Greek letters, the combining overline) are replaced by `?` or,
//! for the overline, drawn as a line above the preceding letter. Translucent fills use
//! `ExtGState` alpha.

use super::{arc_points, marker_geom, MarkerGeom};
use crate::core::font::text_width;
use crate::core::geom::Point;
use crate::core::scene::{Anchor, Scene, Shape};
use crate::core::style::{Color, Dash, Stroke};
use std::fmt::Write as _;

fn n(v: f64) -> String {
    if v.is_finite() {
        let s = format!("{v:.3}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s.is_empty() || s == "-" || s == "-0" {
            "0".into()
        } else {
            s.into()
        }
    } else {
        "0".into()
    }
}

fn rgb(c: Color) -> String {
    format!("{} {} {}", n(f64::from(c.r) / 255.0), n(f64::from(c.g) / 255.0), n(f64::from(c.b) / 255.0))
}

fn pdf_text(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in s.chars() {
        let code = ch as u32;
        let b = match ch {
            '(' | ')' | '\\' => {
                out.push(b'\\');
                ch as u8
            }
            '\u{2212}' => b'-',
            '\u{2013}' | '\u{2014}' => b'-',
            '\u{2026}' => b'.',
            c if (32..127).contains(&(c as u32)) => c as u8,
            _ if (160..256).contains(&code) => {
                out.extend_from_slice(format!("\\{code:03o}").as_bytes());
                continue;
            }
            _ => b'?',
        };
        out.push(b);
    }
    out
}

fn is_overbar(c: char) -> bool {
    matches!(c, '\u{0304}' | '\u{0305}' | '\u{203E}')
}

struct Page {
    content: String,
    alphas: Vec<u8>,
    clips: usize,
}

impl Page {
    fn alpha(&mut self, a: u8) -> usize {
        if let Some(i) = self.alphas.iter().position(|&x| x == a) {
            return i;
        }
        self.alphas.push(a);
        self.alphas.len() - 1
    }

    fn set_fill(&mut self, c: Color) {
        if c.a < 255 {
            let i = self.alpha(c.a);
            let _ = writeln!(self.content, "/GS{i} gs");
        }
        let _ = writeln!(self.content, "{} rg", rgb(c));
    }

    fn set_stroke(&mut self, s: &Stroke) {
        if s.color.a < 255 {
            let i = self.alpha(s.color.a);
            let _ = writeln!(self.content, "/GS{i} gs");
        }
        let _ = writeln!(self.content, "{} RG {} w", rgb(s.color), n(s.width));
        let d = s.dash.pattern(s.width);
        if d.is_empty() || matches!(s.dash, Dash::Solid) {
            self.content.push_str("[] 0 d\n");
        } else {
            let v: Vec<String> = d.iter().map(|x| n(*x)).collect();
            let _ = writeln!(self.content, "[{}] 0 d", v.join(" "));
        }
    }

    fn path(&mut self, pts: &[Point], close: bool) {
        for (i, p) in pts.iter().enumerate() {
            let _ = writeln!(self.content, "{} {} {}", n(p.x), n(p.y), if i == 0 { "m" } else { "l" });
        }
        if close {
            self.content.push_str("h\n");
        }
    }

    /// Paint the current path with the item's fill and stroke.
    fn paint(&mut self, fill: Option<Color>, stroke: Option<Stroke>) {
        self.content.push_str("q\n");
        let op = match (fill.is_some(), stroke.is_some()) {
            (true, true) => "B",
            (true, false) => "f",
            (false, true) => "S",
            (false, false) => "n",
        };
        // Colours must be set before the path is painted but after it is built; PDF allows the
        // state to be set while the path is under construction.
        if let Some(f) = fill {
            self.set_fill(f);
        }
        if let Some(s) = stroke {
            self.set_stroke(&s);
        }
        let _ = writeln!(self.content, "{op}\nQ");
    }

    fn circle(&mut self, c: Point, r: f64) {
        let k = 0.552_284_75 * r;
        let _ = writeln!(self.content, "{} {} m", n(c.x + r), n(c.y));
        let seg = [
            (c.x + r, c.y + k, c.x + k, c.y + r, c.x, c.y + r),
            (c.x - k, c.y + r, c.x - r, c.y + k, c.x - r, c.y),
            (c.x - r, c.y - k, c.x - k, c.y - r, c.x, c.y - r),
            (c.x + k, c.y - r, c.x + r, c.y - k, c.x + r, c.y),
        ];
        for s in seg {
            let _ = writeln!(self.content, "{} {} {} {} {} {} c", n(s.0), n(s.1), n(s.2), n(s.3), n(s.4), n(s.5));
        }
        self.content.push_str("h\n");
    }

    fn text(&mut self, at: Point, text: &str, st: &crate::core::scene::TextStyle) {
        // Split off overbars.
        let chars: Vec<char> = text.chars().collect();
        let mut plain = String::new();
        let mut bars: Vec<(usize, char)> = Vec::new(); // (index in `plain`, letter)
        let mut i = 0;
        while i < chars.len() {
            if i + 1 < chars.len() && is_overbar(chars[i + 1]) {
                bars.push((plain.chars().count(), chars[i]));
                plain.push(chars[i]);
                i += 2;
            } else if is_overbar(chars[i]) {
                i += 1;
            } else {
                plain.push(chars[i]);
                i += 1;
            }
        }
        let w = text_width(&plain, st.size, st.bold);
        let x0 = match st.anchor {
            Anchor::Start => 0.0,
            Anchor::Middle => -w / 2.0,
            Anchor::End => -w,
        };
        let font = if st.bold { "F2" } else { "F1" };
        let (c, s) = if st.rotate == 0.0 {
            (1.0, 0.0)
        } else {
            let r = st.rotate.to_radians();
            (r.cos(), r.sin())
        };
        // The page is flipped (y down); a counter-clockwise rotation on screen is a clockwise one
        // in PDF space, and the text matrix must flip y back so glyphs are upright.
        let ty = text_colour_state(self, st.color);
        let _ = writeln!(
            self.content,
            "q\n{ty}BT\n/{font} {} Tf\n{} {} {} {} {} {} Tm",
            n(st.size),
            n(c),
            n(-s),
            n(-s),
            n(-c),
            n(at.x + x0 * c),
            n(at.y - x0 * s)
        );
        let mut bytes = pdf_text(&plain);
        let mut line = Vec::from(&b"("[..]);
        line.append(&mut bytes);
        line.extend_from_slice(b") Tj\nET\n");
        self.content.push_str(&String::from_utf8_lossy(&line));
        // Overlines.
        let pl: Vec<char> = plain.chars().collect();
        for (idx, _) in bars {
            let before: String = pl[..idx].iter().collect();
            let letter: String = pl[idx..=idx].iter().collect();
            let a = x0 + text_width(&before, st.size, st.bold);
            let b = a + text_width(&letter, st.size, st.bold);
            let h = st.size * 0.82;
            let (p0, p1) = ((at.x + a * c - h * s, at.y - a * s - h * c), (at.x + b * c - h * s, at.y - b * s - h * c));
            let _ = writeln!(
                self.content,
                "{} RG {} w {} {} m {} {} l S",
                rgb(st.color),
                n(st.size * 0.07),
                n(p0.0),
                n(p0.1),
                n(p1.0),
                n(p1.1)
            );
        }
        self.content.push_str("Q\n");
    }
}

fn text_colour_state(p: &mut Page, c: Color) -> String {
    let mut s = String::new();
    if c.a < 255 {
        let i = p.alpha(c.a);
        let _ = writeln!(s, "/GS{i} gs");
    }
    let _ = writeln!(s, "{} rg", rgb(c));
    s
}

/// Render a scene as a one-page PDF document.
pub fn to_pdf(scene: &Scene) -> Vec<u8> {
    let (w, h) = (scene.width.max(1.0), scene.height.max(1.0));
    let mut pg = Page { content: String::new(), alphas: Vec::new(), clips: 0 };
    // Flip to a top-left origin.
    let _ = writeln!(pg.content, "1 0 0 -1 0 {} cm", n(h));
    pg.set_fill(scene.background);
    let _ = writeln!(pg.content, "0 0 {} {} re f", n(w), n(h));
    for it in &scene.items {
        match &it.shape {
            Shape::Line(a, b) => {
                if let Some(s) = it.stroke {
                    pg.path(&[*a, *b], false);
                    pg.paint(None, Some(s));
                }
            }
            Shape::Polyline(v) => {
                if let Some(s) = it.stroke {
                    pg.path(v, false);
                    pg.paint(None, Some(s));
                }
            }
            Shape::Polygon(v) => {
                pg.path(v, true);
                pg.paint(it.fill, it.stroke);
            }
            Shape::Rect(r) => {
                let _ = writeln!(pg.content, "{} {} {} {} re", n(r.x), n(r.y), n(r.w), n(r.h));
                pg.paint(it.fill, it.stroke);
            }
            Shape::Circle { c, r } => {
                pg.circle(*c, *r);
                pg.paint(it.fill, it.stroke);
            }
            Shape::Wedge { c, r, a0, a1 } => {
                if (a1 - a0).abs() >= std::f64::consts::TAU - 1e-9 {
                    pg.circle(*c, *r);
                } else {
                    let segs = (((a1 - a0).abs() / 0.1).ceil() as usize).clamp(2, 240);
                    let mut pts = vec![*c];
                    pts.extend(arc_points(*c, *r, *a0, *a1, segs));
                    pg.path(&pts, true);
                }
                pg.paint(it.fill, it.stroke);
            }
            Shape::Text { at, text, style } => pg.text(*at, text, style),
            Shape::Marker { at, kind, size } => match marker_geom(*kind, *at, *size) {
                MarkerGeom::Disc(c, r) => {
                    pg.circle(c, r);
                    pg.paint(it.fill, it.stroke);
                }
                MarkerGeom::Ring(c, r) => {
                    pg.circle(c, r);
                    pg.paint(None, Some(Stroke::new(it.fill.unwrap_or(Color::BLACK), 1.5)));
                }
                MarkerGeom::Polygon(v) => {
                    pg.path(&v, true);
                    pg.paint(it.fill, it.stroke);
                }
                MarkerGeom::Segments(v) => {
                    for (a, b) in v {
                        pg.path(&[a, b], false);
                    }
                    pg.paint(None, Some(Stroke::new(it.fill.unwrap_or(Color::BLACK), 1.6)));
                }
            },
            Shape::ClipPush(r) => {
                pg.clips += 1;
                let _ = writeln!(pg.content, "q\n{} {} {} {} re W n", n(r.x), n(r.y), n(r.w), n(r.h));
            }
            Shape::ClipPop => {
                if pg.clips > 0 {
                    pg.clips -= 1;
                    pg.content.push_str("Q\n");
                }
            }
        }
    }
    for _ in 0..pg.clips {
        pg.content.push_str("Q\n");
    }

    // Objects: 1 catalog, 2 pages, 3 page, 4 content, 5 F1, 6 F2, 7 resources.
    let mut gs = String::new();
    for (i, a) in pg.alphas.iter().enumerate() {
        let _ = write!(
            gs,
            "/GS{i} << /Type /ExtGState /ca {} /CA {} >> ",
            n(f64::from(*a) / 255.0),
            n(f64::from(*a) / 255.0)
        );
    }
    let content = pg.content.into_bytes();
    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
    let obj = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &[u8]| {
        offsets.push(out.len());
        let id = offsets.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };
    obj(&mut out, &mut offsets, b"<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut out, &mut offsets, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut out,
        &mut offsets,
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Contents 4 0 R /Resources 7 0 R >>", n(w), n(h))
            .as_bytes(),
    );
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(&content);
    stream.extend_from_slice(b"\nendstream");
    obj(&mut out, &mut offsets, &stream);
    obj(&mut out, &mut offsets, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
    obj(
        &mut out,
        &mut offsets,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    );
    obj(&mut out, &mut offsets, format!("<< /Font << /F1 5 0 R /F2 6 0 R >> /ExtGState << {gs}>> >>").as_bytes());
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes());
    for o in &offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", offsets.len() + 1).as_bytes(),
    );
    out
}
