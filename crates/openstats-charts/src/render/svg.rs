//! Standalone SVG output.

use super::{marker_geom, MarkerGeom};
use crate::core::font::is_overbar;
use crate::core::geom::Point;
use crate::core::scene::{Anchor, Scene, Shape, TextStyle};
use crate::core::style::{Color, Stroke};
use std::fmt::Write;

fn n(v: f64) -> String {
    let s = format!("{:.2}", if v == 0.0 { 0.0 } else { v });
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() {
        "0".into()
    } else {
        s.to_string()
    }
}

/// Escape text for XML content and attribute values.
pub fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&apos;"),
            c if (c as u32) < 0x20 && c != '\t' && c != '\n' => {}
            c => o.push(c),
        }
    }
    o
}

fn paint(fill: Option<Color>, stroke: Option<Stroke>) -> String {
    let mut s = String::new();
    match fill {
        Some(c) if c.a > 0 => {
            let _ = write!(s, " fill=\"{}\"", c.css());
            if c.a < 255 {
                let _ = write!(s, " fill-opacity=\"{}\"", n(c.opacity()));
            }
        }
        _ => s.push_str(" fill=\"none\""),
    }
    if let Some(st) = stroke {
        let _ = write!(s, " stroke=\"{}\" stroke-width=\"{}\"", st.color.css(), n(st.width));
        if st.color.a < 255 {
            let _ = write!(s, " stroke-opacity=\"{}\"", n(st.color.opacity()));
        }
        let d = st.dash.pattern(st.width);
        if !d.is_empty() {
            let _ = write!(s, " stroke-dasharray=\"{}\"", d.iter().map(|v| n(*v)).collect::<Vec<_>>().join(" "));
        }
        s.push_str(" stroke-linejoin=\"round\"");
    }
    s
}

fn pts(v: &[Point]) -> String {
    v.iter().map(|p| format!("{},{}", n(p.x), n(p.y))).collect::<Vec<_>>().join(" ")
}

fn text_element(at: Point, text: &str, st: &TextStyle) -> String {
    let anchor = match st.anchor {
        Anchor::Start => "start",
        Anchor::Middle => "middle",
        Anchor::End => "end",
    };
    let mut o = format!(
        "<text x=\"{}\" y=\"{}\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{}\" fill=\"{}\" text-anchor=\"{}\"",
        n(at.x),
        n(at.y),
        n(st.size),
        st.color.css(),
        anchor
    );
    if st.bold {
        o.push_str(" font-weight=\"bold\"");
    }
    if st.rotate != 0.0 {
        let _ = write!(o, " transform=\"rotate({} {} {})\"", n(-st.rotate), n(at.x), n(at.y));
    }
    o.push('>');
    // A combining macron / overline decorates the preceding character.
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut plain = String::new();
    while i < chars.len() {
        if i + 1 < chars.len() && is_overbar(chars[i + 1]) {
            o.push_str(&escape(&plain));
            plain.clear();
            let _ = write!(o, "<tspan text-decoration=\"overline\">{}</tspan>", escape(&chars[i].to_string()));
            i += 2;
        } else if is_overbar(chars[i]) {
            i += 1;
        } else {
            plain.push(chars[i]);
            i += 1;
        }
    }
    o.push_str(&escape(&plain));
    o.push_str("</text>");
    o
}

/// Render a scene as a standalone SVG document.
pub fn to_svg(scene: &Scene) -> String {
    let mut o = String::with_capacity(4096 + scene.items.len() * 96);
    let _ = writeln!(
        o,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" role=\"img\">",
        w = n(scene.width),
        h = n(scene.height)
    );
    if !scene.title.is_empty() {
        let _ = writeln!(o, "<title>{}</title>", escape(&scene.title));
    }
    let _ = writeln!(
        o,
        "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
        n(scene.width),
        n(scene.height),
        scene.background.css()
    );
    let mut clip_id = 0;
    for it in &scene.items {
        let pa = paint(it.fill, it.stroke);
        match &it.shape {
            Shape::Line(a, b) => {
                let _ = writeln!(
                    o,
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"{}/>",
                    n(a.x),
                    n(a.y),
                    n(b.x),
                    n(b.y),
                    paint(None, it.stroke)
                );
            }
            Shape::Polyline(v) => {
                let _ = writeln!(o, "<polyline points=\"{}\"{}/>", pts(v), paint(None, it.stroke));
            }
            Shape::Polygon(v) => {
                let _ = writeln!(o, "<polygon points=\"{}\"{}/>", pts(v), pa);
            }
            Shape::Rect(r) => {
                let _ = writeln!(
                    o,
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"{}/>",
                    n(r.x),
                    n(r.y),
                    n(r.w),
                    n(r.h),
                    pa
                );
            }
            Shape::Circle { c, r } => {
                let _ = writeln!(o, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>", n(c.x), n(c.y), n(*r), pa);
            }
            Shape::Wedge { c, r, a0, a1 } => {
                let (p0, p1) = (
                    Point::new(c.x + r * a0.cos(), c.y + r * a0.sin()),
                    Point::new(c.x + r * a1.cos(), c.y + r * a1.sin()),
                );
                let large = i32::from((a1 - a0).abs() > std::f64::consts::PI);
                if (a1 - a0).abs() >= 2.0 * std::f64::consts::PI - 1e-9 {
                    let _ = writeln!(o, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>", n(c.x), n(c.y), n(*r), pa);
                } else {
                    let _ = writeln!(
                        o,
                        "<path d=\"M{} {} L{} {} A{} {} 0 {} 1 {} {} Z\"{}/>",
                        n(c.x),
                        n(c.y),
                        n(p0.x),
                        n(p0.y),
                        n(*r),
                        n(*r),
                        large,
                        n(p1.x),
                        n(p1.y),
                        pa
                    );
                }
            }
            Shape::Text { at, text, style } => {
                let _ = writeln!(o, "{}", text_element(*at, text, style));
            }
            Shape::Marker { at, kind, size } => match marker_geom(*kind, *at, *size) {
                MarkerGeom::Disc(c, r) => {
                    let _ = writeln!(o, "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>", n(c.x), n(c.y), n(r), pa);
                }
                MarkerGeom::Ring(c, r) => {
                    let col = it.fill.unwrap_or(Color::BLACK);
                    let _ = writeln!(
                        o,
                        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>",
                        n(c.x),
                        n(c.y),
                        n(r),
                        paint(None, Some(Stroke::new(col, 1.5)))
                    );
                }
                MarkerGeom::Polygon(v) => {
                    let _ = writeln!(o, "<polygon points=\"{}\"{}/>", pts(&v), pa);
                }
                MarkerGeom::Segments(v) => {
                    let col = it.fill.unwrap_or(Color::BLACK);
                    let d: Vec<String> =
                        v.iter().map(|(a, b)| format!("M{} {} L{} {}", n(a.x), n(a.y), n(b.x), n(b.y))).collect();
                    let _ = writeln!(o, "<path d=\"{}\"{}/>", d.join(" "), paint(None, Some(Stroke::new(col, 1.6))));
                }
            },
            Shape::ClipPush(r) => {
                clip_id += 1;
                let _ = writeln!(
                    o,
                    "<clipPath id=\"clip{id}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath><g clip-path=\"url(#clip{id})\">",
                    n(r.x),
                    n(r.y),
                    n(r.w),
                    n(r.h),
                    id = clip_id
                );
            }
            Shape::ClipPop => o.push_str("</g>\n"),
        }
    }
    o.push_str("</svg>\n");
    o
}
