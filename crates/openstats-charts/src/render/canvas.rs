//! Canvas 2D draw-command stream.
//!
//! A scene becomes a flat list of [`Command`]s that map one-to-one onto the
//! HTML5 `CanvasRenderingContext2D` API (`moveTo`, `lineTo`, `rect`, `arc`,
//! `fillText`, `clip`, …), so a WebAssembly host can replay it with no further
//! geometry. Two encodings are provided: JSON text and a compact binary form
//! (little-endian, `f32` coordinates), both lossless up to `f32` precision.

use super::{marker_geom, MarkerGeom};
use crate::core::font::is_overbar;
use crate::core::scene::{Anchor, Scene, Shape};
use crate::core::style::{Color, Stroke};

/// One drawing command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Resize the canvas and fill the background.
    Clear {
        /// Width.
        w: f32,
        /// Height.
        h: f32,
        /// Background colour.
        color: Color,
    },
    /// `ctx.save()`.
    Save,
    /// `ctx.restore()`.
    Restore,
    /// Intersect the clip region with a rectangle.
    Clip {
        /// Left.
        x: f32,
        /// Top.
        y: f32,
        /// Width.
        w: f32,
        /// Height.
        h: f32,
    },
    /// Set the fill colour.
    FillStyle(Color),
    /// Set the stroke colour, width and dash pattern.
    StrokeStyle {
        /// Colour.
        color: Color,
        /// Line width.
        width: f32,
        /// Dash pattern (empty = solid).
        dash: Vec<f32>,
    },
    /// `ctx.beginPath()`.
    BeginPath,
    /// `ctx.moveTo`.
    MoveTo(f32, f32),
    /// `ctx.lineTo`.
    LineTo(f32, f32),
    /// `ctx.arc(x, y, r, a0, a1)`.
    Arc {
        /// Centre x.
        x: f32,
        /// Centre y.
        y: f32,
        /// Radius.
        r: f32,
        /// Start angle (radians).
        a0: f32,
        /// End angle (radians).
        a1: f32,
    },
    /// `ctx.rect` (adds a rectangle sub-path).
    Rect {
        /// Left.
        x: f32,
        /// Top.
        y: f32,
        /// Width.
        w: f32,
        /// Height.
        h: f32,
    },
    /// `ctx.closePath()`.
    ClosePath,
    /// `ctx.fill()`.
    Fill,
    /// `ctx.stroke()`.
    Stroke,
    /// `ctx.fillText` with the given font and alignment.
    Text {
        /// Anchor x.
        x: f32,
        /// Baseline y.
        y: f32,
        /// String.
        text: String,
        /// Font size.
        size: f32,
        /// Colour.
        color: Color,
        /// Bold.
        bold: bool,
        /// 0 = start, 1 = middle, 2 = end.
        anchor: u8,
        /// Rotation in degrees counter-clockwise.
        rotate: f32,
        /// Draw a bar over the string's characters flagged by the combining macron (indices of characters).
        overbars: Vec<u16>,
    },
}

fn f(v: f64) -> f32 {
    v as f32
}

fn stroke_cmd(s: &Stroke) -> Command {
    Command::StrokeStyle {
        color: s.color,
        width: f(s.width),
        dash: s.dash.pattern(s.width).into_iter().map(f).collect(),
    }
}

/// Convert a scene to draw commands.
pub fn to_commands(scene: &Scene) -> Vec<Command> {
    let mut c = vec![Command::Clear { w: f(scene.width), h: f(scene.height), color: scene.background }];
    for it in &scene.items {
        let paint = |c: &mut Vec<Command>| {
            if let Some(fl) = it.fill {
                c.push(Command::FillStyle(fl));
                c.push(Command::Fill);
            }
            if let Some(s) = &it.stroke {
                c.push(stroke_cmd(s));
                c.push(Command::Stroke);
            }
        };
        match &it.shape {
            Shape::Line(a, b) => {
                c.push(Command::BeginPath);
                c.push(Command::MoveTo(f(a.x), f(a.y)));
                c.push(Command::LineTo(f(b.x), f(b.y)));
                if let Some(s) = &it.stroke {
                    c.push(stroke_cmd(s));
                    c.push(Command::Stroke);
                }
            }
            Shape::Polyline(v) | Shape::Polygon(v) => {
                c.push(Command::BeginPath);
                for (i, p) in v.iter().enumerate() {
                    c.push(if i == 0 { Command::MoveTo(f(p.x), f(p.y)) } else { Command::LineTo(f(p.x), f(p.y)) });
                }
                if matches!(it.shape, Shape::Polygon(_)) {
                    c.push(Command::ClosePath);
                    paint(&mut c);
                } else if let Some(s) = &it.stroke {
                    c.push(stroke_cmd(s));
                    c.push(Command::Stroke);
                }
            }
            Shape::Rect(r) => {
                c.push(Command::BeginPath);
                c.push(Command::Rect { x: f(r.x), y: f(r.y), w: f(r.w), h: f(r.h) });
                paint(&mut c);
            }
            Shape::Circle { c: ce, r } => {
                c.push(Command::BeginPath);
                c.push(Command::Arc { x: f(ce.x), y: f(ce.y), r: f(*r), a0: 0.0, a1: f(2.0 * std::f64::consts::PI) });
                paint(&mut c);
            }
            Shape::Wedge { c: ce, r, a0, a1 } => {
                c.push(Command::BeginPath);
                c.push(Command::MoveTo(f(ce.x), f(ce.y)));
                c.push(Command::Arc { x: f(ce.x), y: f(ce.y), r: f(*r), a0: f(*a0), a1: f(*a1) });
                c.push(Command::ClosePath);
                paint(&mut c);
            }
            Shape::Text { at, text, style } => {
                let mut clean = String::new();
                let mut bars = Vec::new();
                for ch in text.chars() {
                    if is_overbar(ch) {
                        if !clean.is_empty() {
                            bars.push((clean.chars().count() - 1) as u16);
                        }
                    } else {
                        clean.push(ch);
                    }
                }
                c.push(Command::Text {
                    x: f(at.x),
                    y: f(at.y),
                    text: clean,
                    size: f(style.size),
                    color: style.color,
                    bold: style.bold,
                    anchor: match style.anchor {
                        Anchor::Start => 0,
                        Anchor::Middle => 1,
                        Anchor::End => 2,
                    },
                    rotate: f(style.rotate),
                    overbars: bars,
                });
            }
            Shape::Marker { at, kind, size } => match marker_geom(*kind, *at, *size) {
                MarkerGeom::Disc(ce, r) => {
                    c.push(Command::BeginPath);
                    c.push(Command::Arc {
                        x: f(ce.x),
                        y: f(ce.y),
                        r: f(r),
                        a0: 0.0,
                        a1: f(2.0 * std::f64::consts::PI),
                    });
                    paint(&mut c);
                }
                MarkerGeom::Ring(ce, r) => {
                    c.push(Command::BeginPath);
                    c.push(Command::Arc {
                        x: f(ce.x),
                        y: f(ce.y),
                        r: f(r),
                        a0: 0.0,
                        a1: f(2.0 * std::f64::consts::PI),
                    });
                    c.push(stroke_cmd(&Stroke::new(it.fill.unwrap_or(Color::BLACK), 1.5)));
                    c.push(Command::Stroke);
                }
                MarkerGeom::Polygon(v) => {
                    c.push(Command::BeginPath);
                    for (i, p) in v.iter().enumerate() {
                        c.push(if i == 0 { Command::MoveTo(f(p.x), f(p.y)) } else { Command::LineTo(f(p.x), f(p.y)) });
                    }
                    c.push(Command::ClosePath);
                    paint(&mut c);
                }
                MarkerGeom::Segments(v) => {
                    c.push(Command::BeginPath);
                    for (a, b) in v {
                        c.push(Command::MoveTo(f(a.x), f(a.y)));
                        c.push(Command::LineTo(f(b.x), f(b.y)));
                    }
                    c.push(stroke_cmd(&Stroke::new(it.fill.unwrap_or(Color::BLACK), 1.6)));
                    c.push(Command::Stroke);
                }
            },
            Shape::ClipPush(r) => {
                c.push(Command::Save);
                c.push(Command::BeginPath);
                c.push(Command::Rect { x: f(r.x), y: f(r.y), w: f(r.w), h: f(r.h) });
                c.push(Command::Clip { x: f(r.x), y: f(r.y), w: f(r.w), h: f(r.h) });
            }
            Shape::ClipPop => c.push(Command::Restore),
        }
    }
    c
}

fn jf(v: f32) -> String {
    if v.is_finite() {
        let s = format!("{v}");
        s
    } else {
        "0".into()
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn jcolor(c: &Color) -> String {
    format!("[{},{},{},{}]", c.r, c.g, c.b, c.a)
}

/// JSON encoding: an array of `{"op": ..., ...}` objects.
pub fn to_json(cmds: &[Command]) -> String {
    let mut o = String::from("[");
    for (i, c) in cmds.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        let s = match c {
            Command::Clear { w, h, color } => format!("{{\"op\":\"Clear\",\"w\":{},\"h\":{},\"color\":{}}}", jf(*w), jf(*h), jcolor(color)),
            Command::Save => "{\"op\":\"Save\"}".into(),
            Command::Restore => "{\"op\":\"Restore\"}".into(),
            Command::Clip { x, y, w, h } => format!("{{\"op\":\"Clip\",\"x\":{},\"y\":{},\"w\":{},\"h\":{}}}", jf(*x), jf(*y), jf(*w), jf(*h)),
            Command::FillStyle(c) => format!("{{\"op\":\"FillStyle\",\"color\":{}}}", jcolor(c)),
            Command::StrokeStyle { color, width, dash } => format!(
                "{{\"op\":\"StrokeStyle\",\"color\":{},\"width\":{},\"dash\":[{}]}}",
                jcolor(color),
                jf(*width),
                dash.iter().map(|d| jf(*d)).collect::<Vec<_>>().join(",")
            ),
            Command::BeginPath => "{\"op\":\"BeginPath\"}".into(),
            Command::MoveTo(x, y) => format!("{{\"op\":\"MoveTo\",\"x\":{},\"y\":{}}}", jf(*x), jf(*y)),
            Command::LineTo(x, y) => format!("{{\"op\":\"LineTo\",\"x\":{},\"y\":{}}}", jf(*x), jf(*y)),
            Command::Arc { x, y, r, a0, a1 } => {
                format!("{{\"op\":\"Arc\",\"x\":{},\"y\":{},\"r\":{},\"a0\":{},\"a1\":{}}}", jf(*x), jf(*y), jf(*r), jf(*a0), jf(*a1))
            }
            Command::Rect { x, y, w, h } => format!("{{\"op\":\"Rect\",\"x\":{},\"y\":{},\"w\":{},\"h\":{}}}", jf(*x), jf(*y), jf(*w), jf(*h)),
            Command::ClosePath => "{\"op\":\"ClosePath\"}".into(),
            Command::Fill => "{\"op\":\"Fill\"}".into(),
            Command::Stroke => "{\"op\":\"Stroke\"}".into(),
            Command::Text { x, y, text, size, color, bold, anchor, rotate, overbars } => format!(
                "{{\"op\":\"Text\",\"x\":{},\"y\":{},\"text\":{},\"size\":{},\"color\":{},\"bold\":{},\"anchor\":{},\"rotate\":{},\"overbars\":[{}]}}",
                jf(*x),
                jf(*y),
                jstr(text),
                jf(*size),
                jcolor(color),
                bold,
                anchor,
                jf(*rotate),
                overbars.iter().map(u16::to_string).collect::<Vec<_>>().join(",")
            ),
        };
        o.push_str(&s);
    }
    o.push(']');
    o
}

const MAGIC: &[u8; 4] = b"OTC1";

fn put_f(o: &mut Vec<u8>, v: f32) {
    o.extend_from_slice(&v.to_le_bytes());
}

fn put_c(o: &mut Vec<u8>, c: &Color) {
    o.extend_from_slice(&[c.r, c.g, c.b, c.a]);
}

/// Compact binary encoding: `"OTC1"`, then per command an opcode byte and its operands.
pub fn to_binary(cmds: &[Command]) -> Vec<u8> {
    let mut o = MAGIC.to_vec();
    o.extend_from_slice(&(cmds.len() as u32).to_le_bytes());
    for c in cmds {
        match c {
            Command::Clear { w, h, color } => {
                o.push(0);
                put_f(&mut o, *w);
                put_f(&mut o, *h);
                put_c(&mut o, color);
            }
            Command::Save => o.push(1),
            Command::Restore => o.push(2),
            Command::Clip { x, y, w, h } => {
                o.push(3);
                for v in [x, y, w, h] {
                    put_f(&mut o, *v);
                }
            }
            Command::FillStyle(c) => {
                o.push(4);
                put_c(&mut o, c);
            }
            Command::StrokeStyle { color, width, dash } => {
                o.push(5);
                put_c(&mut o, color);
                put_f(&mut o, *width);
                o.push(dash.len() as u8);
                for d in dash {
                    put_f(&mut o, *d);
                }
            }
            Command::BeginPath => o.push(6),
            Command::MoveTo(x, y) => {
                o.push(7);
                put_f(&mut o, *x);
                put_f(&mut o, *y);
            }
            Command::LineTo(x, y) => {
                o.push(8);
                put_f(&mut o, *x);
                put_f(&mut o, *y);
            }
            Command::Arc { x, y, r, a0, a1 } => {
                o.push(9);
                for v in [x, y, r, a0, a1] {
                    put_f(&mut o, *v);
                }
            }
            Command::Rect { x, y, w, h } => {
                o.push(10);
                for v in [x, y, w, h] {
                    put_f(&mut o, *v);
                }
            }
            Command::ClosePath => o.push(11),
            Command::Fill => o.push(12),
            Command::Stroke => o.push(13),
            Command::Text { x, y, text, size, color, bold, anchor, rotate, overbars } => {
                o.push(14);
                put_f(&mut o, *x);
                put_f(&mut o, *y);
                put_f(&mut o, *size);
                put_f(&mut o, *rotate);
                put_c(&mut o, color);
                o.push(u8::from(*bold));
                o.push(*anchor);
                let b = text.as_bytes();
                o.extend_from_slice(&(b.len() as u16).to_le_bytes());
                o.extend_from_slice(b);
                o.push(overbars.len() as u8);
                for v in overbars {
                    o.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    o
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.i..self.i + n)?;
        self.i += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn f(&mut self) -> Option<f32> {
        let s = self.take(4)?;
        Some(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn c(&mut self) -> Option<Color> {
        let s = self.take(4)?;
        Some(Color { r: s[0], g: s[1], b: s[2], a: s[3] })
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.take(2)?;
        Some(u16::from_le_bytes([s[0], s[1]]))
    }
}

/// Decode the binary encoding (`None` for malformed input).
pub fn from_binary(b: &[u8]) -> Option<Vec<Command>> {
    if b.len() < 8 || &b[..4] != MAGIC {
        return None;
    }
    let n = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let mut r = Rd { b, i: 8 };
    let mut out = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        let op = r.u8()?;
        out.push(match op {
            0 => Command::Clear { w: r.f()?, h: r.f()?, color: r.c()? },
            1 => Command::Save,
            2 => Command::Restore,
            3 => Command::Clip { x: r.f()?, y: r.f()?, w: r.f()?, h: r.f()? },
            4 => Command::FillStyle(r.c()?),
            5 => {
                let color = r.c()?;
                let width = r.f()?;
                let k = r.u8()?;
                let mut dash = Vec::new();
                for _ in 0..k {
                    dash.push(r.f()?);
                }
                Command::StrokeStyle { color, width, dash }
            }
            6 => Command::BeginPath,
            7 => Command::MoveTo(r.f()?, r.f()?),
            8 => Command::LineTo(r.f()?, r.f()?),
            9 => Command::Arc { x: r.f()?, y: r.f()?, r: r.f()?, a0: r.f()?, a1: r.f()? },
            10 => Command::Rect { x: r.f()?, y: r.f()?, w: r.f()?, h: r.f()? },
            11 => Command::ClosePath,
            12 => Command::Fill,
            13 => Command::Stroke,
            14 => {
                let (x, y, size, rotate) = (r.f()?, r.f()?, r.f()?, r.f()?);
                let color = r.c()?;
                let bold = r.u8()? != 0;
                let anchor = r.u8()?;
                let len = r.u16()? as usize;
                let text = String::from_utf8(r.take(len)?.to_vec()).ok()?;
                let k = r.u8()?;
                let mut overbars = Vec::new();
                for _ in 0..k {
                    overbars.push(r.u16()?);
                }
                Command::Text { x, y, text, size, color, bold, anchor, rotate, overbars }
            }
            _ => return None,
        });
    }
    (r.i == b.len()).then_some(out)
}

/// A small JavaScript player that replays the JSON stream on a 2D canvas context
/// (`play(ctx, commands)`). Provided for the browser target; it is plain data to this crate.
pub const PLAYER_JS: &str = r#"function play(ctx, cmds) {
  const rgba = c => `rgba(${c[0]},${c[1]},${c[2]},${c[3] / 255})`;
  for (const c of cmds) {
    switch (c.op) {
      case 'Clear': ctx.canvas.width = c.w; ctx.canvas.height = c.h; ctx.fillStyle = rgba(c.color); ctx.fillRect(0, 0, c.w, c.h); break;
      case 'Save': ctx.save(); break;
      case 'Restore': ctx.restore(); break;
      case 'Clip': ctx.clip(); break;
      case 'FillStyle': ctx.fillStyle = rgba(c.color); break;
      case 'StrokeStyle': ctx.strokeStyle = rgba(c.color); ctx.lineWidth = c.width; ctx.setLineDash(c.dash); ctx.lineJoin = 'round'; break;
      case 'BeginPath': ctx.beginPath(); break;
      case 'MoveTo': ctx.moveTo(c.x, c.y); break;
      case 'LineTo': ctx.lineTo(c.x, c.y); break;
      case 'Arc': ctx.arc(c.x, c.y, c.r, c.a0, c.a1); break;
      case 'Rect': ctx.rect(c.x, c.y, c.w, c.h); break;
      case 'ClosePath': ctx.closePath(); break;
      case 'Fill': ctx.fill(); break;
      case 'Stroke': ctx.stroke(); break;
      case 'Text': {
        ctx.save();
        ctx.font = `${c.bold ? 'bold ' : ''}${c.size}px Helvetica, Arial, sans-serif`;
        ctx.fillStyle = rgba(c.color);
        ctx.textAlign = ['left', 'center', 'right'][c.anchor];
        ctx.translate(c.x, c.y);
        if (c.rotate) ctx.rotate(-c.rotate * Math.PI / 180);
        ctx.fillText(c.text, 0, 0);
        const total = ctx.measureText(c.text).width;
        for (const i of c.overbars) {
          const before = ctx.measureText(c.text.slice(0, i)).width, w = ctx.measureText(c.text[i]).width;
          const off = c.anchor === 1 ? -total / 2 : c.anchor === 2 ? -total : 0;
          ctx.fillRect(off + before, -c.size * 0.9, w, Math.max(1, c.size * 0.07));
        }
        ctx.restore();
        break;
      }
    }
  }
}
"#;
