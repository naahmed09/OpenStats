//! Software anti-aliased rasteriser.
//!
//! * Polygons: scanline fill with 4 sub-scanlines per pixel and exact horizontal
//!   coverage, even-odd rule.
//! * Lines: Xiaolin Wu for hairlines (≤ 1 px), axis-aligned lines snapped to the
//!   pixel grid for crisp frames, quads with round joins for thicker strokes,
//!   dash patterns applied along the path.
//! * Text: 5 × 8 bitmap glyphs scaled by exact area coverage, placed with the
//!   proportional metrics of [`crate::core::font`]; 0° and ±90° rotations.

use super::png::{encode, Image};
use super::{arc_points, marker_geom, MarkerGeom};
use crate::core::font::{char_units, glyph_bitmap, is_overbar};
use crate::core::geom::Point;
use crate::core::scene::{Anchor, Scene, Shape, TextStyle};
use crate::core::style::{Color, Stroke};

const SUB: usize = 4;

/// One run of text rasterised by an outside font engine, in device pixels.
///
/// `left` and `top` locate the top-left of `coverage` relative to the pen position on the
/// baseline (`left` to the right, `top` upwards), so a glyph with a 10 px ascent has `top = 10`.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRun {
    /// Total advance of the run in pixels.
    pub advance: f32,
    /// Horizontal offset of the bitmap from the pen position.
    pub left: i32,
    /// Distance from the baseline up to the bitmap's first row.
    pub top: i32,
    /// Bitmap width.
    pub width: usize,
    /// Bitmap height.
    pub height: usize,
    /// 8-bit coverage, row-major, `width × height`.
    pub coverage: Vec<u8>,
}

/// A font engine plugged into [`rasterize_with`]; native shells use it to draw text with the
/// operating system's fonts instead of the built-in 5 × 8 bitmap face.
pub trait TextPainter {
    /// Rasterise `text` at `size_px` pixels. `None` makes the rasteriser fall back to its own
    /// bitmap glyphs for this run (for example when the font lacks a character).
    fn render(&mut self, text: &str, size_px: f32, bold: bool) -> Option<TextRun>;
}

struct Canvas<'a> {
    w: usize,
    h: usize,
    px: Vec<u8>,
    clips: Vec<(i64, i64, i64, i64)>,
    sc: f64,
    painter: Option<&'a mut dyn TextPainter>,
}

impl Canvas<'_> {
    fn clip(&self) -> (i64, i64, i64, i64) {
        *self.clips.last().unwrap_or(&(0, 0, self.w as i64, self.h as i64))
    }

    fn blend(&mut self, x: i64, y: i64, c: Color, cov: f64) {
        let (cx0, cy0, cx1, cy1) = self.clip();
        if x < cx0 || x >= cx1 || y < cy0 || y >= cy1 {
            return;
        }
        let a = (cov.clamp(0.0, 1.0) * c.opacity()) as f32;
        if a <= 0.0 {
            return;
        }
        let i = 4 * (y as usize * self.w + x as usize);
        // Straight-alpha "over": identical to a plain blend while the destination is opaque, and
        // keeps transparent backgrounds transparent (app icons).
        let da = f32::from(self.px[i + 3]) / 255.0;
        let oa = a + da * (1.0 - a);
        for (k, s) in [c.r, c.g, c.b].into_iter().enumerate() {
            let d = f32::from(self.px[i + k]);
            let v = (f32::from(s) * a + d * da * (1.0 - a)) / oa;
            self.px[i + k] = v.round().clamp(0.0, 255.0) as u8;
        }
        self.px[i + 3] = (oa * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    fn fill_polygon(&mut self, pts: &[Point], c: Color) {
        if pts.len() < 3 || pts.iter().any(|p| !p.is_finite()) {
            return;
        }
        let (cx0, cy0, cx1, cy1) = self.clip();
        let ymin = pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let ymax = pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let y0 = (ymin.floor() as i64).max(cy0);
        let y1 = (ymax.ceil() as i64).min(cy1);
        let mut cov = vec![0f64; self.w + 2];
        let n = pts.len();
        for y in y0..y1 {
            let (mut lo, mut hi) = (usize::MAX, 0usize);
            for s in 0..SUB {
                let sy = y as f64 + (s as f64 + 0.5) / SUB as f64;
                let mut xs: Vec<f64> = Vec::new();
                for i in 0..n {
                    let (p, q) = (pts[i], pts[(i + 1) % n]);
                    if (p.y <= sy && q.y > sy) || (q.y <= sy && p.y > sy) {
                        xs.push(p.x + (sy - p.y) / (q.y - p.y) * (q.x - p.x));
                    }
                }
                xs.sort_by(f64::total_cmp);
                for pair in xs.chunks(2) {
                    if pair.len() < 2 {
                        continue;
                    }
                    let a = pair[0].max(cx0 as f64);
                    let b = pair[1].min(cx1 as f64);
                    if b <= a {
                        continue;
                    }
                    let (ia, ib) = (a.floor() as usize, b.floor() as usize);
                    let wt = 1.0 / SUB as f64;
                    lo = lo.min(ia);
                    hi = hi.max(ib);
                    if ia == ib {
                        cov[ia] += (b - a) * wt;
                    } else {
                        cov[ia] += (ia as f64 + 1.0 - a) * wt;
                        for v in cov.iter_mut().take(ib).skip(ia + 1) {
                            *v += wt;
                        }
                        cov[ib] += (b - ib as f64) * wt;
                    }
                }
            }
            if lo != usize::MAX {
                for x in lo..=hi.min(self.w - 1) {
                    if cov[x] > 0.0 {
                        self.blend(x as i64, y, c, cov[x]);
                    }
                    cov[x] = 0.0;
                }
                for v in cov.iter_mut().take(hi + 2).skip(lo) {
                    *v = 0.0;
                }
            }
        }
    }

    fn wu_line(&mut self, a: Point, b: Point, c: Color, strength: f64) {
        let steep = (b.y - a.y).abs() > (b.x - a.x).abs();
        let (mut x0, mut y0, mut x1, mut y1) = if steep { (a.y, a.x, b.y, b.x) } else { (a.x, a.y, b.x, b.y) };
        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }
        let dx = x1 - x0;
        let grad = if dx.abs() < 1e-12 { 1.0 } else { (y1 - y0) / dx };
        let plot = |s: &mut Canvas, x: i64, y: i64, cv: f64| {
            let (px, py) = if steep { (y, x) } else { (x, y) };
            s.blend(px, py, c, cv * strength);
        };
        let rfpart = |v: f64| 1.0 - (v - v.floor());
        let xend = x0.round();
        let yend = y0 + grad * (xend - x0);
        let xgap = rfpart(x0 + 0.5);
        let xpxl1 = xend as i64;
        plot(self, xpxl1, yend.floor() as i64, rfpart(yend) * xgap);
        plot(self, xpxl1, yend.floor() as i64 + 1, (yend - yend.floor()) * xgap);
        let mut intery = yend + grad;
        let xend2 = x1.round();
        let yend2 = y1 + grad * (xend2 - x1);
        let xgap2 = (x1 + 0.5) - (x1 + 0.5).floor();
        let xpxl2 = xend2 as i64;
        for x in xpxl1 + 1..xpxl2 {
            plot(self, x, intery.floor() as i64, rfpart(intery));
            plot(self, x, intery.floor() as i64 + 1, intery - intery.floor());
            intery += grad;
        }
        plot(self, xpxl2, yend2.floor() as i64, rfpart(yend2) * xgap2);
        plot(self, xpxl2, yend2.floor() as i64 + 1, (yend2 - yend2.floor()) * xgap2);
    }

    fn segment(&mut self, a: Point, b: Point, width: f64, c: Color) {
        if !(a.is_finite() && b.is_finite()) {
            return;
        }
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return;
        }
        let axis = dx.abs() < 1e-9 || dy.abs() < 1e-9;
        if axis && width <= 2.5 {
            // Snap to the pixel grid so frame lines are crisp.
            let wi = width.round().max(1.0);
            let snap = |v: f64| if (wi as i64) % 2 == 1 { v.floor() + 0.5 } else { v.round() };
            let poly = if dy.abs() < 1e-9 {
                let y = snap(a.y);
                vec![
                    Point::new(a.x.min(b.x), y - wi / 2.0),
                    Point::new(a.x.max(b.x), y - wi / 2.0),
                    Point::new(a.x.max(b.x), y + wi / 2.0),
                    Point::new(a.x.min(b.x), y + wi / 2.0),
                ]
            } else {
                let x = snap(a.x);
                vec![
                    Point::new(x - wi / 2.0, a.y.min(b.y)),
                    Point::new(x + wi / 2.0, a.y.min(b.y)),
                    Point::new(x + wi / 2.0, a.y.max(b.y)),
                    Point::new(x - wi / 2.0, a.y.max(b.y)),
                ]
            };
            self.fill_polygon(&poly, c);
        } else if width <= 1.05 {
            self.wu_line(a, b, c, width.min(1.0));
        } else {
            let (nx, ny) = (-dy / len * width / 2.0, dx / len * width / 2.0);
            self.fill_polygon(
                &[
                    Point::new(a.x + nx, a.y + ny),
                    Point::new(b.x + nx, b.y + ny),
                    Point::new(b.x - nx, b.y - ny),
                    Point::new(a.x - nx, a.y - ny),
                ],
                c,
            );
        }
    }

    fn disc(&mut self, c: Point, r: f64, col: Color) {
        let n = ((r * 3.0).ceil() as usize).clamp(10, 96);
        let pts: Vec<Point> = arc_points(c, r, 0.0, std::f64::consts::TAU, n).into_iter().take(n).collect();
        self.fill_polygon(&pts, col);
    }

    fn polyline(&mut self, pts: &[Point], s: &Stroke, closed: bool) {
        let w = s.width * self.sc;
        let mut path: Vec<Point> = pts.to_vec();
        if closed && !pts.is_empty() {
            path.push(pts[0]);
        }
        let pattern: Vec<f64> = s.dash.pattern(s.width).iter().map(|d| d * self.sc).collect();
        if pattern.is_empty() {
            for w2 in path.windows(2) {
                self.segment(w2[0], w2[1], w, s.color);
            }
            if w >= 2.5 {
                for p in &path[1..path.len().saturating_sub(1)] {
                    self.disc(*p, w / 2.0, s.color);
                }
            }
        } else {
            let (mut idx, mut left, mut on) = (0usize, pattern[0], true);
            for w2 in path.windows(2) {
                let (a, b) = (w2[0], w2[1]);
                let len = a.dist(b);
                let mut t = 0.0;
                while t < len {
                    let step = left.min(len - t);
                    if on {
                        let f0 = t / len;
                        let f1 = (t + step) / len;
                        self.segment(
                            Point::new(a.x + (b.x - a.x) * f0, a.y + (b.y - a.y) * f0),
                            Point::new(a.x + (b.x - a.x) * f1, a.y + (b.y - a.y) * f1),
                            w,
                            s.color,
                        );
                    }
                    t += step;
                    left -= step;
                    if left <= 1e-12 {
                        idx = (idx + 1) % pattern.len();
                        left = pattern[idx];
                        on = idx % 2 == 0;
                    }
                }
            }
        }
    }

    /// Draw a run produced by an outside font engine (0° and ±90°, like the bitmap path).
    fn blit_run(&mut self, at: Point, run: &TextRun, st: &TextStyle) {
        let u0 = match st.anchor {
            Anchor::Start => 0.0,
            Anchor::Middle => -f64::from(run.advance) / 2.0,
            Anchor::End => -f64::from(run.advance),
        }
        .round();
        for iy in 0..run.height {
            for ix in 0..run.width {
                let v = run.coverage[iy * run.width + ix];
                if v == 0 {
                    continue;
                }
                // Pixel centre in text space: u along the baseline, v perpendicular (down).
                let u = u0 + f64::from(run.left) + ix as f64 + 0.5;
                let w = -f64::from(run.top) + iy as f64 + 0.5;
                let (dx, dy) = if st.rotate > 45.0 && st.rotate < 135.0 {
                    (at.x + w, at.y - u)
                } else if st.rotate < -45.0 && st.rotate > -135.0 {
                    (at.x - w, at.y + u)
                } else {
                    (at.x + u, at.y + w)
                };
                self.blend(dx.floor() as i64, dy.floor() as i64, st.color, f64::from(v) / 255.0);
            }
        }
    }

    fn text(&mut self, at: Point, text: &str, st: &TextStyle) {
        let size = st.size * self.sc;
        let run = match self.painter.as_mut() {
            Some(p) if !text.chars().any(is_overbar) => p.render(text, size as f32, st.bold),
            _ => None,
        };
        if let Some(run) = run {
            self.blit_run(at, &run, st);
            return;
        }
        let s = 0.72 * size / 7.0; // pixel size of one glyph cell
        let units = |c: char| f64::from(char_units(c)) / 1000.0 * size * if st.bold { 1.06 } else { 1.0 };
        let total: f64 = text.chars().map(units).sum();
        let u0 = match st.anchor {
            Anchor::Start => 0.0,
            Anchor::Middle => -total / 2.0,
            Anchor::End => -total,
        };
        // Collect device-space cell rectangles.
        let to_dev = |u: f64, v: f64| -> Point {
            if st.rotate > 45.0 && st.rotate < 135.0 {
                Point::new(at.x + v, at.y - u)
            } else if st.rotate < -45.0 && st.rotate > -135.0 {
                Point::new(at.x - v, at.y + u)
            } else {
                Point::new(at.x + u, at.y + v)
            }
        };
        let mut rects: Vec<(f64, f64, f64, f64)> = Vec::new();
        let mut add = |u0: f64, v0: f64, u1: f64, v1: f64| {
            let (a, b) = (to_dev(u0, v0), to_dev(u1, v1));
            rects.push((a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y)));
        };
        let mut pen = u0;
        let mut last: Option<(f64, f64)> = None;
        let weight = if st.bold { 1.0 } else { 0.0 };
        for ch in text.chars() {
            if is_overbar(ch) {
                if let Some((gu, gw)) = last {
                    add(gu, -9.4 * s, gu + gw, -9.4 * s + s.max(1.0));
                }
                continue;
            }
            let adv = units(ch);
            let g = glyph_bitmap(ch);
            let cols: Vec<usize> = (0..5).filter(|&x| (0..8).any(|y| g[y][x])).collect();
            if let (Some(&c0), Some(&c1)) = (cols.first(), cols.last()) {
                let ink_w = (c1 + 1 - c0) as f64 * s;
                let gx = pen + (adv - ink_w) / 2.0 - c0 as f64 * s;
                for (y, row) in g.iter().enumerate() {
                    for (x, &on) in row.iter().enumerate() {
                        if on {
                            let (u, v) = (gx + x as f64 * s, (y as f64 - 7.0) * s);
                            add(u, v, u + s + weight * 0.15 * s, v + s);
                        }
                    }
                }
                last = Some((gx + c0 as f64 * s, ink_w));
            } else {
                last = Some((pen, adv));
            }
            pen += adv;
        }
        if rects.is_empty() {
            return;
        }
        let x0 = rects.iter().map(|r| r.0).fold(f64::INFINITY, f64::min).floor() as i64;
        let y0 = rects.iter().map(|r| r.1).fold(f64::INFINITY, f64::min).floor() as i64;
        let x1 = rects.iter().map(|r| r.2).fold(f64::NEG_INFINITY, f64::max).ceil() as i64;
        let y1 = rects.iter().map(|r| r.3).fold(f64::NEG_INFINITY, f64::max).ceil() as i64;
        let (mw, mh) = ((x1 - x0).max(1) as usize, (y1 - y0).max(1) as usize);
        if mw * mh > 4_000_000 {
            return;
        }
        let mut mask = vec![0f64; mw * mh];
        for r in &rects {
            let (ix0, ix1) = ((r.0.floor() as i64).max(x0), (r.2.ceil() as i64).min(x1));
            let (iy0, iy1) = ((r.1.floor() as i64).max(y0), (r.3.ceil() as i64).min(y1));
            for y in iy0..iy1 {
                let oy = (r.3.min(y as f64 + 1.0) - r.1.max(y as f64)).max(0.0);
                for x in ix0..ix1 {
                    let ox = (r.2.min(x as f64 + 1.0) - r.0.max(x as f64)).max(0.0);
                    let m = &mut mask[(y - y0) as usize * mw + (x - x0) as usize];
                    *m = (*m).max(ox * oy);
                }
            }
        }
        for y in 0..mh {
            for x in 0..mw {
                let v = mask[y * mw + x];
                if v > 0.0 {
                    self.blend(x0 + x as i64, y0 + y as i64, st.color, v);
                }
            }
        }
    }
}

/// Rasterise a scene at `scale` (1.0 = one pixel per scene unit).
pub fn rasterize(scene: &Scene, scale: f64) -> Image {
    rasterize_with(scene, scale, None)
}

/// Like [`rasterize`], drawing text through `painter` when one is given.
pub fn rasterize_with(scene: &Scene, scale: f64, painter: Option<&mut dyn TextPainter>) -> Image {
    let sc = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let w = ((scene.width * sc).ceil() as usize).clamp(1, 16_384);
    let h = ((scene.height * sc).ceil() as usize).clamp(1, 16_384);
    let mut px = vec![0u8; w * h * 4];
    for p in px.chunks_mut(4) {
        p.copy_from_slice(&[scene.background.r, scene.background.g, scene.background.b, scene.background.a]);
    }
    let mut cv = Canvas { w, h, px, clips: Vec::new(), sc, painter };
    let s = |p: Point| Point::new(p.x * sc, p.y * sc);
    for it in &scene.items {
        match &it.shape {
            Shape::Line(a, b) => {
                if let Some(st) = &it.stroke {
                    cv.polyline(&[s(*a), s(*b)], st, false);
                }
            }
            Shape::Polyline(v) => {
                if let Some(st) = &it.stroke {
                    let v: Vec<Point> = v.iter().map(|p| s(*p)).collect();
                    cv.polyline(&v, st, false);
                }
            }
            Shape::Polygon(v) => {
                let v: Vec<Point> = v.iter().map(|p| s(*p)).collect();
                if let Some(f) = it.fill {
                    cv.fill_polygon(&v, f);
                }
                if let Some(st) = &it.stroke {
                    cv.polyline(&v, st, true);
                }
            }
            Shape::Rect(r) => {
                let v = [
                    s(Point::new(r.x, r.y)),
                    s(Point::new(r.right(), r.y)),
                    s(Point::new(r.right(), r.bottom())),
                    s(Point::new(r.x, r.bottom())),
                ];
                if let Some(f) = it.fill {
                    cv.fill_polygon(&v, f);
                }
                if let Some(st) = &it.stroke {
                    cv.polyline(&v, st, true);
                }
            }
            Shape::Circle { c, r } => {
                let n = ((r * sc * 3.0).ceil() as usize).clamp(12, 128);
                let v: Vec<Point> =
                    arc_points(s(*c), r * sc, 0.0, std::f64::consts::TAU, n).into_iter().take(n).collect();
                if let Some(f) = it.fill {
                    cv.fill_polygon(&v, f);
                }
                if let Some(st) = &it.stroke {
                    cv.polyline(&v, st, true);
                }
            }
            Shape::Wedge { c, r, a0, a1 } => {
                let n = (((a1 - a0).abs() * r * sc / 3.0).ceil() as usize).clamp(4, 256);
                let mut v = vec![s(*c)];
                v.extend(arc_points(s(*c), r * sc, *a0, *a1, n));
                if let Some(f) = it.fill {
                    cv.fill_polygon(&v, f);
                }
                if let Some(st) = &it.stroke {
                    cv.polyline(&v, st, true);
                }
            }
            Shape::Text { at, text, style } => cv.text(s(*at), text, style),
            Shape::Marker { at, kind, size } => match marker_geom(*kind, *at, *size) {
                MarkerGeom::Disc(c, r) => {
                    if let Some(f) = it.fill {
                        cv.disc(s(c), r * sc, f);
                    }
                    if let Some(st) = &it.stroke {
                        let n = 24;
                        let v: Vec<Point> =
                            arc_points(s(c), r * sc, 0.0, std::f64::consts::TAU, n).into_iter().take(n).collect();
                        cv.polyline(&v, st, true);
                    }
                }
                MarkerGeom::Ring(c, r) => {
                    let n = 24;
                    let v: Vec<Point> =
                        arc_points(s(c), r * sc, 0.0, std::f64::consts::TAU, n).into_iter().take(n).collect();
                    cv.polyline(&v, &Stroke::new(it.fill.unwrap_or(Color::BLACK), 1.5), true);
                }
                MarkerGeom::Polygon(v) => {
                    let v: Vec<Point> = v.iter().map(|p| s(*p)).collect();
                    if let Some(f) = it.fill {
                        cv.fill_polygon(&v, f);
                    }
                    if let Some(st) = &it.stroke {
                        cv.polyline(&v, st, true);
                    }
                }
                MarkerGeom::Segments(v) => {
                    let st = Stroke::new(it.fill.unwrap_or(Color::BLACK), 1.6);
                    for (a, b) in v {
                        cv.polyline(&[s(a), s(b)], &st, false);
                    }
                }
            },
            Shape::ClipPush(r) => {
                let (x0, y0) = ((r.x * sc).round() as i64, (r.y * sc).round() as i64);
                let (x1, y1) = ((r.right() * sc).round() as i64, (r.bottom() * sc).round() as i64);
                let (px0, py0, px1, py1) = cv.clip();
                cv.clips.push((x0.max(px0), y0.max(py0), x1.min(px1), y1.min(py1)));
            }
            Shape::ClipPop => {
                cv.clips.pop();
            }
        }
    }
    Image { width: cv.w, height: cv.h, rgba: cv.px }
}

/// Render a scene to PNG bytes.
pub fn render_png(scene: &Scene, scale: f64) -> Vec<u8> {
    encode(&rasterize(scene, scale))
}
