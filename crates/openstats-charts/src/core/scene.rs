//! The scene graph: a flat, ordered display list that every renderer consumes.

use super::geom::{Point, Rect};
use super::hit::{HitTarget, Region};
use super::scale::Scale;
use super::style::{Color, Marker, Stroke};

/// Horizontal text anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// Text starts at the anchor point.
    Start,
    /// Text is centred on it.
    Middle,
    /// Text ends at it.
    End,
}

/// How text is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextStyle {
    /// Font size in pixels.
    pub size: f64,
    /// Colour.
    pub color: Color,
    /// Bold weight.
    pub bold: bool,
    /// Horizontal anchor.
    pub anchor: Anchor,
    /// Rotation in degrees counter-clockwise (0 or 90 are supported by every renderer).
    pub rotate: f64,
}

impl TextStyle {
    /// Regular text, start-anchored, unrotated.
    pub fn new(size: f64, color: Color) -> TextStyle {
        TextStyle { size, color, bold: false, anchor: Anchor::Start, rotate: 0.0 }
    }

    /// With an anchor.
    pub fn anchored(mut self, a: Anchor) -> TextStyle {
        self.anchor = a;
        self
    }

    /// Bold.
    pub fn bold(mut self) -> TextStyle {
        self.bold = true;
        self
    }

    /// Rotated by `deg` (counter-clockwise).
    pub fn rotated(mut self, deg: f64) -> TextStyle {
        self.rotate = deg;
        self
    }
}

/// A geometric primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// Straight segment.
    Line(Point, Point),
    /// Open polyline.
    Polyline(Vec<Point>),
    /// Closed polygon.
    Polygon(Vec<Point>),
    /// Rectangle.
    Rect(Rect),
    /// Circle.
    Circle {
        /// Centre.
        c: Point,
        /// Radius.
        r: f64,
    },
    /// Pie wedge; angles in radians, 0 = east, increasing clockwise on screen.
    Wedge {
        /// Centre.
        c: Point,
        /// Radius.
        r: f64,
        /// Start angle.
        a0: f64,
        /// End angle.
        a1: f64,
    },
    /// Text whose anchor (baseline) point is `at`.
    Text {
        /// Anchor point (baseline).
        at: Point,
        /// The string.
        text: String,
        /// Style.
        style: TextStyle,
    },
    /// Marker glyph centred at `at`, `size` pixels across.
    Marker {
        /// Centre.
        at: Point,
        /// Shape.
        kind: Marker,
        /// Size.
        size: f64,
    },
    /// Start clipping to a rectangle.
    ClipPush(Rect),
    /// End the innermost clip.
    ClipPop,
}

/// A shape with its paint.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    /// Geometry.
    pub shape: Shape,
    /// Outline (none for fill-only).
    pub stroke: Option<Stroke>,
    /// Fill colour.
    pub fill: Option<Color>,
}

/// Where the data axes of a panel are, for mapping pointer positions to data.
#[derive(Debug, Clone, PartialEq)]
pub struct AxesMeta {
    /// Plot rectangle in scene pixels.
    pub plot: Rect,
    /// x scale.
    pub x: Scale,
    /// y scale.
    pub y: Scale,
}

impl AxesMeta {
    /// Data coordinates of a scene point (NaN outside the plot area).
    pub fn to_data(&self, p: Point) -> Point {
        if !self.plot.contains(p) {
            return Point::new(f64::NAN, f64::NAN);
        }
        let tx = (p.x - self.plot.x) / self.plot.w;
        let ty = 1.0 - (p.y - self.plot.y) / self.plot.h;
        Point::new(self.x.invert(tx), self.y.invert(ty))
    }

    /// Scene point of a data coordinate.
    pub fn to_pixel(&self, d: Point) -> Point {
        Point::new(self.x.map(d.x, self.plot.x, self.plot.right()), self.y.map(d.y, self.plot.bottom(), self.plot.y))
    }
}

/// A complete drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// Width in pixels.
    pub width: f64,
    /// Height in pixels.
    pub height: f64,
    /// Background fill.
    pub background: Color,
    /// Title (an accessible description for SVG).
    pub title: String,
    /// Display list, back to front.
    pub items: Vec<Item>,
    /// Interactive targets (hovered / clicked data points).
    pub targets: Vec<HitTarget>,
    /// Data axes of each panel.
    pub axes: Vec<AxesMeta>,
}

impl Scene {
    /// Empty scene.
    pub fn new(width: f64, height: f64, background: Color) -> Scene {
        Scene {
            width,
            height,
            background,
            title: String::new(),
            items: Vec::new(),
            targets: Vec::new(),
            axes: Vec::new(),
        }
    }

    /// Append an item.
    pub fn push(&mut self, shape: Shape, stroke: Option<Stroke>, fill: Option<Color>) {
        self.items.push(Item { shape, stroke, fill });
    }

    /// Straight line.
    pub fn line(&mut self, a: Point, b: Point, stroke: Stroke) {
        if a.is_finite() && b.is_finite() {
            self.push(Shape::Line(a, b), Some(stroke), None);
        }
    }

    /// Polyline broken at non-finite points (each finite run becomes one polyline).
    pub fn polyline_gaps(&mut self, pts: &[Point], stroke: Stroke) {
        let mut run: Vec<Point> = Vec::new();
        for p in pts.iter().chain(std::iter::once(&Point::new(f64::NAN, f64::NAN))) {
            if p.is_finite() {
                run.push(*p);
            } else {
                if run.len() >= 2 {
                    self.push(Shape::Polyline(std::mem::take(&mut run)), Some(stroke), None);
                }
                run.clear();
            }
        }
    }

    /// Filled and/or outlined rectangle.
    pub fn rect(&mut self, r: Rect, fill: Option<Color>, stroke: Option<Stroke>) {
        if r.x.is_finite() && r.y.is_finite() && r.w.is_finite() && r.h.is_finite() {
            self.push(Shape::Rect(r), stroke, fill);
        }
    }

    /// Filled polygon.
    pub fn polygon(&mut self, pts: Vec<Point>, fill: Option<Color>, stroke: Option<Stroke>) {
        if pts.len() >= 3 && pts.iter().all(Point::is_finite) {
            self.push(Shape::Polygon(pts), stroke, fill);
        }
    }

    /// Text.
    pub fn text(&mut self, at: Point, text: impl Into<String>, style: TextStyle) {
        let text = text.into();
        if at.is_finite() && !text.is_empty() {
            self.push(Shape::Text { at, text, style }, None, None);
        }
    }

    /// Marker.
    pub fn marker(&mut self, at: Point, kind: Marker, size: f64, fill: Color, stroke: Option<Stroke>) {
        if at.is_finite() {
            self.push(Shape::Marker { at, kind, size }, stroke, Some(fill));
        }
    }

    /// Begin clipping.
    pub fn clip_push(&mut self, r: Rect) {
        self.push(Shape::ClipPush(r), None, None);
    }

    /// End clipping.
    pub fn clip_pop(&mut self) {
        self.push(Shape::ClipPop, None, None);
    }

    /// Register a hit target.
    pub fn target(&mut self, region: Region, series: usize, index: usize, label: impl Into<String>) {
        self.targets.push(HitTarget { region, series, index, label: label.into() });
    }

    /// Copy `other` into this scene at offset `(dx, dy)` (items, targets, axes).
    pub fn embed(&mut self, other: &Scene, dx: f64, dy: f64) {
        for it in &other.items {
            self.items.push(Item { shape: translate_shape(&it.shape, dx, dy), stroke: it.stroke, fill: it.fill });
        }
        for t in &other.targets {
            self.targets.push(HitTarget { region: t.region.translate(dx, dy), ..t.clone() });
        }
        for a in &other.axes {
            self.axes.push(AxesMeta { plot: a.plot.translate(dx, dy), ..a.clone() });
        }
    }

    /// Number of drawn primitives (clip markers excluded).
    pub fn primitive_count(&self) -> usize {
        self.items.iter().filter(|i| !matches!(i.shape, Shape::ClipPush(_) | Shape::ClipPop)).count()
    }

    /// Whether clip pushes and pops balance.
    pub fn clips_balanced(&self) -> bool {
        let mut d = 0i64;
        for i in &self.items {
            match i.shape {
                Shape::ClipPush(_) => d += 1,
                Shape::ClipPop => {
                    d -= 1;
                    if d < 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }
        d == 0
    }
}

fn translate_shape(s: &Shape, dx: f64, dy: f64) -> Shape {
    let tp = |p: &Point| Point::new(p.x + dx, p.y + dy);
    match s {
        Shape::Line(a, b) => Shape::Line(tp(a), tp(b)),
        Shape::Polyline(v) => Shape::Polyline(v.iter().map(tp).collect()),
        Shape::Polygon(v) => Shape::Polygon(v.iter().map(tp).collect()),
        Shape::Rect(r) => Shape::Rect(r.translate(dx, dy)),
        Shape::Circle { c, r } => Shape::Circle { c: tp(c), r: *r },
        Shape::Wedge { c, r, a0, a1 } => Shape::Wedge { c: tp(c), r: *r, a0: *a0, a1: *a1 },
        Shape::Text { at, text, style } => Shape::Text { at: tp(at), text: text.clone(), style: *style },
        Shape::Marker { at, kind, size } => Shape::Marker { at: tp(at), kind: *kind, size: *size },
        Shape::ClipPush(r) => Shape::ClipPush(r.translate(dx, dy)),
        Shape::ClipPop => Shape::ClipPop,
    }
}
