//! Points and rectangles.

/// A point in scene (pixel) or data coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate (down in scene space).
    pub y: f64,
}

impl Point {
    /// New point.
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    /// Whether both coordinates are finite.
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Euclidean distance.
    pub fn dist(&self, o: Point) -> f64 {
        ((self.x - o.x).powi(2) + (self.y - o.y).powi(2)).sqrt()
    }
}

/// Axis-aligned rectangle (`x`, `y` is the top-left corner).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    /// Left edge.
    pub x: f64,
    /// Top edge.
    pub y: f64,
    /// Width (≥ 0).
    pub w: f64,
    /// Height (≥ 0).
    pub h: f64,
}

impl Rect {
    /// New rectangle (negative sizes are normalised).
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Rect {
        let (x, w) = if w < 0.0 { (x + w, -w) } else { (x, w) };
        let (y, h) = if h < 0.0 { (y + h, -h) } else { (y, h) };
        Rect { x, y, w, h }
    }

    /// Right edge.
    pub fn right(&self) -> f64 {
        self.x + self.w
    }

    /// Bottom edge.
    pub fn bottom(&self) -> f64 {
        self.y + self.h
    }

    /// Centre.
    pub fn center(&self) -> Point {
        Point::new(self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Whether `p` lies inside (edges included).
    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.x && p.x <= self.right() && p.y >= self.y && p.y <= self.bottom()
    }

    /// Intersection (empty rectangle when disjoint).
    pub fn intersect(&self, o: &Rect) -> Rect {
        let x0 = self.x.max(o.x);
        let y0 = self.y.max(o.y);
        let x1 = self.right().min(o.right());
        let y1 = self.bottom().min(o.bottom());
        Rect { x: x0, y: y0, w: (x1 - x0).max(0.0), h: (y1 - y0).max(0.0) }
    }

    /// Whether the rectangles overlap with positive area.
    pub fn overlaps(&self, o: &Rect) -> bool {
        let i = self.intersect(o);
        i.w > 0.0 && i.h > 0.0
    }

    /// Shrink by `d` on every side.
    pub fn inset(&self, d: f64) -> Rect {
        Rect::new(self.x + d, self.y + d, (self.w - 2.0 * d).max(0.0), (self.h - 2.0 * d).max(0.0))
    }

    /// Translate.
    pub fn translate(&self, dx: f64, dy: f64) -> Rect {
        Rect { x: self.x + dx, y: self.y + dy, w: self.w, h: self.h }
    }
}
