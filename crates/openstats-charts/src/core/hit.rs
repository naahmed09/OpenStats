//! Hit-testing for hover tooltips and data brushing, and zoom/pan viewports.

use super::geom::{Point, Rect};
use super::scale::Scale;

/// Area of the scene that answers to the pointer.
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    /// A point marker (centre, pick radius in pixels).
    Point(Point, f64),
    /// A bar, box or similar rectangle.
    Rect(Rect),
}

impl Region {
    /// Translate.
    pub fn translate(&self, dx: f64, dy: f64) -> Region {
        match self {
            Region::Point(p, r) => Region::Point(Point::new(p.x + dx, p.y + dy), *r),
            Region::Rect(r) => Region::Rect(r.translate(dx, dy)),
        }
    }

    fn bounds(&self) -> Rect {
        match self {
            Region::Point(p, r) => Rect::new(p.x - r, p.y - r, 2.0 * r, 2.0 * r),
            Region::Rect(r) => *r,
        }
    }

    fn distance(&self, q: Point) -> f64 {
        match self {
            Region::Point(p, r) => (p.dist(q) - r).max(0.0),
            Region::Rect(r) => {
                let dx = (r.x - q.x).max(0.0).max(q.x - r.right());
                let dy = (r.y - q.y).max(0.0).max(q.y - r.bottom());
                (dx * dx + dy * dy).sqrt()
            }
        }
    }
}

/// A pickable element tied to a data row.
#[derive(Debug, Clone, PartialEq)]
pub struct HitTarget {
    /// Geometry.
    pub region: Region,
    /// Series index in the chart.
    pub series: usize,
    /// Row index within the series (the worksheet row for brushing).
    pub index: usize,
    /// Tooltip text.
    pub label: String,
}

/// Result of a hit test.
#[derive(Debug, Clone, PartialEq)]
pub struct HitResult {
    /// Series index.
    pub series: usize,
    /// Row index.
    pub index: usize,
    /// Tooltip text.
    pub label: String,
    /// Distance from the query point to the target (0 inside).
    pub distance: f64,
}

/// Uniform-grid spatial index over hit targets.
#[derive(Debug, Clone)]
pub struct HitIndex {
    cell: f64,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
    targets: Vec<HitTarget>,
}

impl HitIndex {
    /// Index the targets of a scene of the given size.
    pub fn new(width: f64, height: f64, targets: &[HitTarget]) -> HitIndex {
        let cell = 32.0;
        let cols = ((width / cell).ceil() as usize).max(1);
        let rows = ((height / cell).ceil() as usize).max(1);
        let mut cells = vec![Vec::new(); cols * rows];
        for (i, t) in targets.iter().enumerate() {
            let b = t.region.bounds();
            let (c0, c1) = (to_cell(b.x, cell, cols), to_cell(b.right(), cell, cols));
            let (r0, r1) = (to_cell(b.y, cell, rows), to_cell(b.bottom(), cell, rows));
            for r in r0..=r1 {
                for c in c0..=c1 {
                    cells[r * cols + c].push(i);
                }
            }
        }
        HitIndex { cell, cols, rows, cells, targets: targets.to_vec() }
    }

    /// Nearest target within `tolerance` pixels of `p` (ties: the later, i.e. the front-most, one).
    pub fn query(&self, p: Point, tolerance: f64) -> Option<HitResult> {
        let (c0, c1) = (to_cell(p.x - tolerance, self.cell, self.cols), to_cell(p.x + tolerance, self.cell, self.cols));
        let (r0, r1) = (to_cell(p.y - tolerance, self.cell, self.rows), to_cell(p.y + tolerance, self.cell, self.rows));
        let mut best: Option<(f64, usize)> = None;
        for r in r0..=r1 {
            for c in c0..=c1 {
                for &i in &self.cells[r * self.cols + c] {
                    let d = self.targets[i].region.distance(p);
                    if d <= tolerance && best.is_none_or(|(bd, bi)| d < bd || (d == bd && i > bi)) {
                        best = Some((d, i));
                    }
                }
            }
        }
        best.map(|(d, i)| {
            let t = &self.targets[i];
            HitResult { series: t.series, index: t.index, label: t.label.clone(), distance: d }
        })
    }

    /// Every `(series, index)` whose target centre lies in `r` (rubber-band brushing).
    pub fn query_rect(&self, r: Rect) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self
            .targets
            .iter()
            .filter(|t| {
                let c = t.region.bounds().center();
                r.contains(c)
            })
            .map(|t| (t.series, t.index))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Number of indexed targets.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether there are no targets.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

fn to_cell(v: f64, cell: f64, n: usize) -> usize {
    if v.is_nan() {
        return 0;
    }
    ((v / cell).floor().max(0.0) as usize).min(n - 1)
}

/// A zoom/pan window in data coordinates; `None` keeps the automatic range.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Viewport {
    /// Visible x range.
    pub x: Option<(f64, f64)>,
    /// Visible y range.
    pub y: Option<(f64, f64)>,
}

impl Viewport {
    /// Zoom by `factor` (> 1 zooms in) about the fractional position `(tx, ty)` of the
    /// current window (`ty` measured upwards), in each scale's own (e.g. log) space.
    pub fn zoomed(&self, sx: &Scale, sy: &Scale, tx: f64, ty: f64, factor: f64) -> Viewport {
        let f = factor.max(1e-6);
        let z = |s: &Scale, t: f64| {
            let (a, b) = (t - t / f, t + (1.0 - t) / f);
            (s.invert(a), s.invert(b))
        };
        Viewport { x: Some(z(sx, tx)), y: Some(z(sy, ty)) }
    }

    /// Pan by fractions `(dx, dy)` of the current window (positive = towards larger values).
    pub fn panned(&self, sx: &Scale, sy: &Scale, dx: f64, dy: f64) -> Viewport {
        Viewport { x: Some((sx.invert(dx), sx.invert(1.0 + dx))), y: Some((sy.invert(dy), sy.invert(1.0 + dy))) }
    }

    /// Back to the automatic range.
    pub fn reset() -> Viewport {
        Viewport::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_and_rect_targets() {
        let t = vec![
            HitTarget { region: Region::Point(Point::new(10.0, 10.0), 3.0), series: 0, index: 0, label: "a".into() },
            HitTarget { region: Region::Point(Point::new(40.0, 10.0), 3.0), series: 0, index: 1, label: "b".into() },
            HitTarget {
                region: Region::Rect(Rect::new(100.0, 50.0, 20.0, 30.0)),
                series: 1,
                index: 7,
                label: "bar".into(),
            },
        ];
        let h = HitIndex::new(200.0, 100.0, &t);
        assert_eq!(h.query(Point::new(12.0, 11.0), 4.0).unwrap().index, 0);
        assert_eq!(h.query(Point::new(39.0, 9.0), 4.0).unwrap().label, "b");
        assert!(h.query(Point::new(25.0, 10.0), 4.0).is_none());
        let r = h.query(Point::new(110.0, 60.0), 0.0).unwrap();
        assert_eq!((r.series, r.index, r.distance), (1, 7, 0.0));
        assert_eq!(h.query_rect(Rect::new(0.0, 0.0, 50.0, 20.0)), vec![(0, 0), (0, 1)]);
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn zoom_and_pan() {
        let s = Scale::Linear { min: 0.0, max: 10.0 };
        let v = Viewport::default().zoomed(&s, &s, 0.5, 0.5, 2.0);
        let (a, b) = v.x.unwrap();
        assert!((a - 2.5).abs() < 1e-12 && (b - 7.5).abs() < 1e-12);
        let p = Viewport::default().panned(&s, &s, 0.1, 0.0);
        assert!((p.x.unwrap().0 - 1.0).abs() < 1e-12);
        let l = Scale::Log { min: 1.0, max: 100.0 };
        let z = Viewport::default().zoomed(&l, &l, 0.5, 0.5, 2.0);
        let (a, b) = z.x.unwrap();
        assert!((a - 10f64.powf(0.5)).abs() < 1e-9 && (b - 10f64.powf(1.5)).abs() < 1e-9);
    }
}
