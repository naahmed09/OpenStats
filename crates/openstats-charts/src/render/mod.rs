//! Rendering back ends for a [`Scene`](crate::core::Scene).
//!
//! * [`svg`] — standalone SVG text.
//! * [`pdf`] — a one-page vector PDF.
//! * [`canvas`] — a draw-command stream (JSON and compact binary) for HTML5 Canvas / WebAssembly.
//! * [`raster`] + [`png`] — a software anti-aliased rasteriser and PNG encoder / decoder.

pub mod canvas;
pub mod pdf;
pub mod png;
pub mod raster;
pub mod svg;

use crate::core::geom::Point;
use crate::core::style::Marker;

/// Geometry of a marker glyph, shared by every back end.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MarkerGeom {
    /// Filled disc.
    Disc(Point, f64),
    /// Hollow circle (stroked in the fill colour).
    Ring(Point, f64),
    /// Filled polygon.
    Polygon(Vec<Point>),
    /// Stroked segments (cross, plus), drawn in the fill colour.
    Segments(Vec<(Point, Point)>),
}

pub(crate) fn marker_geom(kind: Marker, at: Point, size: f64) -> MarkerGeom {
    let h = size / 2.0;
    let p = |dx: f64, dy: f64| Point::new(at.x + dx, at.y + dy);
    match kind {
        Marker::Circle => MarkerGeom::Disc(at, h),
        Marker::OpenCircle => MarkerGeom::Ring(at, h),
        Marker::Square => MarkerGeom::Polygon(vec![p(-h, -h), p(h, -h), p(h, h), p(-h, h)]),
        Marker::Diamond => {
            let d = h * 1.25;
            MarkerGeom::Polygon(vec![p(0.0, -d), p(d, 0.0), p(0.0, d), p(-d, 0.0)])
        }
        Marker::TriangleUp => MarkerGeom::Polygon(vec![p(0.0, -h * 1.15), p(h * 1.1, h * 0.85), p(-h * 1.1, h * 0.85)]),
        Marker::TriangleDown => {
            MarkerGeom::Polygon(vec![p(0.0, h * 1.15), p(h * 1.1, -h * 0.85), p(-h * 1.1, -h * 0.85)])
        }
        Marker::Cross => {
            let d = h * 0.95;
            MarkerGeom::Segments(vec![(p(-d, -d), p(d, d)), (p(-d, d), p(d, -d))])
        }
        Marker::Plus => {
            let d = h * 1.1;
            MarkerGeom::Segments(vec![(p(-d, 0.0), p(d, 0.0)), (p(0.0, -d), p(0.0, d))])
        }
    }
}

/// Points of the arc of a circle, `n` segments from `a0` to `a1` (radians, clockwise on screen).
pub(crate) fn arc_points(c: Point, r: f64, a0: f64, a1: f64, n: usize) -> Vec<Point> {
    (0..=n)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f64 / n as f64;
            Point::new(c.x + r * a.cos(), c.y + r * a.sin())
        })
        .collect()
}
