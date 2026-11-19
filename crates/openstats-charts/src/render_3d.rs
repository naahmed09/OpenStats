//! Software 3-D engine for response surfaces (Phase 3c), std only.
//!
//! * [`Vec3`] / [`Mat4`]: right-handed, z-up; column-vector convention (`p' = M p`).
//! * [`Camera`]: orbit camera with perspective or orthographic projection, zoom and pan;
//!   interaction hooks [`Camera::orbit`], [`Camera::zoom_by`], [`Camera::pan_by`].
//! * [`SurfaceGrid`]: a response grid (from a `ChartSpec`, or sampled from a closure such as a
//!   fitted CCD / Box–Behnken model).
//! * [`render_surface`]: depth-sorted (painter's algorithm) triangle mesh with Lambert +
//!   Blinn–Phong lighting, flat or smooth shading, wireframe with hidden-line removal,
//!   bounding box with ticks, floor contour projection and a colour bar. Hit targets and
//!   [`pick`] map pointer positions back to grid cells.
//! * [`contour_lines`]: marching squares with saddle disambiguation and segment stitching;
//!   [`contour_plot`] draws labelled (optionally banded) contours with design-point overlays.

use crate::core::geom::{Point, Rect};
use crate::core::hit::Region;
use crate::core::layout::{Frame, FrameSpec};
use crate::core::scale::Scale;
use crate::core::scene::{Anchor, Scene, TextStyle};
use crate::core::style::{Color, Marker, Stroke, Theme};
use crate::core::ticks::extended_ticks;
use crate::plots::common::{num, ramp, MAX_TARGETS};
use crate::plots::PlotOptions;
use openstats_stats::report::ChartSpec;
use std::collections::BTreeMap;

// ------------------------------------------------------------------ linear algebra

/// A 3-vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// x.
    pub x: f64,
    /// y.
    pub y: f64,
    /// z.
    pub z: f64,
}

impl Vec3 {
    /// Construct.
    pub const fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }
    /// Sum.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    /// Difference.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    /// Scalar multiple.
    pub fn scale(self, k: f64) -> Vec3 {
        Vec3::new(self.x * k, self.y * k, self.z * k)
    }
    /// Dot product.
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    /// Cross product.
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    /// Euclidean length.
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    /// Unit vector (the zero vector stays zero).
    pub fn normalized(self) -> Vec3 {
        let l = self.length();
        if l > 0.0 {
            self.scale(1.0 / l)
        } else {
            self
        }
    }
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// A 4×4 matrix, row-major, acting on column vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// Rows.
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    /// Identity.
    pub fn identity() -> Mat4 {
        let mut m = [[0.0; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Mat4 { m }
    }
    /// Translation.
    pub fn translation(t: Vec3) -> Mat4 {
        let mut a = Mat4::identity();
        a.m[0][3] = t.x;
        a.m[1][3] = t.y;
        a.m[2][3] = t.z;
        a
    }
    /// Non-uniform scaling.
    pub fn scaling(s: Vec3) -> Mat4 {
        let mut a = Mat4::identity();
        a.m[0][0] = s.x;
        a.m[1][1] = s.y;
        a.m[2][2] = s.z;
        a
    }
    /// Rotation about the x axis by `rad` (right-handed).
    pub fn rotation_x(rad: f64) -> Mat4 {
        let (s, c) = rad.sin_cos();
        let mut a = Mat4::identity();
        a.m[1][1] = c;
        a.m[1][2] = -s;
        a.m[2][1] = s;
        a.m[2][2] = c;
        a
    }
    /// Rotation about the y axis.
    pub fn rotation_y(rad: f64) -> Mat4 {
        let (s, c) = rad.sin_cos();
        let mut a = Mat4::identity();
        a.m[0][0] = c;
        a.m[0][2] = s;
        a.m[2][0] = -s;
        a.m[2][2] = c;
        a
    }
    /// Rotation about the z axis.
    pub fn rotation_z(rad: f64) -> Mat4 {
        let (s, c) = rad.sin_cos();
        let mut a = Mat4::identity();
        a.m[0][0] = c;
        a.m[0][1] = -s;
        a.m[1][0] = s;
        a.m[1][1] = c;
        a
    }
    /// Matrix product `self · o` (apply `o` first).
    pub fn mul(&self, o: &Mat4) -> Mat4 {
        let mut r = [[0.0; 4]; 4];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..4).map(|k| self.m[i][k] * o.m[k][j]).sum();
            }
        }
        Mat4 { m: r }
    }
    /// Transpose.
    pub fn transpose(&self) -> Mat4 {
        let mut r = [[0.0; 4]; 4];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.m[j][i];
            }
        }
        Mat4 { m: r }
    }
    /// Transform a point; returns `(x, y, z, w)` (not divided by `w`).
    pub fn transform4(&self, p: Vec3) -> [f64; 4] {
        let v = [p.x, p.y, p.z, 1.0];
        let mut out = [0.0; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = (0..4).map(|k| self.m[i][k] * v[k]).sum();
        }
        out
    }
    /// Transform a point of an affine matrix (w ignored).
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let r = self.transform4(p);
        Vec3::new(r[0], r[1], r[2])
    }
    /// Transform a direction (translation ignored).
    pub fn transform_dir(&self, d: Vec3) -> Vec3 {
        let v = [d.x, d.y, d.z];
        let mut out = [0.0; 3];
        for (i, o) in out.iter_mut().enumerate() {
            *o = (0..3).map(|k| self.m[i][k] * v[k]).sum();
        }
        Vec3::new(out[0], out[1], out[2])
    }
    /// Look-at view matrix (camera at `eye` looking at `target`, `up` roughly vertical).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = target.sub(eye).normalized();
        let s = f.cross(up).normalized();
        let u = s.cross(f);
        let mut a = Mat4::identity();
        a.m[0] = [s.x, s.y, s.z, -s.dot(eye)];
        a.m[1] = [u.x, u.y, u.z, -u.dot(eye)];
        a.m[2] = [-f.x, -f.y, -f.z, f.dot(eye)];
        a
    }
    /// OpenGL-style perspective projection (`fov_y` in radians).
    pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
        let f = 1.0 / (fov_y / 2.0).tan();
        let mut a = Mat4 { m: [[0.0; 4]; 4] };
        a.m[0][0] = f / aspect;
        a.m[1][1] = f;
        a.m[2][2] = (far + near) / (near - far);
        a.m[2][3] = 2.0 * far * near / (near - far);
        a.m[3][2] = -1.0;
        a
    }
    /// OpenGL-style orthographic projection with half-height `half_h`.
    pub fn orthographic(half_h: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
        let mut a = Mat4::identity();
        a.m[0][0] = 1.0 / (half_h * aspect);
        a.m[1][1] = 1.0 / half_h;
        a.m[2][2] = -2.0 / (far - near);
        a.m[2][3] = -(far + near) / (far - near);
        a
    }
}

// ------------------------------------------------------------------ camera

/// Projection type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    /// Perspective with the vertical field of view in degrees.
    Perspective(f64),
    /// Orthographic.
    Orthographic,
}

/// Orbit camera. The azimuth is measured in degrees from +x towards +y, the elevation from
/// the xy plane towards +z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Azimuth (degrees).
    pub azimuth: f64,
    /// Elevation (degrees, clamped to ±89).
    pub elevation: f64,
    /// Distance from the target.
    pub distance: f64,
    /// Look-at point.
    pub target: Vec3,
    /// Projection.
    pub projection: Projection,
    /// Zoom factor (1 = fit).
    pub zoom: f64,
}

/// Field of view used by orthographic cameras to size the view.
const ORTHO_FOV: f64 = 30.0;

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            azimuth: -50.0,
            elevation: 28.0,
            distance: 5.0,
            target: Vec3::default(),
            projection: Projection::Perspective(30.0),
            zoom: 1.0,
        }
    }
}

impl Camera {
    /// Orthographic variant of the default camera.
    pub fn orthographic() -> Camera {
        Camera { projection: Projection::Orthographic, ..Camera::default() }
    }

    /// Camera position.
    pub fn eye(&self) -> Vec3 {
        let (az, el) = (self.azimuth.to_radians(), self.elevation.to_radians());
        self.target.add(Vec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin()).scale(self.distance))
    }

    /// World → view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(self.eye(), self.target, Vec3::new(0.0, 0.0, 1.0))
    }

    fn half(&self) -> f64 {
        let fov = match self.projection {
            Projection::Perspective(f) => f,
            Projection::Orthographic => ORTHO_FOV,
        };
        (fov.to_radians() / 2.0).tan() / self.zoom
    }

    /// View → clip matrix for a square viewport.
    pub fn projection_matrix(&self) -> Mat4 {
        let (near, far) = (0.05, 200.0);
        match self.projection {
            Projection::Perspective(_) => Mat4::perspective(2.0 * self.half().atan(), 1.0, near, far),
            Projection::Orthographic => Mat4::orthographic(self.distance * self.half(), 1.0, near, far),
        }
    }

    /// Rotate by a pointer drag of `(dx, dy)` pixels (0.5° per pixel); elevation is clamped.
    pub fn orbit(&mut self, dx: f64, dy: f64) {
        self.azimuth = (self.azimuth - dx * 0.5).rem_euclid(360.0);
        if self.azimuth > 180.0 {
            self.azimuth -= 360.0;
        }
        self.elevation = (self.elevation + dy * 0.5).clamp(-89.0, 89.0);
    }

    /// Multiply the zoom (result clamped to 0.2 … 20).
    pub fn zoom_by(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.zoom = (self.zoom * factor).clamp(0.2, 20.0);
        }
    }

    /// Pan the view by `(dx, dy)` in normalised device units (right and up positive).
    pub fn pan_by(&mut self, dx: f64, dy: f64) {
        let v = self.view_matrix();
        let right = Vec3::new(v.m[0][0], v.m[0][1], v.m[0][2]);
        let up = Vec3::new(v.m[1][0], v.m[1][1], v.m[1][2]);
        let unit = self.distance * self.half();
        self.target = self.target.sub(right.scale(dx * unit)).sub(up.scale(dy * unit));
    }

    /// Back to the default orientation, zoom and target (projection kept).
    pub fn reset(&mut self) {
        *self = Camera { projection: self.projection, ..Camera::default() };
    }

    /// Normalised device coordinates and view depth (distance along the view axis) of a world
    /// point; `None` behind the near plane.
    pub fn project_ndc(&self, p: Vec3) -> Option<(f64, f64, f64)> {
        self.project_view(self.view_matrix().transform_point(p))
    }

    fn project_view(&self, v: Vec3) -> Option<(f64, f64, f64)> {
        if -v.z < 0.05 {
            return None;
        }
        let c = self.projection_matrix().transform4(v);
        if c[3].abs() < 1e-12 {
            return None;
        }
        Some((c[0] / c[3], c[1] / c[3], -v.z))
    }
}

/// Maps NDC to pixels so that the reference box fits the drawing area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewTransform {
    k: f64,
    ox: f64,
    oy: f64,
    cx: f64,
    cy: f64,
}

impl ViewTransform {
    /// Fit `points` (seen from `camera` at zoom 1, no pan) into `area`.
    pub fn fit(camera: &Camera, points: &[Vec3], area: Rect) -> ViewTransform {
        let reference = Camera { zoom: 1.0, target: Vec3::default(), ..*camera };
        let ndc: Vec<(f64, f64)> =
            points.iter().filter_map(|p| reference.project_ndc(*p).map(|q| (q.0, q.1))).collect();
        let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (x, y) in &ndc {
            x0 = x0.min(*x);
            x1 = x1.max(*x);
            y0 = y0.min(*y);
            y1 = y1.max(*y);
        }
        if ndc.is_empty() || !(x1 > x0 || y1 > y0) {
            return ViewTransform {
                k: area.w.min(area.h) / 2.0,
                ox: area.center().x,
                oy: area.center().y,
                cx: 0.0,
                cy: 0.0,
            };
        }
        let k = (area.w / (x1 - x0).max(1e-9)).min(area.h / (y1 - y0).max(1e-9)) * 0.92;
        ViewTransform { k, ox: area.center().x, oy: area.center().y, cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0 }
    }

    /// NDC → pixel.
    pub fn to_pixel(&self, ndc: (f64, f64)) -> Point {
        Point::new(self.ox + (ndc.0 - self.cx) * self.k, self.oy - (ndc.1 - self.cy) * self.k)
    }

    /// Pixels → NDC offset (for pans).
    pub fn pixels_to_ndc(&self, dx: f64, dy: f64) -> (f64, f64) {
        (dx / self.k, -dy / self.k)
    }
}

// ------------------------------------------------------------------ grid

/// A response grid: `z[j][i]` is the value at `(xs[i], ys[j])`.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceGrid {
    /// x coordinates (increasing).
    pub xs: Vec<f64>,
    /// y coordinates (increasing).
    pub ys: Vec<f64>,
    /// Values, one row per y.
    pub z: Vec<Vec<f64>>,
    /// x axis label.
    pub x_label: String,
    /// y axis label.
    pub y_label: String,
    /// z axis label.
    pub z_label: String,
}

impl SurfaceGrid {
    /// Validate and build a grid (at least 2×2, increasing coordinates, rectangular).
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, z: Vec<Vec<f64>>) -> Result<SurfaceGrid, String> {
        if xs.len() < 2 || ys.len() < 2 {
            return Err("a surface needs at least a 2 x 2 grid".into());
        }
        if z.len() != ys.len() || z.iter().any(|r| r.len() != xs.len()) {
            return Err("z must have one row per y and one value per x".into());
        }
        let inc = |v: &[f64]| v.iter().all(|a| a.is_finite()) && v.windows(2).all(|w| w[1] > w[0]);
        if !inc(&xs) || !inc(&ys) {
            return Err("grid coordinates must be finite and strictly increasing".into());
        }
        Ok(SurfaceGrid { xs, ys, z, x_label: String::new(), y_label: String::new(), z_label: String::new() })
    }

    /// Sample `f(x, y)` on an `nx` × `ny` grid over the given ranges, for instance a fitted
    /// CCD or Box–Behnken model in natural or coded units.
    pub fn from_fn(
        xr: (f64, f64),
        yr: (f64, f64),
        nx: usize,
        ny: usize,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<SurfaceGrid, String> {
        if nx < 2 || ny < 2 {
            return Err("a surface needs at least a 2 x 2 grid".into());
        }
        let line = |r: (f64, f64), n: usize| -> Vec<f64> {
            (0..n).map(|i| r.0 + (r.1 - r.0) * i as f64 / (n - 1) as f64).collect()
        };
        let (xs, ys) = (line(xr, nx), line(yr, ny));
        let z = ys.iter().map(|&y| xs.iter().map(|&x| f(x, y)).collect()).collect();
        SurfaceGrid::new(xs, ys, z)
    }

    /// From a `ChartKind::Surface` spec (series name = row y, `x` = columns, `y` = z values).
    pub fn from_spec(spec: &ChartSpec) -> Result<SurfaceGrid, String> {
        let mut rows: Vec<(f64, &[f64], &[f64])> = spec
            .series
            .iter()
            .filter_map(|s| s.name.trim().parse::<f64>().ok().map(|y| (y, s.x.as_slice(), s.y.as_slice())))
            .collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let first = rows.first().ok_or("the surface has no rows")?;
        let xs = first.1.to_vec();
        let ys = rows.iter().map(|r| r.0).collect();
        let z = rows.iter().map(|r| r.2.to_vec()).collect();
        let mut g = SurfaceGrid::new(xs, ys, z)?;
        g.x_label = spec.x_label.clone();
        g.y_label = spec.categories.first().cloned().unwrap_or_default();
        g.z_label = spec.y_label.clone();
        Ok(g)
    }

    /// Finite minimum and maximum of z.
    pub fn z_range(&self) -> Option<(f64, f64)> {
        crate::plots::common::range_of(self.z.iter().flatten())
    }

    /// Bilinear interpolation at `(x, y)` (`None` outside the grid or at missing corners).
    pub fn sample(&self, x: f64, y: f64) -> Option<f64> {
        let loc = |v: &[f64], t: f64| -> Option<(usize, f64)> {
            if t < v[0] || t > v[v.len() - 1] || t.is_nan() {
                return None;
            }
            let k = v.partition_point(|&a| a <= t).clamp(1, v.len() - 1) - 1;
            Some((k, (t - v[k]) / (v[k + 1] - v[k])))
        };
        let (i, u) = loc(&self.xs, x)?;
        let (j, w) = loc(&self.ys, y)?;
        let (a, b, c, d) = (self.z[j][i], self.z[j][i + 1], self.z[j + 1][i], self.z[j + 1][i + 1]);
        let v = a * (1.0 - u) * (1.0 - w) + b * u * (1.0 - w) + c * (1.0 - u) * w + d * u * w;
        v.is_finite().then_some(v)
    }
}

// ------------------------------------------------------------------ marching squares

/// A contour polyline in data coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourLine {
    /// Level.
    pub level: f64,
    /// Vertices `(x, y)`; a closed contour repeats its first vertex at the end.
    pub points: Vec<(f64, f64)>,
    /// Whether the contour is closed.
    pub closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeKey(u8, usize, usize);

/// Contour lines of `z` at `level` by marching squares (corner value ≥ level counts as
/// "high"; saddles are resolved with the cell-centre mean). Cells with a missing corner are
/// skipped. Segments are stitched into polylines.
pub fn contour_lines(xs: &[f64], ys: &[f64], z: &[Vec<f64>], level: f64) -> Vec<ContourLine> {
    let (ny, nx) = (ys.len(), xs.len());
    if nx < 2 || ny < 2 || z.len() != ny || !level.is_finite() {
        return Vec::new();
    }
    // Crossing point on an edge.
    let pt = |k: EdgeKey| -> (f64, f64) {
        let EdgeKey(kind, j, i) = k;
        if kind == 0 {
            let (a, b) = (z[j][i], z[j][i + 1]);
            let t = if b == a { 0.5 } else { (level - a) / (b - a) };
            (xs[i] + (xs[i + 1] - xs[i]) * t, ys[j])
        } else {
            let (a, b) = (z[j][i], z[j + 1][i]);
            let t = if b == a { 0.5 } else { (level - a) / (b - a) };
            (xs[i], ys[j] + (ys[j + 1] - ys[j]) * t)
        }
    };
    let mut segs: Vec<(EdgeKey, EdgeKey)> = Vec::new();
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let c = [z[j][i], z[j][i + 1], z[j + 1][i + 1], z[j + 1][i]];
            if c.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let e = [EdgeKey(0, j, i), EdgeKey(1, j, i + 1), EdgeKey(0, j + 1, i), EdgeKey(1, j, i)];
            let idx = c.iter().enumerate().fold(0usize, |m, (k, &v)| m | (usize::from(v >= level) << k));
            let center = (c[0] + c[1] + c[2] + c[3]) / 4.0 >= level;
            let pairs: &[(usize, usize)] = match idx {
                1 | 14 => &[(3, 0)],
                2 | 13 => &[(0, 1)],
                3 | 12 => &[(3, 1)],
                4 | 11 => &[(1, 2)],
                6 | 9 => &[(0, 2)],
                7 | 8 => &[(3, 2)],
                5 => {
                    if center {
                        &[(0, 1), (2, 3)]
                    } else {
                        &[(3, 0), (1, 2)]
                    }
                }
                10 => {
                    if center {
                        &[(3, 0), (1, 2)]
                    } else {
                        &[(0, 1), (2, 3)]
                    }
                }
                _ => &[],
            };
            for &(a, b) in pairs {
                segs.push((e[a], e[b]));
            }
        }
    }
    // Stitch.
    let mut adj: BTreeMap<EdgeKey, Vec<usize>> = BTreeMap::new();
    for (s, (a, b)) in segs.iter().enumerate() {
        adj.entry(*a).or_default().push(s);
        adj.entry(*b).or_default().push(s);
    }
    let mut used = vec![false; segs.len()];
    let walk = |used: &mut Vec<bool>, mut key: EdgeKey, stop: Option<EdgeKey>| -> (Vec<EdgeKey>, bool) {
        let mut out = Vec::new();
        loop {
            let next = adj.get(&key).and_then(|v| v.iter().copied().find(|&s| !used[s]));
            let Some(s) = next else { return (out, false) };
            used[s] = true;
            let (a, b) = segs[s];
            key = if a == key { b } else { a };
            out.push(key);
            if Some(key) == stop {
                return (out, true);
            }
        }
    };
    let mut lines = Vec::new();
    for s in 0..segs.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let (a, b) = segs[s];
        let (fwd, closed) = walk(&mut used, b, Some(a));
        let mut keys: Vec<EdgeKey> = vec![a, b];
        keys.extend(fwd);
        if !closed {
            let (back, _) = walk(&mut used, a, None);
            let mut pre: Vec<EdgeKey> = back.into_iter().rev().collect();
            pre.extend(keys);
            keys = pre;
        }
        lines.push(ContourLine { level, points: keys.into_iter().map(pt).collect(), closed });
    }
    lines
}

/// `n` equally spaced levels strictly between `lo` and `hi`.
pub fn equal_levels(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (1..=n).map(|i| lo + (hi - lo) * i as f64 / (n + 1) as f64).collect()
}

/// Round-number levels (Wilkinson ticks) inside `[lo, hi]`, about `target` of them.
pub fn nice_levels(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    extended_ticks(lo, hi, target).into_iter().map(|t| t.value).collect()
}

// ------------------------------------------------------------------ lighting

/// Directional light with Blinn–Phong parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// Direction *towards* the light (world space).
    pub direction: Vec3,
    /// Ambient term.
    pub ambient: f64,
    /// Diffuse weight.
    pub diffuse: f64,
    /// Specular weight.
    pub specular: f64,
    /// Specular exponent.
    pub shininess: f64,
}

impl Default for Light {
    fn default() -> Light {
        Light {
            direction: Vec3::new(0.5, -0.6, 0.8).normalized(),
            ambient: 0.35,
            diffuse: 0.65,
            specular: 0.25,
            shininess: 24.0,
        }
    }
}

impl Light {
    /// Lit colour of a surface point with unit `normal` seen along `view` (towards the eye);
    /// shading is two-sided.
    pub fn shade(&self, base: Color, normal: Vec3, view: Vec3) -> Color {
        let n = if normal.dot(view) < 0.0 { normal.scale(-1.0) } else { normal };
        let l = self.direction.normalized();
        let diff = n.dot(l).max(0.0);
        let h = l.add(view).normalized();
        let spec = if diff > 0.0 { n.dot(h).max(0.0).powf(self.shininess) } else { 0.0 };
        let k = self.ambient + self.diffuse * diff;
        let ch = |c: u8| -> u8 { (f64::from(c) * k + 255.0 * self.specular * spec).round().clamp(0.0, 255.0) as u8 };
        Color { r: ch(base.r), g: ch(base.g), b: ch(base.b), a: base.a }
    }
}

// ------------------------------------------------------------------ surface rendering

/// What to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Shaded faces only.
    Surface,
    /// Shaded faces with mesh lines.
    SurfaceWireframe,
    /// Mesh lines with hidden-line removal.
    Wireframe,
}

/// Shading model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shading {
    /// One normal per face.
    Flat,
    /// Normals averaged at the vertices, colour averaged per face.
    Smooth,
}

/// Options of [`render_surface`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceStyle {
    /// Drawing mode.
    pub mode: RenderMode,
    /// Shading.
    pub shading: Shading,
    /// Light.
    pub light: Light,
    /// Half-height of the box relative to its half-width (default 0.6).
    pub z_scale: f64,
    /// Draw the bounding box, ticks and axis titles.
    pub show_box: bool,
    /// Project this many contour levels onto the floor (0 = none).
    pub floor_contours: usize,
    /// Draw the colour bar.
    pub color_bar: bool,
}

impl Default for SurfaceStyle {
    fn default() -> SurfaceStyle {
        SurfaceStyle {
            mode: RenderMode::SurfaceWireframe,
            shading: Shading::Smooth,
            light: Light::default(),
            z_scale: 0.6,
            show_box: true,
            floor_contours: 8,
            color_bar: true,
        }
    }
}

struct Mesh {
    /// World position per vertex (`None` when z is missing).
    pos: Vec<Option<Vec3>>,
    normal: Vec<Vec3>,
    nx: usize,
    ny: usize,
}

fn build_mesh(g: &SurfaceGrid, zs: f64) -> Mesh {
    let (nx, ny) = (g.xs.len(), g.ys.len());
    let (zlo, zhi) = g.z_range().unwrap_or((0.0, 1.0));
    let zspan = if zhi > zlo { zhi - zlo } else { 1.0 };
    let (x0, x1, y0, y1) = (g.xs[0], g.xs[nx - 1], g.ys[0], g.ys[ny - 1]);
    let mut pos = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let z = g.z[j][i];
            pos.push(z.is_finite().then(|| {
                Vec3::new(
                    2.0 * (g.xs[i] - x0) / (x1 - x0) - 1.0,
                    2.0 * (g.ys[j] - y0) / (y1 - y0) - 1.0,
                    zs * (2.0 * (z - zlo) / zspan - 1.0),
                )
            }));
        }
    }
    let at = |j: usize, i: usize| pos[j * nx + i];
    let mut normal = vec![Vec3::new(0.0, 0.0, 1.0); nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let (il, ir) = (i.saturating_sub(1), (i + 1).min(nx - 1));
            let (jl, jr) = (j.saturating_sub(1), (j + 1).min(ny - 1));
            if let (Some(a), Some(b), Some(c), Some(d)) = (at(j, il), at(j, ir), at(jl, i), at(jr, i)) {
                let n = b.sub(a).cross(d.sub(c)).normalized();
                if n.is_finite() && n.length() > 0.0 {
                    normal[j * nx + i] = n;
                }
            }
        }
    }
    Mesh { pos, normal, nx, ny }
}

struct Tri {
    /// Vertex indices.
    v: [usize; 3],
    /// Projected polygon (after near clipping) in pixels.
    poly: Vec<Point>,
    depth: f64,
    cell: (usize, usize),
    /// True when the polygon is the unclipped triangle (vertex order preserved).
    exact: bool,
}

fn box_corners(zs: f64) -> Vec<Vec3> {
    let mut v = Vec::new();
    for &x in &[-1.0, 1.0] {
        for &y in &[-1.0, 1.0] {
            for &z in &[-zs, zs] {
                v.push(Vec3::new(x, y, z));
            }
        }
    }
    v
}

/// Clip a polygon in view space to the near plane.
fn clip_near(poly: &[Vec3]) -> Vec<Vec3> {
    let inside = |p: &Vec3| -p.z >= 0.05;
    if poly.iter().all(inside) {
        return poly.to_vec();
    }
    let mut out = Vec::new();
    for k in 0..poly.len() {
        let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
        match (inside(&a), inside(&b)) {
            (true, true) => out.push(b),
            (true, false) | (false, true) => {
                let t = (-0.05 - a.z) / (b.z - a.z);
                let p = a.add(b.sub(a).scale(t));
                out.push(p);
                if inside(&b) {
                    out.push(b);
                }
            }
            (false, false) => {}
        }
    }
    out
}

fn triangles(mesh: &Mesh, cam: &Camera, vt: &ViewTransform) -> Vec<Tri> {
    let view = cam.view_matrix();
    let (nx, ny) = (mesh.nx, mesh.ny);
    let mut tris = Vec::with_capacity(2 * (nx - 1) * (ny - 1));
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let ids = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i];
            for t in [[0usize, 1, 2], [0, 2, 3]] {
                let v = [ids[t[0]], ids[t[1]], ids[t[2]]];
                let ps: Vec<Option<Vec3>> = v.iter().map(|&k| mesh.pos[k]).collect();
                if ps.iter().any(Option::is_none) {
                    continue;
                }
                let vs: Vec<Vec3> = ps.iter().map(|p| view.transform_point(p.unwrap_or_default())).collect();
                let depth = vs.iter().map(|p| -p.z).sum::<f64>() / 3.0;
                let clipped = clip_near(&vs);
                if clipped.len() < 3 {
                    continue;
                }
                let poly: Vec<Point> =
                    clipped.iter().filter_map(|p| cam.project_view(*p)).map(|q| vt.to_pixel((q.0, q.1))).collect();
                if poly.len() >= 3 && poly.iter().all(Point::is_finite) {
                    let exact = poly.len() == 3 && clipped.len() == 3 && vs.iter().all(|p| -p.z >= 0.05);
                    tris.push(Tri { v, poly, depth, cell: (j, i), exact });
                }
            }
        }
    }
    // Painter's algorithm: farthest first; ties broken by cell order for determinism.
    tris.sort_by(|a, b| b.depth.total_cmp(&a.depth).then(a.cell.cmp(&b.cell)));
    tris
}

fn height_t(g: &SurfaceGrid, v: usize, nx: usize) -> f64 {
    let (lo, hi) = g.z_range().unwrap_or((0.0, 1.0));
    let z = g.z[v / nx][v % nx];
    if hi > lo {
        (z - lo) / (hi - lo)
    } else {
        0.5
    }
}

/// Render a response surface. `title` is drawn above; hit targets are registered on a
/// decimated subset of the grid vertices (`series` 0, `index` = `row * nx + col`).
pub fn render_surface(
    g: &SurfaceGrid,
    style: &SurfaceStyle,
    cam: &Camera,
    theme: &Theme,
    w: f64,
    h: f64,
    title: &str,
) -> Scene {
    let mut sc = Scene::new(w, h, theme.background);
    sc.title = title.to_string();
    let top = if title.is_empty() { 8.0 } else { 14.0 + theme.title_size * 1.4 };
    if !title.is_empty() {
        sc.text(
            Point::new(w / 2.0, 8.0 + theme.title_size * 0.95),
            title,
            TextStyle::new(theme.title_size, theme.ink).anchored(Anchor::Middle).bold(),
        );
    }
    let bar_w = if style.color_bar { 74.0 } else { 0.0 };
    let area = Rect::new(8.0, top, (w - bar_w - 16.0).max(40.0), (h - top - 8.0).max(40.0));
    let zs = style.z_scale.clamp(0.05, 3.0);
    let corners = box_corners(zs);
    let vt = ViewTransform::fit(cam, &corners, area);
    let mesh = build_mesh(g, zs);
    let (zlo, zhi) = g.z_range().unwrap_or((0.0, 1.0));
    let proj = |p: Vec3| -> Option<Point> { cam.project_ndc(p).map(|q| vt.to_pixel((q.0, q.1))) };
    let eye = cam.eye();
    sc.clip_push(area);
    // Floor and back frame.
    let mut floor_edges: Vec<(Vec3, Vec3)> = Vec::new();
    if style.show_box {
        let c = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
        let fl = [c(-1.0, -1.0, -zs), c(1.0, -1.0, -zs), c(1.0, 1.0, -zs), c(-1.0, 1.0, -zs)];
        if let Some(poly) = fl.iter().map(|p| proj(*p)).collect::<Option<Vec<Point>>>() {
            sc.polygon(poly, Some(theme.grid.with_alpha(0.25)), None);
        }
        for k in 0..4 {
            floor_edges.push((fl[k], fl[(k + 1) % 4]));
        }
        // The two vertical edges farthest from the eye form the back frame.
        let mut vert: Vec<(f64, f64, f64)> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .iter()
            .map(|&(x, y)| (c(x, y, 0.0).sub(eye).length(), x, y))
            .collect();
        vert.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, x, y) in vert.iter().take(2) {
            floor_edges.push((c(x, y, -zs), c(x, y, zs)));
        }
        // Floor grid lines at the x/y ticks.
        let gl = Stroke::new(theme.grid, 0.8);
        for t in extended_ticks(g.xs[0], g.xs[g.xs.len() - 1], 5) {
            let u = 2.0 * (t.value - g.xs[0]) / (g.xs[g.xs.len() - 1] - g.xs[0]) - 1.0;
            if let (Some(a), Some(b)) = (proj(c(u, -1.0, -zs)), proj(c(u, 1.0, -zs))) {
                sc.line(a, b, gl);
            }
        }
        for t in extended_ticks(g.ys[0], g.ys[g.ys.len() - 1], 5) {
            let u = 2.0 * (t.value - g.ys[0]) / (g.ys[g.ys.len() - 1] - g.ys[0]) - 1.0;
            if let (Some(a), Some(b)) = (proj(c(-1.0, u, -zs)), proj(c(1.0, u, -zs))) {
                sc.line(a, b, gl);
            }
        }
        for (a, b) in &floor_edges {
            if let (Some(p), Some(q)) = (proj(*a), proj(*b)) {
                sc.line(p, q, Stroke::new(theme.ink.lighten(0.45), 1.0));
            }
        }
    }
    // Contours projected on the floor.
    if style.floor_contours > 0 && zhi > zlo {
        let (x0, x1, y0, y1) = (g.xs[0], g.xs[g.xs.len() - 1], g.ys[0], g.ys[g.ys.len() - 1]);
        for lv in equal_levels(zlo, zhi, style.floor_contours) {
            let col = ramp((lv - zlo) / (zhi - zlo));
            for line in contour_lines(&g.xs, &g.ys, &g.z, lv) {
                let pts: Vec<Point> = line
                    .points
                    .iter()
                    .filter_map(|&(x, y)| {
                        proj(Vec3::new(2.0 * (x - x0) / (x1 - x0) - 1.0, 2.0 * (y - y0) / (y1 - y0) - 1.0, -zs))
                    })
                    .collect();
                sc.polyline_gaps(&pts, Stroke::new(col, 1.2));
            }
        }
    }
    // Mesh.
    let tris = triangles(&mesh, cam, &vt);
    let wire = Stroke::new(theme.ink.lighten(0.25).with_alpha(0.55), 0.6);
    for t in &tris {
        let verts = t.v;
        let base_t = verts.iter().map(|&v| height_t(g, v, mesh.nx)).sum::<f64>() / 3.0;
        let centroid = verts.iter().filter_map(|&v| mesh.pos[v]).fold(Vec3::default(), Vec3::add).scale(1.0 / 3.0);
        let view_dir = eye.sub(centroid).normalized();
        let fill = match style.shading {
            Shading::Flat => {
                let p: Vec<Vec3> = verts.iter().filter_map(|&v| mesh.pos[v]).collect();
                let n = p[1].sub(p[0]).cross(p[2].sub(p[0])).normalized();
                style.light.shade(ramp(base_t), n, view_dir)
            }
            Shading::Smooth => {
                let cs: Vec<Color> = verts
                    .iter()
                    .map(|&v| style.light.shade(ramp(height_t(g, v, mesh.nx)), mesh.normal[v], view_dir))
                    .collect();
                let avg = |f: fn(&Color) -> u8| (cs.iter().map(|c| u32::from(f(c))).sum::<u32>() / 3) as u8;
                Color { r: avg(|c| c.r), g: avg(|c| c.g), b: avg(|c| c.b), a: 255 }
            }
        };
        match style.mode {
            RenderMode::Surface => sc.polygon(t.poly.clone(), Some(fill), Some(Stroke::new(fill, 0.5))),
            RenderMode::SurfaceWireframe => sc.polygon(t.poly.clone(), Some(fill), Some(wire)),
            RenderMode::Wireframe => {
                sc.polygon(t.poly.clone(), Some(theme.background), Some(Stroke::new(theme.ink.lighten(0.15), 0.8)))
            }
        }
    }
    // Axes: ticks and titles on the edges nearest the eye.
    if style.show_box {
        let ts = TextStyle::new(theme.tick_size, theme.ink);
        let centre = proj(Vec3::default()).unwrap_or_else(|| area.center());
        let out = |p: Point, d: f64| -> Point {
            let v = Point::new(p.x - centre.x, p.y - centre.y);
            let l = (v.x * v.x + v.y * v.y).sqrt().max(1e-9);
            Point::new(p.x + v.x / l * d, p.y + v.y / l * d)
        };
        let near = |a: Vec3, b: Vec3| a.add(b).scale(0.5).sub(eye).length();
        // x axis along the nearer of y = ±1 floor edges, y axis along the nearer x = ±1 edge.
        let ye = if near(Vec3::new(-1.0, -1.0, -zs), Vec3::new(1.0, -1.0, -zs))
            <= near(Vec3::new(-1.0, 1.0, -zs), Vec3::new(1.0, 1.0, -zs))
        {
            -1.0
        } else {
            1.0
        };
        let xe = if near(Vec3::new(-1.0, -1.0, -zs), Vec3::new(-1.0, 1.0, -zs))
            <= near(Vec3::new(1.0, -1.0, -zs), Vec3::new(1.0, 1.0, -zs))
        {
            -1.0
        } else {
            1.0
        };
        let (xl, xh, yl, yh) = (g.xs[0], g.xs[g.xs.len() - 1], g.ys[0], g.ys[g.ys.len() - 1]);
        for t in extended_ticks(xl, xh, 5) {
            let u = 2.0 * (t.value - xl) / (xh - xl) - 1.0;
            if let Some(p) = proj(Vec3::new(u, ye, -zs)) {
                sc.text(out(p, 12.0), t.label.clone(), ts.anchored(Anchor::Middle));
            }
        }
        for t in extended_ticks(yl, yh, 5) {
            let u = 2.0 * (t.value - yl) / (yh - yl) - 1.0;
            if let Some(p) = proj(Vec3::new(xe, u, -zs)) {
                sc.text(out(p, 12.0), t.label.clone(), ts.anchored(Anchor::Middle));
            }
        }
        let lab = TextStyle::new(theme.label_size, theme.ink).anchored(Anchor::Middle);
        if let Some(p) = proj(Vec3::new(0.0, ye, -zs)) {
            sc.text(out(p, 30.0), g.x_label.clone(), lab);
        }
        if let Some(p) = proj(Vec3::new(xe, 0.0, -zs)) {
            sc.text(out(p, 30.0), g.y_label.clone(), lab);
        }
        // z axis on the vertical edge that is leftmost on screen.
        let mut best: Option<(f64, Vec3)> = None;
        for &(x, y) in &[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            if let Some(p) = proj(Vec3::new(x, y, -zs)) {
                if best.is_none_or(|b| p.x < b.0) {
                    best = Some((p.x, Vec3::new(x, y, 0.0)));
                }
            }
        }
        if let Some((_, e)) = best {
            for t in extended_ticks(zlo, zhi, 5) {
                let z = zs * (2.0 * (t.value - zlo) / (zhi - zlo).max(1e-300) - 1.0);
                if let Some(p) = proj(Vec3::new(e.x, e.y, z)) {
                    sc.text(
                        Point::new(p.x - 6.0, p.y + theme.tick_size * 0.35),
                        t.label.clone(),
                        ts.anchored(Anchor::End),
                    );
                }
            }
        }
    }
    // Hit targets on a decimated vertex set.
    let stride = (((mesh.nx * mesh.ny) as f64 / (MAX_TARGETS.min(2500)) as f64).sqrt().ceil() as usize).max(1);
    for j in (0..mesh.ny).step_by(stride) {
        for i in (0..mesh.nx).step_by(stride) {
            if let Some(p) = mesh.pos[j * mesh.nx + i].and_then(proj) {
                if area.contains(p) {
                    sc.target(
                        Region::Point(p, 6.0),
                        0,
                        j * mesh.nx + i,
                        format!(
                            "{} = {}, {} = {}, {} = {}",
                            name_or(&g.x_label, "x"),
                            num(g.xs[i]),
                            name_or(&g.y_label, "y"),
                            num(g.ys[j]),
                            name_or(&g.z_label, "z"),
                            num(g.z[j][i])
                        ),
                    );
                }
            }
        }
    }
    sc.clip_pop();
    if style.color_bar && zhi > zlo {
        let x = w - bar_w + 6.0;
        let (y0, hgt) = (top + 10.0, (h - top - 30.0).max(40.0));
        let n = 48;
        for k in 0..n {
            let t = 1.0 - (k as f64 + 0.5) / n as f64;
            sc.rect(Rect::new(x, y0 + hgt * k as f64 / n as f64, 14.0, hgt / n as f64 + 0.6), Some(ramp(t)), None);
        }
        for t in extended_ticks(zlo, zhi, 5) {
            let y = y0 + hgt * (1.0 - (t.value - zlo) / (zhi - zlo));
            sc.text(
                Point::new(x + 18.0, y + theme.tick_size * 0.35),
                t.label,
                TextStyle::new(theme.tick_size, theme.ink),
            );
        }
    }
    sc
}

fn name_or<'a>(s: &'a str, d: &'a str) -> &'a str {
    if s.is_empty() {
        d
    } else {
        s
    }
}

/// Result of [`pick`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Picked {
    /// Grid row (y index) of the cell containing the point.
    pub row: usize,
    /// Grid column (x index).
    pub col: usize,
    /// Data x at the hit.
    pub x: f64,
    /// Data y at the hit.
    pub y: f64,
    /// Interpolated surface value at the hit.
    pub z: f64,
}

/// The visible surface point under the pixel `(px, py)` (nearest triangle along the view ray),
/// using exactly the transform of [`render_surface`] with the same arguments.
pub fn pick(
    g: &SurfaceGrid,
    style: &SurfaceStyle,
    cam: &Camera,
    w: f64,
    h: f64,
    title_present: bool,
    px: f64,
    py: f64,
) -> Option<Picked> {
    let top = if title_present { 14.0 + 15.0 * 1.4 } else { 8.0 };
    let bar_w = if style.color_bar { 74.0 } else { 0.0 };
    let area = Rect::new(8.0, top, (w - bar_w - 16.0).max(40.0), (h - top - 8.0).max(40.0));
    let zs = style.z_scale.clamp(0.05, 3.0);
    let vt = ViewTransform::fit(cam, &box_corners(zs), area);
    let mesh = build_mesh(g, zs);
    let q = Point::new(px, py);
    let mut best: Option<(f64, Picked)> = None;
    for t in triangles(&mesh, cam, &vt) {
        if !t.exact {
            continue;
        }
        let (a, b, c) = (t.poly[0], t.poly[1], t.poly[2]);
        let den = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if den.abs() < 1e-12 {
            continue;
        }
        let l1 = ((b.y - c.y) * (q.x - c.x) + (c.x - b.x) * (q.y - c.y)) / den;
        let l2 = ((c.y - a.y) * (q.x - c.x) + (a.x - c.x) * (q.y - c.y)) / den;
        let l3 = 1.0 - l1 - l2;
        if l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9 {
            continue;
        }
        if best.is_none_or(|b| t.depth < b.0) {
            let at = |v: usize| (g.xs[v % mesh.nx], g.ys[v / mesh.nx], g.z[v / mesh.nx][v % mesh.nx]);
            let (p1, p2, p3) = (at(t.v[0]), at(t.v[1]), at(t.v[2]));
            best = Some((
                t.depth,
                Picked {
                    row: t.cell.0,
                    col: t.cell.1,
                    x: l1 * p1.0 + l2 * p2.0 + l3 * p3.0,
                    y: l1 * p1.1 + l2 * p2.1 + l3 * p3.1,
                    z: l1 * p1.2 + l2 * p2.2 + l3 * p3.2,
                },
            ));
        }
    }
    best.map(|b| b.1)
}

// ------------------------------------------------------------------ contour plot

/// A design point drawn on a contour plot.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignPoint {
    /// x (same units as the grid).
    pub x: f64,
    /// y.
    pub y: f64,
    /// Kind: 0 factorial/cube, 1 axial/star, 2 centre.
    pub kind: u8,
}

/// Options of [`contour_plot`].
#[derive(Debug, Clone, PartialEq)]
pub struct ContourOptions {
    /// Levels to draw (empty = about eight round levels).
    pub levels: Vec<f64>,
    /// Shade the grid cells by level band underneath the lines.
    pub filled: bool,
    /// Label each level on its longest line.
    pub labels: bool,
    /// Design points to overlay.
    pub points: Vec<DesignPoint>,
}

impl Default for ContourOptions {
    fn default() -> ContourOptions {
        ContourOptions { levels: Vec::new(), filled: true, labels: true, points: Vec::new() }
    }
}

/// Labelled contour plot of a grid, with optional band shading and design points.
pub fn contour_plot(g: &SurfaceGrid, o: &ContourOptions, opts: &PlotOptions, title: &str) -> Scene {
    let Some((zlo, zhi)) = g.z_range() else {
        return crate::plots::common::empty_scene(opts, title, "No data to display");
    };
    let levels: Vec<f64> = if o.levels.is_empty() { nice_levels(zlo, zhi, 8) } else { o.levels.clone() };
    let (xl, xh, yl, yh) = (g.xs[0], g.xs[g.xs.len() - 1], g.ys[0], g.ys[g.ys.len() - 1]);
    let mut fs = FrameSpec::new(
        opts.width,
        opts.height,
        Scale::linear_padded(xl, xh, 0.0),
        Scale::linear_padded(yl, yh, 0.0),
        &opts.theme,
    );
    fs.title = title.into();
    fs.x_label = g.x_label.clone();
    fs.y_label = g.y_label.clone();
    let mut f = Frame::new(fs);
    f.begin_data();
    let t_of = |z: f64| if zhi > zlo { (z - zlo) / (zhi - zlo) } else { 0.5 };
    if o.filled {
        for j in 0..g.ys.len() - 1 {
            for i in 0..g.xs.len() - 1 {
                let c = [g.z[j][i], g.z[j][i + 1], g.z[j + 1][i], g.z[j + 1][i + 1]];
                if c.iter().any(|v| !v.is_finite()) {
                    continue;
                }
                let m = c.iter().sum::<f64>() / 4.0;
                let band = levels.iter().filter(|&&l| l <= m).count();
                let bt = if levels.is_empty() { t_of(m) } else { (band as f64 + 0.5) / (levels.len() as f64 + 1.0) };
                let (a, b) = (f.pt(g.xs[i], g.ys[j]), f.pt(g.xs[i + 1], g.ys[j + 1]));
                f.scene.rect(
                    Rect::new(a.x.min(b.x), a.y.min(b.y), (a.x - b.x).abs() + 0.6, (a.y - b.y).abs() + 0.6),
                    Some(ramp(bt).lighten(0.35)),
                    None,
                );
            }
        }
    }
    for &lv in &levels {
        let col = if o.filled { opts.theme.ink.lighten(0.15) } else { ramp(t_of(lv)) };
        let lines = contour_lines(&g.xs, &g.ys, &g.z, lv);
        for l in &lines {
            let pts: Vec<Point> = l.points.iter().map(|&(x, y)| f.pt(x, y)).collect();
            f.scene.polyline_gaps(&pts, Stroke::new(col, 1.3));
        }
        if o.labels {
            if let Some(l) = lines.iter().max_by_key(|l| l.points.len()).filter(|l| l.points.len() >= 4) {
                let (x, y) = l.points[l.points.len() / 2];
                let p = f.pt(x, y);
                f.scene.text(
                    Point::new(p.x + 2.0, p.y - 2.0),
                    num(lv),
                    TextStyle::new(opts.theme.tick_size, opts.theme.ink).bold(),
                );
            }
        }
    }
    for (k, d) in o.points.iter().enumerate() {
        let p = f.pt(d.x, d.y);
        let (m, size) = match d.kind {
            0 => (Marker::Square, 6.0),
            1 => (Marker::Diamond, 7.0),
            _ => (Marker::Cross, 8.0),
        };
        f.scene.marker(
            p,
            m,
            size,
            opts.theme.palette.color(usize::from(d.kind) + 1),
            Some(Stroke::new(opts.theme.ink, 0.8)),
        );
        if k < MAX_TARGETS {
            f.scene.target(Region::Point(p, 7.0), 0, k, format!("({}, {})", num(d.x), num(d.y)));
        }
    }
    f.end_data();
    f.finish()
}

/// Default rendering of a `ChartKind::Surface` spec (used by `render_spec`).
pub(crate) fn surface_spec_scene(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    match SurfaceGrid::from_spec(spec) {
        Ok(g) => render_surface(
            &g,
            &SurfaceStyle::default(),
            &Camera::default(),
            &opts.theme,
            opts.width,
            opts.height,
            &spec.title,
        ),
        Err(_) => crate::plots::common::empty_scene(opts, &spec.title, "No data to display"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scene::Shape;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    fn plane() -> SurfaceGrid {
        SurfaceGrid::from_fn((-1.0, 1.0), (-1.0, 1.0), 9, 9, |x, _| 0.2 * x).unwrap()
    }

    #[test]
    fn matrices_compose_and_invert() {
        let r = Mat4::rotation_z(0.7).mul(&Mat4::rotation_x(-0.3)).mul(&Mat4::rotation_y(1.1));
        let i = r.mul(&r.transpose());
        for a in 0..4 {
            for b in 0..4 {
                close(i.m[a][b], f64::from(u8::from(a == b)), 1e-12);
            }
        }
        let p = Vec3::new(1.0, 2.0, 3.0);
        close(r.transform_point(p).length(), p.length(), 1e-12);
        let t = Mat4::translation(Vec3::new(1.0, -2.0, 0.5)).mul(&Mat4::scaling(Vec3::new(2.0, 2.0, 2.0)));
        assert_eq!(t.transform_point(p), Vec3::new(3.0, 2.0, 6.5));
        assert_eq!(t.transform_dir(p), Vec3::new(2.0, 4.0, 6.0));
        let q = Mat4::rotation_z(std::f64::consts::FRAC_PI_2).transform_point(Vec3::new(1.0, 0.0, 0.0));
        close(q.y, 1.0, 1e-12);
        close(q.x, 0.0, 1e-12);
        assert_eq!(Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0)), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(Vec3::default().normalized(), Vec3::default());
    }

    #[test]
    fn look_at_and_projections() {
        let v = Mat4::look_at(Vec3::new(0.0, -5.0, 0.0), Vec3::default(), Vec3::new(0.0, 0.0, 1.0));
        let t = v.transform_point(Vec3::default());
        close(t.x, 0.0, 1e-12);
        close(t.z, -5.0, 1e-12);
        // Perspective: nearer objects appear larger; ortho: size independent of depth.
        let persp = Camera { elevation: 0.0, azimuth: -90.0, ..Camera::default() };
        let ortho = Camera { projection: Projection::Orthographic, ..persp };
        let (near, far) = (Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        let (pn, pf) = (persp.project_ndc(near).unwrap(), persp.project_ndc(far).unwrap());
        assert!(pn.0 > pf.0 && pn.2 < pf.2);
        let (on, of) = (ortho.project_ndc(near).unwrap(), ortho.project_ndc(far).unwrap());
        close(on.0, of.0, 1e-12);
        let c = persp.project_ndc(Vec3::default()).unwrap();
        close(c.0, 0.0, 1e-12);
        close(c.1, 0.0, 1e-12);
        // Behind the camera.
        assert!(persp.project_ndc(Vec3::new(0.0, -50.0, 0.0)).is_none());
        // Zoom doubles the image size.
        let z2 = Camera { zoom: 2.0, ..persp };
        close(z2.project_ndc(near).unwrap().0, 2.0 * pn.0, 1e-9);
    }

    #[test]
    fn camera_hooks() {
        let mut c = Camera::default();
        c.orbit(10.0, 1000.0);
        assert_eq!(c.elevation, 89.0);
        c.orbit(0.0, -5000.0);
        assert_eq!(c.elevation, -89.0);
        c.orbit(1000.0, 0.0);
        assert!((-180.0..=180.0).contains(&c.azimuth));
        c.zoom_by(1e9);
        assert_eq!(c.zoom, 20.0);
        c.zoom_by(0.0);
        assert_eq!(c.zoom, 20.0);
        c.zoom_by(f64::NAN);
        assert_eq!(c.zoom, 20.0);
        let mut p = Camera::default();
        let before = p.target;
        p.pan_by(0.1, 0.0);
        let moved = p.target.sub(before);
        // Panning right moves the target left in the view plane, perpendicular to the view axis.
        let axis = p.eye().sub(p.target).normalized();
        close(moved.dot(axis), 0.0, 1e-12);
        assert!(moved.length() > 0.0);
        p.reset();
        assert_eq!(p, Camera::default());
        let mut o = Camera::orthographic();
        o.orbit(3.0, 3.0);
        o.reset();
        assert_eq!(o.projection, Projection::Orthographic);
    }

    #[test]
    fn marching_squares_circle_is_one_closed_loop() {
        let g = SurfaceGrid::from_fn((-2.0, 2.0), (-2.0, 2.0), 81, 81, |x, y| x * x + y * y).unwrap();
        let lines = contour_lines(&g.xs, &g.ys, &g.z, 1.0);
        assert_eq!(lines.len(), 1);
        let l = &lines[0];
        assert!(l.closed);
        assert_eq!(l.points.first(), l.points.last());
        for &(x, y) in &l.points {
            close((x * x + y * y).sqrt(), 1.0, 2e-3);
        }
        // Circumference ≈ 2π.
        let len: f64 =
            l.points.windows(2).map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt()).sum();
        close(len, std::f64::consts::TAU, 0.01);
        assert!(contour_lines(&g.xs, &g.ys, &g.z, 100.0).is_empty());
        assert!(contour_lines(&g.xs, &g.ys, &g.z, -1.0).is_empty());
        // Nested levels give nested loops.
        assert_eq!(contour_lines(&g.xs, &g.ys, &g.z, 3.0).len(), 1);
    }

    #[test]
    fn marching_squares_open_lines_saddles_and_missing() {
        let g = SurfaceGrid::from_fn((0.0, 1.0), (0.0, 1.0), 11, 11, |x, _| x).unwrap();
        let l = contour_lines(&g.xs, &g.ys, &g.z, 0.35);
        assert_eq!(l.len(), 1);
        assert!(!l[0].closed);
        assert_eq!(l[0].points.len(), 11);
        for &(x, _) in &l[0].points {
            close(x, 0.35, 1e-12);
        }
        // Saddle: corners 1,0 / 0,1.
        let (xs, ys) = (vec![0.0, 1.0], vec![0.0, 1.0]);
        let z = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let hi = contour_lines(&xs, &ys, &z, 0.5); // centre = 0.5 >= level: high corners joined
        assert_eq!(hi.len(), 2);
        let lo = contour_lines(&xs, &ys, &z, 0.6); // centre below: high corners separated
        assert_eq!(lo.len(), 2);
        assert_ne!(hi[0].points, lo[0].points);
        // A missing corner skips only its cells.
        let mut gz = g.z.clone();
        gz[5][3] = f64::NAN;
        let holed = contour_lines(&g.xs, &g.ys, &gz, 0.35);
        assert!(holed.iter().map(|l| l.points.len()).sum::<usize>() < 11);
        assert!(contour_lines(&xs[..1], &ys, &z, 0.5).is_empty());
        assert!(contour_lines(&xs, &ys, &z, f64::NAN).is_empty());
    }

    #[test]
    fn level_helpers() {
        let e = equal_levels(0.0, 10.0, 4);
        assert_eq!(e.len(), 4);
        close(e[0], 2.0, 1e-12);
        let n = nice_levels(0.0, 10.0, 5);
        assert!(n.iter().all(|v| (0.0..=10.0).contains(v)) && n.len() >= 3);
    }

    #[test]
    fn lighting() {
        let l = Light::default();
        let base = Color::hex(0x808080);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let toward = l.direction;
        let away = l.direction.scale(-1.0);
        let view = Vec3::new(0.0, -1.0, 1.0).normalized();
        let bright = l.shade(base, toward, toward);
        let dark = l.shade(base, away.scale(-1.0), view);
        assert!(bright.r > dark.r);
        // Ambient-only floor and two-sided symmetry.
        let back = Light { specular: 0.0, ..l }.shade(base, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let flipped = Light { specular: 0.0, ..l }.shade(base, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(back, flipped);
        assert!(back.r >= (f64::from(base.r) * l.ambient).round() as u8);
        let _ = up;
    }

    #[test]
    fn grid_validation_and_sampling() {
        assert!(SurfaceGrid::new(vec![0.0], vec![0.0, 1.0], vec![vec![0.0], vec![0.0]]).is_err());
        assert!(SurfaceGrid::new(vec![0.0, 0.0], vec![0.0, 1.0], vec![vec![0.0; 2]; 2]).is_err());
        assert!(SurfaceGrid::new(vec![0.0, 1.0], vec![0.0, 1.0], vec![vec![0.0; 2]; 3]).is_err());
        assert!(SurfaceGrid::from_fn((0.0, 1.0), (0.0, 1.0), 1, 5, |_, _| 0.0).is_err());
        let g = plane();
        close(g.sample(0.37, -0.52).unwrap(), 0.2 * 0.37, 1e-12);
        assert!(g.sample(2.0, 0.0).is_none());
        assert!(g.sample(f64::NAN, 0.0).is_none());
        assert_eq!(g.z_range(), Some((-0.2, 0.2)));
    }

    #[test]
    fn painter_order_is_far_to_near_and_deterministic() {
        let g = SurfaceGrid::from_fn((-1.0, 1.0), (-1.0, 1.0), 12, 12, |x, y| (3.0 * x).sin() * y).unwrap();
        let cam = Camera::default();
        let mesh = build_mesh(&g, 0.6);
        let vt = ViewTransform::fit(&cam, &box_corners(0.6), Rect::new(0.0, 0.0, 400.0, 300.0));
        let t1 = triangles(&mesh, &cam, &vt);
        assert_eq!(t1.len(), 2 * 11 * 11);
        assert!(t1.windows(2).all(|w| w[0].depth >= w[1].depth));
        let t2 = triangles(&mesh, &cam, &vt);
        assert!(t1.iter().zip(&t2).all(|(a, b)| a.v == b.v));
    }

    #[test]
    fn render_modes_and_validity() {
        let g = plane();
        let th = Theme::light();
        let polys = |s: &Scene| s.items.iter().filter(|i| matches!(i.shape, Shape::Polygon(_))).count();
        for (mode, shading) in [
            (RenderMode::Surface, Shading::Flat),
            (RenderMode::SurfaceWireframe, Shading::Smooth),
            (RenderMode::Wireframe, Shading::Flat),
        ] {
            let st = SurfaceStyle { mode, shading, ..SurfaceStyle::default() };
            let s = render_surface(&g, &st, &Camera::default(), &th, 640.0, 420.0, "Plane");
            assert!(s.clips_balanced());
            assert!(polys(&s) >= 2 * 8 * 8, "{mode:?}");
            assert!(!s.targets.is_empty());
            let svg = crate::render::svg::to_svg(&s);
            assert!(openstats_core::io::xml::events(&svg).is_ok());
        }
        let a = render_surface(&g, &SurfaceStyle::default(), &Camera::default(), &th, 640.0, 420.0, "A");
        let b = render_surface(&g, &SurfaceStyle::default(), &Camera::default(), &th, 640.0, 420.0, "A");
        assert_eq!(a, b);
        let o = render_surface(&g, &SurfaceStyle::default(), &Camera::orthographic(), &th, 640.0, 420.0, "A");
        assert_ne!(a, o);
        let mut zoomed = Camera::default();
        zoomed.zoom_by(2.0);
        assert_ne!(a, render_surface(&g, &SurfaceStyle::default(), &zoomed, &th, 640.0, 420.0, "A"));
        let bare = SurfaceStyle { show_box: false, floor_contours: 0, color_bar: false, ..SurfaceStyle::default() };
        assert!(
            render_surface(&g, &bare, &Camera::default(), &th, 640.0, 420.0, "").primitive_count()
                < a.primitive_count()
        );
    }

    #[test]
    fn missing_values_are_skipped() {
        let mut g = plane();
        g.z[4][4] = f64::NAN;
        let s = render_surface(&g, &SurfaceStyle::default(), &Camera::default(), &Theme::light(), 500.0, 400.0, "Hole");
        let polys = s.items.iter().filter(|i| matches!(i.shape, Shape::Polygon(_))).count();
        assert!(polys < 2 * 8 * 8 + 10);
        assert!(s.clips_balanced());
        // An all-missing grid still renders the frame.
        let allnan = SurfaceGrid::new(vec![0.0, 1.0], vec![0.0, 1.0], vec![vec![f64::NAN; 2]; 2]).unwrap();
        assert!(render_surface(
            &allnan,
            &SurfaceStyle::default(),
            &Camera::default(),
            &Theme::light(),
            300.0,
            200.0,
            ""
        )
        .clips_balanced());
    }

    #[test]
    fn picking_matches_projection() {
        let g = plane();
        let (w, h) = (640.0, 420.0);
        let cam = Camera::default();
        let st = SurfaceStyle::default();
        let top = 14.0 + 15.0 * 1.4;
        let area = Rect::new(8.0, top, w - 74.0 - 16.0, h - top - 8.0);
        let vt = ViewTransform::fit(&cam, &box_corners(0.6), area);
        let mesh = build_mesh(&g, 0.6);
        for &(j, i) in &[(2usize, 3usize), (5, 5), (7, 1)] {
            let p = mesh.pos[j * 9 + i].unwrap();
            let q = cam.project_ndc(p).unwrap();
            let px = vt.to_pixel((q.0, q.1));
            let hit = pick(&g, &st, &cam, w, h, true, px.x, px.y).expect("hit");
            close(hit.x, g.xs[i], 1e-6);
            close(hit.y, g.ys[j], 1e-6);
            close(hit.z, g.z[j][i], 1e-6);
        }
        assert!(pick(&g, &st, &cam, w, h, true, 1.0, 1.0).is_none());
    }

    #[test]
    fn contour_plot_draws_lines_labels_and_points() {
        let g = SurfaceGrid::from_fn((-2.0, 2.0), (-2.0, 2.0), 41, 41, |x, y| x * x + y * y).unwrap();
        let o = ContourOptions {
            points: vec![
                DesignPoint { x: 0.0, y: 0.0, kind: 2 },
                DesignPoint { x: 1.0, y: 1.0, kind: 0 },
                DesignPoint { x: 1.4, y: 0.0, kind: 1 },
            ],
            ..ContourOptions::default()
        };
        let s = contour_plot(&g, &o, &PlotOptions::default(), "Contours");
        assert!(s.clips_balanced());
        assert_eq!(s.targets.len(), 3);
        let svg = crate::render::svg::to_svg(&s);
        assert!(svg.contains("Contours"));
        let plain = contour_plot(
            &g,
            &ContourOptions { filled: false, labels: false, levels: vec![1.0, 2.0], points: vec![] },
            &PlotOptions::default(),
            "",
        );
        assert!(plain.primitive_count() < s.primitive_count());
    }
}
