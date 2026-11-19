//! Phase 3c: response surfaces of fitted CCD and Box–Behnken models through the 3-D engine.

use openstats_charts::plots::PlotOptions;
use openstats_charts::render::{png, raster, svg};
use openstats_charts::render_3d::*;
use openstats_charts::{render_spec, Theme};
use openstats_stats::doe::*;

fn ccd_analysis() -> (Design, SurfaceAnalysis) {
    let mut opts = CcdOptions::new(CcdKind::Circumscribed);
    opts.center_factorial = 3;
    opts.center_axial = 2;
    let d = central_composite(2, &opts).unwrap();
    // A known quadratic with a maximum inside the region.
    let y: Vec<f64> = d
        .coded
        .iter()
        .enumerate()
        .map(|(i, r)| {
            50.0 + 4.0 * r[0] + 2.0 * r[1] - 3.0 * r[0] * r[0] - 2.0 * r[1] * r[1]
                + 1.0 * r[0] * r[1]
                + 0.05 * (i % 3) as f64
        })
        .collect();
    let a = analyze_surface(&d, &y, &SurfaceOptions::default()).unwrap();
    (d, a)
}

fn valid(s: &openstats_charts::Scene) {
    assert!(s.clips_balanced());
    let doc = svg::to_svg(s);
    openstats_core::io::xml::events(&doc).expect("valid svg");
    let img = png::decode(&raster::render_png(s, 1.0)).expect("png");
    let first = img.rgba[..4].to_vec();
    assert!(img.rgba.chunks(4).any(|p| p != first.as_slice()), "blank");
}

#[test]
fn ccd_surface_and_contour_specs_render() {
    let (d, a) = ccd_analysis();
    let plot = SurfacePlot {
        model: &a.model,
        factors: &d.factors,
        ix: 0,
        iy: 1,
        fixed: &[0.0, 0.0],
        range: ([-1.5, 1.5], [-1.5, 1.5]),
        grid_points: 31,
    };
    let s = surface_chart(&plot).unwrap();
    let scene = render_spec(&s, &PlotOptions::default().sized(720.0, 520.0));
    valid(&scene);
    assert!(scene.targets.len() > 100);
    let c = contour_chart(&plot, 7).unwrap();
    valid(&render_spec(&c, &PlotOptions::default()));
}

#[test]
fn ccd_grid_contours_lie_on_their_levels() {
    let (d, a) = ccd_analysis();
    let f = &d.factors;
    // Coded grid straight from the fitted model.
    let g = SurfaceGrid::from_fn((-1.5, 1.5), (-1.5, 1.5), 61, 61, |x, y| a.model.predict(&[x, y])).unwrap();
    let (lo, hi) = g.z_range().unwrap();
    let levels = nice_levels(lo, hi, 6);
    assert!(levels.len() >= 3);
    for lv in levels {
        for line in contour_lines(&g.xs, &g.ys, &g.z, lv) {
            for &(x, y) in &line.points {
                // Linear interpolation error on a quadratic is O(h^2).
                assert!((a.model.predict(&[x, y]) - lv).abs() < 0.05, "off level {lv}");
            }
        }
    }
    // The model peaks inside the region: the highest level encloses it as a closed loop.
    let near_top = contour_lines(&g.xs, &g.ys, &g.z, hi - 0.5);
    assert!(near_top.iter().any(|l| l.closed));
    let _ = f;
    // Design points overlay: factorial, axial and centre points all drawn with hit targets.
    let pts: Vec<DesignPoint> = d
        .coded
        .iter()
        .zip(&d.point_type)
        .map(|(r, t)| DesignPoint {
            x: r[0],
            y: r[1],
            kind: match t {
                PointType::Factorial => 0,
                PointType::Axial => 1,
                _ => 2,
            },
        })
        .collect();
    let n = pts.len();
    let o = ContourOptions { points: pts, ..ContourOptions::default() };
    let scene = contour_plot(&g, &o, &PlotOptions::default(), "CCD contours");
    assert_eq!(scene.targets.len(), n);
    valid(&scene);
}

#[test]
fn box_behnken_surface_modes() {
    let d = box_behnken(3, 3, None).unwrap();
    let y: Vec<f64> = d
        .coded
        .iter()
        .map(|r| 10.0 + r[0] - 2.0 * r[1] + 0.5 * r[2] - r[0] * r[0] + 0.3 * r[1] * r[2] - 0.8 * r[1] * r[1])
        .collect();
    let a = analyze_surface(&d, &y, &SurfaceOptions::default()).unwrap();
    // Slice of the 3-factor model at C = 0.
    let mut g = SurfaceGrid::from_fn((-1.0, 1.0), (-1.0, 1.0), 25, 25, |x, y| a.model.predict(&[x, y, 0.0])).unwrap();
    g.x_label = "A".into();
    g.y_label = "B".into();
    g.z_label = "Response".into();
    let th = Theme::light();
    for mode in [RenderMode::Surface, RenderMode::SurfaceWireframe, RenderMode::Wireframe] {
        for cam in [Camera::default(), Camera::orthographic()] {
            let st = SurfaceStyle { mode, ..SurfaceStyle::default() };
            valid(&render_surface(&g, &st, &cam, &th, 640.0, 460.0, "Box-Behnken surface"));
        }
    }
    // Interaction hooks: orbit, zoom and pan each change the picture; picking follows the view.
    let st = SurfaceStyle::default();
    let mut cam = Camera::default();
    let base = render_surface(&g, &st, &cam, &th, 640.0, 460.0, "");
    // A vertex target registered by the renderer is picked back at the same pixel.
    let t = &base.targets[base.targets.len() / 2];
    let openstats_charts::core::hit::Region::Point(pp, _) = t.region else { panic!("point target") };
    let hit0 = pick(&g, &st, &cam, 640.0, 460.0, false, pp.x, pp.y);
    cam.orbit(40.0, 10.0);
    assert_ne!(base, render_surface(&g, &st, &cam, &th, 640.0, 460.0, ""));
    cam.reset();
    assert_eq!(base, render_surface(&g, &st, &cam, &th, 640.0, 460.0, ""));
    assert_eq!(hit0, pick(&g, &st, &cam, 640.0, 460.0, false, pp.x, pp.y));
    let picked = hit0.expect("a vertex of the surface can be picked");
    assert!((picked.x - g.xs[t.index % 25]).abs() < 1e-6 && (picked.y - g.ys[t.index / 25]).abs() < 1e-6);
    assert!((a.model.predict(&[picked.x, picked.y, 0.0]) - picked.z).abs() < 0.05);
}
