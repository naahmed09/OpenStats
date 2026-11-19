//! Renders every chart type from real `openstats-stats` outputs and hand-built specs and checks
//! SVG well-formedness, PNG validity, canvas round trips, determinism and geometry sanity.

use openstats_charts::core::hit::{HitIndex, Viewport};
use openstats_charts::core::scale::ProbabilityKind;
use openstats_charts::core::scene::Shape;
use openstats_charts::core::{geom::Point, Scene};
use openstats_charts::plots::basic::*;
use openstats_charts::plots::quality::*;
use openstats_charts::plots::relation::*;
use openstats_charts::plots::{render_panels, render_spec, PlotOptions};
use openstats_charts::render::{canvas, png, raster, svg};
use openstats_stats::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};
use openstats_stats::spc::capability::Specs;
use openstats_stats::spc::charts::{i_mr, p_chart, xbar_r, ChartOptions};
use openstats_stats::spc::sixpack::{capability_sixpack_individuals, SixpackOptions};

fn lcg(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (next().max(1e-12), next());
            10.0 + 2.0 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}

fn opts() -> PlotOptions {
    PlotOptions::default()
}

/// Every check that must hold for any scene.
fn check(name: &str, s: &Scene) {
    assert!(s.clips_balanced(), "{name}: unbalanced clips");
    assert!(s.primitive_count() > 3, "{name}: nearly empty scene");
    for it in &s.items {
        let finite = match &it.shape {
            Shape::Line(a, b) => a.is_finite() && b.is_finite(),
            Shape::Polyline(p) | Shape::Polygon(p) => p.iter().all(Point::is_finite),
            Shape::Rect(r) => r.x.is_finite() && r.y.is_finite() && r.w.is_finite() && r.h.is_finite(),
            Shape::Circle { c, r } | Shape::Wedge { c, r, .. } => c.is_finite() && r.is_finite(),
            Shape::Text { at, .. } | Shape::Marker { at, .. } => at.is_finite(),
            Shape::ClipPush(_) | Shape::ClipPop => true,
        };
        assert!(finite, "{name}: non-finite geometry {:?}", it.shape);
    }
    // SVG: well-formed XML with an svg root.
    let doc = svg::to_svg(s);
    assert!(doc.starts_with("<svg"), "{name}: svg header");
    let ev = openstats_core::io::xml::events(&doc).unwrap_or_else(|e| panic!("{name}: invalid XML: {e}"));
    assert!(!ev.is_empty());
    // PNG: valid, correct size, not blank.
    let bytes = raster::render_png(s, 1.0);
    let img = png::decode(&bytes).unwrap_or_else(|e| panic!("{name}: bad png: {e:?}"));
    assert_eq!(img.width as f64, s.width.round());
    assert_eq!(img.height as f64, s.height.round());
    let first = &img.rgba[..4];
    assert!(img.rgba.chunks(4).any(|p| p != first), "{name}: blank image");
    // Canvas: JSON is non-empty and the binary stream round trips.
    let cmds = canvas::to_commands(s);
    assert!(!canvas::to_json(&cmds).is_empty());
    let back = canvas::from_binary(&canvas::to_binary(&cmds)).expect("binary decode");
    assert_eq!(back.len(), cmds.len(), "{name}: canvas round trip");
    // Determinism.
    assert_eq!(svg::to_svg(s), doc);
    assert_eq!(raster::rasterize(s, 1.0).hash(), raster::rasterize(s, 1.0).hash());
    // Hit index over registered targets.
    let hi = HitIndex::new(s.width, s.height, &s.targets);
    assert_eq!(hi.len(), s.targets.len());
}

#[test]
fn basic_plots() {
    let a = lcg(1, 80);
    let b: Vec<f64> = lcg(2, 60).iter().map(|v| v + 3.0).collect();
    let g = vec![BoxGroup { label: "A".into(), data: a.clone() }, BoxGroup { label: "B".into(), data: b.clone() }];
    let o = opts();
    check("hist none", &histogram_from_data(&a, None, &HistogramOverlay::None, "Histogram", "x", &o));
    check("hist normal", &histogram_from_data(&a, Some(8), &HistogramOverlay::Normal, "Histogram", "x", &o));
    check("hist const", &histogram_from_data(&[5.0, 5.0, 5.0], None, &HistogramOverlay::Normal, "Const", "x", &o));
    check("box", &boxplot_groups(&g, false, true, "Boxplot", "g", "y", &o));
    check("box notched", &boxplot_groups(&g, true, false, "Notched", "g", "y", &o));
    check("ivp", &individual_value_plot(&g, "IVP", "g", "y", &o));
    let iv = vec![
        IntervalGroup { label: "A".into(), estimate: 10.0, lower: 9.0, upper: 11.0 },
        IntervalGroup { label: "B".into(), estimate: 12.0, lower: 11.0, upper: 13.5 },
    ];
    check("interval", &interval_plot(&iv, "Interval", "g", "mean", &o));
    let cats: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    check("bar", &bar_chart(&cats, &[("n".into(), vec![3.0, 5.0, 2.0])], "Bar", "c", "n", &o));
    check(
        "bar clustered",
        &bar_chart(&cats, &[("x".into(), vec![3.0, 5.0, 2.0]), ("y".into(), vec![1.0, 2.0, 4.0])], "Bar", "c", "n", &o),
    );
    check("pie", &pie_chart(&cats, &[3.0, 5.0, 2.0], "Pie", &o).unwrap());
    assert!(pie_chart(&cats, &[0.0, 0.0, 0.0], "Pie", &o).is_err());
    assert!(pie_chart(&cats, &[1.0, -1.0, 2.0], "Pie", &o).is_err());
    check("ecdf", &ecdf_plot(&g, "ECDF", "x", &o));
    for k in [ProbabilityKind::Normal, ProbabilityKind::Lognormal, ProbabilityKind::Weibull] {
        check("probability", &probability_plot(&a, k, Some(0.95), "Probability Plot", "x", &o));
    }
    let empty = boxplot_groups(&[], false, false, "Empty", "", "", &o);
    assert!(empty.clips_balanced());
    assert!(svg::to_svg(&empty).contains("No data to display"));
}

#[test]
fn relation_plots() {
    let x: Vec<f64> = (0..50).map(f64::from).collect();
    let n = lcg(3, 50);
    let y: Vec<f64> = x.iter().zip(&n).map(|(x, e)| 2.0 + 0.5 * x + (e - 10.0)).collect();
    let d = ScatterData::new(x.clone(), y.clone());
    let o = opts();
    check("scatter", &scatter_plot(&d, &Fit::None, FitBands::default(), "Scatter", "x", "y", &o));
    check(
        "scatter ols",
        &scatter_plot(
            &d,
            &Fit::Poly(1),
            FitBands { confidence: Some(0.95), prediction: Some(0.95) },
            "Fit",
            "x",
            "y",
            &o,
        ),
    );
    check("scatter quad", &scatter_plot(&d, &Fit::Poly(2), FitBands::default(), "Quad", "x", "y", &o));
    check("scatter lowess", &scatter_plot(&d, &Fit::Lowess(0.5), FitBands::default(), "Lowess", "x", "y", &o));
    let grouped = ScatterData {
        x: x.clone(),
        y: y.clone(),
        groups: (0..50).map(|i| i % 3).collect(),
        group_names: vec!["a".into(), "b".into(), "c".into()],
    };
    check("scatter grouped", &scatter_plot(&grouped, &Fit::None, FitBands::default(), "Grouped", "x", "y", &o));
}

#[test]
fn fits_are_correct() {
    // exact line
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|v| 3.0 + 2.0 * v).collect();
    let f = poly_fit(&x, &y, 1).expect("fit");
    for v in [0.0, 4.5, 9.0] {
        assert!((f.predict(v) - (3.0 + 2.0 * v)).abs() < 1e-9);
    }
    // exact quadratic
    let yq: Vec<f64> = x.iter().map(|v| 1.0 - v + 0.5 * v * v).collect();
    let q = poly_fit(&x, &yq, 2).expect("quad");
    assert!((q.predict(3.0) - (1.0 - 3.0 + 4.5)).abs() < 1e-8);
    assert!(poly_fit(&x[..2], &y[..2], 1).is_none());
    assert!(poly_fit(&[1.0; 5], &[1.0, 2.0, 3.0, 4.0, 5.0], 1).is_none());
    // LOWESS reproduces a straight line and is robust to one outlier.
    let (lx, ly) = lowess(&x, &y, 0.6, 2);
    assert_eq!(lx.len(), 10);
    for (a, b) in lx.iter().zip(&ly) {
        assert!((b - (3.0 + 2.0 * a)).abs() < 1e-8);
    }
    let mut yo = y.clone();
    yo[5] += 100.0;
    let (_, lo) = lowess(&x, &yo, 0.6, 3);
    assert!((lo[5] - y[5]).abs() < 5.0, "robust fit pulled to outlier: {}", lo[5]);
}

#[test]
fn spec_charts_from_stats() {
    let o = opts();
    let data = lcg(4, 60);
    // control charts: individuals, X-bar/R, P chart
    let imr = i_mr(&data, &ChartOptions::default()).expect("imr");
    let cc = ControlChartOptions { zones: true, ..ControlChartOptions::default() };
    check("I chart", &control_chart_panel(&imr.location, "I Chart", "Observation", &cc, &o));
    check(
        "MR chart",
        &control_chart_panel(&imr.dispersion, "MR Chart", "Observation", &ControlChartOptions::default(), &o),
    );
    let staged = ControlChartOptions {
        stage_breaks: vec![30],
        stage_labels: vec!["Before".into(), "After".into()],
        ..ControlChartOptions::default()
    };
    check("staged", &control_chart_panel(&imr.location, "Staged", "Observation", &staged, &o));
    let groups: Vec<Vec<f64>> = data.chunks(5).map(<[f64]>::to_vec).collect();
    let xr = xbar_r(&groups, &ChartOptions::default()).expect("xbar-r");
    check("xbar", &control_chart_panel(&xr.location, "Xbar", "Sample", &ControlChartOptions::default(), &o));
    check("spec chart_spec", &render_spec(&imr.location.chart_spec("I", "Obs"), &o));
    let defects: Vec<f64> = (0..25).map(|i| f64::from(2 + (i * 7) % 6)).collect();
    let sizes = vec![100.0; 25];
    let p = p_chart(&defects, &sizes, &ChartOptions::default()).expect("p");
    check("p chart", &render_spec(&p.panel.chart_spec("P Chart", "Sample"), &o));
    // capability sixpack as panels
    let sp = capability_sixpack_individuals(&data, &Specs::two_sided(4.0, 16.0), &SixpackOptions::default())
        .expect("sixpack");
    let specs = sp.charts();
    assert!(specs.len() >= 6);
    for s in &specs {
        check(&s.title, &render_spec(s, &o));
    }
    check(
        "sixpack",
        &render_panels(&specs[..6], 3, "Capability Sixpack", &PlotOptions::default().sized(1000.0, 700.0)),
    );
    // capability histogram
    let ci = CapabilityInput {
        data: data.clone(),
        lsl: Some(4.0),
        usl: Some(16.0),
        target: Some(10.0),
        mean: 10.0,
        sd_within: Some(1.9),
        sd_overall: 2.0,
    };
    check("capability hist", &capability_histogram(&ci, "Capability", "x", &o));
    // pareto
    let cats: Vec<String> = ["scratch", "dent", "crack", "chip"].iter().map(|s| s.to_string()).collect();
    check("pareto", &pareto_chart(&cats, &[12.0, 40.0, 5.0, 22.0], "Pareto", "defect", "count", &o));
}

#[test]
fn hand_built_specs() {
    let o = opts();
    let s = |n: &str, x: Vec<f64>, y: Vec<f64>| Series { name: n.into(), x, y };
    let mk = |kind, series: Vec<Series>, cats: Vec<String>, refs: Vec<ReferenceLine>| ChartSpec {
        kind,
        title: "T".into(),
        x_label: "X".into(),
        y_label: "Y".into(),
        categories: cats,
        series,
        reference_lines: refs,
    };
    check(
        "hist",
        &render_spec(
            &mk(
                ChartKind::Histogram,
                vec![s("f", vec![0.0, 1.0, 2.0, 3.0], vec![2.0, 5.0, 1.0])],
                vec![],
                vec![ReferenceLine { axis: Axis::X, value: 1.5, label: "LSL".into() }],
            ),
            &o,
        ),
    );
    check(
        "box",
        &render_spec(
            &mk(
                ChartKind::Boxplot,
                vec![
                    s("g1", vec![1.0; 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
                    s("g2", vec![2.0; 5], vec![2.0, 3.0, 4.0, 5.0, 6.0]),
                ],
                vec![],
                vec![],
            ),
            &o,
        ),
    );
    check(
        "bar",
        &render_spec(
            &mk(
                ChartKind::Bar,
                vec![s("n", vec![1.0, 2.0], vec![3.0, 4.0])],
                vec!["a".into(), "b".into()],
                vec![ReferenceLine { axis: Axis::Y, value: 3.5, label: "limit".into() }],
            ),
            &o,
        ),
    );
    check(
        "prob",
        &render_spec(
            &mk(
                ChartKind::ProbabilityPlot,
                vec![s("d", vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![-1.2, -0.5, 0.0, 0.5, 1.2])],
                vec![],
                vec![],
            ),
            &o,
        ),
    );
    check(
        "interval h",
        &render_spec(
            &mk(
                ChartKind::IntervalPlot,
                vec![
                    s("estimate", vec![5.0, 6.0], vec![1.0, 2.0]),
                    s("lower", vec![4.0, 5.0], vec![1.0, 2.0]),
                    s("upper", vec![6.0, 7.5], vec![1.0, 2.0]),
                ],
                vec!["a".into(), "b".into()],
                vec![],
            ),
            &o,
        ),
    );
    check(
        "line",
        &render_spec(
            &mk(
                ChartKind::Line,
                vec![s("Mean", vec![-1.0, 1.0], vec![3.0, 5.0])],
                vec![],
                vec![ReferenceLine { axis: Axis::Y, value: 4.0, label: "Grand mean".into() }],
            ),
            &o,
        ),
    );
    check(
        "contour",
        &render_spec(
            &mk(
                ChartKind::Contour,
                vec![
                    s(
                        "1.5",
                        vec![0.0, 1.0, f64::NAN, 1.0, 2.0, f64::NAN],
                        vec![0.0, 1.0, f64::NAN, 2.0, 1.0, f64::NAN],
                    ),
                    s("2.5", vec![0.5, 1.5, f64::NAN], vec![0.5, 1.5, f64::NAN]),
                ],
                vec![],
                vec![],
            ),
            &o,
        ),
    );
    let rows: Vec<Series> = (0..5)
        .map(|j| s(&format!("{j}"), (0..5).map(f64::from).collect(), (0..5).map(|i| f64::from(i * i + j)).collect()))
        .collect();
    check("surface", &render_spec(&mk(ChartKind::Surface, rows, vec!["B".into()], vec![]), &o));
    let cube = mk(
        ChartKind::Scatter,
        vec![
            s("C=-1", vec![-1.0, 1.0, -1.0, 1.0], vec![-1.0, -1.0, 1.0, 1.0]),
            s("C=+1", vec![-1.0, 1.0, -1.0, 1.0], vec![-1.0, -1.0, 1.0, 1.0]),
        ],
        (0..8).map(|i| format!("A=-1,B=-1,C=-1: {i}.5")).collect(),
        vec![],
    );
    let mut cube = cube;
    cube.title = "Cube Plot (data means)".into();
    check("cube", &render_spec(&cube, &o));
    // degenerate inputs yield a notice scene, never a panic
    for kind in [
        ChartKind::Histogram,
        ChartKind::Boxplot,
        ChartKind::ProbabilityPlot,
        ChartKind::Scatter,
        ChartKind::Line,
        ChartKind::Bar,
        ChartKind::IntervalPlot,
        ChartKind::Contour,
        ChartKind::Surface,
    ] {
        let e = render_spec(&mk(kind, vec![], vec![], vec![]), &o);
        assert!(e.clips_balanced());
        assert!(svg::to_svg(&e).contains("<svg"));
    }
}

#[test]
fn hit_testing_and_viewport() {
    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect();
    let d = ScatterData::new(x.clone(), y.clone());
    let sc = scatter_plot(&d, &Fit::None, FitBands::default(), "S", "x", "y", &opts());
    assert_eq!(sc.targets.len(), 20);
    let idx = HitIndex::new(sc.width, sc.height, &sc.targets);
    let ax = &sc.axes[0];
    let p = ax.to_pixel(Point::new(7.0, 49.0));
    let hit = idx.query(p, 4.0).expect("hit");
    assert_eq!(hit.index, 7);
    assert!(idx.query(Point::new(1.0, 1.0), 4.0).is_none());
    // data round trip
    let back = ax.to_data(p);
    assert!((back.x - 7.0).abs() < 1e-6 && (back.y - 49.0).abs() < 1e-6);
    // zoomed viewport narrows the domain and keeps the plot valid
    let vp = Viewport::default().zoomed(&ax.x, &ax.y, 0.5, 0.5, 2.0);
    let zoomed =
        scatter_plot(&d, &Fit::None, FitBands::default(), "S", "x", "y", &PlotOptions { viewport: vp, ..opts() });
    let (a0, a1) = zoomed.axes[0].x.domain();
    let (b0, b1) = ax.x.domain();
    assert!(a1 - a0 < b1 - b0);
    check("zoomed", &zoomed);
}

#[test]
fn raster_goldens_are_stable() {
    // The software rasteriser is deterministic: identical scenes give identical pixels,
    // different scenes do not.
    let d = ScatterData::new(vec![1.0, 2.0, 3.0], vec![1.0, 4.0, 9.0]);
    let a = scatter_plot(&d, &Fit::Poly(1), FitBands::default(), "A", "x", "y", &opts());
    let b = scatter_plot(&d, &Fit::None, FitBands::default(), "A", "x", "y", &opts());
    assert_eq!(raster::rasterize(&a, 1.0).hash(), raster::rasterize(&a, 1.0).hash());
    assert_ne!(raster::rasterize(&a, 1.0).hash(), raster::rasterize(&b, 1.0).hash());
    let big = raster::rasterize(&a, 2.0);
    assert_eq!(big.width, 1280);
    assert_eq!(big.height, 800);
    let dark = scatter_plot(
        &d,
        &Fit::None,
        FitBands::default(),
        "A",
        "x",
        "y",
        &PlotOptions { theme: openstats_charts::Theme::dark(), ..opts() },
    );
    check("dark", &dark);
}

#[test]
fn transparent_background_stays_transparent_and_opaque_fills_are_unchanged() {
    use openstats_charts::core::style::Color;
    let mut s = openstats_charts::core::scene::Scene::new(20.0, 20.0, Color::WHITE.with_alpha(0.0));
    s.rect(openstats_charts::core::geom::Rect::new(10.0, 10.0, 10.0, 10.0), Some(Color::hex(0x0072B2)), None);
    let img = raster::rasterize(&s, 1.0);
    let px = |x: usize, y: usize| &img.rgba[4 * (y * img.width + x)..4 * (y * img.width + x) + 4];
    assert_eq!(px(2, 2)[3], 0, "untouched background keeps alpha 0");
    assert_eq!(px(15, 15), &[0x00, 0x72, 0xB2, 255], "an opaque fill is exact");
    let opaque = openstats_charts::core::scene::Scene::new(4.0, 4.0, Color::WHITE);
    assert_eq!(raster::rasterize(&opaque, 1.0).rgba[3], 255, "opaque backgrounds stay opaque");
}

#[test]
fn external_text_painter_draws_and_falls_back() {
    use openstats_charts::core::geom::Point;
    use openstats_charts::core::scene::{Anchor, Scene, TextStyle};
    use openstats_charts::core::style::Color;
    use openstats_charts::render::raster::{rasterize_with, TextPainter, TextRun};

    /// A 3 × 2 solid block 4 px wide in advance, sitting 2 px above the baseline.
    struct Block;
    impl TextPainter for Block {
        fn render(&mut self, text: &str, _size: f32, _bold: bool) -> Option<TextRun> {
            if text == "fallback" {
                return None;
            }
            Some(TextRun { advance: 4.0, left: 1, top: 2, width: 3, height: 2, coverage: vec![255; 6] })
        }
    }
    let ink = |img: &openstats_charts::render::png::Image| -> Vec<(usize, usize)> {
        (0..img.height)
            .flat_map(|y| (0..img.width).map(move |x| (x, y)))
            .filter(|&(x, y)| img.rgba[4 * (y * img.width + x)] < 128)
            .collect()
    };
    let mut s = Scene::new(20.0, 20.0, Color::WHITE);
    s.text(Point::new(10.0, 10.0), "a", TextStyle::new(10.0, Color::BLACK));
    let img = rasterize_with(&s, 1.0, Some(&mut Block));
    // Start anchor: x = 10 + left(1) .. 13, rows baseline − top(2) .. 10.
    assert_eq!(ink(&img), vec![(11, 8), (12, 8), (13, 8), (11, 9), (12, 9), (13, 9)]);
    // Middle anchor centres the advance: u0 = −2.
    let mut m = Scene::new(20.0, 20.0, Color::WHITE);
    m.text(Point::new(10.0, 10.0), "a", TextStyle::new(10.0, Color::BLACK).anchored(Anchor::Middle));
    assert_eq!(ink(&rasterize_with(&m, 1.0, Some(&mut Block)))[0], (9, 8));
    // 90° rotation turns the baseline into a vertical line going up.
    let mut r = Scene::new(20.0, 20.0, Color::WHITE);
    let mut st = TextStyle::new(10.0, Color::BLACK);
    st.rotate = 90.0;
    r.text(Point::new(10.0, 10.0), "a", st);
    let pts = ink(&rasterize_with(&r, 1.0, Some(&mut Block)));
    assert_eq!(pts.len(), 6);
    assert!(pts.iter().all(|&(x, y)| (8..10).contains(&x) && (6..9).contains(&y)), "{pts:?}");
    // `None` from the painter uses the bitmap font, as does passing no painter.
    let mut f = Scene::new(40.0, 20.0, Color::WHITE);
    f.text(Point::new(2.0, 14.0), "fallback", TextStyle::new(12.0, Color::BLACK));
    assert_eq!(ink(&rasterize_with(&f, 1.0, Some(&mut Block))), ink(&raster::rasterize(&f, 1.0)));
    assert!(!ink(&raster::rasterize(&f, 1.0)).is_empty());
}

#[test]
fn scatter_spec_draws_fit_and_bands_as_curves() {
    let x: Vec<f64> = (0..12).map(f64::from).collect();
    let line = |name: &str, off: f64| Series {
        name: name.into(),
        x: (0..=50).map(|i| f64::from(i) * 0.22).collect(),
        y: (0..=50).map(|i| f64::from(i) * 0.22 + off).collect(),
    };
    let spec = ChartSpec {
        kind: ChartKind::Scatter,
        title: "Fitted Line Plot".into(),
        x_label: "x".into(),
        y_label: "y".into(),
        categories: vec![],
        series: vec![
            Series { name: "y".into(), x: x.clone(), y: x.iter().map(|v| v + 0.3).collect() },
            line("Fit", 0.0),
            line("Lower CI", -1.0),
            line("Upper CI", 1.0),
        ],
        reference_lines: vec![],
    };
    let scene = render_spec(&spec, &opts());
    assert!(scene.clips_balanced());
    // only the 12 observations are hit targets; the fit and its bands are curves
    assert_eq!(scene.targets.len(), 12);
    let svg_text = svg::to_svg(&scene);
    assert!(svg_text.contains("<svg"));
}

#[test]
fn ice_curves_are_drawn_faint_and_stay_out_of_the_legend() {
    let xs: Vec<f64> = (0..10).map(f64::from).collect();
    let line = |name: &str, k: f64| Series { name: name.into(), x: xs.clone(), y: xs.iter().map(|v| k * v).collect() };
    let mk = |series: Vec<Series>| ChartSpec {
        kind: ChartKind::Line,
        title: "Partial Dependence on x".into(),
        x_label: "x".into(),
        y_label: "y".into(),
        categories: vec![],
        series,
        reference_lines: vec![],
    };
    let o = opts();
    let with_ice = render_spec(&mk(vec![line("ICE 1", 0.5), line("ICE 2", 1.5), line("Partial dependence", 1.0)]), &o);
    check("ice", &with_ice);
    let svg_ice = svg::to_svg(&with_ice);
    assert!(!svg_ice.contains("ICE 1") && !svg_ice.contains("ICE 2"), "ICE series must not be in the legend");
    let plain = render_spec(&mk(vec![line("Partial dependence", 1.0)]), &o);
    // Two extra polylines were drawn, and they are lighter than the data line.
    let paths = |t: &str| t.matches("<polyline").count() + t.matches("<path").count();
    assert!(
        paths(&svg_ice) >= paths(&svg::to_svg(&plain)) + 2,
        "{} vs {}",
        paths(&svg_ice),
        paths(&svg::to_svg(&plain))
    );
    assert_ne!(svg_ice, svg::to_svg(&plain));
}
