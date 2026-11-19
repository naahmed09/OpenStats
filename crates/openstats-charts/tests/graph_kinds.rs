//! The chart kinds of the Graph menu: every kind renders, and the PDF writer produces a
//! structurally valid file.

use openstats_charts::plots::{render_spec, PlotOptions};
use openstats_charts::render::raster::render_png;
use openstats_charts::render::{pdf, svg};
use openstats_stats::report::{ChartKind, ChartSpec, Series};

fn s(name: &str, x: Vec<f64>, y: Vec<f64>) -> Series {
    Series { name: name.into(), x, y }
}

fn spec(kind: ChartKind, cats: &[&str], series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind,
        title: "T (1)".into(),
        x_label: "X".into(),
        y_label: "Y".into(),
        categories: cats.iter().map(|c| (*c).to_string()).collect(),
        series,
        reference_lines: vec![],
    }
}

fn seq(n: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..n).map(|i| f(i as f64)).collect()
}

fn all_specs() -> Vec<(&'static str, ChartSpec)> {
    let x = seq(30, |i| i);
    let y = seq(30, |i| (i * 0.4).sin() * 5.0 + i * 0.3);
    let z = seq(30, |i| (i * 0.9).cos() + 2.0);
    let grid: Vec<Series> =
        (0..6).map(|j| s(&format!("{j}"), seq(8, |i| i), seq(8, |i| i * (j as f64 + 1.0) % 7.0))).collect();
    let mut m = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                m.push(s(&format!("{i}|{j}"), seq(20, |t| t + j as f64), seq(20, |t| (t * (i as f64 + 1.0)).sin())));
            }
        }
    }
    vec![
        ("pie", spec(ChartKind::Pie, &["a", "b", "c"], vec![s("n", vec![], vec![3.0, 5.0, 2.0])])),
        ("ecdf", spec(ChartKind::Ecdf, &[], vec![s("g1", y.clone(), vec![]), s("g2", z.clone(), vec![])])),
        ("ivp", spec(ChartKind::IndividualValue, &[], vec![s("g1", vec![], y.clone()), s("g2", vec![], z.clone())])),
        ("area", spec(ChartKind::Area, &[], vec![s("a", x.clone(), y.clone()), s("b", x.clone(), z.clone())])),
        ("bubble", spec(ChartKind::Bubble, &[], vec![s("p", x.clone(), y.clone()), s("Size", vec![], z.clone())])),
        ("heat", spec(ChartKind::Heatmap, &[], grid)),
        ("dot1", spec(ChartKind::Dotplot, &[], vec![s("a", y.clone(), vec![])])),
        ("dot2", spec(ChartKind::Dotplot, &[], vec![s("a", y.clone(), vec![]), s("b", z.clone(), vec![])])),
        ("matrix", spec(ChartKind::Matrix, &["u", "v", "w"], m)),
        ("marginal", spec(ChartKind::Marginal, &[], vec![s("p", x.clone(), y.clone())])),
        (
            "stacked",
            spec(
                ChartKind::StackedBar,
                &["a", "b"],
                vec![s("s1", vec![], vec![1.0, 2.0]), s("s2", vec![], vec![3.0, 1.0])],
            ),
        ),
        ("prob-w", spec(ChartKind::DataProbability, &[], vec![s("Weibull", seq(30, |i| i + 1.0), vec![])])),
        ("prob-n", spec(ChartKind::DataProbability, &[], vec![s("Normal", y.clone(), vec![])])),
        ("3d", spec(ChartKind::Scatter3D, &[], vec![s("p", x.clone(), y.clone()), s("Z", vec![], z.clone())])),
    ]
}

#[test]
fn every_new_kind_renders_sane_scenes() {
    let o = PlotOptions::default();
    for (name, sp) in all_specs() {
        let sc = render_spec(&sp, &o);
        assert!(sc.clips_balanced(), "{name}: clips");
        assert!(sc.primitive_count() > 5, "{name}: almost empty ({})", sc.primitive_count());
        let t = svg::to_svg(&sc);
        assert!(t.starts_with("<svg") && t.trim_end().ends_with("</svg>"), "{name}");
        assert!(!t.contains("NaN") && !t.contains("inf"), "{name}: non-finite number in SVG");
        let p = render_png(&sc, 1.0);
        assert_eq!(&p[..8], b"\x89PNG\r\n\x1a\n", "{name}");
        // Deterministic.
        assert_eq!(t, svg::to_svg(&render_spec(&sp, &o)), "{name}");
    }
}

#[test]
fn degenerate_inputs_give_notice_scenes() {
    let o = PlotOptions::default();
    for kind in [
        ChartKind::Pie,
        ChartKind::Ecdf,
        ChartKind::IndividualValue,
        ChartKind::Area,
        ChartKind::Bubble,
        ChartKind::Heatmap,
        ChartKind::Dotplot,
        ChartKind::Matrix,
        ChartKind::Marginal,
        ChartKind::StackedBar,
        ChartKind::DataProbability,
        ChartKind::Scatter3D,
    ] {
        let sc = render_spec(&spec(kind, &[], vec![]), &o);
        assert!(sc.clips_balanced(), "{kind:?}");
        assert!(svg::to_svg(&sc).contains("<svg"), "{kind:?}");
    }
    // Negative pie values are refused with a message, not a panic.
    let sc = render_spec(&spec(ChartKind::Pie, &["a", "b"], vec![s("n", vec![], vec![-1.0, 2.0])]), &o);
    assert!(svg::to_svg(&sc).contains("non-negative"));
}

#[test]
fn bubble_area_scales_with_size_and_dots_are_stacked() {
    let o = PlotOptions::default();
    let sp =
        spec(ChartKind::Bubble, &[], vec![s("p", vec![1.0, 2.0], vec![1.0, 2.0]), s("Size", vec![], vec![1.0, 4.0])]);
    let sc = render_spec(&sp, &o);
    let mut radii: Vec<f64> = sc
        .items
        .iter()
        .filter_map(|i| match i.shape {
            openstats_charts::core::scene::Shape::Circle { r, .. } if r > 2.5 && i.fill.is_some_and(|f| f.a < 255) => {
                Some(r)
            }
            _ => None,
        })
        .collect();
    radii.sort_by(f64::total_cmp);
    assert_eq!(radii.len(), 2, "{radii:?}");
    // r = 3 + 22 sqrt(size/max): sizes 1 and 4 -> 3 + 11 = 14 and 25.
    assert!((radii[0] - 14.0).abs() < 1e-9 && (radii[1] - 25.0).abs() < 1e-9, "{radii:?}");
    // Dotplot: ten equal values stack into ten dots in one column.
    let dp = spec(ChartKind::Dotplot, &[], vec![s("a", vec![5.0; 10], vec![])]);
    let sc = render_spec(&dp, &o);
    let dots: Vec<(f64, f64)> = sc
        .items
        .iter()
        .filter_map(|i| match i.shape {
            openstats_charts::core::scene::Shape::Circle { c, .. } => Some((c.x, c.y)),
            _ => None,
        })
        .collect();
    assert_eq!(dots.len(), 10);
    assert!(dots.iter().all(|d| (d.0 - dots[0].0).abs() < 1e-9));
    let mut ys: Vec<f64> = dots.iter().map(|d| d.1).collect();
    ys.sort_by(f64::total_cmp);
    assert!(ys.windows(2).all(|w| w[1] - w[0] > 1.0));
}

// ------------------------------------------------------------------ PDF

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

fn check_pdf(b: &[u8]) {
    assert!(b.starts_with(b"%PDF-1.4\n"));
    assert!(b.ends_with(b"%%EOF\n"));
    // startxref points at the xref table.
    let sx = find(b, b"startxref\n", 0).unwrap();
    let tail = std::str::from_utf8(&b[sx + 10..]).unwrap();
    let off: usize = tail.lines().next().unwrap().trim().parse().unwrap();
    assert_eq!(&b[off..off + 4], b"xref");
    // Every xref entry points at "<id> 0 obj".
    let xr = std::str::from_utf8(&b[off..]).unwrap();
    let mut lines = xr.lines();
    lines.next();
    let hdr: Vec<usize> = lines.next().unwrap().split(' ').map(|x| x.parse().unwrap()).collect();
    assert_eq!(hdr[0], 0);
    lines.next(); // free entry
    for id in 1..hdr[1] {
        let e = lines.next().unwrap();
        let o: usize = e[..10].parse().unwrap();
        let want = format!("{id} 0 obj");
        assert_eq!(&b[o..o + want.len()], want.as_bytes(), "object {id}");
    }
    // The stream length is right.
    let st = find(b, b"/Length ", 0).unwrap();
    let rest = std::str::from_utf8(&b[st + 8..st + 24]).unwrap();
    let len: usize = rest.split(' ').next().unwrap().parse().unwrap();
    let s0 = find(b, b"stream\n", st).unwrap() + 7;
    assert_eq!(&b[s0 + len..s0 + len + 10], b"\nendstream");
    // q / Q are balanced inside the content stream.
    let content = std::str::from_utf8(&b[s0..s0 + len]).unwrap();
    let q = content.lines().filter(|l| *l == "q").count();
    let cap_q = content.lines().filter(|l| *l == "Q").count();
    assert_eq!(q, cap_q, "unbalanced graphics state");
    assert!(!content.contains("NaN") && !content.contains("inf"));
}

#[test]
fn pdf_is_structurally_valid_for_every_kind() {
    let o = PlotOptions::default();
    for (name, sp) in all_specs() {
        let sc = render_spec(&sp, &o);
        let b = pdf::to_pdf(&sc);
        check_pdf(&b);
        assert!(find(&b, b"/MediaBox [0 0 640 400]", 0).is_some(), "{name}");
    }
}

#[test]
fn pdf_escapes_text_and_declares_alpha() {
    let o = PlotOptions::default();
    let mut sp =
        spec(ChartKind::Bubble, &[], vec![s("p", vec![1.0, 2.0], vec![1.0, 2.0]), s("Size", vec![], vec![1.0, 2.0])]);
    sp.title = "Title (with) back\\slash \u{00e9}".into();
    let b = pdf::to_pdf(&render_spec(&sp, &o));
    check_pdf(&b);
    assert!(find(&b, b"(Title \\(with\\) back\\\\slash \\351)", 0).is_some());
    assert!(find(&b, b"/ExtGState", 0).is_some() && find(&b, b"/ca 0.451", 0).is_some());
    // Bold titles use the bold font.
    assert!(find(&b, b"/F2 ", 0).is_some() && find(&b, b"Helvetica-Bold", 0).is_some());
}

/// With `OPENSTATS_PDF_DIR` set, the PDFs are also written there for external validators.
#[test]
fn write_pdfs_for_external_validation() {
    let Ok(dir) = std::env::var("OPENSTATS_PDF_DIR") else { return };
    let o = PlotOptions::default();
    for (name, sp) in all_specs() {
        std::fs::write(format!("{dir}/{name}.pdf"), pdf::to_pdf(&render_spec(&sp, &o))).unwrap();
    }
}
