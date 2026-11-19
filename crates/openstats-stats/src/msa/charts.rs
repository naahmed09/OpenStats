//! Six-panel chart data of a Gage R&R study.

use super::types::{Source, VarianceRow};
use crate::error::Result;
use crate::graphical::boxplot;
use crate::report::{ChartKind, ChartSpec, Series};
use crate::spc::constants::{constants, MAX_SIZE, MIN_SIZE};

/// Cells of a study arranged for plotting.
pub(crate) struct Panels<'a> {
    pub operators: &'a [String],
    /// `[operator][cell]` → replicate values.
    pub cells: Vec<Vec<Vec<f64>>>,
    /// `[operator][cell]` → x position of the cell's part.
    pub part_x: Vec<Vec<f64>>,
    /// Category labels of the part axis (empty when the parts are not shared).
    pub part_labels: Vec<String>,
    /// Whether every operator measures the same parts (adds the interaction panel).
    pub crossed: bool,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn range(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max) - v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn chart(
    kind: ChartKind,
    title: &str,
    x_label: &str,
    y_label: &str,
    categories: Vec<String>,
    series: Vec<Series>,
) -> ChartSpec {
    ChartSpec {
        kind,
        title: title.into(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        categories,
        series,
        reference_lines: Vec::new(),
    }
}

fn series(name: &str, x: Vec<f64>, y: Vec<f64>) -> Series {
    Series { name: name.into(), x, y }
}

pub(crate) fn panels(p: &Panels<'_>, comps: &[VarianceRow]) -> Result<Vec<ChartSpec>> {
    let mut out = Vec::new();

    // 1. Components of variation.
    let pick = |s: Source| comps.iter().find(|r| r.source == s);
    let order = [Source::TotalGageRr, Source::Repeatability, Source::Reproducibility, Source::PartToPart];
    let cats = vec!["Gage R&R".to_string(), "Repeat".into(), "Reprod".into(), "Part-to-Part".into()];
    let xs: Vec<f64> = (1..=4).map(f64::from).collect();
    let mut comp_series = vec![
        series(
            "% Contribution",
            xs.clone(),
            order.iter().map(|&s| pick(s).map_or(f64::NAN, |r| r.pct_contribution)).collect(),
        ),
        series(
            "% Study Var",
            xs.clone(),
            order.iter().map(|&s| pick(s).map_or(f64::NAN, |r| r.pct_study_var)).collect(),
        ),
    ];
    if comps.iter().any(|r| r.pct_tolerance.is_some()) {
        comp_series.push(series(
            "% Tolerance",
            xs,
            order.iter().map(|&s| pick(s).and_then(|r| r.pct_tolerance).unwrap_or(f64::NAN)).collect(),
        ));
    }
    out.push(chart(ChartKind::Bar, "Components of Variation", "Source", "Percent", cats, comp_series));

    // 2./3. Range and X-bar charts by operator.
    let r = p.cells.iter().flatten().map(Vec::len).next().unwrap_or(0);
    let k = (MIN_SIZE..=MAX_SIZE).contains(&r).then(|| constants(r)).transpose()?;
    let (mut seq, mut rng, mut rng_u, mut rng_c, mut rng_l) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut xb, mut xb_u, mut xb_c, mut xb_l) = (vec![], vec![], vec![], vec![]);
    let mut i = 0.0;
    for cells in &p.cells {
        let ranges: Vec<f64> = cells.iter().map(|c| range(c)).collect();
        let rbar = mean(&ranges);
        let means: Vec<f64> = cells.iter().map(|c| mean(c)).collect();
        let center = mean(&means);
        for (rg, m) in ranges.iter().zip(&means) {
            i += 1.0;
            seq.push(i);
            rng.push(*rg);
            rng_c.push(rbar);
            xb.push(*m);
            xb_c.push(center);
            if let Some(k) = &k {
                rng_u.push(k.big_d4 * rbar);
                rng_l.push(k.big_d3 * rbar);
                xb_u.push(center + k.a2 * rbar);
                xb_l.push(center - k.a2 * rbar);
            }
        }
    }
    let mut rs = vec![series("Range", seq.clone(), rng)];
    let mut xs_ = vec![series("Mean", seq.clone(), xb)];
    if k.is_some() {
        rs.push(series("UCL", seq.clone(), rng_u));
        rs.push(series("Center", seq.clone(), rng_c));
        rs.push(series("LCL", seq.clone(), rng_l));
        xs_.push(series("UCL", seq.clone(), xb_u));
        xs_.push(series("Center", seq.clone(), xb_c));
        xs_.push(series("LCL", seq, xb_l));
    } else {
        rs.push(series("Center", seq.clone(), rng_c));
        xs_.push(series("Center", seq, xb_c));
    }
    out.push(chart(
        ChartKind::Line,
        "R Chart by Operator",
        "Operator / part",
        "Sample Range",
        p.operators.to_vec(),
        rs,
    ));
    out.push(chart(
        ChartKind::Line,
        "Xbar Chart by Operator",
        "Operator / part",
        "Sample Mean",
        p.operators.to_vec(),
        xs_,
    ));

    // 4. Measurements by part.
    let (mut vx, mut vy) = (vec![], vec![]);
    let mut by_x: Vec<(f64, f64, usize)> = Vec::new();
    for (cells, xs) in p.cells.iter().zip(&p.part_x) {
        for (c, &x) in cells.iter().zip(xs) {
            for &v in c {
                vx.push(x);
                vy.push(v);
                match by_x.iter_mut().find(|e| e.0 == x) {
                    Some(e) => {
                        e.1 += v;
                        e.2 += 1;
                    }
                    None => by_x.push((x, v, 1)),
                }
            }
        }
    }
    by_x.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.push(chart(
        ChartKind::Scatter,
        "Measurements by Part",
        "Part",
        "Measurement",
        p.part_labels.clone(),
        vec![
            series("Values", vx, vy),
            series("Mean", by_x.iter().map(|e| e.0).collect(), by_x.iter().map(|e| e.1 / e.2 as f64).collect()),
        ],
    ));

    // 5. Measurements by operator (box plots).
    let mut bs = Vec::new();
    let (mut ox, mut oy) = (vec![], vec![]);
    for (j, cells) in p.cells.iter().enumerate() {
        let all: Vec<f64> = cells.iter().flatten().copied().collect();
        let b = boxplot(&all)?;
        let pos = (j + 1) as f64;
        bs.push(series(&p.operators[j], vec![pos; 5], vec![b.lower_whisker, b.q1, b.median, b.q3, b.upper_whisker]));
        for &o in b.low_outliers.iter().chain(&b.high_outliers) {
            ox.push(pos);
            oy.push(o);
        }
    }
    bs.push(series("outliers", ox, oy));
    out.push(chart(
        ChartKind::Boxplot,
        "Measurements by Operator",
        "Operator",
        "Measurement",
        p.operators.to_vec(),
        bs,
    ));

    // 6. Operator × part interaction (crossed designs).
    if p.crossed {
        let lines = p
            .cells
            .iter()
            .zip(&p.part_x)
            .enumerate()
            .map(|(j, (cells, xs))| series(&p.operators[j], xs.clone(), cells.iter().map(|c| mean(c)).collect()))
            .collect();
        out.push(chart(
            ChartKind::Line,
            "Operator * Part Interaction",
            "Part",
            "Average",
            p.part_labels.clone(),
            lines,
        ));
    }
    Ok(out)
}
