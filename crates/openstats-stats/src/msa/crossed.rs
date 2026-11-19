//! Crossed Gage R&R: every operator measures every part, with replicates.

use super::charts::{panels, Panels};
use super::types::*;
use crate::error::{invalid, Result, StatsError};
use crate::spc::constants::{d2, d2_star};
use openstats_math::dist::{ContinuousDistribution, FisherF};

/// A balanced crossed measurement study.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossedStudy {
    parts: Vec<String>,
    operators: Vec<String>,
    replicates: usize,
    /// Measurement at `(part * operators + operator) * replicates + replicate`.
    y: Vec<f64>,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

impl CrossedStudy {
    /// Build from stacked worksheet columns. Every part × operator cell must
    /// have the same number (≥ 2) of finite measurements; replicates keep their
    /// worksheet order.
    pub fn from_stacked(measurement: &[f64], part: &[String], operator: &[String]) -> Result<CrossedStudy> {
        if measurement.len() != part.len() || measurement.len() != operator.len() {
            return invalid("measurement, part and operator columns differ in length");
        }
        if part.iter().chain(operator).any(String::is_empty) {
            return invalid("part and operator labels must not be missing");
        }
        if measurement.iter().any(|v| !v.is_finite()) {
            return invalid("Gage R&R needs complete data: remove or fill missing and infinite measurements");
        }
        let parts = levels_of(part);
        let operators = levels_of(operator);
        let (p, o) = (parts.len(), operators.len());
        if p < 2 || o < 2 {
            return Err(StatsError::InsufficientData {
                what: "Gage R&R (parts and operators)",
                needed: 2,
                got: p.min(o),
            });
        }
        let mut cells: Vec<Vec<f64>> = vec![Vec::new(); p * o];
        for ((m, pa), op) in measurement.iter().zip(part).zip(operator) {
            let i = parts.iter().position(|x| x == pa).unwrap_or(0);
            let j = operators.iter().position(|x| x == op).unwrap_or(0);
            cells[i * o + j].push(*m);
        }
        let r = cells[0].len();
        if r < 2 || cells.iter().any(|c| c.len() != r) {
            return invalid("Gage R&R needs a balanced design: every part × operator cell must have the same number (≥ 2) of measurements");
        }
        Ok(CrossedStudy { parts, operators, replicates: r, y: cells.into_iter().flatten().collect() })
    }

    /// Part labels.
    pub fn parts(&self) -> &[String] {
        &self.parts
    }

    /// Operator labels.
    pub fn operators(&self) -> &[String] {
        &self.operators
    }

    /// Replicates per cell.
    pub fn replicates(&self) -> usize {
        self.replicates
    }

    fn cell(&self, p: usize, o: usize) -> &[f64] {
        let r = self.replicates;
        let s = (p * self.operators.len() + o) * r;
        &self.y[s..s + r]
    }

    fn panels(&self, comps: &[VarianceRow]) -> Result<Vec<crate::report::ChartSpec>> {
        let (np, no) = (self.parts.len(), self.operators.len());
        let cells = (0..no).map(|o| (0..np).map(|p| self.cell(p, o).to_vec()).collect()).collect();
        let part_x = (0..no).map(|_| (1..=np).map(|i| i as f64).collect()).collect();
        panels(
            &Panels { operators: &self.operators, cells, part_x, part_labels: self.parts.clone(), crossed: true },
            comps,
        )
    }

    /// Gage R&R by the ANOVA method (random Part, Operator and Part × Operator).
    pub fn anova(&self, opts: &GageOptions) -> Result<GageResult> {
        let tol = opts.validate()?;
        let (np, no, r) = (self.parts.len(), self.operators.len(), self.replicates);
        let (pf, of, rf) = (np as f64, no as f64, r as f64);
        let gm = mean(&self.y);
        let part_means: Vec<f64> =
            (0..np).map(|p| mean(&(0..no).flat_map(|o| self.cell(p, o).iter().copied()).collect::<Vec<_>>())).collect();
        let op_means: Vec<f64> =
            (0..no).map(|o| mean(&(0..np).flat_map(|p| self.cell(p, o).iter().copied()).collect::<Vec<_>>())).collect();
        let ss_p = of * rf * part_means.iter().map(|m| (m - gm).powi(2)).sum::<f64>();
        let ss_o = pf * rf * op_means.iter().map(|m| (m - gm).powi(2)).sum::<f64>();
        let mut ss_po = 0.0;
        for p in 0..np {
            for o in 0..no {
                let c = mean(self.cell(p, o));
                ss_po += rf * (c - part_means[p] - op_means[o] + gm).powi(2);
            }
        }
        let ss_t: f64 = self.y.iter().map(|v| (v - gm).powi(2)).sum();
        let ss_e = (ss_t - ss_p - ss_o - ss_po).max(0.0);
        let (df_p, df_o, df_po, df_e) = (np - 1, no - 1, (np - 1) * (no - 1), np * no * (r - 1));
        let (ms_p, ms_o, ms_po, ms_e) =
            (ss_p / df_p as f64, ss_o / df_o as f64, ss_po / df_po as f64, ss_e / df_e as f64);
        let ftest = |num: f64, den: f64, d1: usize, d2: usize| -> Result<(Option<f64>, Option<f64>)> {
            if den > 0.0 {
                let f = num / den;
                Ok((Some(f), Some(FisherF::new(d1 as f64, d2 as f64)?.sf(f))))
            } else {
                Ok((None, None))
            }
        };
        let (f_po, p_po) = ftest(ms_po, ms_e, df_po, df_e)?;
        let keep = match opts.interaction {
            Interaction::Keep => true,
            Interaction::Drop => false,
            Interaction::Auto => p_po.map_or(ms_po > 0.0, |p| p < opts.alpha_interaction),
        };
        let row = |source: &str, df: usize, ss: f64, ms: Option<f64>, f: Option<f64>, p: Option<f64>| AnovaRow {
            source: source.into(),
            df,
            ss,
            ms,
            f,
            p,
        };
        let (f_p, p_p) = ftest(ms_p, ms_po, df_p, df_po)?;
        let (f_o, p_o) = ftest(ms_o, ms_po, df_o, df_po)?;
        let full = vec![
            row("Part", df_p, ss_p, Some(ms_p), f_p, p_p),
            row("Operator", df_o, ss_o, Some(ms_o), f_o, p_o),
            row("Part * Operator", df_po, ss_po, Some(ms_po), f_po, p_po),
            row("Repeatability", df_e, ss_e, Some(ms_e), None, None),
            row("Total", np * no * r - 1, ss_t, None, None, None),
        ];
        let op_scale = if opts.operator_fixed { (of - 1.0) / of } else { 1.0 };
        let mut negative = Vec::new();
        let mut clamp = |v: f64, s: Source| {
            if v < 0.0 {
                negative.push(s);
                0.0
            } else {
                v
            }
        };
        let (anova, dropped_from, est) = if keep {
            let est = Estimates {
                repeat: ms_e,
                operator: clamp((ms_o - ms_po) / (pf * rf), Source::Operator) * op_scale,
                op_part: Some(clamp((ms_po - ms_e) / rf, Source::OperatorByPart)),
                part: clamp((ms_p - ms_po) / (of * rf), Source::PartToPart),
            };
            (full, None, est)
        } else {
            let df_pool = df_po + df_e;
            let ms_pool = (ss_po + ss_e) / df_pool as f64;
            let (f_p2, p_p2) = ftest(ms_p, ms_pool, df_p, df_pool)?;
            let (f_o2, p_o2) = ftest(ms_o, ms_pool, df_o, df_pool)?;
            let reduced = vec![
                row("Part", df_p, ss_p, Some(ms_p), f_p2, p_p2),
                row("Operator", df_o, ss_o, Some(ms_o), f_o2, p_o2),
                row("Repeatability", df_pool, ss_po + ss_e, Some(ms_pool), None, None),
                row("Total", np * no * r - 1, ss_t, None, None, None),
            ];
            let est = Estimates {
                repeat: ms_pool,
                operator: clamp((ms_o - ms_pool) / (pf * rf), Source::Operator) * op_scale,
                op_part: None,
                part: clamp((ms_p - ms_pool) / (of * rf), Source::PartToPart),
            };
            (reduced, Some(full), est)
        };
        let (comps, ndc) = components(&est, true, opts, tol);
        let charts = self.panels(&comps)?;
        Ok(GageResult {
            method: GageMethod::CrossedAnova,
            parts: np,
            operators: no,
            replicates: r,
            anova,
            anova_with_interaction: dropped_from,
            interaction_p: p_po,
            interaction_dropped: !keep,
            components: comps,
            ndc,
            study_sigma: opts.study_sigma,
            tolerance: tol,
            negative_set_to_zero: negative,
            operator_fixed: opts.operator_fixed,
            xbar_r: None,
            charts,
        })
    }

    /// Gage R&R by the X̄ and R method: repeatability from the average range of
    /// the cells (`R̄/d2`), operator variation from the range of the operator
    /// means (`d2*` for a single range), part variation from the range of the
    /// part means.
    pub fn xbar_r(&self, opts: &GageOptions) -> Result<GageResult> {
        let tol = opts.validate()?;
        let (np, no, r) = (self.parts.len(), self.operators.len(), self.replicates);
        let range = |v: &[f64]| {
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max) - v.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let ranges: Vec<f64> =
            (0..np).flat_map(|p| (0..no).map(move |o| (p, o))).map(|(p, o)| range(self.cell(p, o))).collect();
        let r_bar = mean(&ranges);
        let ev = r_bar / d2(r)?;
        let op_means: Vec<f64> =
            (0..no).map(|o| mean(&(0..np).flat_map(|p| self.cell(p, o).iter().copied()).collect::<Vec<_>>())).collect();
        let part_means: Vec<f64> =
            (0..np).map(|p| mean(&(0..no).flat_map(|o| self.cell(p, o).iter().copied()).collect::<Vec<_>>())).collect();
        let op_range = range(&op_means);
        let part_range = range(&part_means);
        let raw_av2 = (op_range / d2_star(no, 1.0)?).powi(2) - ev * ev / (np * r) as f64;
        let mut negative = Vec::new();
        let av = if raw_av2 < 0.0 {
            negative.push(Source::Reproducibility);
            0.0
        } else {
            raw_av2.sqrt()
        };
        let pv = part_range / d2_star(np, 1.0)?;
        let est = Estimates { repeat: ev * ev, operator: av * av, op_part: None, part: pv * pv };
        let (comps, ndc) = components(&est, false, opts, tol);
        let charts = self.panels(&comps)?;
        Ok(GageResult {
            method: GageMethod::CrossedXbarR,
            parts: np,
            operators: no,
            replicates: r,
            anova: Vec::new(),
            anova_with_interaction: None,
            interaction_p: None,
            interaction_dropped: false,
            components: comps,
            ndc,
            study_sigma: opts.study_sigma,
            tolerance: tol,
            negative_set_to_zero: negative,
            operator_fixed: false,
            xbar_r: Some(XbarRDetail {
                r_bar,
                operator_range: op_range,
                part_range,
                equipment_variation: ev,
                appraiser_variation: av,
                part_variation: pv,
            }),
            charts,
        })
    }
}
