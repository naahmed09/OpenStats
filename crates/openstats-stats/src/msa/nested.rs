//! Nested Gage R&R: each operator measures different parts (destructive testing).

use super::charts::{panels, Panels};
use super::types::*;
use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{ContinuousDistribution, FisherF};

/// A balanced nested measurement study: parts within operators, replicates within parts.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedStudy {
    operators: Vec<String>,
    parts_per_operator: usize,
    replicates: usize,
    /// Measurement at `(operator * parts + part) * replicates + replicate`.
    y: Vec<f64>,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

impl NestedStudy {
    /// Build from stacked columns. Part labels need only be unique within an
    /// operator. Every operator must have the same number of parts and every
    /// part the same number (≥ 2) of measurements.
    pub fn from_stacked(measurement: &[f64], part: &[String], operator: &[String]) -> Result<NestedStudy> {
        if measurement.len() != part.len() || measurement.len() != operator.len() {
            return invalid("measurement, part and operator columns differ in length");
        }
        if part.iter().chain(operator).any(String::is_empty) {
            return invalid("part and operator labels must not be missing");
        }
        if measurement.iter().any(|v| !v.is_finite()) {
            return invalid("Gage R&R needs complete data: remove or fill missing and infinite measurements");
        }
        let operators = levels_of(operator);
        if operators.len() < 2 {
            return Err(StatsError::InsufficientData {
                what: "nested Gage R&R (operators)",
                needed: 2,
                got: operators.len(),
            });
        }
        let mut per_op: Vec<Vec<(String, Vec<f64>)>> = vec![Vec::new(); operators.len()];
        for ((m, pa), op) in measurement.iter().zip(part).zip(operator) {
            let j = operators.iter().position(|x| x == op).unwrap_or(0);
            match per_op[j].iter_mut().find(|(l, _)| l == pa) {
                Some((_, v)) => v.push(*m),
                None => per_op[j].push((pa.clone(), vec![*m])),
            }
        }
        let p = per_op[0].len();
        let r = per_op[0].first().map_or(0, |(_, v)| v.len());
        if p < 2 || r < 2 || per_op.iter().any(|o| o.len() != p || o.iter().any(|(_, v)| v.len() != r)) {
            return invalid(
                "nested Gage R&R needs the same number (≥ 2) of parts per operator and of measurements per part (≥ 2)",
            );
        }
        let y = per_op.into_iter().flatten().flat_map(|(_, v)| v).collect();
        Ok(NestedStudy { operators, parts_per_operator: p, replicates: r, y })
    }

    fn cell(&self, o: usize, p: usize) -> &[f64] {
        let r = self.replicates;
        let s = (o * self.parts_per_operator + p) * r;
        &self.y[s..s + r]
    }

    /// Operator labels.
    pub fn operators(&self) -> &[String] {
        &self.operators
    }

    /// Gage R&R by the nested ANOVA method.
    pub fn anova(&self, opts: &GageOptions) -> Result<GageResult> {
        let tol = opts.validate()?;
        let (no, np, r) = (self.operators.len(), self.parts_per_operator, self.replicates);
        let (pf, rf) = (np as f64, r as f64);
        let gm = mean(&self.y);
        let op_means: Vec<f64> =
            (0..no).map(|o| mean(&(0..np).flat_map(|p| self.cell(o, p).iter().copied()).collect::<Vec<_>>())).collect();
        let ss_o = pf * rf * op_means.iter().map(|m| (m - gm).powi(2)).sum::<f64>();
        let mut ss_p = 0.0;
        for (o, om) in op_means.iter().enumerate() {
            for p in 0..np {
                ss_p += rf * (mean(self.cell(o, p)) - om).powi(2);
            }
        }
        let ss_t: f64 = self.y.iter().map(|v| (v - gm).powi(2)).sum();
        let ss_e = (ss_t - ss_o - ss_p).max(0.0);
        let (df_o, df_p, df_e) = (no - 1, no * (np - 1), no * np * (r - 1));
        let (ms_o, ms_p, ms_e) = (ss_o / df_o as f64, ss_p / df_p as f64, ss_e / df_e as f64);
        let ftest = |num: f64, den: f64, d1: usize, d2: usize| -> Result<(Option<f64>, Option<f64>)> {
            if den > 0.0 {
                let f = num / den;
                Ok((Some(f), Some(FisherF::new(d1 as f64, d2 as f64)?.sf(f))))
            } else {
                Ok((None, None))
            }
        };
        let (f_o, p_o) = ftest(ms_o, ms_p, df_o, df_p)?;
        let (f_p, p_p) = ftest(ms_p, ms_e, df_p, df_e)?;
        let row = |source: &str, df: usize, ss: f64, ms: Option<f64>, f: Option<f64>, p: Option<f64>| AnovaRow {
            source: source.into(),
            df,
            ss,
            ms,
            f,
            p,
        };
        let anova = vec![
            row("Operator", df_o, ss_o, Some(ms_o), f_o, p_o),
            row("Part (Operator)", df_p, ss_p, Some(ms_p), f_p, p_p),
            row("Repeatability", df_e, ss_e, Some(ms_e), None, None),
            row("Total", no * np * r - 1, ss_t, None, None, None),
        ];
        let mut negative = Vec::new();
        let mut clamp = |v: f64, s: Source| {
            if v < 0.0 {
                negative.push(s);
                0.0
            } else {
                v
            }
        };
        let est = Estimates {
            repeat: ms_e,
            operator: clamp((ms_o - ms_p) / (pf * rf), Source::Reproducibility),
            op_part: None,
            part: clamp((ms_p - ms_e) / rf, Source::PartToPart),
        };
        let (comps, ndc) = components(&est, false, opts, tol);
        let cells: Vec<Vec<Vec<f64>>> = (0..no).map(|o| (0..np).map(|p| self.cell(o, p).to_vec()).collect()).collect();
        let part_x: Vec<Vec<f64>> = (0..no).map(|o| (0..np).map(|p| (o * np + p + 1) as f64).collect()).collect();
        let charts = panels(
            &Panels { operators: &self.operators, cells, part_x, part_labels: Vec::new(), crossed: false },
            &comps,
        )?;
        Ok(GageResult {
            method: GageMethod::NestedAnova,
            parts: np,
            operators: no,
            replicates: r,
            anova,
            anova_with_interaction: None,
            interaction_p: None,
            interaction_dropped: false,
            components: comps,
            ndc,
            study_sigma: opts.study_sigma,
            tolerance: tol,
            negative_set_to_zero: negative,
            operator_fixed: false,
            xbar_r: None,
            charts,
        })
    }
}
