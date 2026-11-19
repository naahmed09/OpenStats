//! Weighted least squares: `min Σ wᵢ (yᵢ − xᵢβ)²`.

use super::*;

/// Fit a design matrix by weighted least squares. The weights must be
/// positive and finite. Sums of squares, R² and the F test are the weighted
/// ones (about the weighted mean when the model has a constant); the
/// residuals and fits are on the original response scale.
pub fn fit_weighted(dm: &DesignMatrix, weights: &[f64], opts: &RegressionOptions) -> Result<RegressionFit> {
    let n = dm.x.rows();
    if weights.len() != n {
        return invalid(format!("{} weights for {} observations", weights.len(), n));
    }
    if weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
        return invalid("weights must be positive and finite");
    }
    let y_raw = dm.y.as_ref().ok_or_else(|| StatsError::InvalidArgument("the design matrix has no response".into()))?;
    let sw: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut dm2 = dm.clone();
    for i in 0..n {
        for j in 0..dm.x.cols() {
            dm2.x[(i, j)] *= sw[i];
        }
    }
    dm2.y = Some(y_raw.iter().zip(&sw).map(|(y, s)| y * s).collect());
    dm2.offset = dm.offset.iter().zip(&sw).map(|(o, s)| o * s).collect();
    let had_const = dm.intercept;
    dm2.intercept = false;
    let mut fit = fit_design(&dm2, opts)?;

    let y: Vec<f64> = y_raw.iter().zip(&dm.offset).map(|(a, o)| a - o).collect();
    let p = fit.kept.len();
    let has_const = had_const && fit.kept.first() == Some(&0);
    fit.intercept = has_const;
    // Residuals and fits on the original scale.
    for i in 0..n {
        let raw = fit.diagnostics.residuals[i] / sw[i];
        fit.diagnostics.fitted[i] = y_raw[i] - raw;
        fit.diagnostics.residuals[i] = raw;
    }
    if has_const {
        let sum_w: f64 = weights.iter().sum();
        let mean = y.iter().zip(weights).map(|(a, w)| a * w).sum::<f64>() / sum_w;
        let sst: f64 = y.iter().zip(weights).map(|(a, w)| w * (a - mean) * (a - mean)).sum();
        let df_total = n - 1;
        let df_model = p - 1;
        let ssr = (sst - fit.sse).max(0.0);
        fit.sst = sst;
        fit.ssr = ssr;
        fit.df_model = df_model;
        fit.r2 = if sst > 0.0 { 1.0 - fit.sse / sst } else { f64::NAN };
        fit.r2_adj = if sst > 0.0 && df_total > 0 { 1.0 - fit.mse / (sst / df_total as f64) } else { f64::NAN };
        fit.r2_pred = if sst > 0.0 { 1.0 - fit.press / sst } else { f64::NAN };
        let (f, fp) = if df_model > 0 && fit.mse > 0.0 {
            let f = (ssr / df_model as f64) / fit.mse;
            (Some(f), Some(FisherF::new(df_model as f64, fit.df_error as f64)?.sf(f)))
        } else {
            (None, None)
        };
        fit.f = f;
        fit.f_p = fp;
        fit.anova.retain(|r| r.source != "Regression" && r.source != "Total");
        if df_model > 0 {
            fit.anova.insert(
                0,
                AnovaRow {
                    source: "Regression".into(),
                    df: df_model,
                    seq_ss: None,
                    adj_ss: ssr,
                    adj_ms: Some(ssr / df_model as f64),
                    f,
                    p_value: fp,
                },
            );
        }
        fit.anova.push(AnovaRow {
            source: "Total".into(),
            df: df_total,
            seq_ss: None,
            adj_ss: sst,
            adj_ms: None,
            f: None,
            p_value: None,
        });
        // Variance inflation factors are not defined for the weighted design here.
        fit.coefficients.iter_mut().for_each(|c| c.vif = None);
    }
    Ok(fit)
}

/// Parse `formula` and fit it by weighted least squares.
pub fn regress_weighted(
    formula: &str,
    data: &Data,
    design: &DesignOptions,
    weights: &[f64],
    opts: &RegressionOptions,
) -> Result<RegressionFit> {
    let (_spec, dm) = build_design(formula, data, design)?;
    if weights.len() != data.n_rows() {
        return invalid("one weight per worksheet row is required");
    }
    let w: Vec<f64> = dm.rows_used.iter().map(|&i| weights[i]).collect();
    fit_weighted(&dm, &w, opts)
}
