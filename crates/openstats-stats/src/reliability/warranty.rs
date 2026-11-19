//! Warranty-return forecasting from shipment and return counts.

use super::data::{LifeData, Obs};
use super::dist::Family;
use super::fit::{fit_distribution, LifeFit};
use crate::error::{invalid, Result};

/// Shipment and return counts. Shipment period `i` shipped `shipments[i]` units, and
/// `returns[i][l]` of them were returned during service period `l + 1` (the number of entries
/// is the number of periods observed so far).
#[derive(Debug, Clone)]
pub struct WarrantyInput {
    /// Units shipped in each shipment period, oldest first.
    pub shipments: Vec<f64>,
    /// Returns by service period for each shipment period.
    pub returns: Vec<Vec<f64>>,
    /// Units that will ship in each future period (may be empty).
    pub future_shipments: Vec<f64>,
}

/// Fit and forecast.
#[derive(Debug, Clone)]
pub struct WarrantyForecast {
    /// The fitted life distribution (time in periods).
    pub fit: LifeFit,
    /// Total returns observed so far.
    pub returned: f64,
    /// Units still in service.
    pub in_service: f64,
    /// Forecast periods 1, 2, … after the last observed period.
    pub periods: Vec<usize>,
    /// Expected returns in each forecast period.
    pub expected: Vec<f64>,
    /// Cumulative expected returns.
    pub cumulative: Vec<f64>,
}

/// Fit `family` to the return data (interval- and right-censored) and forecast `horizon` periods.
pub fn warranty_forecast(input: &WarrantyInput, family: Family, horizon: usize) -> Result<WarrantyForecast> {
    let k = input.shipments.len();
    if k == 0 || input.returns.len() != k {
        return invalid("give one row of returns for every shipment period");
    }
    if !family.is_log() {
        return invalid(
            "warranty analysis uses a distribution of positive times (Weibull, lognormal, exponential or log-logistic)",
        );
    }
    if horizon == 0 || horizon > 1000 {
        return invalid("the forecast horizon must be between 1 and 1000 periods");
    }
    let mut obs = Vec::new();
    let mut returned = 0.0;
    let mut surv = Vec::with_capacity(k);
    let mut ages = Vec::with_capacity(k);
    for i in 0..k {
        let s = input.shipments[i];
        if !(s >= 0.0) || s.fract() != 0.0 {
            return invalid(format!("shipment {}: units shipped must be a non-negative whole number", i + 1));
        }
        let mut r_tot = 0.0;
        for (l, &r) in input.returns[i].iter().enumerate() {
            if !(r >= 0.0) || r.fract() != 0.0 {
                return invalid(format!("shipment {}: returns must be non-negative whole numbers", i + 1));
            }
            if r > 0.0 {
                obs.push(Obs { lower: l as f64, upper: (l + 1) as f64, weight: r, row: i });
            }
            r_tot += r;
        }
        if r_tot > s {
            return invalid(format!("shipment {}: more units were returned than shipped", i + 1));
        }
        let age = input.returns[i].len();
        if s - r_tot > 0.0 && age > 0 {
            obs.push(Obs { lower: age as f64, upper: f64::INFINITY, weight: s - r_tot, row: i });
        }
        returned += r_tot;
        surv.push(s - r_tot);
        ages.push(age);
    }
    // First interval has lower bound 0 and is therefore a left-censored observation.
    let data = LifeData { obs, n_skipped: 0 };
    let fit = fit_distribution(&data, family)?;
    let in_service: f64 = surv.iter().sum();
    let mut expected = vec![0.0; horizon];
    for m in 1..=horizon {
        let mut e = 0.0;
        for i in 0..k {
            let a = (ages[i] + m - 1) as f64;
            let s_a = fit.survival(ages[i] as f64);
            if surv[i] > 0.0 && s_a > 0.0 {
                e += surv[i] * (fit.survival(a) - fit.survival(a + 1.0)) / s_a;
            }
        }
        for (j, &fs) in input.future_shipments.iter().enumerate() {
            // Shipped at the start of forecast period j + 1; age window within this period.
            if m > j {
                let age = (m - j - 1) as f64;
                e += fs * (fit.survival(age) - fit.survival(age + 1.0));
            }
        }
        expected[m - 1] = e;
    }
    let mut cum = 0.0;
    let cumulative = expected
        .iter()
        .map(|e| {
            cum += e;
            cum
        })
        .collect();
    Ok(WarrantyForecast { fit, returned, in_service, periods: (1..=horizon).collect(), expected, cumulative })
}
