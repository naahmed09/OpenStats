//! Tick generation: Wilkinson's extended algorithm (Talbot, Lin & Hanrahan 2010)
//! for linear axes, decade ticks for log axes and calendar ticks for dates.

/// One axis tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Tick {
    /// Position in data units.
    pub value: f64,
    /// Label (empty for unlabelled minor ticks).
    pub label: String,
    /// Major ticks get a label and a grid line.
    pub major: bool,
}

const Q: [f64; 6] = [1.0, 5.0, 2.0, 2.5, 4.0, 3.0];
const W: [f64; 4] = [0.25, 0.2, 0.5, 0.05];

fn fmod(a: f64, b: f64) -> f64 {
    a - (a / b).floor() * b
}

fn simplicity(i: usize, j: f64, lmin: f64, lmax: f64, lstep: f64) -> f64 {
    let n = Q.len() as f64;
    let eps = 1e-10;
    let r = fmod(lmin, lstep);
    let v = if (r < eps || lstep - r < eps) && lmin <= 0.0 && lmax >= 0.0 { 1.0 } else { 0.0 };
    1.0 - i as f64 / (n - 1.0) - j + v
}

fn simplicity_max(i: usize, j: f64) -> f64 {
    1.0 - i as f64 / (Q.len() as f64 - 1.0) - j + 1.0
}

fn coverage(dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let r = 0.1 * (dmax - dmin);
    1.0 - 0.5 * ((dmax - lmax).powi(2) + (dmin - lmin).powi(2)) / (r * r)
}

fn coverage_max(dmin: f64, dmax: f64, span: f64) -> f64 {
    let range = dmax - dmin;
    if span > range {
        let half = (span - range) / 2.0;
        let r = 0.1 * range;
        1.0 - 0.5 * (2.0 * half * half) / (r * r)
    } else {
        1.0
    }
}

fn density(k: f64, m: f64, dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let r = (k - 1.0) / (lmax - lmin);
    let rt = (m - 1.0) / (lmax.max(dmax) - dmin.min(lmin));
    2.0 - (r / rt).max(rt / r)
}

fn density_max(k: f64, m: f64) -> f64 {
    if k >= m {
        2.0 - (k - 1.0) / (m - 1.0)
    } else {
        1.0
    }
}

/// Decimals needed to print multiples of `step` exactly.
fn decimals_for(step: f64) -> usize {
    for d in 0..=12usize {
        let s = step * 10f64.powi(d as i32);
        if (s - s.round()).abs() < 1e-7 * s.abs().max(1.0) {
            return d;
        }
    }
    12
}

/// Format `v` for an axis whose tick spacing is `step`.
pub fn format_tick(v: f64, step: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if a >= 1e7 || (a < 1e-4 && a > 0.0) {
        return format!("{v:e}");
    }
    let d = decimals_for(step.abs());
    let s = format!("{v:.d$}");
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

/// Wilkinson-extended ticks inside `[lo, hi]`, aiming at about `target` of them.
pub fn extended_ticks(lo: f64, hi: f64, target: usize) -> Vec<Tick> {
    if !(lo.is_finite() && hi.is_finite()) {
        return Vec::new();
    }
    let (dmin, dmax) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    if dmax - dmin < 1e-12 * dmax.abs().max(1e-300) {
        return vec![Tick { value: dmin, label: format_tick(dmin, dmin.abs().max(1e-3) / 10.0), major: true }];
    }
    let m = target.max(2) as f64;
    let mut best: Option<(f64, f64, f64)> = None; // (score, lmin, step) with k stored below
    let mut best_k = 0.0;
    let mut best_score = -2.0;
    let mut j = 1.0;
    'outer: while j < 20.0 {
        for (qi, &q) in Q.iter().enumerate() {
            let sm = simplicity_max(qi, j);
            if W[0] * sm + W[1] + W[2] + W[3] < best_score {
                break 'outer;
            }
            let mut k = 2.0;
            while k < 30.0 {
                let dm = density_max(k, m);
                if W[0] * sm + W[1] + W[2] * dm + W[3] < best_score {
                    break;
                }
                let delta = (dmax - dmin) / (k + 1.0) / j / q;
                let mut z = delta.log10().ceil();
                while z < 400.0 {
                    let step = j * q * 10f64.powf(z);
                    let cm = coverage_max(dmin, dmax, step * (k - 1.0));
                    if W[0] * sm + W[1] * cm + W[2] * dm + W[3] < best_score {
                        break;
                    }
                    let min_start = (dmax / step).floor() * j - (k - 1.0) * j;
                    let max_start = (dmin / step).ceil() * j;
                    if min_start > max_start {
                        z += 1.0;
                        continue;
                    }
                    let mut start = min_start;
                    while start <= max_start {
                        let lmin = start * (step / j);
                        let lmax = lmin + step * (k - 1.0);
                        if lmin >= dmin - 1e-12 * step && lmax <= dmax + 1e-12 * step {
                            let s = simplicity(qi, j, lmin, lmax, step);
                            let c = coverage(dmin, dmax, lmin, lmax);
                            let g = density(k, m, dmin, dmax, lmin, lmax);
                            let score = W[0] * s + W[1] * c + W[2] * g + W[3];
                            if score > best_score {
                                best_score = score;
                                best = Some((score, lmin, step));
                                best_k = k;
                            }
                        }
                        start += 1.0;
                    }
                    z += 1.0;
                }
                k += 1.0;
            }
        }
        j += 1.0;
    }
    match best {
        Some((_, lmin, step)) => (0..best_k as usize)
            .map(|i| {
                let v = lmin + step * i as f64;
                let v = if v.abs() < step * 1e-9 { 0.0 } else { v };
                Tick { value: v, label: format_tick(v, step), major: true }
            })
            .collect(),
        None => vec![
            Tick { value: dmin, label: format_tick(dmin, dmax - dmin), major: true },
            Tick { value: dmax, label: format_tick(dmax, dmax - dmin), major: true },
        ],
    }
}

fn pow10_label(e: i32) -> String {
    if (-3..=6).contains(&e) {
        let v = 10f64.powi(e);
        if e >= 0 {
            format!("{v:.0}")
        } else {
            format!("{v:.*}", (-e) as usize)
        }
    } else {
        format!("1e{e}")
    }
}

/// Decade ticks (and 2, 5 minor ticks when the span is short) on `[lo, hi]`, both > 0.
pub fn log_ticks(lo: f64, hi: f64) -> Vec<Tick> {
    if !(lo > 0.0 && hi > lo && hi.is_finite()) {
        return Vec::new();
    }
    let e0 = lo.log10().floor() as i32;
    let e1 = hi.log10().ceil() as i32;
    let decades = (e1 - e0).max(1);
    let every = ((decades as f64) / 8.0).ceil() as i32;
    let mut out = Vec::new();
    for e in e0..=e1 {
        let base = 10f64.powi(e);
        if (e - e0) % every == 0 && base >= lo * (1.0 - 1e-12) && base <= hi * (1.0 + 1e-12) {
            out.push(Tick { value: base, label: pow10_label(e), major: true });
        }
        if decades <= 3 {
            for m in [2.0, 5.0] {
                let v = base * m;
                if v >= lo && v <= hi {
                    out.push(Tick { value: v, label: String::new(), major: false });
                }
            }
        }
    }
    out.sort_by(|a, b| a.value.total_cmp(&b.value));
    if out.iter().filter(|t| t.major).count() < 2 {
        // Narrow range inside one decade: label the 1-2-5 values.
        for t in out.iter_mut() {
            if !t.major {
                t.major = true;
                t.label = format_tick(t.value, t.value / 2.0);
            }
        }
        if out.len() < 2 {
            out = extended_ticks(lo, hi, 4);
        }
    }
    out
}

/// Probability ticks (fractions in (0, 1)) inside `[lo, hi]`, labelled in percent.
pub fn probability_ticks(lo: f64, hi: f64) -> Vec<Tick> {
    const P: [f64; 21] = [
        0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.0001, 0.9999, 0.00001,
        0.99999, 0.02, 0.98,
    ];
    let mut v: Vec<f64> = P.iter().copied().filter(|&p| p >= lo * (1.0 - 1e-9) && p <= hi * (1.0 + 1e-9)).collect();
    v.retain(|&p| ![0.02, 0.98].contains(&p) || (hi - lo) < 0.2);
    v.sort_by(f64::total_cmp);
    v.into_iter()
        .map(|p| {
            let pct = p * 100.0;
            let label = if (1.0..=99.0).contains(&pct) {
                format!("{pct:.0}")
            } else {
                format!("{}", (pct * 1e6).round() / 1e6)
            };
            Tick { value: p, label, major: true }
        })
        .collect()
}

// ---------------------------------------------------------------- calendar

/// Civil date of an Excel serial day number (valid from 1900-03-01).
pub fn civil_from_serial(serial: f64) -> (i64, u32, u32) {
    let z = serial.floor() as i64 - 25569 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Excel serial day number of a civil date.
pub fn serial_from_civil(y: i64, m: u32, d: u32) -> f64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + doe - 719_468 + 25569) as f64
}

#[derive(Clone, Copy)]
enum Unit {
    Minutes(f64),
    Hours(f64),
    Days(f64),
    Months(i64),
    Years(i64),
}

fn unit_days(u: Unit) -> f64 {
    match u {
        Unit::Minutes(n) => n / 1440.0,
        Unit::Hours(n) => n / 24.0,
        Unit::Days(n) => n,
        Unit::Months(n) => n as f64 * 30.4375,
        Unit::Years(n) => n as f64 * 365.25,
    }
}

/// Calendar ticks on a date axis in Excel serial days.
pub fn time_ticks(lo: f64, hi: f64, target: usize) -> Vec<Tick> {
    if !(lo.is_finite() && hi > lo) {
        return Vec::new();
    }
    let units = [
        Unit::Minutes(1.0),
        Unit::Minutes(5.0),
        Unit::Minutes(15.0),
        Unit::Minutes(30.0),
        Unit::Hours(1.0),
        Unit::Hours(3.0),
        Unit::Hours(6.0),
        Unit::Hours(12.0),
        Unit::Days(1.0),
        Unit::Days(2.0),
        Unit::Days(7.0),
        Unit::Days(14.0),
        Unit::Months(1),
        Unit::Months(3),
        Unit::Months(6),
        Unit::Years(1),
        Unit::Years(2),
        Unit::Years(5),
        Unit::Years(10),
        Unit::Years(25),
        Unit::Years(50),
        Unit::Years(100),
    ];
    let span = hi - lo;
    let want = target.max(2) as f64;
    let unit = *units
        .iter()
        .min_by(|a, b| {
            let ca = (span / unit_days(**a)).max(1e-9);
            let cb = (span / unit_days(**b)).max(1e-9);
            (ca / want).ln().abs().total_cmp(&(cb / want).ln().abs())
        })
        .unwrap_or(&units[0]);
    let mut out = Vec::new();
    match unit {
        Unit::Minutes(_) | Unit::Hours(_) | Unit::Days(_) => {
            let step = unit_days(unit);
            let mut t = (lo / step).ceil() * step;
            while t <= hi + 1e-9 && out.len() < 200 {
                let (y, mo, d) = civil_from_serial(t);
                let frac = t - t.floor();
                let mins = (frac * 1440.0).round() as i64;
                let label = match unit {
                    Unit::Days(_) => format!("{y:04}-{mo:02}-{d:02}"),
                    _ => format!("{mo:02}-{d:02} {:02}:{:02}", mins / 60 % 24, mins % 60),
                };
                out.push(Tick { value: t, label, major: true });
                t += step;
            }
        }
        Unit::Months(n) => {
            let (y0, m0, _) = civil_from_serial(lo);
            let mut idx = y0 * 12 + i64::from(m0) - 1;
            idx = (idx + n - 1).div_euclid(n) * n; // aligned and not before the first month
            loop {
                let (y, m) = (idx.div_euclid(12), (idx.rem_euclid(12) + 1) as u32);
                let t = serial_from_civil(y, m, 1);
                if t > hi || out.len() >= 200 {
                    break;
                }
                if t >= lo {
                    out.push(Tick { value: t, label: format!("{y:04}-{m:02}"), major: true });
                }
                idx += n;
            }
        }
        Unit::Years(n) => {
            let (y0, _, _) = civil_from_serial(lo);
            let mut y = (y0 + n - 1).div_euclid(n) * n;
            loop {
                let t = serial_from_civil(y, 1, 1);
                if t > hi || out.len() >= 200 {
                    break;
                }
                if t >= lo {
                    out.push(Tick { value: t, label: format!("{y:04}"), major: true });
                }
                y += n;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(t: &[Tick]) -> Vec<f64> {
        t.iter().map(|t| t.value).collect()
    }

    #[test]
    fn extended_ticks_are_nice() {
        assert_eq!(vals(&extended_ticks(0.0, 100.0, 5)), vec![0.0, 25.0, 50.0, 75.0, 100.0]);
        let t = extended_ticks(0.0, 1.0, 5);
        assert_eq!(vals(&t), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(t[1].label, "0.25");
        let t = extended_ticks(-3.2, 7.9, 6);
        assert!(t.len() >= 4 && t.len() <= 8);
        assert!(t.iter().all(|x| x.value >= -3.2 && x.value <= 7.9));
        let step = t[1].value - t[0].value;
        assert!([1.0, 2.0, 2.5, 5.0].iter().any(|m| (step - m).abs() < 1e-9), "{step}");
        let t = extended_ticks(1000.0, 1000.003, 5);
        assert!(t.len() >= 3);
        assert_eq!(extended_ticks(5.0, 5.0, 5).len(), 1);
        assert!(extended_ticks(f64::NAN, 1.0, 5).is_empty());
    }

    #[test]
    fn log_and_probability_ticks() {
        let t = log_ticks(0.5, 5000.0);
        let majors: Vec<f64> = t.iter().filter(|x| x.major).map(|x| x.value).collect();
        assert_eq!(majors, vec![1.0, 10.0, 100.0, 1000.0]);
        assert_eq!(t.iter().find(|x| x.value == 1000.0).unwrap().label, "1000");
        let t = log_ticks(1.5, 8.0);
        assert!(t.iter().filter(|x| x.major).count() >= 2);
        let p = probability_ticks(0.001, 0.999);
        assert_eq!(p.first().unwrap().label, "0.1");
        assert_eq!(p.last().unwrap().label, "99.9");
        assert!(p.iter().any(|t| t.label == "50"));
    }

    #[test]
    fn calendar() {
        assert_eq!(civil_from_serial(25569.0), (1970, 1, 1));
        assert_eq!(civil_from_serial(45292.0), (2024, 1, 1));
        assert_eq!(serial_from_civil(2024, 2, 29), 45351.0);
        for s in [36000.0, 40000.5, 45000.0, 50000.0] {
            let (y, m, d) = civil_from_serial(s);
            assert_eq!(serial_from_civil(y, m, d), s.floor());
        }
        let t = time_ticks(serial_from_civil(2020, 1, 1), serial_from_civil(2024, 6, 1), 5);
        assert!(t.iter().all(|x| x.label.len() == 4), "{t:?}");
        let t = time_ticks(serial_from_civil(2024, 1, 1), serial_from_civil(2024, 12, 31), 6);
        assert!(t.iter().all(|x| x.label.contains('-')));
        let t = time_ticks(45000.0, 45000.5, 5);
        assert!(t.len() >= 2 && t[0].label.contains(':'));
    }
}
