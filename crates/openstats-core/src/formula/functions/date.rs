//! Date/time functions.

use super::{def, Category::DateTime as Cat, FunctionRegistry};
use crate::column::ColumnData;
use crate::datetime::{self, add_months, days_in_month, DateTime, MICROS_PER_DAY};
use crate::error::{CoreError, Result};
use crate::formula::array::{at, at_i, common_len, dt_missing, Arr};
use crate::missing::DATETIME_MISSING;
use crate::ops::extract::{extract_date_part, DatePart};

fn dates(a: &Arr) -> Result<Vec<i64>> {
    a.to_datetimes("date")
}

fn out_dt(v: Vec<i64>, scalar: bool) -> Arr {
    Arr { data: ColumnData::DateTime(v), scalar }
}

fn part(a: &Arr, p: DatePart) -> Result<Arr> {
    let d = ColumnData::DateTime(dates(a)?);
    Ok(Arr { data: extract_date_part(&d, p)?, scalar: a.scalar })
}

/// Build a date from numeric parts; month overflow rolls into years and day
/// overflow into following months (spreadsheet semantics).
fn make_date(y: f64, m: f64, d: f64) -> i64 {
    if !(y.is_finite() && m.is_finite() && d.is_finite()) || y.fract() != 0.0 || m.fract() != 0.0 || d.fract() != 0.0 {
        return DATETIME_MISSING;
    }
    let total = y as i64 * 12 + (m as i64 - 1);
    let (yy, mm) = (total.div_euclid(12), (total.rem_euclid(12) + 1) as u32);
    (datetime::days_from_civil(yy, mm, 1) + d as i64 - 1) * MICROS_PER_DAY
}

fn now(ctx: &crate::formula::eval::EvalCtx) -> Result<i64> {
    ctx.now.ok_or_else(|| CoreError::invalid("the current time is not available on this platform"))
}

pub(super) fn register(r: &mut FunctionRegistry) {
    r.register(def("DATE", 3, Some(3), Cat, "DATE(year, month, day): construct a date", |a, _| {
        let refs = [&a[0], &a[1], &a[2]];
        let (n, scalar) = common_len(&refs, "DATE")?;
        let (y, m, d) = (a[0].to_nums("year")?, a[1].to_nums("month")?, a[2].to_nums("day")?);
        Ok(out_dt((0..n).map(|i| make_date(at(&y, i), at(&m, i), at(&d, i))).collect(), scalar))
    }));
    r.register(def(
        "TIME",
        3,
        Some(3),
        Cat,
        "TIME(hour, minute, second): time as a fraction of a day (add it to a date)",
        |a, _| {
            let refs = [&a[0], &a[1], &a[2]];
            let (n, scalar) = common_len(&refs, "TIME")?;
            let (h, m, s) = (a[0].to_nums("hour")?, a[1].to_nums("minute")?, a[2].to_nums("second")?);
            Ok(Arr::nums(
                (0..n).map(|i| (at(&h, i) * 3600.0 + at(&m, i) * 60.0 + at(&s, i)) / 86_400.0).collect(),
                scalar,
            ))
        },
    ));
    r.register(def("NOW", 0, Some(0), Cat, "NOW(): current date and time", |_, ctx| Ok(Arr::datetime(now(ctx)?))));
    r.register(def("TODAY", 0, Some(0), Cat, "TODAY(): current date", |_, ctx| {
        let t = now(ctx)?;
        Ok(Arr::datetime(t - t.rem_euclid(MICROS_PER_DAY)))
    }));
    macro_rules! part_fn {
        ($name:literal, $help:literal, $p:expr) => {
            r.register(def($name, 1, Some(1), Cat, $help, |a, _| part(&a[0], $p)));
        };
    }
    part_fn!("YEAR", "YEAR(date): year", DatePart::Year);
    part_fn!("QUARTER", "QUARTER(date): quarter 1–4", DatePart::Quarter);
    part_fn!("MONTH", "MONTH(date): month 1–12", DatePart::Month);
    part_fn!("DAY", "DAY(date): day of month", DatePart::Day);
    part_fn!("HOUR", "HOUR(date): hour 0–23", DatePart::Hour);
    part_fn!("MINUTE", "MINUTE(date): minute", DatePart::Minute);
    part_fn!("SECOND", "SECOND(date): second (with fraction)", DatePart::FractionalSecond);
    part_fn!("WEEKDAY", "WEEKDAY(date): 1 = Sunday … 7 = Saturday", DatePart::Weekday);
    part_fn!("DAYOFYEAR", "DAYOFYEAR(date): 1–366", DatePart::DayOfYear);
    part_fn!("WEEK", "WEEK(date): ISO-8601 week number", DatePart::IsoWeek);
    part_fn!("MONTHNAME", "MONTHNAME(date): month name", DatePart::MonthName);
    part_fn!("DAYNAME", "DAYNAME(date): weekday name", DatePart::WeekdayName);
    part_fn!("DATEONLY", "DATEONLY(datetime): date part", DatePart::DateOnly);
    part_fn!("TIMEONLY", "TIMEONLY(datetime): time-of-day part", DatePart::TimeOnly);
    r.register(def(
        "DATEVALUE",
        1,
        Some(1),
        Cat,
        "DATEVALUE(text): parse text as a date/time (missing if invalid)",
        |a, _| {
            let s = super::texts(&a[0], "text")?;
            Ok(out_dt(s.iter().map(|t| datetime::parse(t).unwrap_or(DATETIME_MISSING)).collect(), a[0].scalar))
        },
    ));
    r.register(def("DATETEXT", 1, Some(2), Cat, "DATETEXT(date, [pattern]): format a date as text", |a, _| {
        let d = dates(&a[0])?;
        let pat = match a.get(1) {
            Some(p) => Some(super::texts(p, "pattern")?.first().cloned().unwrap_or_default()),
            None => None,
        };
        let out = d
            .iter()
            .map(|&t| {
                if dt_missing(t) {
                    String::new()
                } else {
                    match &pat {
                        Some(p) => datetime::format(t, p),
                        None => datetime::format_default(t),
                    }
                }
            })
            .collect();
        Ok(Arr { data: ColumnData::Text(out), scalar: a[0].scalar })
    }));
    r.register(def("EDATE", 2, Some(2), Cat, "EDATE(date, months): same day n months later (clamped)", |a, _| {
        let (n, scalar) = common_len(&[&a[0], &a[1]], "EDATE")?;
        let (d, m) = (dates(&a[0])?, a[1].to_nums("months")?);
        Ok(out_dt(
            (0..n)
                .map(|i| {
                    let k = at(&m, i);
                    if k.is_finite() {
                        add_months(at_i(&d, i), k.trunc() as i64)
                    } else {
                        DATETIME_MISSING
                    }
                })
                .collect(),
            scalar,
        ))
    }));
    r.register(def(
        "EOMONTH",
        1,
        Some(2),
        Cat,
        "EOMONTH(date, [months]): last day of the month n months away",
        |a, _| {
            let months = a.get(1).cloned().unwrap_or_else(|| Arr::num(0.0));
            let (n, scalar) = common_len(&[&a[0], &months], "EOMONTH")?;
            let (d, m) = (dates(&a[0])?, months.to_nums("months")?);
            Ok(out_dt(
                (0..n)
                    .map(|i| {
                        let (t, k) = (at_i(&d, i), at(&m, i));
                        if dt_missing(t) || !k.is_finite() {
                            return DATETIME_MISSING;
                        }
                        let s = DateTime::from_timestamp(add_months(t, k.trunc() as i64));
                        datetime::days_from_civil(s.year, s.month, days_in_month(s.year, s.month)) * MICROS_PER_DAY
                    })
                    .collect(),
                scalar,
            ))
        },
    ));
    r.register(def("DAYS", 2, Some(2), Cat, "DAYS(end, start): days between two dates", |a, _| {
        let (n, scalar) = common_len(&[&a[0], &a[1]], "DAYS")?;
        let (e, s) = (dates(&a[0])?, dates(&a[1])?);
        Ok(Arr::nums((0..n).map(|i| datetime::diff_days(at_i(&e, i), at_i(&s, i))).collect(), scalar))
    }));
    r.register(def("ADDDAYS", 2, Some(2), Cat, "ADDDAYS(date, n): date plus n (fractional) days", |a, _| {
        let (n, scalar) = common_len(&[&a[0], &a[1]], "ADDDAYS")?;
        let (d, k) = (dates(&a[0])?, a[1].to_nums("days")?);
        Ok(out_dt((0..n).map(|i| datetime::add_days(at_i(&d, i), at(&k, i))).collect(), scalar))
    }));
    r.register(def("DATESERIAL", 1, Some(1), Cat, "DATESERIAL(date): spreadsheet serial number", |a, _| {
        let d = dates(&a[0])?;
        Ok(Arr::nums(d.iter().map(|&t| datetime::to_excel_serial(t)).collect(), a[0].scalar))
    }));
    r.register(def("SERIALDATE", 1, Some(1), Cat, "SERIALDATE(x): spreadsheet serial number to date", |a, _| {
        let x = a[0].to_nums("serial")?;
        Ok(out_dt(x.iter().map(|&s| datetime::from_excel_serial(s)).collect(), a[0].scalar))
    }));
    r.register(def(
        "WORKDAYS",
        2,
        Some(2),
        Cat,
        "WORKDAYS(start, end): Monday–Friday days in [start, end]",
        |a, _| {
            let (n, scalar) = common_len(&[&a[0], &a[1]], "WORKDAYS")?;
            let (s, e) = (dates(&a[0])?, dates(&a[1])?);
            Ok(Arr::nums(
                (0..n)
                    .map(|i| {
                        let (x, y) = (at_i(&s, i), at_i(&e, i));
                        if dt_missing(x) || dt_missing(y) {
                            return f64::NAN;
                        }
                        let (d0, d1) = (x.div_euclid(MICROS_PER_DAY), y.div_euclid(MICROS_PER_DAY));
                        let (lo, hi, sign) = if d0 <= d1 { (d0, d1, 1.0) } else { (d1, d0, -1.0) };
                        // Count weekdays in [lo, hi]: whole weeks plus the remainder.
                        let total = hi - lo + 1;
                        let mut count = total / 7 * 5;
                        let start_wd = (lo + 4).rem_euclid(7); // 0 = Sunday
                        for k in 0..total % 7 {
                            let wd = (start_wd + k) % 7;
                            if wd != 0 && wd != 6 {
                                count += 1;
                            }
                        }
                        sign * count as f64
                    })
                    .collect(),
                scalar,
            ))
        },
    ));
    r.register(def("AGE", 2, Some(2), Cat, "AGE(birth, asof): whole years elapsed", |a, _| {
        let (n, scalar) = common_len(&[&a[0], &a[1]], "AGE")?;
        let (b, t) = (dates(&a[0])?, dates(&a[1])?);
        Ok(Arr::nums(
            (0..n)
                .map(|i| {
                    let (x, y) = (at_i(&b, i), at_i(&t, i));
                    if dt_missing(x) || dt_missing(y) {
                        return f64::NAN;
                    }
                    let (p, q) = (DateTime::from_timestamp(x), DateTime::from_timestamp(y));
                    let mut years = q.year - p.year;
                    if (q.month, q.day) < (p.month, p.day) {
                        years -= 1;
                    }
                    years as f64
                })
                .collect(),
            scalar,
        ))
    }));
    r.register(def("ISLEAPYEAR", 1, Some(1), Cat, "ISLEAPYEAR(year or date): 1 for leap years", |a, _| {
        let years: Vec<f64> = match a[0].kind() {
            crate::column::ColumnKind::DateTime => dates(&a[0])?
                .iter()
                .map(|&t| if dt_missing(t) { f64::NAN } else { DateTime::from_timestamp(t).year as f64 })
                .collect(),
            _ => a[0].to_nums("year")?,
        };
        Ok(Arr::nums(
            years
                .iter()
                .map(|&y| {
                    if y.is_finite() && y.fract() == 0.0 {
                        f64::from(u8::from(datetime::is_leap_year(y as i64)))
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
            a[0].scalar,
        ))
    }));
    r.alias("DATEDIFF", "DAYS");
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::datetime::{format, parse};
    use crate::formula::{EvalCtx, FunctionRegistry};
    use crate::worksheet::Worksheet;

    #[test]
    fn date_functions() {
        let ws = Worksheet::with_columns(
            "s",
            vec![Column::datetime("d", vec![parse("2024-01-31 15:30").unwrap(), crate::missing::DATETIME_MISSING])],
        )
        .unwrap();
        let reg = FunctionRegistry::standard();
        let ctx = EvalCtx::new(&ws, &reg).with_now(Some(parse("2026-11-16 12:02").unwrap()));
        let f = |s: &str| ctx.eval_str(s).unwrap();
        let fmt = |s: &str| format(f(s).data.as_datetime().unwrap()[0], "yyyy-MM-dd HH:mm");
        assert_eq!(fmt("DATE(2026, 14, 1)"), "2027-02-01 00:00");
        assert_eq!(fmt("DATE(2026, 3, 0)"), "2026-02-28 00:00");
        assert_eq!(fmt("EDATE(d, 1)"), "2024-02-29 15:30");
        assert_eq!(fmt("EOMONTH(d, 1)"), "2024-02-29 00:00");
        assert_eq!(fmt("TODAY()"), "2026-11-16 00:00");
        assert_eq!(fmt("DATEVALUE(\"16-Nov-2026\") + TIME(1, 30, 0)"), "2026-11-16 01:30");
        assert_eq!(f("YEAR(d)").data.to_f64_vec()[0], 2024.0);
        assert!(f("YEAR(d)").data.is_missing(1));
        assert_eq!(f("SECOND(TIME(0, 0, 1.25))").data.to_f64_vec(), [1.25]);
        assert_eq!(f("HOUR(d)").data.to_f64_vec()[0], 15.0);
        assert_eq!(f("WEEKDAY(DATE(2026, 11, 16))").data.to_f64_vec(), [2.0]);
        assert_eq!(f("DAYS(DATE(2026, 3, 1), DATE(2026, 2, 1))").data.to_f64_vec(), [28.0]);
        assert_eq!(f("WORKDAYS(DATE(2026, 11, 16), DATE(2026, 11, 29))").data.to_f64_vec(), [10.0]);
        assert_eq!(f("WORKDAYS(DATE(2026, 11, 22), DATE(2026, 11, 21))").data.to_f64_vec(), [0.0]);
        assert_eq!(f("AGE(DATE(2000, 11, 17), TODAY())").data.to_f64_vec(), [25.0]);
        assert_eq!(f("DATETEXT(d, \"MMM yyyy\")").data.to_text_vec()[0], "Jan 2024");
        assert_eq!(f("MONTHNAME(d)").data.to_text_vec()[0], "January");
        assert_eq!(f("ISLEAPYEAR(2100) + ISLEAPYEAR(d)").data.to_f64_vec()[0], 1.0);
        assert_eq!(f("DATESERIAL(DATE(2026, 11, 16))").data.to_f64_vec(), [46342.0]);
        let none = EvalCtx::new(&ws, &reg).with_now(None);
        assert!(none.eval_str("NOW()").is_err());
    }
}
