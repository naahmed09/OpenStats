//! Date/time support.
//!
//! Date/time values are stored as `i64` **microseconds since
//! 1970-01-01T00:00:00 UTC** on the proleptic Gregorian calendar
//! ([`crate::missing::DATETIME_MISSING`] = `i64::MIN` is missing). The range
//! covers roughly ±292,000 years, and microsecond resolution matches what
//! spreadsheets and Minitab can display.
//!
//! Calendar conversions use Howard Hinnant's `days_from_civil` /
//! `civil_from_days` algorithms, which are exact for every representable day.
//!
//! ## Format patterns
//!
//! | token | meaning | token | meaning |
//! |---|---|---|---|
//! | `yyyy` | 4-digit year | `yy` | 2-digit year |
//! | `MMMM` | month name | `MMM` | month abbrev. |
//! | `MM` / `M` | month, padded / not | `dd` / `d` | day, padded / not |
//! | `EEEE` / `EEE` | weekday name / abbrev. | `DDD` | day of year (padded) |
//! | `HH` / `H` | hour 0–23 | `hh` / `h` | hour 1–12 |
//! | `mm` / `m` | minute | `ss` / `s` | second |
//! | `fff` / `ffffff` | milli / micro seconds | `a` | AM/PM |
//! | `'text'` | literal | other chars | literal |

use crate::error::{CoreError, Result};
use crate::missing::DATETIME_MISSING;

pub const MICROS_PER_SECOND: i64 = 1_000_000;
pub const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SECOND;
pub const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
pub const MICROS_PER_DAY: i64 = 24 * MICROS_PER_HOUR;

/// Default display pattern for values with no time-of-day component.
pub const DEFAULT_DATE_FORMAT: &str = "M/d/yyyy";
/// Default display pattern for values with a time-of-day component.
pub const DEFAULT_DATETIME_FORMAT: &str = "M/d/yyyy H:mm:ss";
/// ISO-8601 round-trip format used for JSON/CSV export.
pub const ISO_FORMAT: &str = "yyyy-MM-dd'T'HH:mm:ss.ffffff";

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const WEEKDAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

/// Day-month order used when interpreting ambiguous numeric dates such as `03/04/2026`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DateOrder {
    /// Month/Day/Year (US, Minitab default).
    #[default]
    Mdy,
    /// Day/Month/Year (most of Europe).
    Dmy,
    /// Year/Month/Day.
    Ymd,
}

/// Broken-down civil date and time (UTC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DateTime {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub micros: u32,
}

/// `true` if `y` is a Gregorian leap year.
#[inline]
pub fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Number of days in month `m` (1–12) of year `y`.
pub fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days since 1970-01-01 for the civil date `y-m-d` (Hinnant).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400; // [0, 399]
    let m = m as i64;
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Civil date for a count of days since 1970-01-01 (Hinnant).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

impl DateTime {
    /// Validated constructor.
    pub fn new(year: i64, month: u32, day: u32, hour: u32, minute: u32, second: u32, micros: u32) -> Result<Self> {
        if !(1..=12).contains(&month) {
            return Err(CoreError::invalid(format!("month {month} out of range")));
        }
        if day == 0 || day > days_in_month(year, month) {
            return Err(CoreError::invalid(format!("day {day} out of range for {year}-{month:02}")));
        }
        if hour > 23 || minute > 59 || second > 59 || micros > 999_999 {
            return Err(CoreError::invalid(format!("time {hour}:{minute:02}:{second:02}.{micros:06} out of range")));
        }
        Ok(DateTime { year, month, day, hour, minute, second, micros })
    }

    /// Midnight on the given date.
    pub fn date(year: i64, month: u32, day: u32) -> Result<Self> {
        Self::new(year, month, day, 0, 0, 0, 0)
    }

    /// Convert a timestamp (µs since epoch) to its civil components.
    pub fn from_timestamp(ts: i64) -> Self {
        let days = ts.div_euclid(MICROS_PER_DAY);
        let rem = ts.rem_euclid(MICROS_PER_DAY);
        let (year, month, day) = civil_from_days(days);
        let secs = rem / MICROS_PER_SECOND;
        DateTime {
            year,
            month,
            day,
            hour: (secs / 3600) as u32,
            minute: ((secs / 60) % 60) as u32,
            second: (secs % 60) as u32,
            micros: (rem % MICROS_PER_SECOND) as u32,
        }
    }

    /// Timestamp in µs since epoch.
    pub fn to_timestamp(&self) -> i64 {
        days_from_civil(self.year, self.month, self.day) * MICROS_PER_DAY
            + self.hour as i64 * MICROS_PER_HOUR
            + self.minute as i64 * MICROS_PER_MINUTE
            + self.second as i64 * MICROS_PER_SECOND
            + self.micros as i64
    }

    /// 0 = Sunday … 6 = Saturday.
    pub fn weekday(&self) -> u32 {
        // 1970-01-01 was a Thursday (4).
        (days_from_civil(self.year, self.month, self.day) + 4).rem_euclid(7) as u32
    }

    /// 1 = January 1st.
    pub fn day_of_year(&self) -> u32 {
        (days_from_civil(self.year, self.month, self.day) - days_from_civil(self.year, 1, 1) + 1) as u32
    }

    /// Calendar quarter 1–4.
    pub fn quarter(&self) -> u32 {
        (self.month - 1) / 3 + 1
    }

    /// ISO-8601 week-numbering year and week (1–53).
    pub fn iso_week(&self) -> (i64, u32) {
        let days = days_from_civil(self.year, self.month, self.day);
        let iso_wd = (days + 3).rem_euclid(7) + 1; // Monday = 1 … Sunday = 7
                                                   // The Thursday of this week determines the ISO year.
        let thursday = days - iso_wd + 4;
        let (ty, _, _) = civil_from_days(thursday);
        let week = (thursday - days_from_civil(ty, 1, 1)) / 7 + 1;
        (ty, week as u32)
    }

    /// `true` if the time-of-day part is exactly midnight.
    pub fn is_midnight(&self) -> bool {
        self.hour == 0 && self.minute == 0 && self.second == 0 && self.micros == 0
    }
}

/// Add whole calendar months, clamping the day to the target month's length
/// (Jan 31 + 1 month = Feb 28/29). Missing propagates.
pub fn add_months(ts: i64, months: i64) -> i64 {
    if ts == DATETIME_MISSING {
        return DATETIME_MISSING;
    }
    let dt = DateTime::from_timestamp(ts);
    let total = dt.year * 12 + (dt.month as i64 - 1) + months;
    let year = total.div_euclid(12);
    let month = (total.rem_euclid(12) + 1) as u32;
    let day = dt.day.min(days_in_month(year, month));
    DateTime { year, month, day, ..dt }.to_timestamp()
}

/// Add a (possibly fractional) number of days. Missing propagates.
pub fn add_days(ts: i64, days: f64) -> i64 {
    if ts == DATETIME_MISSING || days.is_nan() {
        return DATETIME_MISSING;
    }
    ts + (days * MICROS_PER_DAY as f64).round() as i64
}

/// Difference `a - b` in fractional days; NaN if either is missing.
pub fn diff_days(a: i64, b: i64) -> f64 {
    if a == DATETIME_MISSING || b == DATETIME_MISSING {
        return f64::NAN;
    }
    (a - b) as f64 / MICROS_PER_DAY as f64
}

/// Spreadsheet serial number (days since 1899-12-30, Excel 1900 system,
/// including Excel's fictitious 1900-02-29 for serials < 61) to timestamp.
pub fn from_excel_serial(serial: f64) -> i64 {
    if serial.is_nan() {
        return DATETIME_MISSING;
    }
    // Serials 1..60 are offset by one day because Excel believes 1900 was leap.
    let s = if (1.0..61.0).contains(&serial) { serial + 1.0 } else { serial };
    let base = days_from_civil(1899, 12, 30) * MICROS_PER_DAY;
    base + (s * MICROS_PER_DAY as f64).round() as i64
}

/// Timestamp to Excel 1900-system serial number.
pub fn to_excel_serial(ts: i64) -> f64 {
    if ts == DATETIME_MISSING {
        return f64::NAN;
    }
    let base = days_from_civil(1899, 12, 30) * MICROS_PER_DAY;
    let s = (ts - base) as f64 / MICROS_PER_DAY as f64;
    if (2.0..62.0).contains(&s) {
        s - 1.0
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format a timestamp with the default pattern (date only when the time of
/// day is midnight). Missing is rendered as `*`.
pub fn format_default(ts: i64) -> String {
    if ts == DATETIME_MISSING {
        return crate::missing::MISSING_TOKEN.to_string();
    }
    let dt = DateTime::from_timestamp(ts);
    if dt.is_midnight() {
        format(ts, DEFAULT_DATE_FORMAT)
    } else {
        format(ts, DEFAULT_DATETIME_FORMAT)
    }
}

/// Format a timestamp using a pattern (see module docs).
pub fn format(ts: i64, pattern: &str) -> String {
    if ts == DATETIME_MISSING {
        return crate::missing::MISSING_TOKEN.to_string();
    }
    let dt = DateTime::from_timestamp(ts);
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() + 8);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            // Quoted literal; '' is an escaped quote.
            i += 1;
            if i < chars.len() && chars[i] == '\'' {
                out.push('\'');
                i += 1;
                continue;
            }
            while i < chars.len() && chars[i] != '\'' {
                out.push(chars[i]);
                i += 1;
            }
            i += 1;
            continue;
        }
        let mut run = 1;
        while i + run < chars.len() && chars[i + run] == c {
            run += 1;
        }
        match c {
            'y' => {
                if run == 2 {
                    out.push_str(&format!("{:02}", dt.year.rem_euclid(100)));
                } else if dt.year < 0 {
                    out.push_str(&format!("-{:0w$}", -dt.year, w = run.max(4)));
                } else {
                    out.push_str(&format!("{:0w$}", dt.year, w = run.max(if run == 1 { 1 } else { 4 })));
                }
            }
            'M' => match run {
                1 => out.push_str(&dt.month.to_string()),
                2 => out.push_str(&format!("{:02}", dt.month)),
                3 => out.push_str(&MONTHS[dt.month as usize - 1][..3]),
                _ => out.push_str(MONTHS[dt.month as usize - 1]),
            },
            'd' => match run {
                1 => out.push_str(&dt.day.to_string()),
                _ => out.push_str(&format!("{:02}", dt.day)),
            },
            'D' => out.push_str(&format!("{:0w$}", dt.day_of_year(), w = run)),
            'E' => {
                let name = WEEKDAYS[dt.weekday() as usize];
                if run >= 4 {
                    out.push_str(name)
                } else {
                    out.push_str(&name[..3])
                }
            }
            'H' => out.push_str(&format!("{:0w$}", dt.hour, w = run.min(2))),
            'h' => {
                let h12 = if dt.hour % 12 == 0 { 12 } else { dt.hour % 12 };
                out.push_str(&format!("{:0w$}", h12, w = run.min(2)))
            }
            'm' => out.push_str(&format!("{:0w$}", dt.minute, w = run.min(2))),
            's' => out.push_str(&format!("{:0w$}", dt.second, w = run.min(2))),
            'f' => {
                let digits = format!("{:06}", dt.micros);
                out.push_str(&digits[..run.min(6)]);
            }
            'a' => out.push_str(if dt.hour < 12 { "AM" } else { "PM" }),
            _ => {
                for _ in 0..run {
                    out.push(c);
                }
            }
        }
        i += run;
    }
    out
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseOptions {
    pub order: DateOrder,
    /// Two-digit years `< pivot` map to 20xx, others to 19xx (default 30,
    /// the spreadsheet convention).
    pub two_digit_pivot: u32,
}

impl Default for ParseOptions {
    fn default() -> Self {
        ParseOptions { order: DateOrder::Mdy, two_digit_pivot: 30 }
    }
}

/// Parse a date/time string with default options. Missing tokens (`*`, empty,
/// `NA` …) yield [`DATETIME_MISSING`].
pub fn parse(s: &str) -> Result<i64> {
    parse_with(s, ParseOptions::default())
}

/// Parse a date/time string. Accepts ISO-8601 (with optional `T`, fractional
/// seconds and `Z`/offset), numeric dates with `/`, `-` or `.` separators,
/// textual months (`16-Nov-2026`, `Nov 16, 2026`, `Monday, November 16 2026`),
/// optional times with seconds, fractions and AM/PM, and time-only values
/// (placed on 1970-01-01). Offsets are converted to UTC.
pub fn parse_with(s: &str, opts: ParseOptions) -> Result<i64> {
    let s = s.trim();
    if crate::missing::is_missing_token(s) {
        return Ok(DATETIME_MISSING);
    }
    if let Some(ts) = parse_iso(s) {
        return Ok(ts);
    }
    parse_general(s, opts)
}

fn digits_at(b: &[u8], pos: usize, n: usize) -> Option<u32> {
    if pos + n > b.len() {
        return None;
    }
    let mut v = 0u32;
    for &c in &b[pos..pos + n] {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as u32;
    }
    Some(v)
}

/// Strict ISO-8601 extended format: `YYYY-MM-DD[(T| )HH:MM[:SS[.f+]]][Z|±HH[:MM]]`.
fn parse_iso(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let (neg, off) = if b.first() == Some(&b'-') { (true, 1) } else { (false, 0) };
    let year = digits_at(b, off, 4)? as i64;
    let year = if neg { -year } else { year };
    if b.get(off + 4) != Some(&b'-') {
        return None;
    }
    let month = digits_at(b, off + 5, 2)?;
    if b.get(off + 7) != Some(&b'-') {
        return None;
    }
    let day = digits_at(b, off + 8, 2)?;
    let mut p = off + 10;
    let (mut h, mut mi, mut sec, mut us) = (0, 0, 0, 0u32);
    let mut offset_us: i64 = 0;
    if p < b.len() {
        if b[p] != b'T' && b[p] != b' ' && b[p] != b't' {
            return None;
        }
        p += 1;
        h = digits_at(b, p, 2)?;
        if b.get(p + 2) != Some(&b':') {
            return None;
        }
        mi = digits_at(b, p + 3, 2)?;
        p += 5;
        if b.get(p) == Some(&b':') {
            sec = digits_at(b, p + 1, 2)?;
            p += 3;
            if b.get(p) == Some(&b'.') || b.get(p) == Some(&b',') {
                p += 1;
                let start = p;
                while p < b.len() && b[p].is_ascii_digit() {
                    p += 1;
                }
                if p == start {
                    return None;
                }
                us = frac_to_micros(&s[start..p]);
            }
        }
        if p < b.len() {
            match b[p] {
                b'Z' | b'z' if p + 1 == b.len() => {}
                b'+' | b'-' => {
                    let sign = if b[p] == b'-' { -1 } else { 1 };
                    let oh = digits_at(b, p + 1, 2)? as i64;
                    let rest = &b[p + 3..];
                    let om = match rest.len() {
                        0 => 0,
                        2 => digits_at(b, p + 3, 2)? as i64,
                        3 if rest[0] == b':' => digits_at(b, p + 4, 2)? as i64,
                        _ => return None,
                    };
                    offset_us = sign * (oh * MICROS_PER_HOUR + om * MICROS_PER_MINUTE);
                }
                _ => return None,
            }
        }
    }
    let dt = DateTime::new(year, month, day, h, mi, sec, us).ok()?;
    Some(dt.to_timestamp() - offset_us)
}

fn frac_to_micros(digits: &str) -> u32 {
    let mut v = 0u64;
    let mut n = 0;
    for c in digits.bytes().take(7) {
        v = v * 10 + (c - b'0') as u64;
        n += 1;
    }
    // Round to 6 digits.
    while n < 7 {
        v *= 10;
        n += 1;
    }
    let rounded = (v + 5) / 10;
    rounded.min(999_999) as u32
}

#[derive(Debug, PartialEq)]
enum Tok {
    Num(String),
    Word(String),
}

fn month_from_word(w: &str) -> Option<u32> {
    let lw = w.to_ascii_lowercase();
    if lw.len() < 3 {
        return None;
    }
    let lw = if lw == "sept" { "sep".to_string() } else { lw };
    MONTHS
        .iter()
        .position(|m| m.to_ascii_lowercase().starts_with(&lw) && (lw.len() == 3 || m.len() >= lw.len()))
        .map(|i| i as u32 + 1)
}

fn is_weekday_word(w: &str) -> bool {
    let lw = w.to_ascii_lowercase();
    lw.len() >= 3 && WEEKDAYS.iter().any(|d| d.to_ascii_lowercase().starts_with(&lw))
}

fn tokenize_date(s: &str) -> Result<Vec<Tok>> {
    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            let st = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Num(chars[st..i].iter().collect()));
        } else if c.is_alphabetic() {
            let st = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            let w: String = chars[st..i].iter().collect();
            // Ordinal suffixes: 1st, 2nd, 3rd, 4th
            let lw = w.to_ascii_lowercase();
            if matches!(lw.as_str(), "st" | "nd" | "rd" | "th") && matches!(out.last(), Some(Tok::Num(_))) {
                continue;
            }
            out.push(Tok::Word(w));
        } else if matches!(c, '/' | '-' | '.' | ',' | ' ' | '\t') {
            i += 1;
        } else {
            return Err(CoreError::parse("date", format!("unexpected character {c:?}"), Some(i)));
        }
    }
    Ok(out)
}

fn expand_year(y: &str, opts: ParseOptions) -> i64 {
    let v: i64 = y.parse().unwrap_or(0);
    if y.len() <= 2 {
        if (v as u32) < opts.two_digit_pivot {
            2000 + v
        } else {
            1900 + v
        }
    } else {
        v
    }
}

fn parse_general(s: &str, opts: ParseOptions) -> Result<i64> {
    // Split into date part and time part at the digit run preceding the first ':'.
    let (date_part, time_part) = match s.find(':') {
        Some(colon) => {
            let bytes = s.as_bytes();
            let mut st = colon;
            while st > 0 && bytes[st - 1].is_ascii_digit() {
                st -= 1;
            }
            let mut dp = s[..st].trim_end();
            // Allow an ISO-style 'T' separator in non-ISO dates.
            if dp.ends_with('T') {
                dp = &dp[..dp.len() - 1];
            }
            (dp.trim(), Some(s[st..].trim()))
        }
        None => (s, None),
    };

    let (y, m, d) =
        if date_part.is_empty() { (1970, 1, 1) } else { parse_date_tokens(&tokenize_date(date_part)?, opts, s)? };
    let (h, mi, sec, us, offset) = match time_part {
        Some(t) => parse_time(t)?,
        None => (0, 0, 0, 0, 0),
    };
    let dt = DateTime::new(y, m, d, h, mi, sec, us)?;
    Ok(dt.to_timestamp() - offset)
}

fn parse_date_tokens(toks: &[Tok], opts: ParseOptions, original: &str) -> Result<(i64, u32, u32)> {
    let err = || CoreError::parse("date", format!("unrecognised date {original:?}"), None);
    let mut t: &[Tok] = toks;
    if let Some(Tok::Word(w)) = t.first() {
        if is_weekday_word(w) && month_from_word(w).is_none() {
            t = &t[1..];
        }
    }
    let num = |tok: &Tok| -> Option<String> {
        if let Tok::Num(n) = tok {
            Some(n.clone())
        } else {
            None
        }
    };
    let pu = |s: &str| -> Result<u32> { s.parse::<u32>().map_err(|_| err()) };
    match t {
        [a, b, c] => {
            match (a, b, c) {
                (Tok::Num(a), Tok::Num(b), Tok::Num(c)) => {
                    if a.len() >= 3 {
                        Ok((expand_year(a, opts), pu(b)?, pu(c)?))
                    } else {
                        match opts.order {
                            DateOrder::Mdy => Ok((expand_year(c, opts), pu(a)?, pu(b)?)),
                            DateOrder::Dmy => Ok((expand_year(c, opts), pu(b)?, pu(a)?)),
                            DateOrder::Ymd => Ok((expand_year(a, opts), pu(b)?, pu(c)?)),
                        }
                    }
                }
                (Tok::Num(d), Tok::Word(mw), Tok::Num(y)) => {
                    let m = month_from_word(mw).ok_or_else(err)?;
                    if d.len() >= 3 {
                        // 2026-Nov-16
                        Ok((expand_year(d, opts), m, pu(y)?))
                    } else {
                        Ok((expand_year(y, opts), m, pu(d)?))
                    }
                }
                (Tok::Word(mw), Tok::Num(d), Tok::Num(y)) => {
                    let m = month_from_word(mw).ok_or_else(err)?;
                    Ok((expand_year(y, opts), m, pu(d)?))
                }
                _ => Err(err()),
            }
        }
        // "Nov 2026" / "11/2026": first day of month.
        [a, b] => match (a, b) {
            (Tok::Word(mw), Tok::Num(y)) => Ok((expand_year(y, opts), month_from_word(mw).ok_or_else(err)?, 1)),
            (Tok::Num(m), Tok::Num(y)) if y.len() == 4 => Ok((expand_year(y, opts), pu(m)?, 1)),
            _ => Err(err()),
        },
        // Compact "20261116"
        [a] => {
            let a = num(a).ok_or_else(err)?;
            if a.len() == 8 {
                Ok((a[..4].parse().map_err(|_| err())?, pu(&a[4..6])?, pu(&a[6..])?))
            } else {
                Err(err())
            }
        }
        _ => Err(err()),
    }
}

/// Parse `H:MM[:SS[.fff]][ AM|PM][Z|±HH:MM]` → (h, m, s, µs, offset µs).
fn parse_time(t: &str) -> Result<(u32, u32, u32, u32, i64)> {
    let err = |m: &str| CoreError::parse("time", format!("{m} in {t:?}"), None);
    let b = t.as_bytes();
    let mut p = 0;
    let read_num = |p: &mut usize| -> Option<(u32, usize)> {
        let st = *p;
        while *p < b.len() && b[*p].is_ascii_digit() {
            *p += 1;
        }
        if *p == st || *p - st > 2 {
            return None;
        }
        Some((t[st..*p].parse().ok()?, *p - st))
    };
    let (mut h, _) = read_num(&mut p).ok_or_else(|| err("bad hour"))?;
    if b.get(p) != Some(&b':') {
        return Err(err("expected ':'"));
    }
    p += 1;
    let (mi, _) = read_num(&mut p).ok_or_else(|| err("bad minute"))?;
    let mut sec = 0;
    let mut us = 0;
    if b.get(p) == Some(&b':') {
        p += 1;
        sec = read_num(&mut p).ok_or_else(|| err("bad second"))?.0;
        if b.get(p) == Some(&b'.') {
            p += 1;
            let st = p;
            while p < b.len() && b[p].is_ascii_digit() {
                p += 1;
            }
            if st == p {
                return Err(err("bad fraction"));
            }
            us = frac_to_micros(&t[st..p]);
        }
    }
    let rest = t[p..].trim();
    let mut offset = 0i64;
    let mut rest_after_ampm = rest;
    let lower = rest.to_ascii_lowercase();
    for (tag, pm) in [("am", false), ("pm", true), ("a.m.", false), ("p.m.", true), ("a", false), ("p", true)] {
        if lower.starts_with(tag) {
            if h == 0 || h > 12 {
                return Err(err("hour out of range for 12-hour clock"));
            }
            if pm && h != 12 {
                h += 12;
            } else if !pm && h == 12 {
                h = 0;
            }
            rest_after_ampm = rest[tag.len()..].trim();
            break;
        }
    }
    let r = rest_after_ampm;
    if !r.is_empty() {
        if r.eq_ignore_ascii_case("z") || r.eq_ignore_ascii_case("utc") || r.eq_ignore_ascii_case("gmt") {
        } else if r.starts_with('+') || r.starts_with('-') {
            let sign = if r.starts_with('-') { -1 } else { 1 };
            let digits: String = r[1..].chars().filter(|c| c.is_ascii_digit()).collect();
            let (oh, om) = match digits.len() {
                1 | 2 => (digits.parse::<i64>().unwrap_or(0), 0),
                3 | 4 => {
                    let (a, b) = digits.split_at(digits.len() - 2);
                    (a.parse::<i64>().unwrap_or(0), b.parse::<i64>().unwrap_or(0))
                }
                _ => return Err(err("bad offset")),
            };
            offset = sign * (oh * MICROS_PER_HOUR + om * MICROS_PER_MINUTE);
        } else {
            return Err(err("trailing text"));
        }
    }
    Ok((h, mi, sec, us, offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_roundtrip_wide_range() {
        for z in (-1_000_000..1_000_000).step_by(997) {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }

    #[test]
    fn components() {
        let ts = parse("2026-11-16T11:31:05.25").unwrap();
        let dt = DateTime::from_timestamp(ts);
        assert_eq!(
            (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.micros),
            (2026, 11, 16, 11, 31, 5, 250_000)
        );
        assert_eq!(dt.weekday(), 1); // Monday
        assert_eq!(dt.quarter(), 4);
        assert_eq!(dt.day_of_year(), 320);
        assert_eq!(dt.iso_week(), (2026, 47));
        // ISO week edge cases
        assert_eq!(DateTime::date(2021, 1, 3).unwrap().iso_week(), (2020, 53));
        assert_eq!(DateTime::date(2024, 12, 30).unwrap().iso_week(), (2025, 1));
    }

    #[test]
    fn parsing_variants() {
        let target = DateTime::date(2026, 11, 16).unwrap().to_timestamp();
        for s in [
            "2026-11-16",
            "11/16/2026",
            "11-16-26",
            "16-Nov-2026",
            "Nov 16, 2026",
            "November 16th 2026",
            "Monday, November 16, 2026",
            "20261116",
            "2026/11/16",
        ] {
            assert_eq!(parse(s).unwrap(), target, "{s}");
        }
        let dmy = ParseOptions { order: DateOrder::Dmy, ..Default::default() };
        assert_eq!(parse_with("16.11.2026", dmy).unwrap(), target);
        let t = parse("11/16/2026 1:05 PM").unwrap();
        assert_eq!(format(t, "yyyy-MM-dd HH:mm"), "2026-11-16 13:05");
        assert_eq!(parse("2026-11-16T12:00:00+02:00").unwrap(), parse("2026-11-16 10:00:00Z").unwrap());
        assert_eq!(parse("2026-11-16T12:00:00-0530").unwrap(), parse("2026-11-16T17:30:00").unwrap());
        assert_eq!(parse("12:30 AM").unwrap(), 30 * MICROS_PER_MINUTE);
        assert_eq!(parse("*").unwrap(), DATETIME_MISSING);
        assert!(parse("2026-02-30").is_err());
        assert!(parse("13/45/2026").is_err());
        assert!(parse("hello").is_err());
    }

    #[test]
    fn formatting() {
        let t = parse("2026-03-05 07:08:09.123456").unwrap();
        assert_eq!(format(t, "EEE, d MMM yyyy h:mm:ss.fff a"), "Thu, 5 Mar 2026 7:08:09.123 AM");
        assert_eq!(format(t, "'Day' DDD 'of' yy"), "Day 064 of 26");
        assert_eq!(format(t, ISO_FORMAT), "2026-03-05T07:08:09.123456");
        assert_eq!(format_default(DateTime::date(2026, 3, 5).unwrap().to_timestamp()), "3/5/2026");
        assert_eq!(format_default(DATETIME_MISSING), "*");
        let neg = DateTime::date(-44, 3, 15).unwrap().to_timestamp();
        assert_eq!(format(neg, "yyyy-MM-dd"), "-0044-03-15");
    }

    #[test]
    fn arithmetic() {
        let jan31 = DateTime::date(2024, 1, 31).unwrap().to_timestamp();
        assert_eq!(format(add_months(jan31, 1), "yyyy-MM-dd"), "2024-02-29");
        assert_eq!(format(add_months(jan31, -2), "yyyy-MM-dd"), "2023-11-30");
        assert_eq!(format(add_days(jan31, 1.5), "yyyy-MM-dd HH"), "2024-02-01 12");
        assert_eq!(diff_days(add_days(jan31, 10.0), jan31), 10.0);
        assert!(diff_days(DATETIME_MISSING, jan31).is_nan());
    }

    #[test]
    fn excel_serials() {
        assert_eq!(format(from_excel_serial(46342.0), "yyyy-MM-dd"), "2026-11-16");
        assert_eq!(format(from_excel_serial(1.0), "yyyy-MM-dd"), "1900-01-01");
        assert_eq!(format(from_excel_serial(61.0), "yyyy-MM-dd"), "1900-03-01");
        assert_eq!(to_excel_serial(from_excel_serial(46342.5)), 46342.5);
        assert_eq!(to_excel_serial(from_excel_serial(30.0)), 30.0);
    }
}
