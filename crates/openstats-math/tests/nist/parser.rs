//! Parser for NIST StRD `.dat` files (all four procedures).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Univariate,
    Anova,
    Linear,
    Nonlinear,
}

#[derive(Debug, Clone, Default)]
pub struct AnovaRow {
    pub df: f64,
    pub ss: f64,
    pub ms: f64,
    pub f: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub kind: Kind,
    /// Right-hand side of the model equation (without `y =` and `+ e`).
    pub model: String,
    /// `(name, certified estimate, certified standard deviation)`.
    pub params: Vec<(String, f64, f64)>,
    /// Nonlinear starting values: `starts[0]` is Start 1, `starts[1]` Start 2.
    pub starts: Vec<Vec<f64>>,
    pub rss: Option<f64>,
    pub resid_sd: Option<f64>,
    pub dof: Option<f64>,
    pub nobs: Option<usize>,
    pub r2: Option<f64>,
    pub mean: Option<f64>,
    pub sd: Option<f64>,
    pub acf1: Option<f64>,
    pub between: Option<AnovaRow>,
    pub within: Option<AnovaRow>,
    #[allow(dead_code)] // kept for diagnostics
    pub columns: Vec<String>,
    /// Data by column.
    pub data: Vec<Vec<f64>>,
}

fn num(tok: &str) -> Option<f64> {
    let t = tok.trim_matches(|c: char| c == ',' || c == ':');
    if t.eq_ignore_ascii_case("infinity") || t.eq_ignore_ascii_case("inf") {
        return Some(f64::INFINITY);
    }
    t.parse::<f64>().ok()
}

/// Numeric tokens of a line, in order.
fn nums(line: &str) -> Vec<f64> {
    line.split_whitespace().filter_map(num).collect()
}

fn last_num(line: &str) -> Option<f64> {
    nums(line).last().copied()
}

fn is_param_name(t: &str) -> bool {
    let mut c = t.chars();
    matches!(c.next(), Some('b' | 'B')) && t.len() > 1 && c.all(|x| x.is_ascii_digit())
}

pub fn parse(text: &str, name: &str) -> Result<Dataset, String> {
    let lines: Vec<&str> = text.lines().collect();
    let lower = |l: &str| l.trim().to_ascii_lowercase();

    let procedure = lines
        .iter()
        .find_map(|l| l.trim().strip_prefix("Procedure:").map(|r| r.trim().to_ascii_lowercase()))
        .ok_or("no Procedure: line")?;
    let kind = if procedure.starts_with("linear") {
        Kind::Linear
    } else if procedure.starts_with("nonlinear") {
        Kind::Nonlinear
    } else if procedure.starts_with("analysis") {
        Kind::Anova
    } else if procedure.starts_with("univariate") {
        Kind::Univariate
    } else {
        return Err(format!("unknown procedure {procedure:?}"));
    };

    // The data header is the last "Data:" line that is not the descriptive
    // "Data: 1 Response ..." line, i.e. whose remainder does not start with a digit.
    let data_at = lines
        .iter()
        .rposition(|l| {
            l.trim()
                .strip_prefix("Data:")
                .is_some_and(|r| r.trim().chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
        })
        .ok_or("no Data: header")?;

    let mut ds = Dataset {
        name: name.to_string(),
        kind,
        model: String::new(),
        params: Vec::new(),
        starts: vec![Vec::new(), Vec::new()],
        rss: None,
        resid_sd: None,
        dof: None,
        nobs: None,
        r2: None,
        mean: None,
        sd: None,
        acf1: None,
        between: None,
        within: None,
        columns: lines[data_at].trim().trim_start_matches("Data:").split_whitespace().map(str::to_string).collect(),
        data: Vec::new(),
    };

    let head = &lines[..data_at];

    // Model equation: from the first line starting `y =` to the next blank line.
    if let Some(start) = head.iter().position(|l| {
        let t = l.trim_start();
        t.starts_with('y') && t[1..].trim_start().starts_with('=')
    }) {
        let mut s = String::new();
        for l in &head[start..] {
            if l.trim().is_empty() || l.trim_start().starts_with("Note:") {
                break;
            }
            s.push(' ');
            s.push_str(l.trim());
        }
        let s = s.trim();
        let rhs = s[s.find('=').unwrap() + 1..].trim();
        let rhs = rhs.strip_suffix('e').map(str::trim_end).and_then(|r| r.strip_suffix('+')).unwrap_or(rhs);
        ds.model = rhs.trim().to_string();
    }

    let mut resid_label_pending = false;
    for (i, raw) in head.iter().enumerate() {
        let l = raw.trim();
        let low = lower(l);
        let toks: Vec<&str> = l.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        match kind {
            Kind::Univariate => {
                if low.starts_with("sample mean") {
                    ds.mean = last_num(l);
                } else if low.starts_with("sample standard deviation") {
                    ds.sd = last_num(l);
                } else if low.starts_with("sample autocorrelation") {
                    ds.acf1 = last_num(l);
                }
            }
            Kind::Anova => {
                let n = nums(l);
                if low.starts_with("between") && n.len() >= 3 {
                    ds.between = Some(AnovaRow { df: n[0], ss: n[1], ms: n[2], f: n.get(3).copied() });
                } else if low.starts_with("within") && n.len() >= 3 {
                    ds.within = Some(AnovaRow { df: n[0], ss: n[1], ms: n[2], f: None });
                } else if low.starts_with("certified r-squared") {
                    ds.r2 = last_num(l);
                } else if low.starts_with("certified residual") && n.is_empty() {
                    resid_label_pending = true;
                } else if resid_label_pending && low.starts_with("standard deviation") {
                    ds.resid_sd = last_num(l);
                    resid_label_pending = false;
                } else if low.starts_with("certified residual standard deviation") {
                    ds.resid_sd = last_num(l);
                }
            }
            Kind::Linear => {
                if is_param_name(toks[0]) && toks.len() >= 3 {
                    if let (Some(e), Some(s)) = (num(toks[1]), num(toks[2])) {
                        ds.params.push((toks[0].to_string(), e, s));
                    }
                } else if low.starts_with("standard deviation") && last_num(l).is_some() {
                    ds.resid_sd = last_num(l);
                } else if low.starts_with("r-squared") {
                    ds.r2 = last_num(l);
                } else if low == "residual"
                    && head.get(i + 1).is_some_and(|n| lower(n).starts_with("standard deviation"))
                {
                    // value is on the next line; handled above
                }
            }
            Kind::Nonlinear => {
                if is_param_name(toks[0]) && toks.get(1) == Some(&"=") {
                    let n = nums(&l[l.find('=').unwrap() + 1..]);
                    if n.len() >= 4 {
                        ds.starts[0].push(n[0]);
                        ds.starts[1].push(n[1]);
                        ds.params.push((toks[0].to_string(), n[2], n[3]));
                    }
                } else if low.starts_with("residual sum of squares") {
                    ds.rss = last_num(l);
                } else if low.starts_with("residual standard deviation") {
                    ds.resid_sd = last_num(l);
                } else if low.starts_with("degrees of freedom") {
                    ds.dof = last_num(l);
                } else if low.starts_with("number of observations") {
                    ds.nobs = last_num(l).map(|v| v as usize);
                }
            }
        }
    }

    for l in &lines[data_at + 1..] {
        let toks: Vec<&str> = l.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        let row: Option<Vec<f64>> = toks.iter().map(|t| num(t)).collect();
        if let Some(r) = row {
            if ds.data.is_empty() {
                ds.data = vec![Vec::new(); r.len()];
            }
            if r.len() != ds.data.len() {
                return Err(format!("ragged data row {l:?}"));
            }
            for (c, v) in ds.data.iter_mut().zip(r) {
                c.push(v);
            }
        }
    }
    if ds.data.is_empty() {
        return Err("no data rows".into());
    }
    Ok(ds)
}
