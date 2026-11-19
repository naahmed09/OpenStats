//! Text functions. Character positions are 1-based and count Unicode
//! scalar values, not bytes.

use super::{def, texts, Category::Text, FunctionRegistry};
use crate::column::ColumnData;
use crate::error::{CoreError, Result};
use crate::formula::array::{at, at_s, common_len, Arr};
use crate::ops::extract::{text_op, TextOp};

fn out_text(v: Vec<String>, scalar: bool) -> Arr {
    Arr { data: ColumnData::Text(v), scalar }
}

/// Element-wise text → text with one numeric parameter per row.
fn text_num(a: &Arr, n: &Arr, f: impl Fn(&str, f64) -> Option<String>) -> Result<Arr> {
    let (len, scalar) = common_len(&[a, n], "text function")?;
    let s = texts(a, "first argument")?;
    let k = n.to_nums("count")?;
    Ok(out_text(
        (0..len)
            .map(|i| {
                let (t, x) = (at_s(&s, i), at(&k, i));
                if t.is_empty() || x.is_nan() {
                    String::new()
                } else {
                    f(t, x).unwrap_or_default()
                }
            })
            .collect(),
        scalar,
    ))
}

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn find_impl(args: &[Arr], case_insensitive: bool) -> Result<Arr> {
    let refs: Vec<&Arr> = args.iter().collect();
    let (n, scalar) = common_len(&refs, "FIND")?;
    let needle = texts(&args[0], "search text")?;
    let hay = texts(&args[1], "text")?;
    let start = match args.get(2) {
        Some(s) => s.to_nums("start")?,
        None => vec![1.0],
    };
    let out = (0..n)
        .map(|i| {
            let (nd, hy, st) = (at_s(&needle, i), at_s(&hay, i), at(&start, i));
            if hy.is_empty() || st.is_nan() || st < 1.0 {
                return f64::NAN;
            }
            let (nd, hy) = if case_insensitive {
                (nd.to_lowercase(), hy.to_lowercase())
            } else {
                (nd.to_string(), hy.to_string())
            };
            let skip = st as usize - 1;
            let byte_start = match hy.char_indices().nth(skip) {
                Some((b, _)) => b,
                None => return 0.0,
            };
            match hy[byte_start..].find(&nd) {
                Some(b) => (hy[..byte_start + b].chars().count() + 1) as f64,
                None => 0.0,
            }
        })
        .collect();
    Ok(Arr::nums(out, scalar))
}

pub(super) fn register(r: &mut FunctionRegistry) {
    r.register(def(
        "CONCATENATE",
        1,
        None,
        Text,
        "CONCATENATE(a, b, …): join text (numbers are formatted)",
        |a, _| {
            let refs: Vec<&Arr> = a.iter().collect();
            let (n, scalar) = common_len(&refs, "CONCATENATE")?;
            let parts: Vec<Vec<String>> = a.iter().map(Arr::to_texts).collect();
            Ok(out_text((0..n).map(|i| parts.iter().map(|p| at_s(p, i)).collect::<String>()).collect(), scalar))
        },
    ));
    r.register(def("LEFT", 2, Some(2), Text, "LEFT(text, n): first n characters", |a, _| {
        text_num(&a[0], &a[1], |t, k| Some(t.chars().take(k.max(0.0) as usize).collect()))
    }));
    r.register(def("RIGHT", 2, Some(2), Text, "RIGHT(text, n): last n characters", |a, _| {
        text_num(&a[0], &a[1], |t, k| {
            let c = chars(t);
            Some(c[c.len().saturating_sub(k.max(0.0) as usize)..].iter().collect())
        })
    }));
    r.register(def("MID", 3, Some(3), Text, "MID(text, start, n): n characters from position start", |a, _| {
        let (n, scalar) = common_len(&[&a[0], &a[1], &a[2]], "MID")?;
        let s = texts(&a[0], "text")?;
        let st = a[1].to_nums("start")?;
        let ln = a[2].to_nums("length")?;
        Ok(out_text(
            (0..n)
                .map(|i| {
                    let (t, p, l) = (at_s(&s, i), at(&st, i), at(&ln, i));
                    if p.is_nan() || l.is_nan() || p < 1.0 || l < 0.0 {
                        String::new()
                    } else {
                        t.chars().skip(p as usize - 1).take(l as usize).collect()
                    }
                })
                .collect(),
            scalar,
        ))
    }));
    r.register(def("LEN", 1, Some(1), Text, "LEN(text): number of characters", |a, _| {
        let s = texts(&a[0], "text")?;
        Ok(Arr::nums(s.iter().map(|t| t.chars().count() as f64).collect(), a[0].scalar))
    }));
    r.register(def(
        "FIND",
        2,
        Some(3),
        Text,
        "FIND(search, text, [start]): position of search (case-sensitive), 0 if absent",
        |a, _| find_impl(a, false),
    ));
    r.register(def("SEARCH", 2, Some(3), Text, "SEARCH(search, text, [start]): case-insensitive FIND", |a, _| {
        find_impl(a, true)
    }));
    r.register(def(
        "SUBSTITUTE",
        3,
        Some(4),
        Text,
        "SUBSTITUTE(text, old, new, [instance]): replace occurrences",
        |a, _| {
            let refs: Vec<&Arr> = a.iter().collect();
            let (n, scalar) = common_len(&refs, "SUBSTITUTE")?;
            let s = texts(&a[0], "text")?;
            let old = texts(&a[1], "old text")?;
            let new = texts(&a[2], "new text")?;
            let inst = match a.get(3) {
                Some(x) => Some(x.to_nums("instance")?),
                None => None,
            };
            Ok(out_text(
                (0..n)
                    .map(|i| {
                        let (t, o, w) = (at_s(&s, i), at_s(&old, i), at_s(&new, i));
                        if o.is_empty() {
                            return t.to_string();
                        }
                        match &inst {
                            None => t.replace(o, w),
                            Some(k) => {
                                let k = at(k, i);
                                if k.is_nan() || k < 1.0 {
                                    return t.to_string();
                                }
                                match t.match_indices(o).nth(k as usize - 1) {
                                    Some((b, _)) => format!("{}{}{}", &t[..b], w, &t[b + o.len()..]),
                                    None => t.to_string(),
                                }
                            }
                        }
                    })
                    .collect(),
                scalar,
            ))
        },
    ));
    r.register(def(
        "REPLACE",
        4,
        Some(4),
        Text,
        "REPLACE(text, start, n, new): replace n characters at start",
        |a, _| {
            let refs: Vec<&Arr> = a.iter().collect();
            let (n, scalar) = common_len(&refs, "REPLACE")?;
            let s = texts(&a[0], "text")?;
            let st = a[1].to_nums("start")?;
            let ln = a[2].to_nums("count")?;
            let new = texts(&a[3], "new text")?;
            Ok(out_text(
                (0..n)
                    .map(|i| {
                        let c = chars(at_s(&s, i));
                        let (p, l) = (at(&st, i), at(&ln, i));
                        if p.is_nan() || l.is_nan() || p < 1.0 || l < 0.0 {
                            return String::new();
                        }
                        let p = (p as usize - 1).min(c.len());
                        let e = (p + l as usize).min(c.len());
                        c[..p].iter().collect::<String>() + at_s(&new, i) + &c[e..].iter().collect::<String>()
                    })
                    .collect(),
                scalar,
            ))
        },
    ));
    macro_rules! textop {
        ($name:literal, $help:literal, $op:expr) => {
            r.register(def($name, 1, Some(1), Text, $help, |a, _| {
                let s = texts(&a[0], "text")?;
                Ok(Arr { data: text_op(&ColumnData::Text(s), &$op)?, scalar: a[0].scalar })
            }));
        };
    }
    textop!("TRIM", "TRIM(text): remove leading/trailing spaces", TextOp::Trim);
    textop!("UPPER", "UPPER(text): upper case", TextOp::Upper);
    textop!("LOWER", "LOWER(text): lower case", TextOp::Lower);
    textop!("PROPER", "PROPER(text): capitalise each word", TextOp::Proper);
    r.register(def("REPEAT", 2, Some(2), Text, "REPEAT(text, n): text repeated n times", |a, _| {
        text_num(&a[0], &a[1], |t, k| if (0.0..=1e6).contains(&k) { Some(t.repeat(k as usize)) } else { None })
    }));
    r.register(def("NUMTEXT", 1, Some(2), Text, "NUMTEXT(x, [decimals]): number to text", |a, _| {
        let x = a[0].to_nums("number")?;
        let d = match a.get(1) {
            Some(d) => Some(super::scalar_int(d, "decimals")?),
            None => None,
        };
        if d.is_some_and(|d| !(0..=20).contains(&d)) {
            return Err(CoreError::invalid("decimals must be between 0 and 20"));
        }
        Ok(out_text(
            x.iter()
                .map(|&v| {
                    if v.is_nan() {
                        String::new()
                    } else {
                        match d {
                            Some(d) => format!("{:.*}", d as usize, v),
                            None => crate::value::format_number(v),
                        }
                    }
                })
                .collect(),
            a[0].scalar,
        ))
    }));
    r.register(def("TEXTNUM", 1, Some(1), Text, "TEXTNUM(text): text to number (missing if not numeric)", |a, _| {
        let s = texts(&a[0], "text")?;
        Ok(Arr::nums(s.iter().map(|t| crate::value::parse_number(t).unwrap_or(f64::NAN)).collect(), a[0].scalar))
    }));
    r.register(def("EXACT", 2, Some(2), Text, "EXACT(a, b): 1 if texts are identical (case-sensitive)", |a, _| {
        let (n, scalar) = common_len(&[&a[0], &a[1]], "EXACT")?;
        let (x, y) = (a[0].to_texts(), a[1].to_texts());
        Ok(Arr::nums((0..n).map(|i| f64::from(u8::from(at_s(&x, i) == at_s(&y, i)))).collect(), scalar))
    }));
    r.register(def("CHAR", 1, Some(1), Text, "CHAR(code): character with the given Unicode code", |a, _| {
        let x = a[0].to_nums("code")?;
        Ok(out_text(
            x.iter()
                .map(|&c| {
                    if c.is_finite() && c >= 1.0 {
                        char::from_u32(c as u32).map(String::from).unwrap_or_default()
                    } else {
                        String::new()
                    }
                })
                .collect(),
            a[0].scalar,
        ))
    }));
    r.register(def("CODE", 1, Some(1), Text, "CODE(text): Unicode code of the first character", |a, _| {
        let s = texts(&a[0], "text")?;
        Ok(Arr::nums(s.iter().map(|t| t.chars().next().map_or(f64::NAN, |c| c as u32 as f64)).collect(), a[0].scalar))
    }));
    r.register(def("WORD", 2, Some(2), Text, "WORD(text, n): n-th space-separated word", |a, _| {
        text_num(
            &a[0],
            &a[1],
            |t, k| if k >= 1.0 { t.split_whitespace().nth(k as usize - 1).map(String::from) } else { None },
        )
    }));
    r.register(def("WORDCOUNT", 1, Some(1), Text, "WORDCOUNT(text): number of space-separated words", |a, _| {
        let s = texts(&a[0], "text")?;
        Ok(Arr::nums(s.iter().map(|t| t.split_whitespace().count() as f64).collect(), a[0].scalar))
    }));

    r.alias("LENGTH", "LEN");
    r.alias("SUBSTRING", "MID");
    r.alias("REPT", "REPEAT");
    r.alias("VALUE", "TEXTNUM");
    r.alias("TEXT", "NUMTEXT");
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::formula::evaluate;
    use crate::worksheet::Worksheet;

    #[test]
    fn text_functions() {
        let ws = Worksheet::with_columns(
            "s",
            vec![Column::text("t", vec!["Lot-42 ok", "", "Ünïcode"]), Column::numeric("n", vec![2.0, 3.0, 4.0])],
        )
        .unwrap();
        let txt = |f: &str| evaluate(f, &ws).unwrap().data.to_text_vec();
        let num = |f: &str| evaluate(f, &ws).unwrap().data.to_f64_vec();
        assert_eq!(txt("CONCATENATE(t, \"#\", n)"), ["Lot-42 ok#2", "#3", "Ünïcode#4"]);
        assert_eq!(txt("LEFT(t, n)"), ["Lo", "", "Ünïc"]);
        assert_eq!(txt("RIGHT(t, 2)")[0], "ok");
        assert_eq!(txt("MID(t, 5, 2)")[0], "42");
        assert_eq!(num("LEN(t)"), [9.0, 0.0, 7.0]);
        assert_eq!(num("FIND(\"o\", t)")[0], 2.0);
        assert_eq!(num("FIND(\"o\", t, 3)")[0], 8.0);
        assert_eq!(num("SEARCH(\"CODE\", t)")[2], 4.0);
        assert_eq!(num("FIND(\"zz\", t)")[0], 0.0);
        assert!(num("FIND(\"o\", t)")[1].is_nan());
        assert_eq!(txt("SUBSTITUTE(\"a-b-c\", \"-\", \"+\", 2)"), ["a-b+c"]);
        assert_eq!(txt("SUBSTITUTE(\"a-b-c\", \"-\", \"\")"), ["abc"]);
        assert_eq!(txt("REPLACE(\"abcdef\", 2, 3, \"X\")"), ["aXef"]);
        assert_eq!(txt("UPPER(t)")[2], "ÜNÏCODE");
        assert_eq!(txt("NUMTEXT(PI(), 3)"), ["3.142"]);
        assert_eq!(num("TEXTNUM(\" 1,234.5 \")"), [1234.5]);
        assert_eq!(txt("WORD(t, 2)")[0], "ok");
        assert_eq!(num("CODE(\"A\")"), [65.0]);
        assert_eq!(txt("CHAR(233)"), ["é"]);
        assert_eq!(txt("REPEAT(\"ab\", 3)"), ["ababab"]);
        assert!(evaluate("LEN(n)", &ws).is_err());
    }
}
