//! Logical and missing-value functions.

use super::{def, Category::Logical, FunctionRegistry};
use crate::column::{ColumnData, ColumnKind};
use crate::error::{CoreError, Result};
use crate::formula::array::{at, common_len, Arr};
use crate::formula::eval::truth;

/// Common result kind for a set of branches: identical kinds are kept;
/// all-missing branches adopt the others' kind; otherwise an error.
fn unify(branches: &[&Arr]) -> Result<ColumnKind> {
    let mut kind: Option<ColumnKind> = None;
    for b in branches {
        if b.all_missing() && b.kind() == ColumnKind::Numeric {
            continue;
        }
        match kind {
            None => kind = Some(b.kind()),
            Some(k) if k == b.kind() => {}
            Some(k) => {
                return Err(CoreError::invalid(format!(
                    "results must have the same type ({} vs {})",
                    k.name(),
                    b.kind().name()
                )))
            }
        }
    }
    Ok(kind.unwrap_or(ColumnKind::Numeric))
}

/// IF(test1, value1, [test2, value2, …], [else]).
fn if_fn(a: &[Arr]) -> Result<Arr> {
    let pairs = a.len() / 2;
    let has_else = a.len() % 2 == 1;
    let refs: Vec<&Arr> = a.iter().collect();
    let (n, scalar) = common_len(&refs, "IF")?;
    let mut branches: Vec<&Arr> = (0..pairs).map(|p| &a[2 * p + 1]).collect();
    if has_else {
        branches.push(&a[a.len() - 1]);
    }
    let kind = unify(&branches)?;
    let tests = (0..pairs).map(|p| a[2 * p].to_nums("IF test")).collect::<Result<Vec<_>>>()?;
    let vals = branches.iter().map(|b| b.coerce_to(kind, "IF")).collect::<Result<Vec<_>>>()?;
    let mut out = ColumnData::missing(kind, n);
    for i in 0..n {
        let mut chosen: Option<usize> = None;
        let mut undetermined = false;
        for (p, t) in tests.iter().enumerate() {
            match truth(at(t, i)) {
                Some(true) => {
                    chosen = Some(p);
                    break;
                }
                Some(false) => {}
                None => {
                    undetermined = true;
                    break;
                }
            }
        }
        if chosen.is_none() && !undetermined && has_else {
            chosen = Some(pairs);
        }
        if let Some(c) = chosen {
            let v = &vals[c];
            out.set(i, &v.data.get(if v.scalar { 0 } else { i }))?;
        }
    }
    Ok(Arr { data: out, scalar })
}

fn fold_logic(a: &[Arr], and: bool) -> Result<Arr> {
    let refs: Vec<&Arr> = a.iter().collect();
    let (n, scalar) = common_len(&refs, if and { "AND" } else { "OR" })?;
    let cols = a.iter().map(|x| x.to_nums("logical")).collect::<Result<Vec<_>>>()?;
    let out = (0..n)
        .map(|i| {
            let mut unknown = false;
            for c in &cols {
                match truth(at(c, i)) {
                    Some(v) if v != and => return if and { 0.0 } else { 1.0 },
                    None => unknown = true,
                    _ => {}
                }
            }
            if unknown {
                f64::NAN
            } else if and {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Ok(Arr::nums(out, scalar))
}

pub(super) fn register(r: &mut FunctionRegistry) {
    r.register(def(
        "IF",
        2,
        None,
        Logical,
        "IF(test, value, [test2, value2 …], [else]): conditional values",
        |a, _| if_fn(a),
    ));
    r.register(def("AND", 1, None, Logical, "AND(a, b, …): 1 if all are true", |a, _| fold_logic(a, true)));
    r.register(def("OR", 1, None, Logical, "OR(a, b, …): 1 if any is true", |a, _| fold_logic(a, false)));
    r.register(def("NOT", 1, Some(1), Logical, "NOT(a): logical negation", |a, _| {
        super::map1(&a[0], |x| match truth(x) {
            Some(t) => f64::from(u8::from(!t)),
            None => f64::NAN,
        })
    }));
    r.register(def("XOR", 2, Some(2), Logical, "XOR(a, b): 1 if exactly one is true", |a, _| {
        super::map2(&a[0], &a[1], |x, y| match (truth(x), truth(y)) {
            (Some(p), Some(q)) => f64::from(u8::from(p != q)),
            _ => f64::NAN,
        })
    }));
    r.register(def("ISMISSING", 1, Some(1), Logical, "ISMISSING(x): 1 if the value is missing", |a, _| {
        let n = a[0].len();
        Ok(Arr::nums((0..n).map(|i| f64::from(u8::from(a[0].data.is_missing(i)))).collect(), a[0].scalar))
    }));
    r.register(def("COALESCE", 1, None, Logical, "COALESCE(a, b, …): first non-missing value per row", |a, _| {
        let refs: Vec<&Arr> = a.iter().collect();
        let (n, scalar) = common_len(&refs, "COALESCE")?;
        let kind = unify(&refs)?;
        let vals = a.iter().map(|x| x.coerce_to(kind, "COALESCE")).collect::<Result<Vec<_>>>()?;
        let mut out = ColumnData::missing(kind, n);
        for i in 0..n {
            if let Some(v) = vals.iter().map(|v| v.data.get(if v.scalar { 0 } else { i })).find(|v| !v.is_missing()) {
                out.set(i, &v)?;
            }
        }
        Ok(Arr { data: out, scalar })
    }));
    r.register(def("MISSING", 0, Some(0), Logical, "MISSING(): the missing value *", |_, _| Ok(Arr::missing())));
    r.alias("MISS", "ISMISSING");
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::formula::evaluate;
    use crate::worksheet::Worksheet;

    #[test]
    fn conditionals() {
        let ws = Worksheet::with_columns(
            "s",
            vec![Column::numeric("x", vec![1.0, 5.0, 10.0, f64::NAN]), Column::text("t", vec!["a", "", "c", "d"])],
        )
        .unwrap();
        let r = evaluate("IF(x < 3, \"low\", x < 7, \"mid\", \"high\")", &ws).unwrap();
        assert_eq!(r.data.as_text().unwrap(), ["low", "mid", "high", ""]);
        let r = evaluate("IF(x > 2, x * 10)", &ws).unwrap();
        assert!(r.data.is_missing(0));
        assert_eq!(r.data.get(1).as_f64(), 50.0);
        assert!(evaluate("IF(x > 2, \"a\", 1)", &ws).is_err());
        let r = evaluate("COALESCE(t, \"none\")", &ws).unwrap();
        assert_eq!(r.data.as_text().unwrap()[1], "none");
        let r = evaluate("ISMISSING(t) + ISMISSING(x)", &ws).unwrap();
        assert_eq!(r.data.to_f64_vec(), [0.0, 1.0, 0.0, 1.0]);
        let r = evaluate("AND(x > 0, x < 8, 1)", &ws).unwrap();
        assert_eq!(r.data.to_f64_vec()[..3], [1.0, 1.0, 0.0]);
        assert!(r.data.is_missing(3));
        assert_eq!(evaluate("OR(0, *, 1)", &ws).unwrap().data.to_f64_vec(), [1.0]);
        assert_eq!(evaluate("XOR(1, 0)", &ws).unwrap().data.to_f64_vec(), [1.0]);
    }
}
