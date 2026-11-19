//! Row statistics across several columns (missing values ignored).

use super::stats;
use super::{def, Category::RowStatistics, FunctionRegistry};
use crate::error::Result;
use crate::formula::array::{at, common_len, fin, Arr};

/// Apply `f` to the non-missing values of each row of `args`.
fn row_apply(args: &[Arr], f: fn(&[f64]) -> f64) -> Result<Arr> {
    let refs: Vec<&Arr> = args.iter().collect();
    let (n, scalar) = common_len(&refs, "row statistic")?;
    let cols = args.iter().map(|a| a.to_nums("row statistic")).collect::<Result<Vec<_>>>()?;
    let mut buf = Vec::with_capacity(cols.len());
    let out = (0..n)
        .map(|i| {
            buf.clear();
            buf.extend(cols.iter().map(|c| at(c, i)).filter(|v| !v.is_nan()));
            fin(f(&buf))
        })
        .collect();
    Ok(Arr::nums(out, scalar))
}

/// Count per row of missing (or present) values, valid for any column type.
fn row_count(args: &[Arr], missing: bool) -> Result<Arr> {
    let refs: Vec<&Arr> = args.iter().collect();
    let (n, scalar) = common_len(&refs, "row statistic")?;
    let out = (0..n)
        .map(|i| {
            args.iter()
                .filter(|a| {
                    let j = if a.scalar { 0 } else { i };
                    a.data.is_missing(j) == missing
                })
                .count() as f64
        })
        .collect();
    Ok(Arr::nums(out, scalar))
}

pub(super) fn register(r: &mut FunctionRegistry) {
    macro_rules! rowfn {
        ($name:literal, $help:literal, $f:expr) => {
            r.register(def($name, 1, None, RowStatistics, $help, |a, _| row_apply(a, $f)));
        };
    }
    rowfn!("RSUM", "RSUM(c1, c2, …): row sums", |x| if x.is_empty() { f64::NAN } else { stats::sum(x) });
    rowfn!("RMEAN", "RMEAN(c1, c2, …): row means", stats::mean);
    rowfn!("RMEDIAN", "RMEDIAN(c1, c2, …): row medians", stats::median);
    rowfn!("RSTDEV", "RSTDEV(c1, c2, …): row standard deviations", stats::stdev);
    rowfn!("RMINIMUM", "RMINIMUM(c1, c2, …): row minimums", |x| x.iter().copied().fold(f64::NAN, f64::min));
    rowfn!("RMAXIMUM", "RMAXIMUM(c1, c2, …): row maximums", |x| x.iter().copied().fold(f64::NAN, f64::max));
    rowfn!("RRANGE", "RRANGE(c1, c2, …): row ranges", |x| {
        x.iter().copied().fold(f64::NAN, f64::max) - x.iter().copied().fold(f64::NAN, f64::min)
    });
    rowfn!("RSSQ", "RSSQ(c1, c2, …): row sums of squares", |x| {
        if x.is_empty() {
            f64::NAN
        } else {
            stats::sum(&x.iter().map(|v| v * v).collect::<Vec<_>>())
        }
    });
    rowfn!("RPRODUCT", "RPRODUCT(c1, c2, …): row products", |x| if x.is_empty() {
        f64::NAN
    } else {
        x.iter().product()
    });
    r.register(def("RN", 1, None, RowStatistics, "RN(c1, c2, …): non-missing count per row", |a, _| {
        row_count(a, false)
    }));
    r.register(def("RNMISS", 1, None, RowStatistics, "RNMISS(c1, c2, …): missing count per row", |a, _| {
        row_count(a, true)
    }));
    r.register(def("RCOUNT", 1, None, RowStatistics, "RCOUNT(c1, c2, …): number of columns per row", |a, _| {
        let refs: Vec<&Arr> = a.iter().collect();
        let (n, scalar) = common_len(&refs, "RCOUNT")?;
        Ok(Arr::nums(vec![a.len() as f64; n], scalar))
    }));
    r.alias("RMIN", "RMINIMUM");
    r.alias("RMAX", "RMAXIMUM");
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::formula::evaluate;
    use crate::worksheet::Worksheet;

    #[test]
    fn rows() {
        let ws = Worksheet::with_columns(
            "s",
            vec![Column::numeric("a", vec![1.0, f64::NAN, 3.0]), Column::numeric("b", vec![3.0, f64::NAN, 5.0])],
        )
        .unwrap();
        let g = |f: &str| evaluate(f, &ws).unwrap().data.to_f64_vec();
        assert_eq!(g("RMEAN(a, b)")[0], 2.0);
        assert!(g("RMEAN(a, b)")[1].is_nan());
        assert_eq!(g("RSUM(a, b, 10)")[1], 10.0);
        assert_eq!(g("RNMISS(a, b)"), [0.0, 2.0, 0.0]);
        assert_eq!(g("RN(a, b, *)"), [2.0, 0.0, 2.0]);
        assert_eq!(g("RRANGE(a, b)")[2], 2.0);
        assert_eq!(g("RCOUNT(a, b)"), [2.0, 2.0, 2.0]);
        assert!((g("RSTDEV(a, b)")[0] - 2f64.sqrt()).abs() < 1e-15);
    }
}
