//! Data > Rank.

/// How tied values are ranked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TieMethod {
    /// Mean of the ranks the ties span (Minitab default).
    #[default]
    Average,
    /// Lowest rank of the tie group.
    Min,
    /// Highest rank of the tie group.
    Max,
    /// Ranks assigned in order of appearance.
    Ordinal,
    /// Consecutive integers without gaps (1, 2, 2, 3).
    Dense,
}

/// Rank numeric data (1 = smallest, or largest with `descending`). Missing
/// values stay missing and are excluded from ranking.
pub fn rank(x: &[f64], ties: TieMethod, descending: bool) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).filter(|&i| !x[i].is_nan()).collect();
    idx.sort_by(|&a, &b| {
        let o = x[a].total_cmp(&x[b]);
        let o = if descending { o.reverse() } else { o };
        o.then(a.cmp(&b))
    });
    let mut out = vec![f64::NAN; x.len()];
    let mut i = 0;
    let mut dense = 0.0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        dense += 1.0;
        let lo = (i + 1) as f64;
        let hi = j as f64;
        for (k, &r) in idx[i..j].iter().enumerate() {
            out[r] = match ties {
                TieMethod::Average => (lo + hi) / 2.0,
                TieMethod::Min => lo,
                TieMethod::Max => hi,
                TieMethod::Ordinal => lo + k as f64,
                TieMethod::Dense => dense,
            };
        }
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tie_methods() {
        let x = [10.0, 20.0, 10.0, f64::NAN, 5.0];
        let avg = rank(&x, TieMethod::Average, false);
        assert_eq!(&avg[..3], &[2.5, 4.0, 2.5]);
        assert!(avg[3].is_nan());
        assert_eq!(avg[4], 1.0);
        assert_eq!(rank(&x, TieMethod::Min, false)[0], 2.0);
        assert_eq!(rank(&x, TieMethod::Max, false)[2], 3.0);
        let o = rank(&x, TieMethod::Ordinal, false);
        assert_eq!((o[0], o[2]), (2.0, 3.0));
        assert_eq!(rank(&x, TieMethod::Dense, false)[1], 3.0);
        assert_eq!(rank(&x, TieMethod::Average, true)[1], 1.0);
    }
}
