//! Agglomerative clustering (observations and variables) and k-means.

use super::common::{corr_from_cov, covariance, listwise, mean, names_or_default};
use crate::error::{invalid, Result, StatsError};
use openstats_math::random::{Pcg64, Rng, SeedableRng};
use openstats_math::Matrix;

/// Distance between observations (rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    SquaredEuclidean,
    Manhattan,
    /// 1 − r between the two rows treated as profiles.
    Pearson,
    /// 1 − r².
    SquaredPearson,
}

impl Distance {
    fn is_squared(self) -> bool {
        matches!(self, Distance::SquaredEuclidean | Distance::SquaredPearson)
    }

    fn between(self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Distance::Euclidean => a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt(),
            Distance::SquaredEuclidean => a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum(),
            Distance::Manhattan => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
            Distance::Pearson | Distance::SquaredPearson => {
                let ma = mean(a);
                let mb = mean(b);
                let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
                for (x, y) in a.iter().zip(b) {
                    sab += (x - ma) * (y - mb);
                    saa += (x - ma).powi(2);
                    sbb += (y - mb).powi(2);
                }
                let r = if saa > 0.0 && sbb > 0.0 { sab / (saa * sbb).sqrt() } else { f64::NAN };
                if self == Distance::Pearson {
                    1.0 - r
                } else {
                    1.0 - r * r
                }
            }
        }
    }
}

/// Linkage (amalgamation) method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Centroid,
    Median,
    Ward,
    McQuitty,
}

impl LinkageMethod {
    /// Centroid, median and Ward are geometric: their updates act on squared Euclidean distances.
    fn geometric(self) -> bool {
        matches!(self, LinkageMethod::Centroid | LinkageMethod::Median | LinkageMethod::Ward)
    }
}

/// One amalgamation step.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeStep {
    /// Clusters joined, as ids: observations are `0..n`, the cluster created at step `s` (1-based) is `n + s - 1`.
    pub left: usize,
    pub right: usize,
    /// Smallest observation numbers (1-based) of the two clusters, as Minitab labels them.
    pub label_left: usize,
    pub label_right: usize,
    /// Label of the new cluster (the smaller of the two).
    pub label_new: usize,
    pub distance: f64,
    /// 100·(1 − distance / maximum distance).
    pub similarity: f64,
    /// Number of observations in the new cluster.
    pub size: usize,
}

/// A hierarchical clustering.
#[derive(Debug, Clone)]
pub struct Linkage {
    pub n: usize,
    pub method: LinkageMethod,
    pub steps: Vec<MergeStep>,
    /// Item names (observation numbers or variable names).
    pub items: Vec<String>,
    /// Leaf order for a dendrogram.
    pub order: Vec<usize>,
    /// Worksheet rows of the observations (empty for variables).
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    pub max_distance: f64,
}

impl Linkage {
    /// Membership (1-based cluster numbers, numbered by first member) when `k` clusters remain.
    pub fn cut_clusters(&self, k: usize) -> Result<Vec<usize>> {
        if k == 0 || k > self.n {
            return invalid(format!("the number of clusters must be between 1 and {}", self.n));
        }
        let merges = self.n - k;
        let mut parent: Vec<usize> = (0..self.n + self.steps.len()).collect();
        for (s, st) in self.steps.iter().take(merges).enumerate() {
            parent[st.left] = self.n + s;
            parent[st.right] = self.n + s;
        }
        let root = |mut i: usize| {
            while parent[i] != i {
                i = parent[i];
            }
            i
        };
        let mut ids: Vec<usize> = Vec::new();
        let mut out = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let r = root(i);
            let pos = match ids.iter().position(|&x| x == r) {
                Some(p) => p,
                None => {
                    ids.push(r);
                    ids.len() - 1
                }
            };
            out.push(pos + 1);
        }
        Ok(out)
    }

    /// Number of clusters left when merging stops before the first step whose similarity is below `level` (percent).
    pub fn clusters_at_similarity(&self, level: f64) -> usize {
        let done = self.steps.iter().take_while(|s| s.similarity >= level).count();
        self.n - done
    }

    pub fn cut_similarity(&self, level: f64) -> Result<Vec<usize>> {
        self.cut_clusters(self.clusters_at_similarity(level))
    }

    /// Dendrogram polylines: for each merge a U-shaped path (x = leaf position, y = distance).
    pub fn dendrogram_paths(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut pos = vec![0.0; self.n + self.steps.len()];
        for (i, &leaf) in self.order.iter().enumerate() {
            pos[leaf] = i as f64 + 1.0;
        }
        let mut height = vec![0.0; self.n + self.steps.len()];
        let mut out = Vec::with_capacity(self.steps.len());
        for (s, st) in self.steps.iter().enumerate() {
            let id = self.n + s;
            pos[id] = (pos[st.left] + pos[st.right]) / 2.0;
            height[id] = st.distance;
            let (xl, xr) = (pos[st.left], pos[st.right]);
            let (hl, hr) = (height[st.left], height[st.right]);
            out.push((vec![xl, xl, xr, xr], vec![hl, st.distance, st.distance, hr]));
        }
        out
    }
}

fn distance_matrix(rows: &[Vec<f64>], metric: Distance) -> Result<Matrix> {
    let n = rows.len();
    let mut d = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i + 1..n {
            let v = metric.between(&rows[i], &rows[j]);
            if !v.is_finite() {
                return invalid("a distance is undefined (a row has no variation, so its correlation is undefined)");
            }
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
    Ok(d)
}

const MAX_ITEMS: usize = 3000;

/// Agglomerative clustering of observations.
pub fn hierarchical(cols: &[Vec<f64>], method: LinkageMethod, metric: Distance, standardize: bool) -> Result<Linkage> {
    let c = listwise(cols)?;
    let n = c.rows.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "cluster analysis", needed: 2, got: n });
    }
    if n > MAX_ITEMS {
        return Err(StatsError::TooLarge(format!("hierarchical clustering is limited to {MAX_ITEMS} observations")));
    }
    let p = cols.len();
    let mut data: Vec<Vec<f64>> = (0..n).map(|i| (0..p).map(|v| c.cols[v][i]).collect()).collect();
    if standardize {
        standardize_rows(&mut data)?;
    }
    let d = distance_matrix(&data, metric)?;
    let items = c.rows.iter().map(|r| (r + 1).to_string()).collect();
    let mut l = linkage_from(&d, method, metric.is_squared(), items)?;
    l.rows_used = c.rows;
    l.n_dropped = cols[0].len() - n;
    Ok(l)
}

/// Agglomerative clustering of variables, from 1 − r or 1 − |r|.
pub fn cluster_variables(
    cols: &[Vec<f64>],
    names: &[String],
    method: LinkageMethod,
    absolute: bool,
) -> Result<Linkage> {
    let p = cols.len();
    if p < 2 {
        return invalid("clustering variables needs at least two variables");
    }
    let c = listwise(cols)?;
    if c.rows.len() < 3 {
        return Err(StatsError::InsufficientData { what: "variable clustering", needed: 3, got: c.rows.len() });
    }
    let r = corr_from_cov(&covariance(&c.cols))?;
    let d = Matrix::from_fn(p, p, |i, j| {
        if i == j {
            0.0
        } else if absolute {
            1.0 - r[(i, j)].abs()
        } else {
            1.0 - r[(i, j)]
        }
    });
    linkage_from(&d, method, false, names_or_default(names, p, "C"))
}

fn standardize_rows(data: &mut [Vec<f64>]) -> Result<()> {
    let n = data.len();
    let p = data[0].len();
    for v in 0..p {
        let m = data.iter().map(|r| r[v]).sum::<f64>() / n as f64;
        let sd = (data.iter().map(|r| (r[v] - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
        if !(sd > 0.0) {
            return invalid("a variable is constant, so it cannot be standardized");
        }
        for r in data.iter_mut() {
            r[v] = (r[v] - m) / sd;
        }
    }
    Ok(())
}

fn linkage_from(dist: &Matrix, method: LinkageMethod, already_squared: bool, items: Vec<String>) -> Result<Linkage> {
    let n = dist.rows();
    let square = method.geometric() && !already_squared;
    let mut d = if square { dist.map(|x| x * x) } else { dist.clone() };
    let mut active = vec![true; n];
    let mut size = vec![1usize; n];
    let mut id: Vec<usize> = (0..n).collect();
    let mut label: Vec<usize> = (1..=n).collect();
    // Nearest neighbour among active j > i.
    let mut nn = vec![usize::MAX; n];
    let find = |d: &Matrix, active: &[bool], i: usize| -> usize {
        let mut best = usize::MAX;
        let mut bd = f64::INFINITY;
        for j in i + 1..n {
            if active[j] && d[(i, j)] < bd {
                bd = d[(i, j)];
                best = j;
            }
        }
        best
    };
    for i in 0..n {
        nn[i] = find(&d, &active, i);
    }
    let mut steps = Vec::with_capacity(n - 1);
    for s in 0..n - 1 {
        // Global minimum over rows.
        let mut bi = usize::MAX;
        let mut bd = f64::INFINITY;
        for i in 0..n {
            if active[i] && nn[i] != usize::MAX && d[(i, nn[i])] < bd {
                bd = d[(i, nn[i])];
                bi = i;
            }
        }
        if bi == usize::MAX {
            return invalid("the distances are not finite");
        }
        let i = bi;
        let j = nn[i];
        let (na, nb) = (size[i] as f64, size[j] as f64);
        let dij = d[(i, j)];
        let height = if square { dij.max(0.0).sqrt() } else { dij };
        let (la, lb) = (label[i], label[j]);
        steps.push(MergeStep {
            left: id[i],
            right: id[j],
            label_left: la.min(lb),
            label_right: la.max(lb),
            label_new: la.min(lb),
            distance: height,
            similarity: 0.0,
            size: size[i] + size[j],
        });
        // Lance–Williams update into slot i.
        for k in 0..n {
            if !active[k] || k == i || k == j {
                continue;
            }
            let nk = size[k] as f64;
            let (dka, dkb) = (d[(k, i)], d[(k, j)]);
            let nab = na + nb;
            let (aa, ab, beta, gamma) = match method {
                LinkageMethod::Single => (0.5, 0.5, 0.0, -0.5),
                LinkageMethod::Complete => (0.5, 0.5, 0.0, 0.5),
                LinkageMethod::Average => (na / nab, nb / nab, 0.0, 0.0),
                LinkageMethod::McQuitty => (0.5, 0.5, 0.0, 0.0),
                LinkageMethod::Centroid => (na / nab, nb / nab, -na * nb / (nab * nab), 0.0),
                LinkageMethod::Median => (0.5, 0.5, -0.25, 0.0),
                LinkageMethod::Ward => {
                    let t = nab + nk;
                    ((na + nk) / t, (nb + nk) / t, -nk / t, 0.0)
                }
            };
            let v = aa * dka + ab * dkb + beta * dij + gamma * (dka - dkb).abs();
            d[(k, i)] = v;
            d[(i, k)] = v;
        }
        active[j] = false;
        size[i] += size[j];
        label[i] = la.min(lb);
        id[i] = n + s;
        // Refresh nearest neighbours.
        nn[i] = find(&d, &active, i);
        for r in 0..n {
            if !active[r] || r == i {
                continue;
            }
            if r < i {
                if nn[r] == i || nn[r] == j || nn[r] == usize::MAX {
                    nn[r] = find(&d, &active, r);
                } else if d[(r, i)] < d[(r, nn[r])] {
                    nn[r] = i;
                }
            } else if nn[r] == j || nn[r] == i {
                nn[r] = find(&d, &active, r);
            }
        }
    }
    let max_d = steps.iter().map(|s| s.distance).fold(0.0, f64::max);
    for st in steps.iter_mut() {
        st.similarity = if max_d > 0.0 { 100.0 * (1.0 - st.distance / max_d) } else { 100.0 };
    }
    // Leaf order by depth-first traversal from the last merge.
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![n + steps.len() - 1];
    while let Some(node) = stack.pop() {
        if node < n {
            order.push(node);
        } else {
            let st = &steps[node - n];
            stack.push(st.right);
            stack.push(st.left);
        }
    }
    Ok(Linkage { n, method, steps, items, order, rows_used: Vec::new(), n_dropped: 0, max_distance: max_d })
}

/// Summary of a partition computed from the data (Minitab's "Final Partition").
#[derive(Debug, Clone)]
pub struct PartitionSummary {
    pub sizes: Vec<usize>,
    pub within_ss: Vec<f64>,
    pub average_distance: Vec<f64>,
    pub maximum_distance: Vec<f64>,
    /// k × p centroids (original units).
    pub centroids: Matrix,
    /// k × k Euclidean distances between centroids (working scale).
    pub centroid_distances: Matrix,
    pub total_within_ss: f64,
}

impl PartitionSummary {
    /// `labels` are 1-based cluster numbers aligned with the rows of `cols` that were used.
    pub fn from_partition(
        cols: &[Vec<f64>],
        rows_used: &[usize],
        labels: &[usize],
        standardize: bool,
    ) -> Result<PartitionSummary> {
        let p = cols.len();
        let n = rows_used.len();
        if labels.len() != n {
            return invalid("the labels do not match the rows used");
        }
        let k = labels.iter().copied().max().unwrap_or(0);
        let mut data: Vec<Vec<f64>> = rows_used.iter().map(|&r| (0..p).map(|v| cols[v][r]).collect()).collect();
        let raw = data.clone();
        if standardize {
            standardize_rows(&mut data)?;
        }
        summarize(&data, &raw, &labels.iter().map(|l| l - 1).collect::<Vec<_>>(), k)
    }
}

fn summarize(work: &[Vec<f64>], raw: &[Vec<f64>], labels: &[usize], k: usize) -> Result<PartitionSummary> {
    let p = work[0].len();
    let mut sizes = vec![0usize; k];
    let mut cw = vec![vec![0.0; p]; k];
    let mut cr = vec![vec![0.0; p]; k];
    for (i, &l) in labels.iter().enumerate() {
        sizes[l] += 1;
        for v in 0..p {
            cw[l][v] += work[i][v];
            cr[l][v] += raw[i][v];
        }
    }
    for l in 0..k {
        if sizes[l] > 0 {
            for v in 0..p {
                cw[l][v] /= sizes[l] as f64;
                cr[l][v] /= sizes[l] as f64;
            }
        }
    }
    let mut ss = vec![0.0; k];
    let mut avg = vec![0.0; k];
    let mut mx = vec![0.0f64; k];
    for (i, &l) in labels.iter().enumerate() {
        let d2: f64 = (0..p).map(|v| (work[i][v] - cw[l][v]).powi(2)).sum();
        ss[l] += d2;
        avg[l] += d2.sqrt();
        mx[l] = mx[l].max(d2.sqrt());
    }
    for l in 0..k {
        if sizes[l] > 0 {
            avg[l] /= sizes[l] as f64;
        }
    }
    let cd = Matrix::from_fn(k, k, |a, b| (0..p).map(|v| (cw[a][v] - cw[b][v]).powi(2)).sum::<f64>().sqrt());
    let total = ss.iter().sum();
    Ok(PartitionSummary {
        sizes,
        within_ss: ss,
        average_distance: avg,
        maximum_distance: mx,
        centroids: Matrix::from_fn(k, p, |a, v| cr[a][v]),
        centroid_distances: cd,
        total_within_ss: total,
    })
}

/// How k-means chooses its starting centres.
#[derive(Debug, Clone, PartialEq)]
pub enum KMeansInit {
    /// k distinct observations chosen at random.
    Random(u64),
    /// k-means++ seeding.
    PlusPlus(u64),
    /// Observation indices (0-based, among the rows used) to start from.
    Rows(Vec<usize>),
    /// A starting partition: cluster number (1-based) for each row used.
    Partition(Vec<usize>),
}

/// Options of [`kmeans`].
#[derive(Debug, Clone)]
pub struct KMeansOptions {
    pub k: usize,
    pub init: KMeansInit,
    pub standardize: bool,
    pub max_iter: usize,
}

impl KMeansOptions {
    pub fn new(k: usize) -> KMeansOptions {
        KMeansOptions { k, init: KMeansInit::PlusPlus(1), standardize: false, max_iter: 200 }
    }
}

/// Result of [`kmeans`].
#[derive(Debug, Clone)]
pub struct KMeans {
    pub k: usize,
    pub n: usize,
    pub rows_used: Vec<usize>,
    pub n_dropped: usize,
    /// 1-based cluster of each row used.
    pub labels: Vec<usize>,
    pub standardized: bool,
    pub summary: PartitionSummary,
    pub iterations: usize,
    pub converged: bool,
    /// Distance (working scale) of each row used to its centroid.
    pub distances: Vec<f64>,
}

pub fn kmeans(cols: &[Vec<f64>], opts: &KMeansOptions) -> Result<KMeans> {
    let c = listwise(cols)?;
    let n = c.rows.len();
    let p = cols.len();
    let k = opts.k;
    if k < 2 || k > n {
        return invalid(format!("the number of clusters must be between 2 and the number of observations ({n})"));
    }
    let raw: Vec<Vec<f64>> = (0..n).map(|i| (0..p).map(|v| c.cols[v][i]).collect()).collect();
    let mut x = raw.clone();
    if opts.standardize {
        standardize_rows(&mut x)?;
    }
    let mut centers: Vec<Vec<f64>> = match &opts.init {
        KMeansInit::Random(seed) => {
            let mut rng = Pcg64::seed_from_u64(*seed);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + rng.below((n - i) as u64) as usize;
                idx.swap(i, j);
            }
            idx[..k].iter().map(|&i| x[i].clone()).collect()
        }
        KMeansInit::PlusPlus(seed) => {
            let mut rng = Pcg64::seed_from_u64(*seed);
            let mut cs = vec![x[rng.below(n as u64) as usize].clone()];
            let mut d2: Vec<f64> = x.iter().map(|r| sq(r, &cs[0])).collect();
            while cs.len() < k {
                let tot: f64 = d2.iter().sum();
                let pick = if tot > 0.0 {
                    let mut t = rng.next_f64() * tot;
                    let mut ch = n - 1;
                    for (i, &w) in d2.iter().enumerate() {
                        if t < w {
                            ch = i;
                            break;
                        }
                        t -= w;
                    }
                    ch
                } else {
                    rng.below(n as u64) as usize
                };
                cs.push(x[pick].clone());
                for (i, r) in x.iter().enumerate() {
                    d2[i] = d2[i].min(sq(r, cs.last().unwrap()));
                }
            }
            cs
        }
        KMeansInit::Rows(idx) => {
            if idx.len() != k || idx.iter().any(|&i| i >= n) {
                return invalid(format!("give {k} valid observation indices as starting centres"));
            }
            idx.iter().map(|&i| x[i].clone()).collect()
        }
        KMeansInit::Partition(part) => {
            if part.len() != n || part.iter().any(|&l| l == 0 || l > k) {
                return invalid("the starting partition must give a cluster number from 1 to k for every observation");
            }
            let mut cs = vec![vec![0.0; p]; k];
            let mut cnt = vec![0usize; k];
            for (i, &l) in part.iter().enumerate() {
                cnt[l - 1] += 1;
                for v in 0..p {
                    cs[l - 1][v] += x[i][v];
                }
            }
            if cnt.contains(&0) {
                return invalid("every cluster of the starting partition needs at least one observation");
            }
            for l in 0..k {
                for v in 0..p {
                    cs[l][v] /= cnt[l] as f64;
                }
            }
            cs
        }
    };

    let mut lab = vec![0usize; n];
    let mut iterations = 0;
    let mut converged = false;
    // Lloyd.
    for it in 0..opts.max_iter.max(1) {
        iterations = it + 1;
        let mut changed = it == 0;
        for i in 0..n {
            let b = nearest(&x[i], &centers);
            if b != lab[i] {
                lab[i] = b;
                changed = true;
            }
        }
        let mut cnt = vec![0usize; k];
        let mut sums = vec![vec![0.0; p]; k];
        for i in 0..n {
            cnt[lab[i]] += 1;
            for v in 0..p {
                sums[lab[i]][v] += x[i][v];
            }
        }
        for l in 0..k {
            if cnt[l] == 0 {
                // Re-seed an empty cluster at the point farthest from its centre.
                let mut far = 0;
                let mut fd = -1.0;
                for i in 0..n {
                    let d = sq(&x[i], &centers[lab[i]]);
                    if d > fd && cnt[lab[i]] > 1 {
                        fd = d;
                        far = i;
                    }
                }
                cnt[lab[far]] -= 1;
                lab[far] = l;
                cnt[l] = 1;
                sums[l] = x[far].clone();
                changed = true;
            }
        }
        for l in 0..k {
            for v in 0..p {
                centers[l][v] = sums[l][v] / cnt[l] as f64;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    // Hartigan–Wong style optimal transfers until no single move lowers the total within-cluster SS.
    let mut cnt = vec![0usize; k];
    for &l in &lab {
        cnt[l] += 1;
    }
    let mut guard = 0;
    loop {
        let mut moved = false;
        for i in 0..n {
            let a = lab[i];
            if cnt[a] <= 1 {
                continue;
            }
            let na = cnt[a] as f64;
            let rem = na / (na - 1.0) * sq(&x[i], &centers[a]);
            let mut best = a;
            let mut bc = rem;
            for b in 0..k {
                if b == a {
                    continue;
                }
                let nb = cnt[b] as f64;
                let add = nb / (nb + 1.0) * sq(&x[i], &centers[b]);
                if add < bc * (1.0 - 1e-12) {
                    bc = add;
                    best = b;
                }
            }
            if best != a {
                let (na, nb) = (cnt[a] as f64, cnt[best] as f64);
                for v in 0..p {
                    centers[a][v] = (centers[a][v] * na - x[i][v]) / (na - 1.0);
                    centers[best][v] = (centers[best][v] * nb + x[i][v]) / (nb + 1.0);
                }
                cnt[a] -= 1;
                cnt[best] += 1;
                lab[i] = best;
                moved = true;
            }
        }
        guard += 1;
        if !moved || guard > 10_000 {
            converged = converged || !moved;
            break;
        }
    }
    // Recompute centres exactly from the final labels.
    let summary = summarize(&x, &raw, &lab, k)?;
    let mut cw = vec![vec![0.0; p]; k];
    for (i, &l) in lab.iter().enumerate() {
        for v in 0..p {
            cw[l][v] += x[i][v] / summary.sizes[l] as f64;
        }
    }
    let distances = (0..n).map(|i| sq(&x[i], &cw[lab[i]]).sqrt()).collect();
    Ok(KMeans {
        k,
        n,
        n_dropped: cols[0].len() - n,
        rows_used: c.rows,
        labels: lab.iter().map(|l| l + 1).collect(),
        standardized: opts.standardize,
        summary,
        iterations,
        converged,
        distances,
    })
}

fn sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn nearest(x: &[f64], centers: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut bd = f64::INFINITY;
    for (k, c) in centers.iter().enumerate() {
        let d = sq(x, c);
        if d < bd {
            bd = d;
            best = k;
        }
    }
    best
}
