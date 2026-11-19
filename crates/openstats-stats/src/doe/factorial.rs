//! Two-level factorial designs: full 2^k, fractional 2^(k−p) with generator
//! selection and alias structure, and Plackett–Burman screening designs.
//!
//! Factors are labelled A, B, C, … in the defining relation and alias
//! structure. A fractional design takes the first `k − p` letters as base
//! factors (full factorial in standard order, A changing fastest) and each
//! remaining factor is the product of the base factors named by its
//! generator (`"D=ABC"` or just `"ABC"`).

use super::design::{letter, word_label, Defining, Design, DesignKind, PointType};
use super::hadamard::{hadamard, two_level_columns};
use crate::error::{invalid, Result};

/// Options shared by the two-level designs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FactorialOptions {
    /// Replicates of the whole design (default 1).
    pub replicates: usize,
    /// Centre points added to each block.
    pub center_points: usize,
    /// Block generator words ("ABC"): `2^len` blocks.
    pub block_generators: Vec<String>,
    /// Randomize the run order within blocks with this seed.
    pub randomize: Option<u64>,
}

const MAX_FACTORS: usize = 15;

/// (resolution, aberration pattern) of a generator set.
type Score = (usize, Vec<usize>);
/// Best score so far with its generator words.
type Best = Option<(Score, Vec<u32>)>;

fn parse_word(s: &str, k: usize, allow_eq: bool) -> Result<(Option<usize>, u32)> {
    let t: String = s.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_uppercase();
    let (lhs, rhs) = match t.split_once('=') {
        Some((l, r)) if allow_eq => (Some(l.to_string()), r.to_string()),
        Some(_) => return invalid(format!("'{s}' must be a word of factor letters")),
        None => (None, t),
    };
    let mut mask = 0u32;
    for c in rhs.chars() {
        if !c.is_ascii_uppercase() || (c as usize - 'A' as usize) >= k {
            return invalid(format!("'{s}': '{c}' is not one of the {k} factor letters"));
        }
        let b = 1u32 << (c as usize - 'A' as usize);
        if mask & b != 0 {
            return invalid(format!("'{s}' repeats a letter"));
        }
        mask |= b;
    }
    if mask == 0 {
        return invalid(format!("'{s}' is empty"));
    }
    let target = match lhs {
        None => None,
        Some(l) if l.len() == 1 && l.as_bytes()[0].is_ascii_uppercase() => {
            Some(l.as_bytes()[0] as usize - 'A' as usize)
        }
        Some(l) => return invalid(format!("'{l}' is not a single factor letter")),
    };
    Ok((target, mask))
}

fn subgroup(gens: &[u32]) -> Vec<u32> {
    let mut v = Vec::new();
    for s in 1u32..(1 << gens.len()) {
        let mut w = 0;
        for (i, g) in gens.iter().enumerate() {
            if s >> i & 1 == 1 {
                w ^= g;
            }
        }
        v.push(w);
    }
    v
}

pub(crate) fn defining_for_generators(gens: &[u32]) -> Result<Defining> {
    defining_from(gens.to_vec())
}

fn defining_from(gens: Vec<u32>) -> Result<Defining> {
    let words = subgroup(&gens);
    if words.contains(&0) {
        return invalid("the generators are dependent (a product of generator words is the identity)");
    }
    let resolution = words.iter().map(|w| w.count_ones() as usize).min().unwrap_or(usize::MAX);
    if resolution < 3 {
        return invalid("a generator confounds a main effect with another main effect or the mean");
    }
    Ok(Defining { generators: gens, words, resolution })
}

/// Word-length pattern (counts of words of length 3, 4, …) used to rank designs by minimum aberration.
fn pattern(words: &[u32], k: usize) -> Vec<usize> {
    let mut p = vec![0; k + 1];
    for w in words {
        p[w.count_ones() as usize] += 1;
    }
    p
}

fn better(a: &(usize, Vec<usize>), b: &(usize, Vec<usize>)) -> bool {
    // Higher resolution wins; then fewer words of the shortest length, next shortest, ….
    if a.0 != b.0 {
        return a.0 > b.0;
    }
    for len in a.0..a.1.len() {
        if a.1[len] != b.1[len] {
            return a.1[len] < b.1[len];
        }
    }
    false
}

/// Default generators of a 2^(k−p) design: the choice of generator words
/// (each a product of base factors) with the highest resolution and then
/// minimum aberration. Exhaustive when the search space is at most about
/// 2·10⁶ candidate sets, otherwise greedy (one generator at a time).
pub fn default_generators(k: usize, p: usize) -> Result<Vec<u32>> {
    check_kp(k, p)?;
    let base = k - p;
    let cands: Vec<u32> = (1u32..(1 << base)).filter(|m| m.count_ones() >= 2).collect();
    if cands.len() < p {
        return invalid(format!("no 2^({k}-{p}) design exists: only {} usable generator words", cands.len()));
    }
    let word_of = |i: usize, m: u32| m | (1 << (base + i));
    let mut combos = 1f64;
    for i in 0..p {
        combos *= (cands.len() - i) as f64 / (i + 1) as f64;
    }
    let score = |chosen: &[u32]| -> Option<(usize, Vec<usize>)> {
        let gens: Vec<u32> = chosen.iter().enumerate().map(|(i, &m)| word_of(i, m)).collect();
        let w = subgroup(&gens);
        if w.contains(&0) {
            return None;
        }
        let res = w.iter().map(|x| x.count_ones() as usize).min()?;
        Some((res, pattern(&w, k)))
    };
    let mut best: Best = None;
    if combos <= 2.0e6 {
        fn rec(
            cands: &[u32],
            start: usize,
            p: usize,
            cur: &mut Vec<u32>,
            score: &dyn Fn(&[u32]) -> Option<Score>,
            best: &mut Best,
        ) {
            if cur.len() == p {
                if let Some(s) = score(cur) {
                    if best.as_ref().is_none_or(|b| better(&s, &b.0)) {
                        *best = Some((s, cur.clone()));
                    }
                }
                return;
            }
            for i in start..cands.len() {
                cur.push(cands[i]);
                rec(cands, i + 1, p, cur, score, best);
                cur.pop();
            }
        }
        rec(&cands, 0, p, &mut Vec::new(), &score, &mut best);
    } else {
        let mut chosen: Vec<u32> = Vec::new();
        for _ in 0..p {
            let mut step: Option<((usize, Vec<usize>), u32)> = None;
            for &c in &cands {
                if chosen.contains(&c) {
                    continue;
                }
                let mut t = chosen.clone();
                t.push(c);
                // Score the partial design (its own words only).
                let gens: Vec<u32> = t.iter().enumerate().map(|(i, &m)| word_of(i, m)).collect();
                let w = subgroup(&gens);
                if w.contains(&0) {
                    continue;
                }
                let res = w.iter().map(|x| x.count_ones() as usize).min().unwrap_or(usize::MAX);
                let s = (res, pattern(&w, k));
                if step.as_ref().is_none_or(|b| better(&s, &b.0)) {
                    step = Some((s, c));
                }
            }
            match step {
                Some((_, c)) => chosen.push(c),
                None => return invalid("could not find independent generators"),
            }
        }
        best = score(&chosen).map(|s| (s, chosen));
    }
    match best {
        Some((_, m)) => Ok(m.iter().enumerate().map(|(i, &x)| word_of(i, x)).collect()),
        None => invalid("could not find independent generators"),
    }
}

fn check_kp(k: usize, p: usize) -> Result<()> {
    if !(2..=MAX_FACTORS).contains(&k) {
        return invalid(format!("the number of factors must be 2..={MAX_FACTORS} (got {k})"));
    }
    if p >= k || k - p < 2 {
        return invalid(format!("2^({k}-{p}) leaves fewer than 2 base factors"));
    }
    Ok(())
}

fn build(
    kind: DesignKind,
    k: usize,
    defining: Option<Defining>,
    gen_words: &[u32],
    opts: &FactorialOptions,
) -> Result<Design> {
    let p = gen_words.len();
    let base = k - p;
    let reps = opts.replicates.max(1);
    let mut block_words = Vec::new();
    for s in &opts.block_generators {
        let (_, m) = parse_word(s, k, false)?;
        if m.count_ones() < 2 {
            return invalid(format!("block generator '{s}' would confound a main effect with blocks"));
        }
        block_words.push(m);
    }
    if block_words.len() > base.saturating_sub(1) {
        return invalid("too many blocks for the number of base factors");
    }
    let mut all = subgroup(&block_words);
    if let Some(d) = &defining {
        for &b in &all {
            if d.words.contains(&b) {
                return invalid("a block generator is confounded with the defining relation");
            }
            if d.words.iter().any(|&w| (w ^ b).count_ones() < 2 && w ^ b != 0) {
                return invalid("a block generator is aliased with a main effect");
            }
        }
    }
    all.retain(|&w| w != 0);
    if block_words.iter().any(|b| b.count_ones() < 2) || all.is_empty() && !block_words.is_empty() {
        return invalid("invalid block generators");
    }
    let n_blocks = 1usize << block_words.len();
    let n_base = 1usize << base;
    let mut rows: Vec<(Vec<f64>, PointType, usize)> = Vec::new();
    for _ in 0..reps {
        for r in 0..n_base {
            let mut x: Vec<f64> = (0..base).map(|j| if r >> j & 1 == 1 { 1.0 } else { -1.0 }).collect();
            for &g in gen_words {
                let letters = g & ((1u32 << base) - 1);
                let v: f64 = (0..base).filter(|j| letters >> j & 1 == 1).map(|j| x[j]).product();
                x.push(v);
            }
            let mut blk = 0usize;
            for (bi, &bw) in block_words.iter().enumerate() {
                let v: f64 = (0..k).filter(|j| bw >> j & 1 == 1).map(|j| x[j]).product();
                if v > 0.0 {
                    blk |= 1 << bi;
                }
            }
            rows.push((x, PointType::Factorial, blk + 1));
        }
    }
    // Blocks contiguous, each followed by its centre points.
    let mut ordered = Vec::new();
    for b in 1..=n_blocks {
        ordered.extend(rows.iter().filter(|r| r.2 == b).cloned());
        for _ in 0..opts.center_points {
            ordered.push((vec![0.0; k], PointType::Center, b));
        }
    }
    let mut d = Design::assemble(kind, k, ordered, n_blocks, defining, block_words);
    d.std_order = (1..=d.n_runs()).collect();
    Ok(match opts.randomize {
        Some(s) => d.randomized(s),
        None => d,
    })
}

/// 2^k full factorial (2 ≤ k ≤ 15) in standard order.
pub fn full_factorial(k: usize, opts: &FactorialOptions) -> Result<Design> {
    if !(2..=MAX_FACTORS).contains(&k) {
        return invalid(format!("the number of factors must be 2..={MAX_FACTORS} (got {k})"));
    }
    if (1usize << k) * opts.replicates.max(1) > 1 << 20 {
        return invalid("the design has more than 2^20 runs");
    }
    let mut o = opts.clone();
    // Block words may name any factor letter; with no generated factors the base is k.
    o.block_generators = opts.block_generators.clone();
    build(DesignKind::FullFactorial, k, None, &[], &o)
}

/// 2^(k−p) fractional factorial. `generators` are words such as `"D=ABC"`;
/// `None` chooses the highest-resolution, minimum-aberration generators.
pub fn fractional_factorial(
    k: usize,
    p: usize,
    generators: Option<&[&str]>,
    opts: &FactorialOptions,
) -> Result<Design> {
    check_kp(k, p)?;
    if p == 0 {
        return full_factorial(k, opts);
    }
    let base = k - p;
    let words: Vec<u32> = match generators {
        None => default_generators(k, p)?,
        Some(g) => {
            if g.len() != p {
                return invalid(format!("{} generators given for a 2^({k}-{p}) design (need {p})", g.len()));
            }
            let mut out = vec![0u32; p];
            let mut used = vec![false; p];
            for s in g {
                let (target, m) = parse_word(s, k, true)?;
                if m >> base != 0 {
                    return invalid(format!(
                        "generator '{s}' may only use the base factors {}",
                        (0..base).map(letter).collect::<String>()
                    ));
                }
                let slot = match target {
                    Some(t) => {
                        if t < base || t >= k {
                            return invalid(format!("'{s}': {} is not a generated factor", letter(t)));
                        }
                        t - base
                    }
                    None => match used.iter().position(|u| !u) {
                        Some(i) => i,
                        None => return invalid("too many generators"),
                    },
                };
                if used[slot] {
                    return invalid(format!("factor {} has two generators", letter(base + slot)));
                }
                used[slot] = true;
                out[slot] = m | (1 << (base + slot));
            }
            out
        }
    };
    let def = defining_from(words.clone())?;
    build(DesignKind::FractionalFactorial, k, Some(def), &words, opts)
}

/// Alias structure of a fractional factorial design.
#[derive(Debug, Clone, PartialEq)]
pub struct AliasStructure {
    /// Letters and factor names ("A" → "Temp").
    pub letters: Vec<(char, String)>,
    /// Defining relation as words ("ABCD"), each meaning `I = word`.
    pub defining_relation: Vec<String>,
    /// Design resolution (`None` for a full factorial).
    pub resolution: Option<usize>,
    /// Groups of confounded effects (up to the requested order), each sorted by order then letters.
    pub classes: Vec<Vec<String>>,
}

impl Design {
    /// Alias structure listing effects up to interaction order `max_order`
    /// (main effects are order 1). For a full factorial every effect is its own class.
    pub fn alias_structure(&self, max_order: usize) -> Result<AliasStructure> {
        let k = self.n_factors();
        if max_order == 0 || max_order > k {
            return invalid(format!("alias order must be 1..={k}"));
        }
        let mut sub = vec![0u32];
        let mut res = None;
        let mut rel = Vec::new();
        if let Some(d) = &self.defining {
            sub.extend(d.words.iter().copied());
            res = Some(d.resolution);
            let mut w = d.words.clone();
            w.sort_by_key(|x| (x.count_ones(), *x));
            rel = w.iter().map(|&x| word_label(x)).collect();
        }
        let mut classes: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut terms: Vec<u32> = (1u32..(1 << k)).filter(|m| (m.count_ones() as usize) <= max_order).collect();
        terms.sort_by_key(|m| (m.count_ones(), *m));
        for t in terms {
            let key = sub.iter().map(|&g| t ^ g).min().unwrap_or(t);
            match classes.iter_mut().find(|c| c.0 == key) {
                Some(c) => c.1.push(t),
                None => classes.push((key, vec![t])),
            }
        }
        let mut out = Vec::new();
        for (_, mut members) in classes {
            members.sort_by_key(|m| (m.count_ones(), *m));
            out.push(members.into_iter().map(word_label).collect());
        }
        Ok(AliasStructure {
            letters: self.factors.iter().enumerate().map(|(i, f)| (letter(i), f.name.clone())).collect(),
            defining_relation: rel,
            resolution: res,
            classes: out,
        })
    }
}

/// Plackett–Burman design with `runs` runs (a multiple of 4 for which a
/// Hadamard matrix is constructible) and `factors` ≤ runs − 1 two-level
/// factors (the first columns of the normalised Hadamard matrix). The
/// columns are orthogonal; with fewer factors than `runs − 1` the unused
/// columns are dropped.
pub fn plackett_burman(runs: usize, factors: usize, opts: &FactorialOptions) -> Result<Design> {
    if runs < 4 || runs % 4 != 0 {
        return invalid(format!("a Plackett-Burman design needs a multiple of 4 runs (got {runs})"));
    }
    if factors < 2 || factors > runs - 1 {
        return invalid(format!("a {runs}-run design takes 2..={} factors (got {factors})", runs - 1));
    }
    let Some(h) = hadamard(runs) else {
        return invalid(format!(
            "no Hadamard matrix of order {runs} can be constructed here (supported: Paley and Sylvester constructions)"
        ));
    };
    let cols = two_level_columns(&h);
    let reps = opts.replicates.max(1);
    let mut rows = Vec::new();
    for _ in 0..reps {
        for r in &cols {
            rows.push((r[..factors].iter().map(|&v| f64::from(v)).collect::<Vec<f64>>(), PointType::Factorial, 1));
        }
    }
    for _ in 0..opts.center_points {
        rows.push((vec![0.0; factors], PointType::Center, 1));
    }
    let mut d = Design::assemble(DesignKind::PlackettBurman, factors, rows, 1, None, Vec::new());
    d.std_order = (1..=d.n_runs()).collect();
    Ok(match opts.randomize {
        Some(s) => d.randomized(s),
        None => d,
    })
}
