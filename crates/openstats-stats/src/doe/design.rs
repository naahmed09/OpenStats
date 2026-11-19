//! The design data structure shared by every experimental design.

use crate::error::{invalid, Result};
use openstats_math::random::{Pcg64, Rng};

/// A factor with its natural-unit levels (coded −1 and +1).
#[derive(Debug, Clone, PartialEq)]
pub struct FactorSpec {
    /// Name.
    pub name: String,
    /// Natural value of coded −1.
    pub low: f64,
    /// Natural value of coded +1.
    pub high: f64,
}

impl FactorSpec {
    /// Natural value of a coded value.
    pub fn to_natural(&self, coded: f64) -> f64 {
        0.5 * (self.low + self.high) + 0.5 * (self.high - self.low) * coded
    }

    /// Coded value of a natural value.
    pub fn to_coded(&self, natural: f64) -> f64 {
        (natural - 0.5 * (self.low + self.high)) / (0.5 * (self.high - self.low))
    }
}

/// Kind of a design point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointType {
    /// Corner (factorial) point.
    Factorial,
    /// Centre point.
    Center,
    /// Axial (star) point of a central composite design.
    Axial,
    /// Edge midpoint of a Box–Behnken design.
    Edge,
}

impl PointType {
    /// Minitab-style code: 1 corner/edge, 0 centre, −1 axial.
    pub fn code(self) -> i64 {
        match self {
            PointType::Factorial | PointType::Edge => 1,
            PointType::Center => 0,
            PointType::Axial => -1,
        }
    }
}

/// The family a design belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignKind {
    /// 2^k full factorial.
    FullFactorial,
    /// 2^(k−p) fractional factorial.
    FractionalFactorial,
    /// Plackett–Burman screening design.
    PlackettBurman,
    /// Central composite design.
    CentralComposite,
    /// Box–Behnken design.
    BoxBehnken,
}

impl DesignKind {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            DesignKind::FullFactorial => "Full Factorial Design",
            DesignKind::FractionalFactorial => "Fractional Factorial Design",
            DesignKind::PlackettBurman => "Plackett-Burman Design",
            DesignKind::CentralComposite => "Central Composite Design",
            DesignKind::BoxBehnken => "Box-Behnken Design",
        }
    }
}

/// Defining relation of a fractional factorial (or of the generators of a block structure).
#[derive(Debug, Clone, PartialEq)]
pub struct Defining {
    /// Words (bit masks over the factors, bit `i` = factor `i`) of the generators.
    pub generators: Vec<u32>,
    /// Every non-empty product of generator words (the defining relation).
    pub words: Vec<u32>,
    /// Resolution: length of the shortest word.
    pub resolution: usize,
}

/// An experimental design in run order.
#[derive(Debug, Clone, PartialEq)]
pub struct Design {
    /// Family.
    pub kind: DesignKind,
    /// Factors.
    pub factors: Vec<FactorSpec>,
    /// Coded levels, one row per run in run order.
    pub coded: Vec<Vec<f64>>,
    /// Point type of each run.
    pub point_type: Vec<PointType>,
    /// Block (1-based) of each run.
    pub block: Vec<usize>,
    /// Number of blocks.
    pub blocks: usize,
    /// Standard-order number (1-based) of each run.
    pub std_order: Vec<usize>,
    /// Defining relation (fractional factorials, including the fraction of a CCD).
    pub defining: Option<Defining>,
    /// Block generator words (confounded with the blocks).
    pub block_words: Vec<u32>,
}

/// Letter label of factor `i` ("A", "B", …).
pub fn letter(i: usize) -> char {
    (b'A' + i as u8) as char
}

/// The word of a mask ("ABD").
pub fn word_label(mask: u32) -> String {
    (0..32).filter(|i| mask >> i & 1 == 1).map(letter).collect()
}

impl Design {
    pub(crate) fn assemble(
        kind: DesignKind,
        k: usize,
        rows: Vec<(Vec<f64>, PointType, usize)>,
        blocks: usize,
        defining: Option<Defining>,
        block_words: Vec<u32>,
    ) -> Design {
        let factors = (0..k).map(|i| FactorSpec { name: letter(i).to_string(), low: -1.0, high: 1.0 }).collect();
        let n = rows.len();
        Design {
            kind,
            factors,
            point_type: rows.iter().map(|r| r.1).collect(),
            block: rows.iter().map(|r| r.2).collect(),
            coded: rows.into_iter().map(|r| r.0).collect(),
            blocks,
            std_order: (1..=n).collect(),
            defining,
            block_words,
        }
    }

    /// Number of runs.
    pub fn n_runs(&self) -> usize {
        self.coded.len()
    }

    /// Number of factors.
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Coded values of factor `j` in run order.
    pub fn column(&self, j: usize) -> Vec<f64> {
        self.coded.iter().map(|r| r[j]).collect()
    }

    /// Natural-unit settings, one row per run.
    pub fn natural(&self) -> Vec<Vec<f64>> {
        self.coded.iter().map(|r| r.iter().zip(&self.factors).map(|(&c, f)| f.to_natural(c)).collect()).collect()
    }

    /// Rename the factors.
    pub fn with_names(mut self, names: &[&str]) -> Result<Design> {
        if names.len() != self.factors.len() {
            return invalid(format!("{} names given for {} factors", names.len(), self.factors.len()));
        }
        for (i, n) in names.iter().enumerate() {
            if n.is_empty() || names[..i].contains(n) {
                return invalid("factor names must be non-empty and distinct");
            }
        }
        for (f, n) in self.factors.iter_mut().zip(names) {
            f.name = (*n).to_string();
        }
        Ok(self)
    }

    /// Set the natural levels of the factors as `(low, high)` pairs.
    pub fn with_levels(mut self, levels: &[(f64, f64)]) -> Result<Design> {
        if levels.len() != self.factors.len() {
            return invalid(format!("{} level pairs given for {} factors", levels.len(), self.factors.len()));
        }
        for (f, &(lo, hi)) in self.factors.iter_mut().zip(levels) {
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return invalid("each factor needs finite levels low < high");
            }
            f.low = lo;
            f.high = hi;
        }
        Ok(self)
    }

    /// Randomize the run order within each block (Fisher–Yates with a PCG64 stream).
    pub fn randomized(mut self, seed: u64) -> Design {
        let mut rng = Pcg64::new(u128::from(seed), 54);
        let n = self.n_runs();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut start = 0;
        while start < n {
            let b = self.block[perm[start]];
            let mut end = start;
            while end < n && self.block[perm[end]] == b {
                end += 1;
            }
            for i in (start + 1..end).rev() {
                let j = start + rng.below((i - start + 1) as u64) as usize;
                perm.swap(i, j);
            }
            start = end;
        }
        self.coded = perm.iter().map(|&i| self.coded[i].clone()).collect();
        self.point_type = perm.iter().map(|&i| self.point_type[i]).collect();
        self.block = perm.iter().map(|&i| self.block[i]).collect();
        self.std_order = perm.iter().map(|&i| self.std_order[i]).collect();
        self
    }

    /// Check the orthogonality of the coded columns over the corner points
    /// (`xᵢ·xⱼ = 0` for all pairs of two-level columns).
    pub fn columns_orthogonal(&self, tol: f64) -> bool {
        let k = self.n_factors();
        for a in 0..k {
            for b in a + 1..k {
                let s: f64 = self.coded.iter().map(|r| r[a] * r[b]).sum();
                if s.abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}
