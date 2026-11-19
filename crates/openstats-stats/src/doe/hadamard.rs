//! Hadamard matrices by the Paley and Sylvester constructions (used for
//! Plackett–Burman designs and the Taguchi L12 array).

fn is_prime(n: usize) -> bool {
    n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

/// Legendre symbol table of the prime `q`: χ(x) ∈ {−1, 0, 1}.
fn chi(q: usize) -> Vec<i32> {
    let mut c = vec![-1; q];
    c[0] = 0;
    for x in 1..q {
        c[x * x % q] = 1;
    }
    c
}

fn paley_i(q: usize) -> Vec<Vec<i8>> {
    let c = chi(q);
    let n = q + 1;
    let mut h = vec![vec![0i8; n]; n];
    // H = I + S with S = [[0, 1ᵀ], [−1, Q]], Q[i][j] = χ(j − i).
    for i in 0..n {
        for j in 0..n {
            let s = match (i, j) {
                (0, 0) => 0,
                (0, _) => 1,
                (_, 0) => -1,
                (i, j) => c[(j + q - i) % q],
            };
            h[i][j] = (s + i32::from(i == j)) as i8;
        }
    }
    h
}

fn paley_ii(q: usize) -> Vec<Vec<i8>> {
    let c = chi(q);
    let m = q + 1;
    // Symmetric conference matrix C = [[0, 1ᵀ], [1, Q]].
    let conf = |i: usize, j: usize| -> i32 {
        match (i, j) {
            (0, 0) => 0,
            (0, _) | (_, 0) => 1,
            (i, j) => c[(j + q - i) % q],
        }
    };
    let mut h = vec![vec![0i8; 2 * m]; 2 * m];
    for i in 0..m {
        for j in 0..m {
            let id = i32::from(i == j);
            let cij = conf(i, j);
            h[i][j] = (cij + id) as i8;
            h[i][j + m] = (cij - id) as i8;
            h[i + m][j] = (cij - id) as i8;
            h[i + m][j + m] = (-cij - id) as i8;
        }
    }
    h
}

/// A Hadamard matrix of order `n`, when one of the supported constructions applies.
pub fn hadamard(n: usize) -> Option<Vec<Vec<i8>>> {
    match n {
        0 => None,
        1 => Some(vec![vec![1]]),
        2 => Some(vec![vec![1, 1], vec![1, -1]]),
        _ if n % 4 != 0 => None,
        _ => {
            if is_prime(n - 1) && (n - 1) % 4 == 3 {
                return Some(paley_i(n - 1));
            }
            if n % 2 == 0 && n / 2 >= 2 && is_prime(n / 2 - 1) && (n / 2 - 1) % 4 == 1 {
                return Some(paley_ii(n / 2 - 1));
            }
            let half = hadamard(n / 2)?;
            let m = n / 2;
            let mut h = vec![vec![0i8; n]; n];
            for i in 0..m {
                for j in 0..m {
                    h[i][j] = half[i][j];
                    h[i][j + m] = half[i][j];
                    h[i + m][j] = half[i][j];
                    h[i + m][j + m] = -half[i][j];
                }
            }
            Some(h)
        }
    }
}

/// Whether `h` satisfies `H·Hᵀ = n·I`.
pub fn is_hadamard(h: &[Vec<i8>]) -> bool {
    let n = h.len();
    (0..n).all(|a| {
        (0..n).all(|b| {
            let s: i32 = (0..n).map(|k| i32::from(h[a][k]) * i32::from(h[b][k])).sum();
            s == if a == b { n as i32 } else { 0 }
        })
    })
}

/// Normalise so the first row and column are +1, then drop the first column.
pub fn two_level_columns(h: &[Vec<i8>]) -> Vec<Vec<i8>> {
    let n = h.len();
    let mut g = h.to_vec();
    for j in 0..n {
        if g[0][j] < 0 {
            for row in g.iter_mut() {
                row[j] = -row[j];
            }
        }
    }
    for row in g.iter_mut() {
        if row[0] < 0 {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
    }
    g.into_iter().map(|r| r[1..].to_vec()).collect()
}
