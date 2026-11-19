//! Optimizers against classical test functions and SciPy `curve_fit`.

use openstats_math::linalg::Cholesky;
use openstats_math::optim::testfn::*;
use openstats_math::optim::*;
use openstats_math::Matrix;

#[path = "reference/lsq.rs"]
mod lsq;

type GradFn = fn(&[f64]) -> Vec<f64>;

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

fn opts() -> Options {
    Options::default()
}

/// Run every gradient-based method on one problem and check each reaches `target`.
fn all_gradient_methods(label: &str, f: fn(&[f64]) -> f64, g: Option<GradFn>, x0: &[f64], target: &[f64], tol: f64) {
    let check = |name: &str, r: Minimum| {
        assert!(dist(&r.x, target) < tol, "{label}/{name}: x = {:?}, f = {:e}, {:?}", r.x, r.f, r.termination);
        assert!(r.converged, "{label}/{name}: not converged ({:?}, f = {:e})", r.termination, r.f);
    };
    let mut ff = |x: &[f64]| f(x);
    let mut gg = |x: &[f64]| g.expect("gradient")(x);
    let gopt: Option<GradientFn> = if g.is_some() { Some(&mut gg) } else { None };
    check("bfgs", bfgs(&mut ff, gopt, x0, &opts()).unwrap());
    let mut ff = |x: &[f64]| f(x);
    let mut gg = |x: &[f64]| g.expect("gradient")(x);
    let gopt: Option<GradientFn> = if g.is_some() { Some(&mut gg) } else { None };
    check("lbfgs", lbfgs(&mut ff, gopt, x0, &opts()).unwrap());
}

#[test]
fn rosenbrock_all_methods() {
    let target2 = [1.0, 1.0];
    all_gradient_methods("rosenbrock2", rosenbrock, Some(rosenbrock_grad), &[-1.2, 1.0], &target2, 1e-6);
    // Finite-difference gradients.
    all_gradient_methods("rosenbrock2-fd", rosenbrock, None, &[-1.2, 1.0], &target2, 1e-5);
    let x0 = vec![-1.2; 10];
    let t10 = vec![1.0; 10];
    all_gradient_methods("rosenbrock10", rosenbrock, Some(rosenbrock_grad), &x0, &t10, 1e-5);

    let mut f = |x: &[f64]| rosenbrock(x);
    let mut g = |x: &[f64]| rosenbrock_grad(x);
    let mut h = |x: &[f64]| rosenbrock_hess(x);
    let r = newton(&mut f, Some(&mut g), Some(&mut h), &[-1.2, 1.0], &opts()).unwrap();
    assert!(r.converged && dist(&r.x, &target2) < 1e-8, "newton {r:?}");
    assert!(r.iterations < 40);
    let mut f = |x: &[f64]| rosenbrock(x);
    let mut g = |x: &[f64]| rosenbrock_grad(x);
    let mut h = |x: &[f64]| rosenbrock_hess(x);
    let r = newton_dogleg(&mut f, Some(&mut g), Some(&mut h), &[-1.2, 1.0], &opts()).unwrap();
    assert!(r.converged && dist(&r.x, &target2) < 1e-8, "dogleg {r:?}");
    // Newton without any derivatives supplied (finite differences throughout).
    let mut f = |x: &[f64]| rosenbrock(x);
    let r = newton(&mut f, None, None, &[-1.2, 1.0], &opts()).unwrap();
    assert!(dist(&r.x, &target2) < 1e-4, "newton fd {r:?}");
    // Newton / dogleg on the 10-dimensional problem.
    let mut f = |x: &[f64]| rosenbrock(x);
    let mut g = |x: &[f64]| rosenbrock_grad(x);
    let mut h = |x: &[f64]| rosenbrock_hess(x);
    let r = newton_dogleg(&mut f, Some(&mut g), Some(&mut h), &x0, &opts()).unwrap();
    assert!(r.converged && dist(&r.x, &t10) < 1e-6, "dogleg10 {r:?}");

    let mut f = |x: &[f64]| rosenbrock(x);
    let r = nelder_mead(&mut f, &[-1.2, 1.0], &NelderMeadOptions::default()).unwrap();
    assert!(r.converged && dist(&r.x, &target2) < 1e-6 && r.f < 1e-12, "nm {r:?}");
}

#[test]
fn himmelblau_reaches_one_of_four_minima() {
    let minima: [[f64; 2]; 4] = [
        [3.0, 2.0],
        [-2.805118086952745, 3.131312518250573],
        [-3.779310253377747, -3.283185991286170],
        [3.584428340330492, -1.848126526964404],
    ];
    for start in [[0.0, 0.0], [-1.0, 1.0], [-4.0, -4.0], [4.0, -2.0], [-5.0, 5.0]] {
        let near = |x: &[f64]| minima.iter().map(|m| dist(x, m)).fold(f64::INFINITY, f64::min);
        let mut f = |x: &[f64]| himmelblau(x);
        let mut g = |x: &[f64]| himmelblau_grad(x);
        let r = bfgs(&mut f, Some(&mut g), &start, &opts()).unwrap();
        assert!(r.converged && near(&r.x) < 1e-6 && r.f < 1e-12, "bfgs from {start:?}: {r:?}");
        let mut f = |x: &[f64]| himmelblau(x);
        let r = lbfgs(&mut f, None, &start, &opts()).unwrap();
        assert!(near(&r.x) < 1e-5, "lbfgs from {start:?}: {r:?}");
        let mut f = |x: &[f64]| himmelblau(x);
        let r = nelder_mead(&mut f, &start, &NelderMeadOptions::default()).unwrap();
        assert!(near(&r.x) < 1e-6 && r.f < 1e-10, "nm from {start:?}: {r:?}");
        let mut f = |x: &[f64]| himmelblau(x);
        let mut g = |x: &[f64]| himmelblau_grad(x);
        let r = newton_dogleg(&mut f, Some(&mut g), None, &start, &opts()).unwrap();
        assert!(near(&r.x) < 1e-5 && r.f < 1e-10, "dogleg from {start:?}: {r:?}");
    }
}

#[test]
fn beale_wood_goldstein_price_booth() {
    let mut f = |x: &[f64]| beale(x);
    let r = bfgs(&mut f, None, &[1.0, 1.0], &opts()).unwrap();
    assert!(dist(&r.x, &[3.0, 0.5]) < 1e-5 && r.f < 1e-10, "beale bfgs {r:?}");
    let mut f = |x: &[f64]| beale(x);
    let r = nelder_mead(&mut f, &[1.0, 1.0], &NelderMeadOptions::default()).unwrap();
    assert!(dist(&r.x, &[3.0, 0.5]) < 1e-5, "beale nm {r:?}");
    let mut f = |x: &[f64]| beale(x);
    let r = newton(&mut f, None, None, &[1.0, 1.0], &opts()).unwrap();
    assert!(dist(&r.x, &[3.0, 0.5]) < 1e-4, "beale newton {r:?}");

    let ones = [1.0; 4];
    let mut f = |x: &[f64]| wood(x);
    let r = bfgs(&mut f, None, &[-3.0, -1.0, -3.0, -1.0], &opts()).unwrap();
    assert!(dist(&r.x, &ones) < 1e-4 && r.f < 1e-10, "wood bfgs {r:?}");
    let mut f = |x: &[f64]| wood(x);
    let r = lbfgs(&mut f, None, &[-3.0, -1.0, -3.0, -1.0], &opts()).unwrap();
    assert!(dist(&r.x, &ones) < 1e-4, "wood lbfgs {r:?}");
    let mut f = |x: &[f64]| wood(x);
    let r = newton_dogleg(&mut f, None, None, &[-3.0, -1.0, -3.0, -1.0], &opts()).unwrap();
    assert!(dist(&r.x, &ones) < 1e-4, "wood dogleg {r:?}");

    let mut f = |x: &[f64]| goldstein_price(x);
    let r = bfgs(&mut f, None, &[-0.3, -0.8], &opts()).unwrap();
    assert!((r.f - 3.0).abs() < 1e-8 && dist(&r.x, &[0.0, -1.0]) < 1e-5, "goldstein-price {r:?}");

    let mut f = |x: &[f64]| booth(x);
    let r = nelder_mead(&mut f, &[0.0, 0.0], &NelderMeadOptions::default()).unwrap();
    assert!(dist(&r.x, &[1.0, 3.0]) < 1e-6);
}

#[test]
fn powell_singular_function() {
    // The Hessian is singular at the minimiser, so convergence is only linear
    // for gradient methods: demand a tiny objective rather than a tiny x error.
    let x0 = [3.0, -1.0, 0.0, 1.0];
    let mut f = |x: &[f64]| powell_singular(x);
    let r = bfgs(&mut f, None, &x0, &Options { max_iter: 5000, ..opts() }).unwrap();
    assert!(r.f < 1e-10, "bfgs {r:?}");
    let mut f = |x: &[f64]| powell_singular(x);
    let r = lbfgs(&mut f, None, &x0, &Options { max_iter: 5000, ..opts() }).unwrap();
    assert!(r.f < 1e-10, "lbfgs {r:?}");
    let mut f = |x: &[f64]| powell_singular(x);
    let r = nelder_mead(&mut f, &x0, &NelderMeadOptions::default()).unwrap();
    assert!(r.f < 1e-9, "nm {r:?}");
    let mut f = |x: &[f64]| powell_singular(x);
    let r = newton(&mut f, None, None, &x0, &Options { max_iter: 5000, ..opts() }).unwrap();
    assert!(r.f < 1e-9, "newton (singular Hessian, modified Cholesky) {r:?}");
}

#[test]
fn indefinite_hessian_handled_by_modified_cholesky() {
    // f = x⁴ − x² + y² has an indefinite Hessian near x = 0: a plain Newton
    // step would head for the local maximum at x = 0.
    let f = |x: &[f64]| x[0].powi(4) - x[0] * x[0] + x[1] * x[1];
    let g = |x: &[f64]| vec![4.0 * x[0].powi(3) - 2.0 * x[0], 2.0 * x[1]];
    let h = |x: &[f64]| Matrix::from_rows(&[vec![12.0 * x[0] * x[0] - 2.0, 0.0], vec![0.0, 2.0]]).unwrap();
    // (x = 0 exactly is a saddle with zero gradient: Newton rightly stays there.)
    for start in [[0.1, 1.0], [-0.05, 2.0], [0.001, 1.0]] {
        let (mut ff, mut gg, mut hh) = (f, g, h);
        let r = newton(&mut ff, Some(&mut gg), Some(&mut hh), &start, &opts()).unwrap();
        assert!(
            r.converged && (r.x[0].abs() - 0.5f64.sqrt()).abs() < 1e-8 && r.x[1].abs() < 1e-8,
            "newton {start:?}: {r:?}"
        );
        assert!((r.f + 0.25).abs() < 1e-12);
        let (mut ff, mut gg, mut hh) = (f, g, h);
        let r = newton_dogleg(&mut ff, Some(&mut gg), Some(&mut hh), &start, &opts()).unwrap();
        assert!(r.converged && (r.x[0].abs() - 0.5f64.sqrt()).abs() < 1e-8, "dogleg {start:?}: {r:?}");
    }
}

#[test]
fn quadratic_bowl_in_high_dimension() {
    // f = ½xᵀAx − bᵀx with SPD A: all methods must find A⁻¹b.
    let n = 20;
    let a =
        Matrix::from_fn(n, n, |i, j| if i == j { 2.0 + i as f64 } else { 1.0 / (1.0 + (i as f64 - j as f64).abs()) });
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 1.0).collect();
    let xs = Cholesky::new(&a).unwrap().solve(&b).unwrap();
    let f = |x: &[f64]| {
        0.5 * x.iter().zip(a.matvec(x).unwrap()).map(|(u, v)| u * v).sum::<f64>()
            - x.iter().zip(&b).map(|(u, v)| u * v).sum::<f64>()
    };
    let g = |x: &[f64]| a.matvec(x).unwrap().iter().zip(&b).map(|(u, v)| u - v).collect::<Vec<_>>();
    let x0 = vec![0.0; n];
    let (mut ff, mut gg) = (f, g);
    let r = bfgs(&mut ff, Some(&mut gg), &x0, &opts()).unwrap();
    assert!(dist(&r.x, &xs) < 1e-7 && r.iterations <= 3 * n, "bfgs {} iterations", r.iterations);
    let (mut ff, mut gg) = (f, g);
    let r = lbfgs(&mut ff, Some(&mut gg), &x0, &Options { memory: 5, ..opts() }).unwrap();
    assert!(dist(&r.x, &xs) < 1e-6, "lbfgs");
    let (mut ff, mut gg) = (f, g);
    let r = newton(&mut ff, Some(&mut gg), None, &x0, &opts()).unwrap();
    assert!(
        dist(&r.x, &xs) < 1e-7 && r.iterations <= 3,
        "newton: a quadratic takes one Newton step, took {}",
        r.iterations
    );
}

#[test]
fn bounded_minimisation() {
    // Separable quadratic: the solution is the clamp of the unconstrained one.
    let c = [3.0, -2.0, 0.5, 10.0];
    let (lo, hi) = ([-1.0, -1.0, -1.0, -1.0], [2.0, 2.0, 2.0, f64::INFINITY]);
    let mut f = |x: &[f64]| x.iter().zip(&c).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
    let r = lbfgs_bounded(&mut f, None, &[0.0; 4], &lo, &hi, &opts()).unwrap();
    assert!(r.converged && dist(&r.x, &[2.0, -1.0, 0.5, 10.0]) < 1e-6, "{r:?}");

    // Rosenbrock with x ≤ 0.5: the constrained optimum is (0.5, 0.25), f = 0.25.
    let mut f = |x: &[f64]| rosenbrock(x);
    let mut g = |x: &[f64]| rosenbrock_grad(x);
    let r = lbfgs_bounded(&mut f, Some(&mut g), &[-1.2, 1.0], &[-1.5, -1.5], &[0.5, 1.5], &opts()).unwrap();
    assert!(r.converged && dist(&r.x, &[0.5, 0.25]) < 1e-6 && (r.f - 0.25).abs() < 1e-10, "{r:?}");
    // KKT: the gradient pushes x₀ against its upper bound and vanishes in y.
    assert!(r.grad[0] < 0.0 && r.grad[1].abs() < 1e-6);

    // An interior optimum is found as in the unconstrained case.
    let mut f = |x: &[f64]| rosenbrock(x);
    let mut g = |x: &[f64]| rosenbrock_grad(x);
    let r = lbfgs_bounded(&mut f, Some(&mut g), &[-1.2, 1.0], &[-2.0, -2.0], &[2.0, 2.0], &opts()).unwrap();
    assert!(r.converged && dist(&r.x, &[1.0, 1.0]) < 1e-6, "{r:?}");
    // Infeasible start is projected; bad bounds are rejected.
    let mut f = |x: &[f64]| x[0] * x[0];
    let r = lbfgs_bounded(&mut f, None, &[5.0], &[1.0], &[3.0], &opts()).unwrap();
    assert!((r.x[0] - 1.0).abs() < 1e-9);
    let mut f = |x: &[f64]| x[0] * x[0];
    assert!(lbfgs_bounded(&mut f, None, &[0.0], &[2.0], &[1.0], &opts()).is_err());
}

#[test]
fn iteration_limits_and_input_errors() {
    let mut f = |x: &[f64]| rosenbrock(x);
    let r = bfgs(&mut f, None, &[-1.2, 1.0], &Options { max_iter: 3, ..opts() }).unwrap();
    assert!(!r.converged && r.termination == Termination::MaxIterations && r.iterations == 3);
    let mut f = |x: &[f64]| rosenbrock(x);
    let r = nelder_mead(&mut f, &[-1.2, 1.0], &NelderMeadOptions { max_iter: 5, restarts: 0, ..Default::default() })
        .unwrap();
    assert!(!r.converged);
    let mut f = |x: &[f64]| x[0];
    assert!(bfgs(&mut f, None, &[], &opts()).is_err());
    let mut f = |x: &[f64]| x[0];
    assert!(bfgs(&mut f, None, &[f64::NAN], &opts()).is_err());
    let mut f = |_: &[f64]| f64::NAN;
    assert!(lbfgs(&mut f, None, &[1.0], &opts()).is_err());
    let mut f = |x: &[f64]| x[0] * x[0];
    let mut bad = |_: &[f64]| vec![1.0, 2.0];
    assert!(bfgs(&mut f, Some(&mut bad), &[1.0], &opts()).is_err());
    let mut f = |x: &[f64]| x[0] * x[0];
    assert!(lbfgs(&mut f, None, &[1.0], &Options { memory: 0, ..opts() }).is_err());
}

// ---- nonlinear least squares -------------------------------------------------

fn rosen_resid(x: &[f64]) -> Vec<f64> {
    vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
}

#[test]
fn least_squares_classics() {
    let o = LsOptions::default();
    for (name, run_lm) in [("lm", true), ("gn", false)] {
        let mut r = |x: &[f64]| rosen_resid(x);
        let fit = if run_lm {
            levenberg_marquardt(&mut r, None, &[-1.2, 1.0], &o).unwrap()
        } else {
            gauss_newton(&mut r, None, &[-1.2, 1.0], &o).unwrap()
        };
        assert!(fit.converged && dist(&fit.x, &[1.0, 1.0]) < 1e-7 && fit.sse < 1e-14, "{name} rosenbrock: {fit:?}");
    }
    // Analytic Jacobian reproduces the same answer with far fewer evaluations.
    let mut r = |x: &[f64]| rosen_resid(x);
    let mut j = |x: &[f64]| Matrix::from_rows(&[vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]]).unwrap();
    let fit = levenberg_marquardt(&mut r, Some(&mut j), &[-1.2, 1.0], &o).unwrap();
    assert!(fit.converged && dist(&fit.x, &[1.0, 1.0]) < 1e-10, "{fit:?}");

    // Powell singular (residual form): rank-deficient Jacobian at the solution.
    let powell = |x: &[f64]| {
        vec![
            x[0] + 10.0 * x[1],
            5.0f64.sqrt() * (x[2] - x[3]),
            (x[1] - 2.0 * x[2]).powi(2),
            10.0f64.sqrt() * (x[0] - x[3]).powi(2),
        ]
    };
    let mut r = powell;
    let fit = levenberg_marquardt(&mut r, None, &[3.0, -1.0, 0.0, 1.0], &LsOptions { max_iter: 2000, ..o }).unwrap();
    assert!(fit.sse < 1e-14, "powell lm {fit:?}");
    let sv = openstats_math::linalg::Svd::new(&fit.jacobian).unwrap().s;
    assert!(sv[3] < 1e-5 * sv[0] && sv[2] < 1e-5 * sv[0], "Jacobian at the solution is (numerically) singular: {sv:?}");

    // Beale residuals.
    let beale_r = |x: &[f64]| {
        let (a, b) = (x[0], x[1]);
        vec![1.5 - a * (1.0 - b), 2.25 - a * (1.0 - b * b), 2.625 - a * (1.0 - b * b * b)]
    };
    let mut r = beale_r;
    let fit = levenberg_marquardt(&mut r, None, &[1.0, 1.0], &o).unwrap();
    assert!(fit.converged && dist(&fit.x, &[3.0, 0.5]) < 1e-7, "beale {fit:?}");
    let mut r = beale_r;
    let fit = gauss_newton(&mut r, None, &[1.0, 1.0], &o).unwrap();
    assert!(dist(&fit.x, &[3.0, 0.5]) < 1e-6, "beale gn {fit:?}");

    // Himmelblau as a system of residuals.
    let himmel = |x: &[f64]| vec![x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0];
    let mut r = himmel;
    let fit = levenberg_marquardt(&mut r, None, &[1.0, 1.0], &o).unwrap();
    assert!(fit.sse < 1e-20 && dist(&fit.x, &[3.0, 2.0]) < 1e-9, "himmelblau {fit:?}");
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn compare(name: &str, fit: &LeastSquaresFit, p: &[f64], se: &[f64]) {
    assert!(fit.converged, "{name}: {:?}", fit.termination);
    let got_se = fit.std_errors().unwrap().expect("m > n");
    for i in 0..p.len() {
        assert!(rel(fit.x[i], p[i]) < 1e-7, "{name} p{i}: {} vs {}", fit.x[i], p[i]);
        assert!(rel(got_se[i], se[i]) < 1e-6, "{name} se{i}: {} vs {}", got_se[i], se[i]);
    }
}

#[test]
fn curve_fits_match_scipy() {
    let o = LsOptions { central_jacobian: true, ..Default::default() };
    // a·exp(−b·t) + c
    let (t, y) = (lsq::EXP_X, lsq::EXP_Y);
    let mut r = |p: &[f64]| t.iter().zip(y).map(|(t, y)| p[0] * (-p[1] * t).exp() + p[2] - y).collect::<Vec<_>>();
    let fit = levenberg_marquardt(&mut r, None, &[1.0, 1.0, 0.0], &o).unwrap();
    compare("exp/lm", &fit, lsq::EXP_P, lsq::EXP_SE);
    let mut r = |p: &[f64]| t.iter().zip(y).map(|(t, y)| p[0] * (-p[1] * t).exp() + p[2] - y).collect::<Vec<_>>();
    let mut j = |p: &[f64]| {
        Matrix::from_fn(t.len(), 3, |i, k| match k {
            0 => (-p[1] * t[i]).exp(),
            1 => -p[0] * t[i] * (-p[1] * t[i]).exp(),
            _ => 1.0,
        })
    };
    let fit = levenberg_marquardt(&mut r, Some(&mut j), &[1.0, 1.0, 0.0], &LsOptions::default()).unwrap();
    compare("exp/lm-analytic", &fit, lsq::EXP_P, lsq::EXP_SE);
    let mut r = |p: &[f64]| t.iter().zip(y).map(|(t, y)| p[0] * (-p[1] * t).exp() + p[2] - y).collect::<Vec<_>>();
    let fit = gauss_newton(&mut r, None, &[2.0, 1.0, 0.4], &o).unwrap();
    compare("exp/gn", &fit, lsq::EXP_P, lsq::EXP_SE);

    // Michaelis–Menten v·s/(k + s)
    let (s, y) = (lsq::MM_X, lsq::MM_Y);
    let mut r = |p: &[f64]| s.iter().zip(y).map(|(s, y)| p[0] * s / (p[1] + s) - y).collect::<Vec<_>>();
    let fit = levenberg_marquardt(&mut r, None, &[0.1, 0.1], &o).unwrap();
    compare("mm", &fit, lsq::MM_P, lsq::MM_SE);

    // Gaussian peak a·exp(−(x−m)²/(2σ²)) + b
    let (x, y) = (lsq::GAUSS_X, lsq::GAUSS_Y);
    let mut r = |p: &[f64]| {
        x.iter()
            .zip(y)
            .map(|(x, y)| p[0] * (-(x - p[1]).powi(2) / (2.0 * p[2] * p[2])).exp() + p[3] - y)
            .collect::<Vec<_>>()
    };
    let fit = levenberg_marquardt(&mut r, None, &[2.0, 1.0, 1.0, 0.0], &o).unwrap();
    compare("gauss", &fit, lsq::GAUSS_P, lsq::GAUSS_SE);
}

#[test]
fn linear_problem_covariance_and_degeneracy() {
    // Linear model: LM must reproduce the closed-form covariance σ²(XᵀX)⁻¹.
    let t: Vec<f64> = (0..15).map(|i| i as f64 * 0.3).collect();
    let y: Vec<f64> =
        t.iter().enumerate().map(|(i, t)| 1.0 + 2.0 * t + 0.3 * t * t + 0.1 * ((i * 7 % 5) as f64 - 2.0)).collect();
    let x = Matrix::from_fn(15, 3, |i, k| t[i].powi(k as i32));
    let mut r = |p: &[f64]| (0..15).map(|i| p[0] + p[1] * t[i] + p[2] * t[i] * t[i] - y[i]).collect::<Vec<_>>();
    let fit = levenberg_marquardt(&mut r, None, &[0.0; 3], &LsOptions { central_jacobian: true, ..Default::default() })
        .unwrap();
    let xtx = x.t().matmul(&x).unwrap();
    let inv = Cholesky::new(&xtx).unwrap().inverse().unwrap();
    let s2 = fit.mse().unwrap();
    let cov = fit.covariance().unwrap().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(rel(cov[(i, j)], s2 * inv[(i, j)]) < 1e-6, "cov[{i}{j}]: {} vs {}", cov[(i, j)], s2 * inv[(i, j)]);
        }
    }
    assert_eq!(fit.dof(), 12);

    // Over-parameterised model: a and b only enter through a + b.
    let t = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.1, 3.9, 6.2, 7.8, 10.1];
    let mut r = |p: &[f64]| t.iter().zip(&y).map(|(t, y)| (p[0] + p[1]) * t - y).collect::<Vec<_>>();
    // Analytic Jacobian: the two columns are identical, so the rank is exactly 1
    // (a differenced Jacobian would carry noise of order 1e-11 into the rank).
    let mut j = |_: &[f64]| Matrix::from_fn(5, 2, |i, _| t[i]);
    let fit = levenberg_marquardt(&mut r, Some(&mut j), &[0.3, 0.1], &LsOptions::default()).unwrap();
    assert_eq!(fit.rank().unwrap(), 1);
    assert!(((fit.x[0] + fit.x[1]) - 2.01).abs() < 0.01);
    let cov = fit.covariance().unwrap().unwrap();
    assert!(cov.as_slice().iter().all(|v| v.is_finite()));

    // As many residuals as parameters: no covariance.
    let mut r = |p: &[f64]| vec![p[0] - 1.0, p[1] - 2.0];
    let fit = levenberg_marquardt(&mut r, None, &[0.0, 0.0], &LsOptions::default()).unwrap();
    assert!(fit.covariance().unwrap().is_none() && fit.mse().is_none());
    assert!(dist(&fit.x, &[1.0, 2.0]) < 1e-9);
}

#[test]
fn least_squares_error_handling() {
    let mut bad = |_: &[f64]| Vec::<f64>::new();
    assert!(levenberg_marquardt(&mut bad, None, &[1.0], &LsOptions::default()).is_err());
    let mut r = |_: &[f64]| vec![f64::NAN];
    assert!(levenberg_marquardt(&mut r, None, &[1.0], &LsOptions::default()).is_err());
    let mut r = |x: &[f64]| vec![x[0] - 1.0, x[0] - 2.0];
    let mut jbad = |_: &[f64]| Matrix::zeros(3, 1);
    assert!(levenberg_marquardt(&mut r, Some(&mut jbad), &[0.0], &LsOptions::default()).is_err());
    let mut r = |x: &[f64]| vec![x[0] - 1.0];
    assert!(gauss_newton(&mut r, None, &[], &LsOptions::default()).is_err());
    // Iteration cap is reported, not hidden.
    let mut r = |x: &[f64]| rosen_resid(x);
    let fit =
        levenberg_marquardt(&mut r, None, &[-1.2, 1.0], &LsOptions { max_iter: 2, ..Default::default() }).unwrap();
    assert!(!fit.converged && fit.termination == Termination::MaxIterations);
}
