"""Reference values for tests/multivariate.rs (NumPy / SciPy / scikit-learn).

Run: python3 mv_ref.py > ../common/mv_data.rs

Everything is computed with code independent of OpenStats:
* PCA: numpy.linalg.eigh of the correlation / covariance matrix.
* Factor analysis: scipy.optimize (L-BFGS-B) on the ML discrepancy ln|S| ... with Sigma = LL' + Psi (cross-checked
  against sklearn FactorAnalysis in this script); varimax / quartimax by a numpy orthomax implementation.
* Hierarchical clustering: scipy.cluster.hierarchy.linkage / fcluster. K-means: sklearn.
* Discriminant: numpy (pooled covariance, leave-one-out by refitting), sklearn QuadraticDiscriminantAnalysis,
  scipy.linalg.eigh(B, W) for canonical eigenvalues.
* MANOVA: numpy least squares for the SSCP matrices (effect coding, drop-the-term reduced models), SAS/Rao F formulas,
  scipy.stats.f; two groups cross-checked against Hotelling's T^2.
* Cronbach alpha: numpy.
"""
import numpy as np
from scipy import linalg, optimize, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis

out = []


def arr(name, a):
    out.append(f"pub const {name}: &[f64] = &[{', '.join(repr(float(v)).replace("nan", "f64::NAN") for v in np.ravel(a))}];\n")


def iarr(name, a):
    out.append(f"pub const {name}: &[usize] = &[{', '.join(str(int(v)) for v in np.ravel(a))}];\n")


def sarr(name, a):
    out.append(f"pub const {name}: &[&str] = &[{', '.join(chr(34) + str(v) + chr(34) for v in a)}];\n")


def sc(name, v):
    out.append(f"pub const {name}: f64 = {float(v)!r};\n")


iris = load_iris()
X = iris.data
y = iris.target
n, p = X.shape
arr("IRIS_X", X)
iris_names = ["setosa", "versicolor", "virginica"]
sarr("IRIS_SPECIES", [iris_names[i] for i in y])

# ---------------------------------------------------------------- PCA
for tag, mat in (("CORR", np.corrcoef(X.T)), ("COV", np.cov(X.T))):
    w, v = np.linalg.eigh(mat)
    w, v = w[::-1], v[:, ::-1]
    arr(f"PCA_{tag}_EIG", w)
    arr(f"PCA_{tag}_VEC", v)  # p x p row-major, columns = components (sign arbitrary)
    Z = (X - X.mean(0)) / (X.std(0, ddof=1) if tag == "CORR" else 1)
    arr(f"PCA_{tag}_SCORES", Z @ v)  # n x p
out.append("")

# ---------------------------------------------------------------- Factor analysis
rng = np.random.default_rng(20261120)
nf = 200
latent = rng.standard_normal((nf, 2))
load = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.7], [0.3, 0.6]])
FX = latent @ load.T + rng.standard_normal((nf, 6)) * np.array([0.5, 0.6, 0.5, 0.4, 0.6, 0.7])
arr("FA_X", FX)
R = np.corrcoef(FX.T)
S_cov = np.cov(FX.T)
pf = 6


def orient(L):
    L = L.copy()
    for j in range(L.shape[1]):
        if L[:, j].sum() < 0:
            L[:, j] *= -1
    return L


def orthomax(A, gamma, kaiser):
    pp, m = A.shape
    h = np.sqrt((A ** 2).sum(1)) if kaiser else np.ones(pp)
    An = A / h[:, None]
    T = np.eye(m)
    d = 0
    for _ in range(2000):
        B = An @ T
        G = An.T @ (B ** 3 - gamma / pp * B @ np.diag((B ** 2).sum(0)))
        U, s, Vt = np.linalg.svd(G)
        T = U @ Vt
        dn = s.sum()
        if dn < d * (1 + 1e-12):
            break
        d = dn
    Rr = (An @ T) * h[:, None]
    order = np.argsort(-(Rr ** 2).sum(0), kind="stable")
    Rr = Rr[:, order]
    for j in range(m):
        if Rr[:, j].sum() < 0:
            Rr[:, j] *= -1
    return Rr


def pc_loadings(M, m):
    w, v = np.linalg.eigh(M)
    w, v = w[::-1], v[:, ::-1]
    return orient(v[:, :m] * np.sqrt(w[:m])), w


def ml(M, m):
    pp = M.shape[0]

    def fun(psi):
        d = np.sqrt(psi)
        Ss = M / np.outer(d, d)
        w, v = np.linalg.eigh(Ss)
        w, v = w[::-1], v[:, ::-1]
        L = d[:, None] * v[:, :m] * np.sqrt(np.maximum(w[:m] - 1, 0))
        Sig = L @ L.T + np.diag(psi)
        f = np.linalg.slogdet(Sig)[1] + np.trace(M @ np.linalg.inv(Sig)) - np.linalg.slogdet(M)[1] - pp
        return f, L

    best = None
    for start in (np.full(pp, 0.5) * np.diag(M), np.diag(M) * 0.3, np.diag(M) * 0.7):
        r = optimize.minimize(lambda q: fun(q)[0], start, method="L-BFGS-B",
                              bounds=[(0.005 * M[i, i], M[i, i]) for i in range(pp)],
                              options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000, "maxfun": 100000})
        if best is None or r.fun < best.fun:
            best = r
    f, L = fun(best.x)
    return orient(L), best.x, f


arr("FA_PC_L2", pc_loadings(R, 2)[0])
arr("FA_PC_EIG", pc_loadings(R, 2)[1])
L2 = pc_loadings(R, 2)[0]
for tag, g in (("VARIMAX", 1.0), ("QUARTIMAX", 0.0), ("EQUAMAX", 1.0)):  # equamax gamma = m/2 = 1 for m = 2
    pass
arr("FA_PC_VARIMAX", orthomax(L2, 1.0, True))
arr("FA_PC_VARIMAX_RAW", orthomax(L2, 1.0, False))
arr("FA_PC_QUARTIMAX", orthomax(L2, 0.0, True))
L3 = pc_loadings(R, 3)[0]
arr("FA_PC3_L", L3)
arr("FA_PC3_EQUAMAX", orthomax(L3, 1.5, True))
arr("FA_PC3_ORTHOMAX", orthomax(L3, 0.5, False))
# regression scores for the PC / varimax solution
Zs = (FX - FX.mean(0)) / FX.std(0, ddof=1)
Lv = orthomax(L2, 1.0, True)
arr("FA_PC_SCORE_COEF", np.linalg.solve(R, Lv))
arr("FA_PC_SCORES", Zs @ np.linalg.solve(R, Lv))

for m in (1, 2):
    Lm, psi, f = ml(R, m)
    arr(f"FA_ML{m}_L", Lm)
    arr(f"FA_ML{m}_PSI", psi)
    sc(f"FA_ML{m}_F", f)
    df = ((pf - m) ** 2 - (pf + m)) / 2
    if df > 0:
        stat = (nf - 1 - (2 * pf + 5) / 6 - 2 * m / 3) * f
        sc(f"FA_ML{m}_CHI", stat)
        sc(f"FA_ML{m}_DF", df)
        sc(f"FA_ML{m}_P", stats.chi2.sf(stat, df))
    sk = FactorAnalysis(n_components=m, tol=1e-12, max_iter=100000, svd_method="lapack").fit((FX - FX.mean(0)) / FX.std(0))
    assert np.allclose(1 - sk.noise_variance_, (Lm ** 2).sum(1), atol=2e-3), (m, 1 - sk.noise_variance_, (Lm ** 2).sum(1))
Lm2, _, _ = ml(R, 2)
arr("FA_ML2_VARIMAX", orthomax(Lm2, 1.0, True))
# covariance-matrix ML (m = 1)
Lc, psic, fc = ml(S_cov, 1)
arr("FA_MLCOV1_L", Lc)
arr("FA_MLCOV1_PSI", psic)

# ---------------------------------------------------------------- Hierarchical clustering
rng = np.random.default_rng(7)
HX = np.vstack([rng.normal(0, 1, (10, 3)), rng.normal(4, 1, (10, 3)), rng.normal([0, 5, 0], 1, (10, 3))])
rng.shuffle(HX)
arr("HC_X", HX)
for tag, method in (("SINGLE", "single"), ("COMPLETE", "complete"), ("AVERAGE", "average"), ("CENTROID", "centroid"),
                    ("MEDIAN", "median"), ("WARD", "ward"), ("MCQUITTY", "weighted")):
    Z = hierarchy.linkage(HX, method=method, metric="euclidean")
    arr(f"HC_{tag}", Z)
    if tag in ("AVERAGE", "WARD"):
        for k in (2, 3, 5):
            iarr(f"HC_{tag}_CUT{k}", hierarchy.fcluster(Z, k, "maxclust"))
for tag, metric, method in (("SQEUC_AVG", "sqeuclidean", "average"), ("CITY_COMPLETE", "cityblock", "complete"),
                            ("CORR_AVG", "correlation", "average"), ("SQEUC_SINGLE", "sqeuclidean", "single")):
    arr(f"HC_{tag}", hierarchy.linkage(pdist(HX, metric), method=method))
# standardised Euclidean ward
HZ = (HX - HX.mean(0)) / HX.std(0, ddof=1)
arr("HC_STD_WARD", hierarchy.linkage(HZ, method="ward"))
# variable clustering: 1 - r and 1 - |r| with average linkage
Rv = np.corrcoef(FX.T)
for tag, D in (("CORR", 1 - Rv), ("ABS", 1 - np.abs(Rv))):
    iu = np.triu_indices(6, 1)
    arr(f"HC_VAR_{tag}", hierarchy.linkage(D[iu], method="average"))

# ---------------------------------------------------------------- K-means
rng = np.random.default_rng(11)
KX = np.vstack([rng.normal([0, 0], 0.6, (25, 2)), rng.normal([5, 5], 0.6, (30, 2)), rng.normal([0, 6], 0.6, (20, 2))])
rng.shuffle(KX)
arr("KM_X", KX)
km = KMeans(3, n_init=20, random_state=0).fit(KX)
iarr("KM_LABELS", km.labels_)
sc("KM_INERTIA", km.inertia_)
arr("KM_CENTERS", km.cluster_centers_)
# overlapping data: only the optimality conditions are tested
KY = rng.normal(0, 1, (60, 3))
arr("KM_Y", KY)
km2 = KMeans(4, n_init=30, random_state=1).fit(KY)
sc("KM_Y_BEST_INERTIA", km2.inertia_)

# ---------------------------------------------------------------- Discriminant analysis
def lda_fit(Xd, g, ng, priors):
    mu = np.array([Xd[g == k].mean(0) for k in range(ng)])
    W = sum(((Xd[g == k] - mu[k]).T @ (Xd[g == k] - mu[k])) for k in range(ng))
    pooled = W / (len(Xd) - ng)
    coef = np.linalg.solve(pooled, mu.T).T
    const = -0.5 * np.einsum("ij,ij->i", mu, coef) + np.log(priors)
    return mu, pooled, coef, const


def lda_scores(Xd, coef, const):
    return Xd @ coef.T + const


for tag, priors in (("EQ", np.ones(3) / 3), ("PROP", np.ones(3) / 3), ("GIVEN", np.array([0.5, 0.3, 0.2]))):
    mu, pooled, coef, const = lda_fit(X, y, 3, priors)
    sc_ = lda_scores(X, coef, const)
    post = np.exp(sc_ - sc_.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    pred = sc_.argmax(1)
    loo = np.empty(n, int)
    loop = np.empty((n, 3))
    for i in range(n):
        keep = np.arange(n) != i
        m_, _, c_, k_ = lda_fit(X[keep], y[keep], 3, priors)
        s = lda_scores(X[i:i + 1], c_, k_)[0]
        loo[i] = s.argmax()
        e = np.exp(s - s.max())
        loop[i] = e / e.sum()
    arr(f"DA_{tag}_COEF", coef)
    arr(f"DA_{tag}_CONST", const)
    arr(f"DA_{tag}_POST", post)
    iarr(f"DA_{tag}_PRED", pred)
    iarr(f"DA_{tag}_LOO", loo)
    arr(f"DA_{tag}_LOOPOST", loop)
    if tag == "EQ":
        arr("DA_MEANS", mu)
        arr("DA_POOLED", pooled)
        pi = np.linalg.inv(pooled)
        D = np.array([[(mu[a] - mu[b]) @ pi @ (mu[a] - mu[b]) for b in range(3)] for a in range(3)])
        arr("DA_MAHAL", D)
        # canonical eigenvalues: eigh(B, W)
        grand = X.mean(0)
        B = sum(50 * np.outer(mu[k] - grand, mu[k] - grand) for k in range(3))
        W = pooled * (n - 3)
        w = linalg.eigh(B, W, eigvals_only=True)[::-1]
        arr("DA_CANON_EIG", w[:2])
# quadratic (numpy, class covariances with divisor n_k - 1; sklearn's QDA divides by n_k in this version)
def qda_scores(Xtrain, ytrain, Xnew, priors):
    sc = []
    for k in range(3):
        Xk = Xtrain[ytrain == k]
        mu = Xk.mean(0)
        S = np.cov(Xk.T)
        d = Xnew - mu
        sc.append(-0.5 * np.linalg.slogdet(S)[1] - 0.5 * np.einsum("ij,ij->i", d @ np.linalg.inv(S), d) + np.log(priors[k]))
    return np.array(sc).T


for tag, priors in (("QEQ", np.ones(3) / 3), ("QGIVEN", np.array([0.5, 0.3, 0.2]))):
    s_ = qda_scores(X, y, X, priors)
    post = np.exp(s_ - s_.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    arr(f"DA_{tag}_POST", post)
    iarr(f"DA_{tag}_PRED", s_.argmax(1))
    loo = np.empty(n, int)
    for i in range(n):
        keep = np.arange(n) != i
        loo[i] = qda_scores(X[keep], y[keep], X[i:i + 1], priors)[0].argmax()
    iarr(f"DA_{tag}_LOO", loo)


# ---------------------------------------------------------------- MANOVA
def stats_from(theta, p_, q, v):
    lam = np.prod(1 / (1 + theta))
    pil = np.sum(theta / (1 + theta))
    hl = np.sum(theta)
    roy = theta.max()
    s = min(p_, q)
    m = (abs(p_ - q) - 1) / 2
    nn = (v - p_ - 1) / 2
    r = v - (p_ - q + 1) / 2
    u = (p_ * q - 2) / 4
    t = np.sqrt((p_ ** 2 * q ** 2 - 4) / (p_ ** 2 + q ** 2 - 5)) if p_ ** 2 + q ** 2 - 5 > 0 else 1.0
    wd1, wd2 = p_ * q, r * t - 2 * u
    wf = (1 - lam ** (1 / t)) / lam ** (1 / t) * wd2 / wd1
    pd1, pd2 = s * (2 * m + s + 1), s * (2 * nn + s + 1)
    pf_ = (2 * nn + s + 1) / (2 * m + s + 1) * pil / (s - pil)
    hf, hd1, hd2 = 2 * (s * nn + 1) * hl / (s * s * (2 * m + s + 1)), s * (2 * m + s + 1), 2 * (s * nn + 1)
    rr = max(p_, q)
    rd1, rd2 = rr, v - rr + q
    rf = roy * rd2 / rr
    rows = []
    for stat, f, d1, d2 in ((lam, wf, wd1, wd2), (hl, hf, hd1, hd2), (pil, pf_, pd1, pd2), (roy, rf, rd1, rd2)):
        rows += [stat, f, d1, d2, stats.f.sf(f, d1, d2)]
    return rows


def manova_term(Y, Xfull, cols):
    n_, p_ = Y.shape
    def sscp(Xm):
        B, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
        Rr = Y - Xm @ B
        return Rr.T @ Rr
    E = sscp(Xfull)
    rank_full = np.linalg.matrix_rank(Xfull)
    keep = [c for c in range(Xfull.shape[1]) if c not in cols]
    Xr = Xfull[:, keep]
    q = rank_full - np.linalg.matrix_rank(Xr)
    H = sscp(Xr) - E
    theta = np.sort(np.linalg.eigvals(np.linalg.solve(E, H)).real)[::-1]
    return E, H, theta, q, n_ - rank_full


def effect_coding(labels):
    lv = sorted(set(labels))
    M = np.zeros((len(labels), len(lv) - 1))
    for i, l in enumerate(labels):
        k = lv.index(l)
        if k < len(lv) - 1:
            M[i, k] = 1
        else:
            M[i, :] = -1
    return M


# one-way on iris
Xi = np.hstack([np.ones((n, 1)), effect_coding([iris_names[i] for i in y])])
E, H, theta, q, v = manova_term(X, Xi, [1, 2])
arr("MV1_E", E)
arr("MV1_H", H)
arr("MV1_THETA", theta)
arr("MV1_STATS", stats_from(theta, 4, q, v))
# two groups: Hotelling T^2
sub = y < 2
X2 = X[sub]
g2 = y[sub]
d = X2[g2 == 0].mean(0) - X2[g2 == 1].mean(0)
Sp = ((X2[g2 == 0] - X2[g2 == 0].mean(0)).T @ (X2[g2 == 0] - X2[g2 == 0].mean(0)) + (X2[g2 == 1] - X2[g2 == 1].mean(0)).T @ (X2[g2 == 1] - X2[g2 == 1].mean(0))) / 98
T2 = 50 * 50 / 100 * d @ np.linalg.solve(Sp, d)
F2 = (100 - 4 - 1) / (4 * 98) * T2
sc("MV2G_F", F2)
sc("MV2G_P", stats.f.sf(F2, 4, 95))
iarr("MV2G_SUB", sub.astype(int))
# two-factor unbalanced, 3 responses
rng = np.random.default_rng(99)
cellsizes = {("a1", "b1"): 7, ("a1", "b2"): 5, ("a2", "b1"): 9, ("a2", "b2"): 6, ("a3", "b1"): 4, ("a3", "b2"): 8}
A, Bf = [], []
for (a, b), c in cellsizes.items():
    A += [a] * c
    Bf += [b] * c
nn2 = len(A)
perm = rng.permutation(nn2)
A = [A[i] for i in perm]
Bf = [Bf[i] for i in perm]
ea = {"a1": 0.0, "a2": 1.0, "a3": -0.5}
eb = {"b1": 0.0, "b2": 0.8}
base = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
noise = rng.multivariate_normal([0, 0, 0], base, nn2)
Y2 = np.column_stack([noise[:, 0] + np.array([ea[a] for a in A]),
                      noise[:, 1] + 0.5 * np.array([eb[b] for b in Bf]),
                      noise[:, 2] + 0.3 * np.array([ea[a] * eb[b] for a, b in zip(A, Bf)])])
arr("MV3_Y", Y2)
sarr("MV3_A", A)
sarr("MV3_B", Bf)
CA = effect_coding(A)
CB = effect_coding(Bf)
CAB = np.column_stack([CA[:, i] * CB[:, j] for i in range(CA.shape[1]) for j in range(CB.shape[1])])
Xf = np.hstack([np.ones((nn2, 1)), CA, CB, CAB])
colsets = {"A": [1, 2], "B": [3], "AB": [4, 5]}
E3 = None
for tag, cols in colsets.items():
    E, H, theta, q, v = manova_term(Y2, Xf, cols)
    arr(f"MV3_{tag}_STATS", stats_from(theta, 3, q, v))
    arr(f"MV3_{tag}_H", H)
    arr(f"MV3_{tag}_THETA", theta)
    sc(f"MV3_{tag}_DF", q)
    E3 = E
arr("MV3_E", E3)
sc("MV3_EDF", v)

# ---------------------------------------------------------------- Cronbach alpha
rng = np.random.default_rng(5)
lat = rng.standard_normal(60)
CX = np.column_stack([lat * w + rng.standard_normal(60) * 0.8 for w in (1.0, 0.8, 0.6, 0.9, 0.5)])
CX[7, 2] = np.nan
CX[20, 4] = np.nan
arr("CA_X", CX)
ok = ~np.isnan(CX).any(1)
Cc = CX[ok]
k = Cc.shape[1]
cov = np.cov(Cc.T)
alpha = k / (k - 1) * (1 - np.trace(cov) / cov.sum())
rr = np.corrcoef(Cc.T)
rbar = rr[np.triu_indices(k, 1)].mean()
sc("CA_ALPHA", alpha)
sc("CA_STD_ALPHA", k * rbar / (1 + (k - 1) * rbar))
rows = []
for i in range(k):
    tot = np.delete(Cc, i, 1).sum(1)
    r_it = np.corrcoef(tot, Cc[:, i])[0, 1]
    others = np.delete(np.delete(cov, i, 0), i, 1)
    a_om = (k - 1) / (k - 2) * (1 - np.trace(others) / others.sum())
    smc = 1 - 1 / np.linalg.inv(rr)[i, i]
    rows += [tot.mean(), tot.std(ddof=1), r_it, smc, a_om]
arr("CA_ITEMS", rows)

print("// Generated by tests/gen/mv_ref.py (NumPy %s, SciPy %s). Do not edit." % (np.__version__, __import__("scipy").__version__))
print("#![allow(clippy::approx_constant, clippy::excessive_precision)]")
print("".join(out))
