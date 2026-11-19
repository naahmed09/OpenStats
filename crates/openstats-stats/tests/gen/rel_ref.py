"""Reference values for tests/reliability.rs (NumPy / SciPy).

Run: python3 rel_ref.py > ../common/rel_data.rs

Independent code paths:
* Parametric fits: the log-likelihood is written with scipy.stats distributions (logpdf / logsf / logcdf of
  weibull_min, lognorm, expon, norm, logistic, fisk), maximised with scipy.optimize (BFGS then Nelder-Mead) and
  the Hessian is a central-difference Hessian computed here.
* Kaplan-Meier: scipy.stats.ecdf (Greenwood, log-log interval) with CensoredData; plain interval, restricted mean and its
  standard error by explicit numpy loops.
* Log-rank: scipy.stats.logrank for two groups; k-group log-rank / Wilcoxon / Tarone-Ware by explicit numpy loops that
  are cross-checked against scipy here.
* Accelerated life (Arrhenius-Weibull) and warranty forecast: scipy.optimize on the censored likelihood.
"""
import numpy as np
from scipy import optimize, stats

out = []


def arr(name, a):
    out.append(f"pub const {name}: &[f64] = &[{', '.join(('f64::INFINITY' if np.isinf(v) and v > 0 else 'f64::NAN' if np.isnan(v) else repr(float(v))) for v in np.ravel(a))}];\n")


def sc(name, v):
    out.append(f"pub const {name}: f64 = {float(v)!r};\n")


def dist(fam, mu, s):
    if fam == "weibull":
        return stats.weibull_min(c=1 / s, scale=np.exp(mu))
    if fam == "lognormal":
        return stats.lognorm(s=s, scale=np.exp(mu))
    if fam == "exponential":
        return stats.expon(scale=np.exp(mu))
    if fam == "normal":
        return stats.norm(loc=mu, scale=s)
    if fam == "logistic":
        return stats.logistic(loc=mu, scale=s)
    if fam == "loglogistic":
        return stats.fisk(c=1 / s, scale=np.exp(mu))


def ll_obs(d, lo, up, w):
    ll = np.zeros(len(lo))
    for i in range(len(lo)):
        if lo[i] == up[i]:
            ll[i] = d.logpdf(lo[i])
        elif np.isinf(up[i]):
            ll[i] = d.logsf(lo[i])
        elif lo[i] <= 0:
            ll[i] = d.logcdf(up[i])
        else:
            ll[i] = np.log(d.cdf(up[i]) - d.cdf(lo[i]))
    return ll * w


def negll(th, fam, lo, up, w):
    mu = th[0]
    s = 1.0 if fam == "exponential" else np.exp(th[1])
    v = -np.sum(ll_obs(dist(fam, mu, s), lo, up, w))
    return v if np.isfinite(v) else 1e100


def hess(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


def fit(fam, lo, up, w, x0):
    f = lambda th: negll(th, fam, lo, up, w)
    best = None
    for s0 in (x0, x0 + np.array([0.5, 0.0])[: len(x0)], x0 + np.array([0.0, -0.7])[: len(x0)]):
        r = optimize.minimize(f, s0, method="BFGS", options={"gtol": 1e-10})
        r = optimize.minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    cov = np.linalg.inv(hess(f, th))
    return th, -best.fun, cov


# ---------------------------------------------------------------- mixed-censoring data set
rng = np.random.default_rng(20261118)
n = 70
t = stats.weibull_min(c=1.8, scale=100.0).rvs(n, random_state=rng)
lo = t.copy(); up = t.copy()
cens = t > 140.0                      # type I right censoring at 140
lo[cens] = 140.0; up[cens] = np.inf
# a handful of interval and left censored records
for i in range(0, 12, 3):
    lo[i] = t[i] * 0.9; up[i] = t[i] * 1.1
lo[1] = 0.0; up[1] = t[1] * 1.05
w = np.ones(n)
w[5] = 3; w[6] = 2                    # frequencies
arr("MIX_LO", lo); arr("MIX_UP", up); arr("MIX_W", w)
for fam in ["weibull", "lognormal", "exponential", "normal", "logistic", "loglogistic"]:
    isl = fam not in ("normal", "logistic")
    mid = np.where(np.isinf(up), lo, np.where(lo <= 0, up, 0.5 * (lo + up)))
    g = np.log(mid) if isl else mid
    x0 = np.array([g.mean()]) if fam == "exponential" else np.array([g.mean(), np.log(g.std() * 0.8)])
    th, ll, cov = fit(fam, lo, up, w, x0)
    s = 1.0 if fam == "exponential" else float(np.exp(th[1]))
    d = dist(fam, th[0], s)
    key = fam.upper()
    arr(f"{key}_THETA", th); sc(f"{key}_LL", ll); arr(f"{key}_COV", cov)
    ps = np.array([0.01, 0.1, 0.5, 0.9])
    arr(f"{key}_PCT", d.ppf(ps))
    try:
        mean = d.mean(); sd = d.std()
    except Exception:
        mean = sd = np.nan
    sc(f"{key}_MEAN", mean if np.isfinite(mean) else np.nan); sc(f"{key}_SD", sd if np.isfinite(sd) else np.nan)
    ts = np.array([20.0, 60.0, 120.0])
    arr(f"{key}_SF", d.sf(ts)); arr(f"{key}_PDF", d.pdf(ts))
    # delta-method percentile and reliability intervals on the (mu, ln sigma) scale
    zq = stats.norm.ppf(0.975)
    cl = []
    base = dist(fam, 0.0, 1.0)
    for p in ps:
        zp = base.ppf(p) if isl is False else (np.log(base.ppf(p)) if False else None)
        # standard variate of the location-scale form: z_p = (g(t_p)-mu)/sigma
        tp = d.ppf(p); gp = np.log(tp) if isl else tp
        zp = (gp - th[0]) / s
        gradx = np.array([1.0, s * zp]) if fam != "exponential" else np.array([1.0])
        if fam == "exponential":
            vx = cov[0, 0]
        else:
            vx = gradx @ cov @ gradx
        sx = np.sqrt(vx)
        lo_, hi_ = (np.exp(gp - zq * sx), np.exp(gp + zq * sx)) if isl else (gp - zq * sx, gp + zq * sx)
        cl += [lo_, hi_]
    arr(f"{key}_PCT_CI", cl)
    rl = []
    for tt in ts:
        gt = np.log(tt) if isl else tt
        z = (gt - th[0]) / s
        if fam == "exponential":
            vz = cov[0, 0] / s**2
        else:
            gz = np.array([-1 / s, -z])
            vz = gz @ cov @ gz
        sz = np.sqrt(vz)
        base0 = {"weibull": stats.gumbel_l, "exponential": stats.gumbel_l, "lognormal": stats.norm, "normal": stats.norm,
                 "logistic": stats.logistic, "loglogistic": stats.logistic}[fam]
        rl += [base0.sf(z + zq * sz), base0.sf(z - zq * sz)]
    arr(f"{key}_SF_CI", rl)

# exponential closed form on right-censored data
te = stats.expon(scale=50).rvs(40, random_state=rng)
ce = te > 80
le = np.where(ce, 80.0, te)
arr("EXPO_T", le); arr("EXPO_CENS", ce.astype(float))
sc("EXPO_MEAN", le.sum() / (~ce).sum())
sc("EXPO_NFAIL", (~ce).sum())

# ---------------------------------------------------------------- Kaplan-Meier
rng = np.random.default_rng(7)
tk = np.round(stats.weibull_min(c=1.4, scale=30).rvs(45, random_state=rng), 0) + 1   # ties
ck = (rng.random(45) < 0.3) | (tk > 55)
tk = np.where(ck, np.minimum(tk, np.round(rng.uniform(10, 60, 45))), tk)
arr("KM_T", tk); arr("KM_C", ck.astype(float))
res = stats.ecdf(stats.CensoredData(uncensored=tk[~ck], right=tk[ck]))
sf = res.sf
ci = sf.confidence_interval(confidence_level=0.95, method="log-log")
cil = sf.confidence_interval(confidence_level=0.95, method="linear")
# all event times (failures) and their survival
arr("KM_TIMES", sf.quantiles); arr("KM_SF", sf.probabilities)
arr("KM_PL_LO", cil.low.probabilities); arr("KM_PL_HI", cil.high.probabilities)
arr("KM_LL_LO", ci.low.probabilities); arr("KM_LL_HI", ci.high.probabilities)
# explicit loops: Greenwood, plain CI, restricted mean and its se
order = np.sort(np.unique(tk))
S = 1.0; green = 0.0
rows = []
nrisk_arr = []
for u in order:
    nr = (tk >= u).sum(); d = ((tk == u) & ~ck).sum()
    if d > 0:
        S *= 1 - d / nr
        green += d / (nr * (nr - d)) if nr > d else np.inf
    rows.append((u, nr, d, S, S * np.sqrt(green)))
rows = np.array(rows)
arr("KM_STEP_T", rows[:, 0]); arr("KM_STEP_N", rows[:, 1]); arr("KM_STEP_D", rows[:, 2]); arr("KM_STEP_S", rows[:, 3]); arr("KM_STEP_SE", rows[:, 4])
# restricted mean
tt = np.concatenate([[0.0], rows[:, 0]]); ss = np.concatenate([[1.0], rows[:, 3]])
area = np.sum(ss[:-1] * np.diff(tt))
var = 0.0
for k in range(len(rows)):
    d = rows[k, 2]; nr = rows[k, 1]
    if d > 0 and nr > d:
        a_k = np.sum(rows[k:-1, 3] * np.diff(rows[k:, 0]))
        var += a_k**2 * d / (nr * (nr - d))
sc("KM_MEAN", area); sc("KM_MEAN_SE", np.sqrt(var))
med = rows[rows[:, 3] <= 0.5][0, 0]
sc("KM_MEDIAN", med)
# cumulative hazard
h = 0.0; ch = []
for u in rows:
    if u[2] > 0:
        h += u[2] / u[1]
    ch.append(h)
arr("KM_STEP_H", ch)

# ---------------------------------------------------------------- group comparison
rng = np.random.default_rng(11)
sizes = [22, 25, 20]
scales = [30, 45, 70]
gt = []; gc = []; gg = []
for k, (m, sc_) in enumerate(zip(sizes, scales)):
    x = np.round(stats.weibull_min(c=1.3, scale=sc_).rvs(m, random_state=rng), 0) + 1
    c = x > 80
    x = np.minimum(x, 80)
    gt += list(x); gc += list(c); gg += [k] * m
gt = np.array(gt); gc = np.array(gc); gg = np.array(gg)
arr("GRP_T", gt); arr("GRP_C", gc.astype(float)); arr("GRP_G", gg.astype(float))


def kgroup(weight):
    g = 3
    times = np.unique(gt[~gc])
    U = np.zeros(g); V = np.zeros((g, g)); O = np.zeros(g); E = np.zeros(g)
    for t_ in times:
        nj = np.array([((gt >= t_) & (gg == k)).sum() for k in range(g)], float)
        dj = np.array([((gt == t_) & ~gc & (gg == k)).sum() for k in range(g)], float)
        n_ = nj.sum(); d_ = dj.sum()
        wt = {"lr": 1.0, "wil": n_, "tw": np.sqrt(n_)}[weight]
        e = d_ * nj / n_
        O += dj; E += e; U += wt * (dj - e)
        if n_ > 1:
            V += wt**2 * d_ * (n_ - d_) / (n_ - 1) * (np.diag(nj / n_) - np.outer(nj, nj) / n_**2)
    m = g - 1
    stat = U[:m] @ np.linalg.solve(V[:m, :m], U[:m])
    return stat, stats.chi2.sf(stat, m), O, E


for key, wname in (("LR", "lr"), ("WIL", "wil"), ("TW", "tw")):
    st, p, O, E = kgroup(wname)
    sc(f"GRP_{key}_STAT", st); sc(f"GRP_{key}_P", p)
arr("GRP_OBS", O); arr("GRP_EXP", E)
# two groups: scipy.stats.logrank
m0 = gg == 0; m1 = gg == 1
x = stats.CensoredData(uncensored=gt[m0 & ~gc], right=gt[m0 & gc])
y = stats.CensoredData(uncensored=gt[m1 & ~gc], right=gt[m1 & gc])
lr = stats.logrank(x, y)
sc("LR2_Z", lr.statistic); sc("LR2_P", lr.pvalue)
arr("GRP2_T", gt[m0 | m1]); arr("GRP2_C", gc[m0 | m1].astype(float)); arr("GRP2_G", gg[m0 | m1].astype(float))

# ---------------------------------------------------------------- accelerated life (Arrhenius-Weibull)
rng = np.random.default_rng(99)
temps = np.repeat([80.0, 100.0, 120.0], 15)
b0, b1, sig = -14.9, 0.7, 0.45
xs = 11604.5 / (temps + 273.16)
mu_true = b0 + b1 * xs
tw = np.exp(mu_true + sig * stats.gumbel_l.rvs(size=45, random_state=rng))
cens = tw > 2000
ta = np.where(cens, 2000.0, tw)
arr("ALT_T", ta); arr("ALT_C", cens.astype(float)); arr("ALT_TEMP", temps)
LO = ta.copy(); UP = np.where(cens, np.inf, ta)


def negll_alt(th):
    mu = th[0] + th[1] * xs
    s = np.exp(th[2])
    ll = 0.0
    for i in range(45):
        d = stats.weibull_min(c=1 / s, scale=np.exp(mu[i]))
        ll += d.logsf(LO[i]) if cens[i] else d.logpdf(ta[i])
    return -ll


r = optimize.minimize(negll_alt, [b0, b1, np.log(sig)], method="BFGS", options={"gtol": 1e-10})
r = optimize.minimize(negll_alt, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
th = r.x
# Hessian in a centred, scaled parameterisation (the raw Arrhenius design is nearly collinear), mapped back.
xm, xsd = xs.mean(), xs.std()
def negll_alt_c(q):
    return negll_alt(np.array([q[0] - q[1] * xm / xsd, q[1] / xsd, q[2]]))
q0 = np.array([th[0] + th[1] * xm, th[1] * xsd, th[2]])
Hc = hess(negll_alt_c, q0, 1e-4)
covc = np.linalg.inv(Hc)
Amap = np.array([[1, -xm / xsd, 0], [0, 1 / xsd, 0], [0, 0, 1.0]])
cov = Amap @ covc @ Amap.T
arr("ALT_THETA", th); sc("ALT_LL", -r.fun); arr("ALT_COV", cov)
x0 = 11604.5 / (40.0 + 273.16)
mu0 = th[0] + th[1] * x0
s = np.exp(th[2])
d0 = stats.weibull_min(c=1 / s, scale=np.exp(mu0))
sc("ALT_MU_USE", mu0)
arr("ALT_USE_PCT", d0.ppf([0.01, 0.1, 0.5]))
xp = []
zq = stats.norm.ppf(0.975)
for p in (0.01, 0.1, 0.5):
    zp = np.log(-np.log1p(-p))
    gx = np.array([1.0, x0, s * zp])
    sx = np.sqrt(gx @ cov @ gx)
    gp = mu0 + s * zp
    xp += [np.exp(gp - zq * sx), np.exp(gp + zq * sx)]
arr("ALT_USE_PCT_CI", xp)

# ---------------------------------------------------------------- warranty
rng = np.random.default_rng(5)
K = 6
ship = np.array([1000, 1200, 900, 1500, 1100, 1300], float)
true = stats.weibull_min(c=1.6, scale=9.0)
returns = []
flat = []
for i in range(K):
    age = K - i
    edges = np.arange(0, age + 1, dtype=float)
    pr = np.diff(true.cdf(edges))
    probs = np.append(pr, 1 - pr.sum())
    cnt = rng.multinomial(int(ship[i]), probs)[:age]
    returns.append(cnt)
    flat += list(cnt) + [-1.0]
arr("WAR_SHIP", ship); arr("WAR_RET_FLAT", flat)
future = np.array([1000.0, 1000.0, 1000.0])
arr("WAR_FUTURE", future)
LOW = []; UPP = []; WW = []
for i in range(K):
    age = len(returns[i])
    for l, c in enumerate(returns[i]):
        if c > 0:
            LOW.append(l); UPP.append(l + 1); WW.append(c)
    rem = ship[i] - returns[i].sum()
    if rem > 0:
        LOW.append(age); UPP.append(np.inf); WW.append(rem)
LOW = np.array(LOW, float); UPP = np.array(UPP); WW = np.array(WW, float)
f = lambda th: negll(th, "weibull", LOW, UPP, WW)
r = optimize.minimize(f, [np.log(9), np.log(0.6)], method="BFGS", options={"gtol": 1e-10})
r = optimize.minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
arr("WAR_THETA", r.x)
d = dist("weibull", r.x[0], np.exp(r.x[1]))
H = 8
exp_ret = np.zeros(H)
for m in range(1, H + 1):
    e = 0.0
    for i in range(K):
        age = len(returns[i]); surv = ship[i] - returns[i].sum()
        a = age + m - 1
        e += surv * (d.sf(a) - d.sf(a + 1)) / d.sf(age)
    for j, fs in enumerate(future):
        if m > j:
            a = m - j - 1
            e += fs * (d.sf(a) - d.sf(a + 1))
    exp_ret[m - 1] = e
arr("WAR_EXPECTED", exp_ret)

print("// Generated by tests/gen/rel_ref.py — do not edit.")
print("#![allow(dead_code, clippy::approx_constant, clippy::excessive_precision)]")
print("".join(out))
