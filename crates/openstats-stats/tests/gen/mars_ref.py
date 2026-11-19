"""Reference for tests/mars.rs: a deliberately naive NumPy MARS (Friedman 1991) that refits by
least squares for every candidate and every deletion, with the same candidate-knot and tie rules
as OpenStats. Run: python3 mars_ref.py > ../common/mars_data.rs"""
import numpy as np

out = []


def arr(name, a):
    out.append(f"pub const {name}: &[f64] = &[{', '.join(repr(float(v)) for v in np.ravel(a))}];\n")


def sc(name, v):
    out.append(f"pub const {name}: f64 = {float(v)!r};\n")


def hinge(f, X):
    v, k, s = f
    x = X[:, v]
    return np.maximum(0, x - k) if s == 1 else np.maximum(0, k - x) if s == -1 else x


def column(factors, X):
    c = np.ones(len(X))
    for f in factors:
        c = c * hinge(f, X)
    return c


def rss_of(cols, y):
    A = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    return r @ r, b


def gcv(rss, n, m, pen):
    c = m + pen * (m - 1) / 2
    return np.inf if c >= n else (rss / n) / (1 - c / n) ** 2


def mars(X, y, max_terms, degree, pen, min_span, end_span, thresh, allow_linear):
    n, p = X.shape
    terms = [[]]
    cols = [np.ones(n)]
    tss = np.sum((y - y.mean()) ** 2)
    rss, _ = rss_of(cols, y)
    while True:
        if len(terms) + 1 > max_terms or rss / tss < thresh:
            break
        best = None
        for m, par in enumerate(terms):
            if len(par) >= degree:
                continue
            bm = cols[m]
            for v in range(p):
                if any(f[0] == v for f in par):
                    continue
                xs = np.sort(X[bm > 0, v])
                nm = len(xs)
                cand = []
                if nm > 2 * end_span:
                    i = end_span
                    while i < nm - end_span:
                        if not cand or cand[-1] != xs[i]:
                            cand.append(xs[i])
                        i += min_span
                if len(terms) + 2 <= max_terms:
                    for t in cand:
                        a = bm * np.maximum(0, X[:, v] - t)
                        b = bm * np.maximum(0, t - X[:, v])
                        r2, _ = rss_of(cols + [a, b], y)
                        gain = rss - r2
                        if gain > 0 and (best is None or gain > best[0]):
                            best = (gain, m, v, t, True)
                if allow_linear and nm > 0:
                    c = bm * X[:, v]
                    r2, _ = rss_of(cols + [c], y)
                    gain = rss - r2
                    if gain > 0 and (best is None or gain > best[0]):
                        best = (gain, m, v, np.nan, False)
        if best is None:
            break
        gain, m, v, t, pair = best
        if gain / tss < thresh:
            break
        par = terms[m]
        new = [par + [(v, t, 1)], par + [(v, t, -1)]] if pair else [par + [(v, 0.0, 0)]]
        for f in new:
            terms.append(f)
            cols.append(column(f, X))
        rss, _ = rss_of(cols, y)
    tss = float(((y - y.mean()) ** 2).sum())
    keep = list(range(len(terms)))
    path = []
    while True:
        r, b = rss_of([cols[j] for j in keep], y)
        path.append((list(keep), r, gcv(r, n, len(keep), pen)))
        if len(keep) == 1:
            break
        rs = [(rss_of([cols[q] for q in keep[:j] + keep[j + 1:]], y)[0], j) for j in range(1, len(keep))]
        least = min(r for r, _ in rs)
        bestd = next((r, j) for r, j in rs if r <= least + 1e-7 * tss)
        keep.pop(bestd[1])
    bi = 0
    for i, (k, r, g) in enumerate(path):
        if g < path[bi][2] or (g == path[bi][2] and len(k) < len(path[bi][0])):
            bi = i
    k = path[bi][0]
    r, b = rss_of([cols[j] for j in k], y)
    return [terms[j] for j in k], b, r, gcv(r, n, len(k), pen), [(len(q[0]), q[2]) for q in path], len(terms)


def predict(terms, b, X):
    return sum(bi * column(t, X) for t, bi in zip(terms, b))


def run(tag, X, y, **kw):
    terms, b, r, g, path, nfwd = mars(X, y, **kw)
    arr(f"{tag}_COEF", b); sc(f"{tag}_RSS", r); sc(f"{tag}_GCV", g); sc(f"{tag}_NFWD", nfwd)
    flat = []
    for t in terms:
        for q in range(3):
            if q < len(t):
                flat += [t[q][0], t[q][1] if t[q][2] != 0 else 0.0, t[q][2]]
            else:
                flat += [-1, 0.0, 0]
    arr(f"{tag}_TERMS", flat)
    arr(f"{tag}_PATH", [v for pr in path for v in pr])
    Xt = np.array([[0.1, 0.2, 0.9], [0.5, 0.5, 0.5], [0.9, 0.1, 0.3], [0.3, 0.8, 0.6]])
    arr(f"{tag}_PRED", predict(terms, b, Xt))


rng = np.random.default_rng(31)
n = 120
X = rng.random((n, 3))
y1 = 3 + 2 * np.maximum(0, X[:, 0] - 0.4) - 1.5 * np.maximum(0, 0.6 - X[:, 1]) + 0.8 * X[:, 2] + 0.15 * rng.standard_normal(n)
y2 = 1 + 4 * np.maximum(0, X[:, 0] - 0.5) * np.maximum(0, X[:, 1] - 0.3) + np.maximum(0, 0.5 - X[:, 2]) + 0.1 * rng.standard_normal(n)
arr("X", X.T); arr("Y1", y1); arr("Y2", y2)
run("A", X, y1, max_terms=15, degree=1, pen=2.0, min_span=3, end_span=5, thresh=1e-3, allow_linear=False)
run("B", X, y2, max_terms=15, degree=2, pen=3.0, min_span=3, end_span=5, thresh=1e-3, allow_linear=False)
print("// Generated by tests/gen/mars_ref.py — do not edit.")
print("#![allow(dead_code, clippy::approx_constant, clippy::excessive_precision)]")
print("".join(out))
