"""
Complexity-aware neural-network simplification (pure NumPy, no GPU needed)
=========================================================================

Idea
----
Treat a ReLU MLP as a computational graph (a threshold/piecewise-linear circuit) and
remove structure that is PROVABLY redundant on a given input domain D:

  R1  Dead neuron        : pre-activation <= 0 on all of D      -> constant 0, delete.
  R2  Stably-active set  : pre-activation >= 0 on all of D      -> neuron is affine.
                           A set of k affine neurons whose (w|b) rows span only r<k
                           dimensions can be replaced by r neurons EXACTLY.
  R3  Duplicate neurons  : (w_j|b_j) = c * (w_i|b_i), c>0       -> merge (ReLU is positively
                           homogeneous), add c * outgoing weights.

Deciding R1/R2 exactly is coNP-hard for general domains (ReLU net verification), so we
expose three oracles with different cost / completeness:

  'exact' : enumerate the Boolean cube {0,1}^n (2^n evaluations; exact, exponential)
  'ibp'   : interval bound propagation (polynomial; SOUND but INCOMPLETE)
  'data'  : empirical activation statistics (heuristic, NOT certified; baseline only)

Every transformation under 'exact'/'ibp' preserves the function on D exactly, which we
verify by exhaustive comparison on the whole cube.

Baselines: global magnitude pruning (+fine-tune), structured neuron pruning (+fine-tune),
and random-search NAS over widths (train from scratch).

Run:  python complexity_aware_opt.py
"""
import itertools
import time
import numpy as np


# --------------------------------------------------------------------------- model
class MLP:
    """ReLU MLP with linear scalar logit output. layers = [(W[out,in], b[out]), ...]"""

    def __init__(self, sizes=None, rng=None, layers=None):
        if layers is not None:
            self.W = [w.copy() for w, _ in layers]
            self.b = [b.copy() for _, b in layers]
        else:
            rng = rng or np.random.default_rng(0)
            self.W = [rng.normal(0, np.sqrt(2 / i), (o, i)) for i, o in zip(sizes[:-1], sizes[1:])]
            self.b = [rng.normal(0, 0.1, o) for o in sizes[1:]]
        self.mask = [np.ones_like(w) for w in self.W]

    def sizes(self):
        return [self.W[0].shape[1]] + [w.shape[0] for w in self.W]

    def forward(self, X, cache=False):
        hs, pre = [X], []
        h = X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            pre.append(z)
            h = np.maximum(z, 0) if l < len(self.W) - 1 else z
            hs.append(h)
        return (h[:, 0], hs, pre) if cache else h[:, 0]

    def n_params(self, nonzero=False):
        if nonzero:
            return int(sum(np.count_nonzero(w) + np.count_nonzero(b) for w, b in zip(self.W, self.b)))
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def macs(self):
        return int(sum(w.size for w in self.W))


# --------------------------------------------------------------------------- training
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train(net, X, y, epochs=400, lr=3e-3, wd=0.0, group_lam=0.0, seed=0, verbose=False):
    """Full-batch Adam on logistic loss. group_lam>0 applies a proximal group-lasso step
    on each hidden neuron's incoming (w|b) row -> pushes whole neurons to exactly zero
    (a differentiable-free proxy for 'neuron cost' used during training)."""
    m = [np.zeros_like(p) for p in net.W + net.b]
    v = [np.zeros_like(p) for p in net.W + net.b]
    L = len(net.W)
    for t in range(1, epochs + 1):
        out, hs, pre = net.forward(X, cache=True)
        g = ((sigmoid(out) - y) / len(y))[:, None]
        gW, gb = [None] * L, [None] * L
        for l in range(L - 1, -1, -1):
            gW[l] = g.T @ hs[l] + wd * net.W[l]
            gb[l] = g.sum(0)
            if l > 0:
                g = (g @ net.W[l]) * (pre[l - 1] > 0)
        grads = gW + gb
        params = net.W + net.b
        for i, (p, gr) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr ** 2
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        if group_lam > 0:
            for l in range(L - 1):
                row = np.hstack([net.W[l], net.b[l][:, None]])
                nrm = np.linalg.norm(row, axis=1, keepdims=True) + 1e-12
                scale = np.maximum(0, 1 - lr * group_lam / nrm)
                net.W[l] *= scale
                net.b[l] *= scale[:, 0]
        for l in range(L):
            net.W[l] *= net.mask[l]
        if verbose and t % 100 == 0:
            print(f"   epoch {t}: loss={_loss(net, X, y):.4f}")
    return net


def _loss(net, X, y):
    p = sigmoid(net.forward(X))
    return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))


def accuracy(net, X, y):
    return float(np.mean((net.forward(X) > 0) == (y > 0.5)))


# --------------------------------------------------------------------------- oracles
def stability_ibp(net, lo, hi):
    """Interval bound propagation. Returns per-hidden-layer (pre_lo, pre_hi). Sound, incomplete."""
    res = []
    for l in range(len(net.W) - 1):
        W, b = net.W[l], net.b[l]
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        zl = Wp @ lo + Wn @ hi + b
        zh = Wp @ hi + Wn @ lo + b
        res.append((zl, zh))
        lo, hi = np.maximum(zl, 0), np.maximum(zh, 0)
    return res


def stability_exact(net, Xdomain):
    """Exact on a finite domain (e.g. full Boolean cube) by enumeration."""
    _, _, pre = net.forward(Xdomain, cache=True)
    return [(z.min(0), z.max(0)) for z in pre[:-1]]


def stability_data(net, Xdata):
    return stability_exact(net, Xdata)  # same computation, but on a SAMPLE -> not certified


def get_bounds(net, mode, lo, hi, Xdom, Xdata):
    if mode == "ibp":
        return stability_ibp(net, lo, hi)
    if mode == "exact":
        return stability_exact(net, Xdom)
    if mode == "data":
        return stability_data(net, Xdata)
    raise ValueError(mode)


# --------------------------------------------------------------------------- rules
def _drop_neurons(net, l, keep):
    net.W[l], net.b[l] = net.W[l][keep], net.b[l][keep]
    net.W[l + 1] = net.W[l + 1][:, keep]
    net.mask = [np.ones_like(w) for w in net.W]


def rule_dead(net, l, zl, zh):
    keep = ~(zh <= 0)
    if keep.all():
        return 0
    _drop_neurons(net, l, keep)
    return int((~keep).sum())


def rule_active_rank(net, l, zl, zh, tol=1e-8):
    """Stably-active neurons are affine: h_j = w_j.x + b_j.  If their augmented rows are
    linearly dependent, express dependents via a basis subset and fold into the next layer."""
    act = np.where(zl >= 0)[0]
    if len(act) < 2:
        return 0
    A = np.hstack([net.W[l][act], net.b[l][act, None]])  # k x (d+1)
    basis, Q = [], []
    for i in range(len(act)):
        r = A[i].copy()
        for q in Q:
            r -= (r @ q) * q
        if np.linalg.norm(r) > tol * max(1.0, np.linalg.norm(A[i])):
            basis.append(i)
            Q.append(r / np.linalg.norm(r))
    rest = [i for i in range(len(act)) if i not in basis]
    if not rest:
        return 0
    B = A[basis]  # r x (d+1)
    C = np.linalg.lstsq(B.T, A[rest].T, rcond=None)[0].T  # len(rest) x r
    if np.abs(C @ B - A[rest]).max() > 1e-6:
        return 0
    bi, ri = act[basis], act[rest]
    net.W[l + 1][:, bi] += net.W[l + 1][:, ri] @ C
    keep = np.ones(net.W[l].shape[0], bool)
    keep[ri] = False
    _drop_neurons(net, l, keep)
    return len(ri)


def rule_duplicates(net, l, tol=1e-7):
    A = np.hstack([net.W[l], net.b[l][:, None]])
    n, removed = len(A), []
    for i in range(n):
        if i in removed or np.linalg.norm(A[i]) < 1e-12:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue
            c = (A[j] @ A[i]) / (A[i] @ A[i])
            if c > 0 and np.linalg.norm(A[j] - c * A[i]) < tol * max(1, np.linalg.norm(A[j])):
                net.W[l + 1][:, i] += c * net.W[l + 1][:, j]
                removed.append(j)
    if not removed:
        return 0
    keep = np.ones(n, bool)
    keep[removed] = False
    _drop_neurons(net, l, keep)
    return len(removed)


def simplify(net, mode, lo, hi, Xdom=None, Xdata=None, verbose=True):
    """Apply R1-R3 to a fixed point. Returns (new_net, log)."""
    net = MLP(layers=list(zip(net.W, net.b)))
    log = {"dead": 0, "affine_rank": 0, "duplicate": 0}
    changed = True
    while changed:
        changed = False
        for l in range(len(net.W) - 1):
            zl, zh = get_bounds(net, mode, lo, hi, Xdom, Xdata)[l]
            for key, fn in (("dead", lambda: rule_dead(net, l, zl, zh)),):
                k = fn()
                if k:
                    log[key] += k
                    changed = True
                    break
            else:
                zl, zh = get_bounds(net, mode, lo, hi, Xdom, Xdata)[l]
                k = rule_active_rank(net, l, zl, zh)
                if k:
                    log["affine_rank"] += k
                    changed = True
                    continue
                k = rule_duplicates(net, l)
                if k:
                    log["duplicate"] += k
                    changed = True
                continue
            break  # restart sweep after a dead-neuron removal (bounds changed)
    if verbose:
        print(f"   [{mode}] removed: {log}  -> sizes {net.sizes()}")
    return net, log


# --------------------------------------------------------------------------- baselines
def magnitude_prune(net, target_nonzero, X, y, ft_epochs=200, lr=1e-3):
    net = MLP(layers=list(zip(net.W, net.b)))
    allw = np.concatenate([np.abs(w).ravel() for w in net.W])
    k = max(0, len(allw) - (target_nonzero - sum(b.size for b in net.b)))
    thr = np.sort(allw)[min(k, len(allw) - 1)]
    net.mask = [(np.abs(w) >= thr).astype(float) for w in net.W]
    for l in range(len(net.W)):
        net.W[l] *= net.mask[l]
    return train(net, X, y, epochs=ft_epochs, lr=lr)


def structured_prune(net, keep_frac, X, y, ft_epochs=200, lr=1e-3):
    net = MLP(layers=list(zip(net.W, net.b)))
    for l in range(len(net.W) - 1):
        nrm = np.linalg.norm(np.hstack([net.W[l], net.b[l][:, None]]), axis=1) * \
              np.linalg.norm(net.W[l + 1], axis=0)
        k = max(1, int(round(keep_frac * len(nrm))))
        keep = np.zeros(len(nrm), bool)
        keep[np.argsort(-nrm)[:k]] = True
        _drop_neurons(net, l, keep)
    return train(net, X, y, epochs=ft_epochs, lr=lr)


def random_search_nas(n_in, budget_params, Xtr, ytr, Xva, yva, trials=12, epochs=400, seed=0):
    rng = np.random.default_rng(seed)
    best, best_acc = None, -1
    for t in range(trials):
        depth = rng.integers(1, 4)
        sizes = [n_in] + [int(rng.integers(2, 64)) for _ in range(depth)] + [1]
        net = MLP(sizes, rng=np.random.default_rng(t))
        if net.n_params() > budget_params:
            continue
        train(net, Xtr, ytr, epochs=epochs)
        a = accuracy(net, Xva, yva)
        if a > best_acc:
            best, best_acc = net, a
    return best


# --------------------------------------------------------------------------- experiment
def boolean_cube(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)


def target_fn(X):
    # structured Boolean function: (x0 XOR x1) OR (x2 AND x3 AND x4) XOR x5   (other inputs irrelevant)
    a = (X[:, 0] + X[:, 1]) % 2
    b = X[:, 2] * X[:, 3] * X[:, 4]
    return (np.maximum(a, b) + X[:, 5]) % 2


def fidelity(net, ref, Xdom):
    d = np.abs(net.forward(Xdom) - ref.forward(Xdom))
    return float(d.max()), float(np.mean((net.forward(Xdom) > 0) == (ref.forward(Xdom) > 0)))


def report(name, net, ref, Xdom, ydom, Xte, yte):
    md, agree = fidelity(net, ref, Xdom)
    print(f"{name:<34} params(nz)={net.n_params(True):>6}  MACs={net.macs():>6}  sizes={net.sizes()}\n"
          f"{'':<34} test acc={accuracy(net, Xte, yte):.3f}  cube acc={accuracy(net, Xdom, ydom):.3f}  "
          f"max|logit diff|={md:.2e}  agreement={agree:.4f}")


def main(n=12, seed=0):
    rng = np.random.default_rng(seed)
    Xdom = boolean_cube(n)
    ydom = target_fn(Xdom)
    idx = rng.permutation(len(Xdom))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    Xtr, ytr, Xte, yte = Xdom[tr], ydom[tr], Xdom[te], ydom[te]
    lo, hi = np.zeros(n), np.ones(n)

    print("== 1. Train over-parameterised teacher (with group-sparsity regulariser) ==")
    teacher = MLP([n, 128, 128, 1], rng=np.random.default_rng(seed))
    train(teacher, Xtr, ytr, epochs=600, lr=3e-3, wd=1e-4, group_lam=3.0)
    report("teacher", teacher, teacher, Xdom, ydom, Xte, yte)

    print("\n== 2. Certified simplification (function preserved EXACTLY on the domain) ==")
    results = {}
    for mode in ("ibp", "exact"):
        t0 = time.time()
        s, log = simplify(teacher, mode, lo, hi, Xdom=Xdom)
        results[mode] = s
        print(f"   oracle time: {time.time() - t0:.2f}s")
        report(f"complexity-aware ({mode})", s, teacher, Xdom, ydom, Xte, yte)
    s, _ = simplify(teacher, "data", lo, hi, Xdata=Xtr[:200])
    print("   (data oracle is a heuristic: it sees only 200 samples, so no guarantee)")
    report("data-driven (uncertified)", s, teacher, Xdom, ydom, Xte, yte)

    target = results["exact"]
    print("\n== 3. Baselines at matched size ==")
    mp = magnitude_prune(teacher, target.n_params(True), Xtr, ytr)
    report("magnitude pruning (unstructured)", mp, teacher, Xdom, ydom, Xte, yte)
    frac = target.sizes()[1] / teacher.sizes()[1]
    sp = structured_prune(teacher, max(frac, 1 / 128), Xtr, ytr)
    report("structured neuron pruning", sp, teacher, Xdom, ydom, Xte, yte)
    nas = random_search_nas(n, max(target.n_params(), 100), Xtr, ytr, Xte, yte)
    if nas is not None:
        report("random-search NAS", nas, teacher, Xdom, ydom, Xte, yte)

    print("\nKey: 'max|logit diff|' ~ 0 means the simplified net computes the SAME function on D;"
          "\npruning/NAS give no such guarantee (non-zero diff, agreement < 1).")


if __name__ == "__main__":
    main()
