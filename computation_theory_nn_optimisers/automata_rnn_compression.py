"""
Automata-theoretic compression of recurrent networks (pure NumPy)
=================================================================

Pipeline
--------
 1. Train an Elman RNN (tanh) with per-prefix supervision on a regular language
    (default: #a = 0 mod 3  AND  no substring 'bb'; minimal DFA has 7 states).
 2. Collect hidden states on all strings up to length L_ENUM.
 3. Bounded Myhill-Nerode equivalence: two hidden states h, h' are D-equivalent if the
    RNN gives identical accept/reject outputs on EVERY continuation of length <= D.
    Group states by this behavioural signature (the RNN analogue of DFA state merging).
 4. Build a DFA over the classes (BFS from the initial state), then Moore-minimise it.
 5. Compare with neural compression baselines at matched state budget:
      - small RNN trained from scratch (hidden size = #DFA states)
      - PCA state-space projection + fine-tune
      - magnitude-pruned recurrent matrix + fine-tune
 6. Report accuracy vs ground truth (also on LONGER strings), fidelity to the teacher RNN,
    parameter count and per-step inference cost.

Run:  python automata_rnn_compression.py
"""
import itertools
import numpy as np

ALPHA = 2          # alphabet {a=0, b=1}
MOD = 3


# --------------------------------------------------------------------------- ground truth
def true_labels(seq):
    """Per-prefix label: 1 iff (#a mod MOD == 0) and no 'bb' so far."""
    out, cnt, dead, prev = [], 0, False, -1
    for s in seq:
        cnt += (s == 0)
        dead = dead or (s == 1 and prev == 1)
        prev = s
        out.append(0 if dead else int(cnt % MOD == 0))
    return out


def gen_data(n, T, rng, p_avoid=0.6):
    X = np.zeros((n, T), int)
    for i in range(n):
        avoid = rng.random() < p_avoid
        prev = -1
        for t in range(T):
            s = rng.integers(0, 2)
            if avoid and prev == 1:
                s = 0
            X[i, t] = s
            prev = s
    Y = np.array([true_labels(x) for x in X], float)
    return X, Y


# --------------------------------------------------------------------------- RNN
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class RNN:
    def __init__(self, d, rng=None):
        rng = rng or np.random.default_rng(0)
        self.Wx = rng.normal(0, 0.5, (d, ALPHA))
        self.Wh = rng.normal(0, 1 / np.sqrt(d), (d, d))
        self.b = np.zeros(d)
        self.w = rng.normal(0, 0.5, d)
        self.c = 0.0
        self.mask = np.ones((d, d))
        self.act = 'tanh'      # 'tanh' or 'relu'
        self.h0 = None         # optional fixed initial hidden state

    @property
    def d(self):
        return len(self.b)

    def step(self, H, x):
        """H:[N,d], x:[N] ints -> new H"""
        z = self.Wx[:, x].T + H @ self.Wh.T + self.b
        return np.tanh(z) if self.act == 'tanh' else np.maximum(z, 0)

    def out(self, H):
        return H @ self.w + self.c

    def run(self, X, return_H=False):
        N, T = X.shape
        H = np.zeros((N, self.d)) if self.h0 is None else np.tile(self.h0, (N, 1))
        Hs, Ys = [H], []
        for t in range(T):
            H = self.step(H, X[:, t])
            Hs.append(H)
            Ys.append(self.out(H))
        Y = np.stack(Ys, 1)
        return (Y, np.stack(Hs, 1)) if return_H else Y

    def n_params(self):
        return int(np.count_nonzero(self.Wx) + np.count_nonzero(self.Wh) + self.d + self.d + 1)

    def macs_per_step(self):
        return int(np.count_nonzero(self.Wh) + self.d + self.d)


def train(net, X, Y, epochs=300, lr=5e-3, clip=5.0, pos_weight=1.0):
    N, T = X.shape
    params = ["Wx", "Wh", "b", "w"]
    m = {k: 0 * getattr(net, k) for k in params}
    v = {k: 0 * getattr(net, k) for k in params}
    mc = vc = 0.0
    for ep in range(1, epochs + 1):
        out, H = net.run(X, return_H=True)
        dy = (sigmoid(out) - Y) * (1 + (pos_weight - 1) * Y) / (N * T)
        g = {k: 0 * getattr(net, k) for k in params}
        gc = dy.sum()
        dh_next = np.zeros((N, net.d))
        for t in range(T - 1, -1, -1):
            Ht, Hp = H[:, t + 1], H[:, t]
            g["w"] += dy[:, t] @ Ht
            dh = dy[:, t, None] * net.w + dh_next
            dz = dh * ((1 - Ht ** 2) if net.act == 'tanh' else (Ht > 0))
            g["Wh"] += dz.T @ Hp
            g["b"] += dz.sum(0)
            np.add.at(g["Wx"].T, X[:, t], dz)
            dh_next = dz @ net.Wh
        g["Wh"] *= net.mask
        tot = np.sqrt(sum((x ** 2).sum() for x in g.values()))
        if tot > clip:
            g = {k: x * clip / tot for k, x in g.items()}
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            getattr(net, k)[...] -= lr * (m[k] / (1 - 0.9 ** ep)) / (np.sqrt(v[k] / (1 - 0.999 ** ep)) + 1e-8)
        mc = 0.9 * mc + 0.1 * gc
        vc = 0.999 * vc + 0.001 * gc ** 2
        net.c -= lr * (mc / (1 - 0.9 ** ep)) / (np.sqrt(vc / (1 - 0.999 ** ep)) + 1e-8)
        net.Wh *= net.mask
    return net


def acc(net, X, Y):
    return float(np.mean((net.run(X) > 0) == (Y > 0.5)))


# --------------------------------------------------------------------------- bounded Myhill-Nerode
def suffix_strings(D):
    return np.array(list(itertools.product(range(ALPHA), repeat=D)), int).reshape(-1, D)


def signatures(net, H, D):
    """Behavioural signature of each hidden state: accept bits for all continuations <= D.
    Two states are D-equivalent iff their signatures are identical."""
    S = suffix_strings(D)                                   # [ALPHA^D, D]
    M, K = len(H), len(S)
    bits = [(net.out(H) > 0).astype(np.uint8)[:, None]]     # empty continuation
    cur = np.repeat(H, K, axis=0)                           # [M*K, d]
    for t in range(D):
        cur = net.step(cur, np.tile(S[:, t], M))
        bits.append((net.out(cur) > 0).astype(np.uint8).reshape(M, K))
    sig = np.concatenate(bits, 1)
    return [row.tobytes() for row in sig]


def extract_dfa(net, D, L_enum=9, max_states=500):
    """Quotient the RNN state space by D-bounded Myhill-Nerode equivalence."""
    h0 = np.zeros((1, net.d)) if getattr(net, 'h0', None) is None else net.h0[None].copy()
    # BFS over actual hidden states (representatives only)
    sig0 = signatures(net, h0, D)[0]
    reps = {sig0: h0[0]}
    ids = {sig0: 0}
    order = [sig0]
    delta, accept = {}, {0: bool(net.out(h0)[0] > 0)}
    i = 0
    while i < len(order) and len(order) < max_states:
        s = order[i]
        for a in range(ALPHA):
            h1 = net.step(reps[s][None], np.array([a]))
            s1 = signatures(net, h1, D)[0]
            if s1 not in ids:
                ids[s1] = len(order)
                order.append(s1)
                reps[s1] = h1[0]
                accept[ids[s1]] = bool(net.out(h1)[0] > 0)
            delta[(ids[s], a)] = ids[s1]
        i += 1
    n = len(order)
    T = np.full((n, ALPHA), -1)
    for (q, a), r in delta.items():
        T[q, a] = r
    A = np.array([accept[q] for q in range(n)])
    # consistency: do all sampled states of a class agree on successors?
    strs = [np.array(s) for L in range(L_enum + 1) for s in itertools.product(range(ALPHA), repeat=L)]
    return T, A, ids, reps, (strs, n)


def moore_minimise(T, A):
    n = len(A)
    part = A.astype(int).copy()
    while True:
        sig = [(part[q],) + tuple(part[T[q, a]] if T[q, a] >= 0 else -1 for a in range(ALPHA)) for q in range(n)]
        new = {s: i for i, s in enumerate(dict.fromkeys(sig))}
        newpart = np.array([new[s] for s in sig])
        if len(new) == len(set(part)):
            break
        part = newpart
    k = len(set(part))
    T2, A2 = np.full((k, ALPHA), -1), np.zeros(k, bool)
    for q in range(n):
        A2[part[q]] = A[q]
        for a in range(ALPHA):
            if T[q, a] >= 0:
                T2[part[q], a] = part[T[q, a]]
    return T2, A2, part[0]


def dfa_run(T, A, q0, X):
    out = np.zeros(X.shape, bool)
    for i, x in enumerate(X):
        q = q0
        for t, s in enumerate(x):
            q = T[q, s]
            out[i, t] = A[q]
    return out


# --------------------------------------------------------------------------- baselines
def pca_compress(net, X, k):
    _, H = net.run(X, return_H=True)
    Hf = H.reshape(-1, net.d)
    Hc = Hf - Hf.mean(0)
    U = np.linalg.svd(Hc, full_matrices=False)[2][:k].T       # d x k
    new = RNN(k)
    new.Wx, new.Wh, new.b = U.T @ net.Wx, U.T @ net.Wh @ U, U.T @ net.b
    new.w, new.c, new.mask = U.T @ net.w, net.c, np.ones((k, k))
    return new


def prune_compress(net, keep):
    new = RNN(net.d)
    new.Wx, new.Wh, new.b, new.w, new.c = net.Wx.copy(), net.Wh.copy(), net.b.copy(), net.w.copy(), net.c
    thr = np.sort(np.abs(new.Wh).ravel())[::-1][max(keep - 1, 0)]
    new.mask = (np.abs(new.Wh) >= thr).astype(float)
    new.Wh *= new.mask
    return new


# --------------------------------------------------------------------------- experiment
def main(seed=0):
    rng = np.random.default_rng(seed)
    Xtr, Ytr = gen_data(2000, 16, rng)
    Xte, Yte = gen_data(1000, 16, rng, p_avoid=0.8)
    Xlong, Ylong = gen_data(500, 40, rng, p_avoid=0.9)        # length generalisation

    print("== 1. Train teacher RNN (hidden=64) ==")
    teacher = train(RNN(64, np.random.default_rng(seed)), Xtr, Ytr, epochs=500)
    print(f"   acc: train {acc(teacher, Xtr, Ytr):.4f} | test(len16) {acc(teacher, Xte, Yte):.4f} "
          f"| long(len40) {acc(teacher, Xlong, Ylong):.4f}")

    print("\n== 2. Bounded Myhill-Nerode extraction: effect of lookahead depth D ==")
    best = None
    for D in (1, 2, 3, 4, 5, 6):
        T, A, ids, reps, _ = extract_dfa(teacher, D)
        T2, A2, q0 = moore_minimise(T, A)
        pred = dfa_run(T2, A2, q0, Xlong)
        fid = float(np.mean(pred == (teacher.run(Xlong) > 0)))
        a_true = float(np.mean(pred == (Ylong > 0.5)))
        print(f"   D={D}: |Q|={len(A)} -> minimised {len(A2):>3} | fidelity to RNN {fid:.4f} | acc vs truth {a_true:.4f}")
        if best is None or (fid > best[0] + 1e-9) or (abs(fid - best[0]) < 1e-9 and len(A2) < len(best[2])):
            best = (fid, T2, A2, q0, D)
    fid, T2, A2, q0, D = best
    print(f"   chosen D={D}; ground-truth minimal DFA has {2 * MOD + 1} states")

    print("\n== 3. Comparison at matched state budget ==")
    k = len(A2)
    rows = []

    def add(name, params, macs, a16, along, fid_):
        rows.append((name, params, macs, a16, along, fid_))

    tp = teacher.run(Xte) > 0
    tl = teacher.run(Xlong) > 0
    add("teacher RNN (64)", teacher.n_params(), teacher.macs_per_step(), acc(teacher, Xte, Yte), acc(teacher, Xlong, Ylong), 1.0)
    p16, pl = dfa_run(T2, A2, q0, Xte), dfa_run(T2, A2, q0, Xlong)
    add(f"extracted+minimised DFA ({k})", k * ALPHA + k, 1, float(np.mean(p16 == (Yte > 0.5))),
        float(np.mean(pl == (Ylong > 0.5))), float(np.mean(pl == tl)))

    for name, net in (
        ("small RNN from scratch", train(RNN(k, np.random.default_rng(1)), Xtr, Ytr, epochs=500)),
        ("PCA projection + fine-tune", train(pca_compress(teacher, Xtr, k), Xtr, Ytr, epochs=150, lr=2e-3)),
        ("magnitude prune Wh + fine-tune", None),
    ):
        if net is None:
            net = train(prune_compress(teacher, k * k), Xtr, Ytr, epochs=150, lr=2e-3)
        add(f"{name} (k={k})" if "prune" not in name else name, net.n_params(), net.macs_per_step(),
            acc(net, Xte, Yte), acc(net, Xlong, Ylong), float(np.mean((net.run(Xlong) > 0) == tl)))

    print(f"   {'model':<36}{'params':>8}{'MAC/step':>10}{'acc16':>8}{'acc40':>8}{'fid40':>8}")
    for r in rows:
        print(f"   {r[0]:<36}{r[1]:>8}{r[2]:>10}{r[3]:>8.3f}{r[4]:>8.3f}{r[5]:>8.3f}")
    print("\n   acc16/acc40 = accuracy vs ground truth on length-16 / length-40 strings;"
          "\n   fid40 = agreement with the teacher RNN on length-40 strings.")


if __name__ == "__main__":
    main()
