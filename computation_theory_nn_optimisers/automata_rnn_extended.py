"""
Extended experiments for automata-theoretic RNN compression (pure NumPy)
=======================================================================
Requires automata_rnn_compression.py in the same folder.

  Exp A  Imperfect teacher      : does DFA extraction repair or inherit RNN errors on long strings?
  Exp B  Counter language a^n b^n: does the extracted state count grow with lookahead D?
  Exp C  DFA -> sparse ReLU RNN : exact neural realisation of a DFA (+ round-trip + fine-tuning)
  Exp D  GRU + Tomita grammars  : extraction beyond the vanilla tanh RNN

Run:  python automata_rnn_extended.py            (all)
      python automata_rnn_extended.py A C        (subset)
"""
import sys
import numpy as np
import automata_rnn_compression as ac

ALPHA = ac.ALPHA
sig = ac.sigmoid


# =========================================================================== grammars
class Grammar:
    """Complete DFA over {0,1} used as ground truth and data generator."""

    def __init__(self, name, T, A, q0=0, sink=-1):
        self.name, self.T, self.A, self.q0, self.sink = name, np.array(T), np.array(A, bool), q0, sink

    def labels(self, X):
        Y = np.zeros(X.shape)
        for i, x in enumerate(X):
            q = self.q0
            for t, s in enumerate(x):
                q = self.T[q, s]
                Y[i, t] = self.A[q]
        return Y

    def gen(self, n, L, rng, p_avoid=0.8):
        X = np.zeros((n, L), int)
        for i in range(n):
            avoid, q = rng.random() < p_avoid, self.q0
            for t in range(L):
                s = int(rng.integers(0, 2))
                if avoid and self.T[q, s] == self.sink:
                    s = 1 - s
                X[i, t] = s
                q = self.T[q, s]
        return X, self.labels(X)

    def n_min_states(self):
        T2, A2, _ = ac.moore_minimise(self.T, self.A)
        return len(A2)


def mod3_nobb():
    T = np.zeros((7, 2), int)
    for c in range(3):
        for lb in range(2):
            q = c * 2 + lb
            T[q, 0] = ((c + 1) % 3) * 2
            T[q, 1] = 6 if lb else c * 2 + 1
    T[6] = 6
    A = [(q < 6 and q // 2 == 0) for q in range(7)]
    return Grammar("#a%3==0 & no bb", T, A, 0, sink=6)


def tomita():
    G = []
    # T1: 1*       (state0 accept, sink=1)
    G.append(Grammar("T1  1*", [[1, 0], [1, 1]], [1, 0], 0, 1))
    # T2: (10)*    (0 accept, 1 after '1', 2 sink)
    G.append(Grammar("T2  (10)*", [[2, 1], [0, 2], [2, 2]], [1, 0, 0], 0, 2))
    # T4: no 000   (q = trailing zeros 0..2, 3 sink)
    G.append(Grammar("T4  no 000", [[1, 0], [2, 0], [3, 0], [3, 3]], [1, 1, 1, 0], 0, 3))
    # T5: even #0 and even #1
    G.append(Grammar("T5  even#0 & even#1", [[2, 1], [3, 0], [0, 3], [1, 2]], [1, 0, 0, 0], 0, -1))
    # T6: (#0 - #1) % 3 == 0
    G.append(Grammar("T6  (#0-#1)%3==0", [[1, 2], [2, 0], [0, 1]], [1, 0, 0], 0, -1))
    # T7: 0*1*0*1*
    G.append(Grammar("T7  0*1*0*1*", [[0, 1], [2, 1], [2, 3], [4, 3], [4, 4]], [1, 1, 1, 1, 0], 0, 4))
    return G


# =========================================================================== GRU
def _adam(P, g, m, v, t, lr):
    for k in P:
        m[k] = 0.9 * m[k] + 0.1 * g[k]
        v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)


class GRU:
    def __init__(self, d, rng=None):
        rng = rng or np.random.default_rng(0)
        s = 1 / np.sqrt(d)
        self.P = {}
        for g in "zrn":
            self.P["W" + g] = rng.normal(0, 0.5, (d, ALPHA))
            self.P["U" + g] = rng.normal(0, s, (d, d))
            self.P["b" + g] = np.zeros(d)
        self.P["w"] = rng.normal(0, 0.5, d)
        self.P["c"] = np.zeros(1)
        self.h0 = None

    @property
    def d(self):
        return len(self.P["bz"])

    def _gates(self, H, x):
        P = self.P
        z = sig(P["Wz"][:, x].T + H @ P["Uz"].T + P["bz"])
        r = sig(P["Wr"][:, x].T + H @ P["Ur"].T + P["br"])
        n = np.tanh(P["Wn"][:, x].T + (r * H) @ P["Un"].T + P["bn"])
        return z, r, n

    def step(self, H, x):
        z, r, n = self._gates(H, x)
        return (1 - z) * n + z * H

    def out(self, H):
        return H @ self.P["w"] + self.P["c"][0]

    def run(self, X, cache=False):
        N, T = X.shape
        H = np.zeros((N, self.d))
        Hs, Gs, Ys = [H], [], []
        for t in range(T):
            z, r, n = self._gates(H, X[:, t])
            H = (1 - z) * n + z * H
            Hs.append(H)
            Gs.append((z, r, n))
            Ys.append(self.out(H))
        Y = np.stack(Ys, 1)
        return (Y, Hs, Gs) if cache else Y


def train_gru(net, X, Y, epochs=400, lr=1e-2, clip=5.0):
    N, T = X.shape
    P = net.P
    m = {k: 0 * v for k, v in P.items()}
    v_ = {k: 0 * v for k, v in P.items()}
    for ep in range(1, epochs + 1):
        out, Hs, Gs = net.run(X, cache=True)
        dy = (sig(out) - Y) / (N * T)
        g = {k: 0 * v for k, v in P.items()}
        dh_next = np.zeros((N, net.d))
        for t in range(T - 1, -1, -1):
            Ht, Hp = Hs[t + 1], Hs[t]
            z, r, n = Gs[t]
            x = X[:, t]
            g["w"] += dy[:, t] @ Ht
            g["c"] += dy[:, t].sum()
            dh = dy[:, t, None] * P["w"] + dh_next
            dn_pre = dh * (1 - z) * (1 - n ** 2)
            dz_pre = dh * (Hp - n) * z * (1 - z)
            dhp = dh * z
            g["Un"] += dn_pre.T @ (r * Hp)
            g["bn"] += dn_pre.sum(0)
            np.add.at(g["Wn"].T, x, dn_pre)
            drh = dn_pre @ P["Un"]
            dhp += drh * r
            dr_pre = drh * Hp * r * (1 - r)
            for gate, dpre in (("r", dr_pre), ("z", dz_pre)):
                g["U" + gate] += dpre.T @ Hp
                g["b" + gate] += dpre.sum(0)
                np.add.at(g["W" + gate].T, x, dpre)
                dhp += dpre @ P["U" + gate]
            dh_next = dhp
        tot = np.sqrt(sum((a ** 2).sum() for a in g.values()))
        if tot > clip:
            g = {k: a * clip / tot for k, a in g.items()}
        _adam(P, g, m, v_, ep, lr)
    return net


def run_out(net, X):
    return net.run(X)


def acc_model(net, X, Y):
    return float(np.mean((net.run(X) > 0) == (Y > 0.5)))


# =========================================================================== helpers
def extract_and_minimise(net, D):
    T, A, ids, reps, _ = ac.extract_dfa(net, D)
    # The RNN's output on the EMPTY prefix is never trained (labels start after the first symbol), so
    # it can split off a spurious copy of the start state. Treat that one bit as a don't-care:
    # if another class has the same continuation behaviour (signature minus the first bit), merge.
    key0 = next(k for k, v in ids.items() if v == 0)
    for k, v in ids.items():
        if v != 0 and k[1:] == key0[1:]:
            A[0] = A[v]
            break
    T2, A2, q0 = ac.moore_minimise(T, A)
    return T2, A2, q0, len(A)


def choose_depth(net, Xshort, Ds=(2, 3, 4, 5, 6), tol=0.002):
    """Pick the smallest lookahead D whose DFA matches the teacher on SHORT strings (where the
    teacher is trustworthy) to within tol of the best D. Selecting on long strings would reward
    copying the teacher's own long-range errors."""
    cands = []
    for D in Ds:
        T2, A2, q0, nraw = extract_and_minimise(net, D)
        f = float(np.mean(ac.dfa_run(T2, A2, q0, Xshort) == (net.run(Xshort) > 0)))
        cands.append((D, f, T2, A2, q0, nraw))
    best = max(c[1] for c in cands)
    return next(c for c in cands if c[1] >= best - tol)


def dfa_vs(net_or_none, T, A, q0, X, Y):
    pred = ac.dfa_run(T, A, q0, X)
    a_true = float(np.mean(pred == (Y > 0.5)))
    fid = float(np.mean(pred == (net_or_none.run(X) > 0))) if net_or_none is not None else float("nan")
    return a_true, fid


# =========================================================================== Exp A
def exp_A(seed=0):
    print("\n" + "=" * 78 + "\nExp A: imperfect teacher (trained on length 8 only, tested up to length 60)\n" + "=" * 78)
    G = mod3_nobb()
    rng = np.random.default_rng(seed)
    Xtr, Ytr = G.gen(1500, 8, rng, 0.6)
    Xs, Ys = G.gen(800, 8, rng, 0.8)
    Xl, Yl = G.gen(600, 60, rng, 0.9)
    teacher = ac.train(ac.RNN(8, np.random.default_rng(seed)), Xtr, Ytr, epochs=250)
    print(f"   teacher RNN(8): acc len8 {ac.acc(teacher, Xs, Ys):.4f} | len60 {ac.acc(teacher, Xl, Yl):.4f}"
          f"   (ground-truth minimal DFA: {G.n_min_states()} states)")
    print(f"   {'D':>2} {'|Q|':>4} {'min':>4} {'fid len8':>9} {'acc60 (truth)':>14} {'fid60 to RNN':>13}")
    for D in (2, 3, 4, 5, 6):
        T2, A2, q0, nraw = extract_and_minimise(teacher, D)
        a, f = dfa_vs(teacher, T2, A2, q0, Xl, Yl)
        fs = float(np.mean(ac.dfa_run(T2, A2, q0, Xs) == (teacher.run(Xs) > 0)))
        print(f"   {D:>2} {nraw:>4} {len(A2):>4} {fs:>9.4f} {a:>14.4f} {f:>13.4f}")
    D, fs, T2, A2, q0, _ = choose_depth(teacher, Xs)
    a, _ = dfa_vs(teacher, T2, A2, q0, Xl, Yl)
    print(f"   depth chosen on short strings: D={D} -> {len(A2)} states, acc60 {a:.4f} (teacher RNN {ac.acc(teacher, Xl, Yl):.4f})")
    print("   -> acc60(DFA) > acc60(RNN): extraction REPAIRS long-range drift. Large D over-splits states"
          "\n      (|Q| blows up) because deep probes read the teacher's own errors as new states.")


# =========================================================================== Exp B
def anbn_labels(X):
    Y = np.zeros(X.shape)
    for i, x in enumerate(X):
        na = nb = 0
        phase, ok = 0, True
        for t, s in enumerate(x):
            if s == 0:
                if phase == 1:
                    ok = False
                na += 1
            else:
                phase = 1
                nb += 1
                if nb > na:
                    ok = False
            Y[i, t] = float(ok and na == nb and na > 0)
    return Y


def anbn_data(n, L, rng, nmax=8):
    X = rng.integers(0, 2, (n, L))
    for i in range(n):
        u = rng.random()
        if u < 0.6:
            k = int(rng.integers(1, nmax + 1))
            w = [0] * k + [1] * k
            X[i, :min(L, 2 * k)] = w[:L]
        elif u < 0.8:
            k = int(rng.integers(1, nmax + 1))
            X[i, :k] = 0
    return X, anbn_labels(X)


def capacity(predict, nmax=24):
    """largest n such that a^m b^m is handled exactly (accept only at its end) for all m<=n"""
    cap = 0
    for n in range(1, nmax + 1):
        x = np.array([[0] * n + [1] * n + [0]])
        want = anbn_labels(x)[0] > 0.5
        if not np.array_equal(predict(x)[0], want):
            break
        cap = n
    return cap


def exp_B(seed=0):
    print("\n" + "=" * 78 + "\nExp B: counter language a^n b^n (non-regular): state count vs lookahead D\n" + "=" * 78)
    rng = np.random.default_rng(seed)
    X, Y = anbn_data(3000, 20, rng)
    net = ac.train(ac.RNN(32, np.random.default_rng(seed)), X, Y, epochs=900, lr=1e-2, pos_weight=8.0)
    pred_rnn = lambda x: net.run(x) > 0
    print(f"   teacher RNN(32): counting capacity = {capacity(pred_rnn)}  (largest n it handles exactly)")
    print(f"   {'D':>2} {'|Q| raw':>8} {'|Q| min':>8} {'DFA capacity':>13}")
    for D in (1, 2, 4, 6, 8, 10, 11, 12, 13, 14):
        T, A, ids, reps, _ = ac.extract_dfa(net, D, max_states=400)
        T2, A2, q0 = ac.moore_minimise(T, A)
        n_unres = int((T < 0).sum())
        T2c = np.where(T2 < 0, 0, T2)                       # unexplored (capped) edges -> state 0
        cap = capacity(lambda x: ac.dfa_run(T2c, A2, q0, x))
        flag = "  (state cap hit)" if len(A) >= 400 or n_unres else ""
        print(f"   {D:>2} {len(A):>8} {len(A2):>8} {cap:>13}{flag}")
    print("   -> |Q| grows linearly (2D+1) with D while D <= the RNN's counting capacity; a regular"
          "\n      language would plateau instead. Beyond the capacity the DFA capacity stays pinned and |Q|"
          "\n      grows erratically: deep probes are now reading the teacher's own errors, not structure.")


# =========================================================================== Exp C
def dfa_to_relu_rnn(T, A, q0):
    """Exact Elman-RNN realisation of a DFA with |Q|*|Sigma|+1 ReLU units (Minsky/Omlin-Giles style).
    unit (q',a') is 1 iff the current symbol is a' and the previous state q had delta(q,a')=q'."""
    n = len(A)
    d = n * ALPHA + 1
    st = d - 1                                            # start unit, h0[st]=1
    net = ac.RNN(d)
    net.act = "relu"
    net.Wx[:], net.Wh[:], net.b[:], net.w[:] = 0, 0, 0, 0
    for q2 in range(n):
        for a2 in range(ALPHA):
            u = q2 * ALPHA + a2
            net.Wx[u, a2] = 1.0
            net.b[u] = -1.0
            for q in range(n):
                if T[q, a2] == q2:
                    for a in range(ALPHA):
                        net.Wh[u, q * ALPHA + a] = 1.0
            if T[q0, a2] == q2:
                net.Wh[u, st] = 1.0
    for q in range(n):
        for a in range(ALPHA):
            net.w[q * ALPHA + a] = 1.0 if A[q] else -1.0
    net.w[st] = 1.0 if A[q0] else -1.0                    # output of the empty prefix = label of q0
    net.c = 0.0
    net.h0 = np.zeros(d)
    net.h0[st] = 1.0
    net.mask = (net.Wh != 0).astype(float)                # keep it sparse while fine-tuning
    return net


def exp_C(seed=0):
    print("\n" + "=" * 78 + "\nExp C: DFA -> sparse ReLU RNN (neural realisation) + round trip + fine-tune\n" + "=" * 78)
    G = mod3_nobb()
    rng = np.random.default_rng(seed)
    Xs, Ys = G.gen(800, 8, rng, 0.8)
    Xl, Yl = G.gen(600, 60, rng, 0.9)
    teacher = ac.train(ac.RNN(8, np.random.default_rng(seed)), *G.gen(1500, 8, rng, 0.6), epochs=250)
    D, _, T2, A2, q0, _ = choose_depth(teacher, Xs)
    k = len(A2)
    net = dfa_to_relu_rnn(T2, A2, q0)
    dfa_pred = ac.dfa_run(T2, A2, q0, Xl)
    same = float(np.mean((net.run(Xl) > 0) == dfa_pred))
    print(f"   extracted DFA: {k} states -> ReLU RNN with {net.d} units, {net.n_params()} non-zero params,"
          f" {net.macs_per_step()} MAC/step")
    print(f"   realisation == DFA on 60-char strings: {same:.4f}  (should be exactly 1.0)")
    T3, A3, _, nraw = extract_and_minimise(net, D)
    print(f"   round trip: DFA extracted back from the realised RNN has {len(A3)} states (original {k})")
    print(f"   acc vs truth len60:  teacher {ac.acc(teacher, Xl, Yl):.4f} | DFA {np.mean(dfa_pred == (Yl > 0.5)):.4f}"
          f" | realised RNN {ac.acc(net, Xl, Yl):.4f}")
    # fine-tune from the exact initialisation on the (short) training distribution
    Xt, Yt = G.gen(1500, 8, rng, 0.6)
    ft = dfa_to_relu_rnn(T2, A2, q0)
    ac.train(ft, Xt, Yt, epochs=100, lr=1e-3)
    print(f"   fine-tune on len-8 data  : len8 {ac.acc(ft, Xs, Ys):.4f} | len60 {ac.acc(ft, Xl, Yl):.4f}"
          f"  (start {ac.acc(net, Xs, Ys):.4f} / {ac.acc(net, Xl, Yl):.4f})")
    Xm, Ym = G.gen(1500, 30, rng, 0.8)
    ft2 = dfa_to_relu_rnn(T2, A2, q0)
    ac.train(ft2, Xm, Ym, epochs=100, lr=1e-3)
    print(f"   fine-tune on len-30 data : len8 {ac.acc(ft2, Xs, Ys):.4f} | len60 {ac.acc(ft2, Xl, Yl):.4f}")
    print("   -> the exact realisation has nothing to recover; gradient steps on short data can only"
          "\n      break its exactness (see len60). Fine-tuning matters after LOSSY compression, not here.")


# =========================================================================== Exp D
def exp_D(seed=0):
    print("\n" + "=" * 78 + "\nExp D: GRU(24) teachers on Tomita-style grammars (length 20 train, length 50 test)\n" + "=" * 78)
    print(f"   {'grammar':<22}{'min|Q|':>7}{'GRU acc50':>10}{'D':>3}{'|Q| ext':>8}{'DFA acc50':>10}{'fidelity':>9}")
    for G in tomita():
        rng = np.random.default_rng(seed)
        Xtr, Ytr = G.gen(1500, 20, rng, 0.7)
        Xl, Yl = G.gen(500, 50, rng, 0.8)
        net = train_gru(GRU(24, np.random.default_rng(seed)), Xtr, Ytr, epochs=300)
        Xsh, _ = G.gen(500, 12, np.random.default_rng(seed + 7), 0.8)
        D, _, T2, A2, q0, _ = choose_depth(net, Xsh, Ds=(3, 4, 5, 6))
        a, f = dfa_vs(net, T2, A2, q0, Xl, Yl)
        nq = len(A2)
        print(f"   {G.name:<22}{G.n_min_states():>7}{acc_model(net, Xl, Yl):>10.3f}{D:>3}{nq:>8}{a:>10.3f}{f:>9.3f}")


if __name__ == "__main__":
    which = [a.upper() for a in sys.argv[1:]] or list("ABCD")
    for w in which:
        {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D}[w]()
