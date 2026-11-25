"""
Unified comparison of the three computation-theoretic optimisers
================================================================
Imports (does NOT re-implement) the three earlier modules:

    complexity_aware_opt     certified neuron-level redundancy removal (dead / affine-rank / duplicate)
    automata_rnn_compression bounded Myhill-Nerode state merging: RNN -> minimal DFA
    automata_rnn_extended      (depth selection, start-state fix, DFA -> ReLU RNN)
    lambda_nn                typed lambda-term architectures, beta-reduction + cost-guided rewriting

The methods act on different architecture classes, so the comparison has three tracks:

  Track 1  Trained ReLU MLP on a Boolean domain   -> where redundancy is SEMANTIC (domain-dependent)
  Track 2  Hybrid net with STRUCTURAL + SEMANTIC redundancy, authored as a lambda term
  Track 3  Recurrent net on a regular language    -> where redundancy is in the STATE SPACE

All numbers are measured below with the same metrics: params, MACs, max output deviation from the
original on the full domain, and optimiser wall-clock time.

Run:  python unified_comparison.py            (all tracks)
      python unified_comparison.py 1 2        (subset)
"""
import sys
import time
import numpy as np

import complexity_aware_opt as cao
import automata_rnn_compression as arc
import automata_rnn_extended as are
import lambda_nn as lnn


# ============================================================================ bridges between modules
def mlp_to_term(net):
    """cao.MLP  ->  closed lambda term  \\x. Lin(ReLU(Lin(...x)))."""
    body, L = lnn.Var(0), len(net.W)
    for l, (W, b) in enumerate(zip(net.W, net.b)):
        body = lnn.App(lnn.Lin(W, b), body)
        if l < L - 1:
            body = lnn.App(lnn.Relu(W.shape[0]), body)
    return lnn.Lam(lnn.vec(net.sizes()[0]), body)


def term_to_mlp(term):
    """lambda term -> cao.MLP if (and only if) it is an alternating Lin/ReLU chain with scalar output."""
    if not isinstance(term, lnn.Lam):
        return None
    t, toks, layers = term.body, [], []
    while isinstance(t, lnn.App):
        if isinstance(t.f, lnn.Lin):
            toks.append("L")
            layers.append((t.f.W, t.f.b if t.f.b is not None else np.zeros(t.f.W.shape[0])))
        elif isinstance(t.f, lnn.Relu):
            toks.append("R")
        else:
            return None
        t = t.a
    if not (isinstance(t, lnn.Var) and t.i == 0):
        return None
    if toks != (["L", "R"] * len(layers))[:-1] or layers[0][0].shape[0] != 1:
        return None                                     # needs L R L R ... L and a scalar output
    return cao.MLP(layers=layers[::-1])


def lam_optimise(term):
    t0 = time.perf_counter()
    nf, steps = lnn.normalise(term)
    opt, log = lnn.rewrite(nf)
    return opt, steps, {k: v for k, v in log.items() if v}, time.perf_counter() - t0


# ============================================================================ reporting
ROWS = []


def table(title, rows):
    print(f"\n{title}")
    print(f"   {'method':<40}{'params':>8}{'MACs':>9}{'max|Δ|':>10}{'agree':>8}{'time(s)':>9}  note")
    for r in rows:
        dev = "-" if r["dev"] is None else f"{r['dev']:.1e}"
        ag = "-" if r["agree"] is None else f"{r['agree']:.4f}"
        tm = "-" if r["time"] is None else f"{r['time']:.2f}"
        print(f"   {r['name']:<40}{r['params']:>8}{r['macs']:>9}{dev:>10}{ag:>8}{tm:>9}  {r.get('note', '')}")


def mlp_row(name, net, ref_out, X, t=None, note="", sparse=False):
    out = net.forward(X)
    macs = int(sum(np.count_nonzero(w) for w in net.W)) if sparse else net.macs()
    return dict(name=name, params=net.n_params(nonzero=sparse), macs=macs, dev=float(np.abs(out - ref_out).max()),
                agree=float(np.mean((out > 0) == (ref_out > 0))), time=t, note=note)


def term_row(name, term, d, ref_out, X, t=None, note=""):
    out = lnn.run(term, X)[:, 0]
    return dict(name=name, params=lnn.params(term), macs=lnn.macs(term, d), dev=float(np.abs(out - ref_out).max()),
                agree=float(np.mean((out > 0) == (ref_out > 0))), time=t, note=note)


# ============================================================================ Track 1
def track1(seed=0, n=12):
    print("\n" + "=" * 100 + "\nTrack 1: trained ReLU MLP 12-128-128-1 on the Boolean cube {0,1}^12 (semantic redundancy)\n" + "=" * 100)
    rng = np.random.default_rng(seed)
    X = cao.boolean_cube(n)
    y = cao.target_fn(X)
    idx = rng.permutation(len(X))
    tr = idx[: len(idx) // 2]
    lo, hi = np.zeros(n), np.ones(n)
    teacher = cao.MLP([n, 128, 128, 1], rng=np.random.default_rng(seed))
    cao.train(teacher, X[tr], y[tr], epochs=600, lr=3e-3, wd=1e-4, group_lam=3.0)
    ref = teacher.forward(X)
    rows = [mlp_row("original (teacher)", teacher, ref, X, note=f"acc {cao.accuracy(teacher, X, y):.3f}")]

    t0 = time.perf_counter()
    s_ibp, log = cao.simplify(teacher, "ibp", lo, hi, verbose=False)
    rows.append(mlp_row("complexity-aware (IBP oracle)", s_ibp, ref, X, time.perf_counter() - t0, f"sizes {s_ibp.sizes()}"))
    t0 = time.perf_counter()
    s_ex, _ = cao.simplify(teacher, "exact", lo, hi, Xdom=X, verbose=False)
    rows.append(mlp_row("complexity-aware (exact oracle)", s_ex, ref, X, time.perf_counter() - t0, f"sizes {s_ex.sizes()}"))

    term = mlp_to_term(teacher)
    opt, steps, lg, t = lam_optimise(term)
    rows.append(term_row("lambda rewriting alone", opt, n, ref, X, t, f"rules {lg or 'none fired'}"))

    # pipeline: certified neuron removal, then structural rewriting of what is left
    opt2, steps2, lg2, t2 = lam_optimise(mlp_to_term(s_ex))
    rows.append(term_row("complexity-aware -> lambda", opt2, n, ref, X, t2, f"extra rules {lg2 or 'none fired'}"))

    mp = cao.magnitude_prune(teacher, s_ex.n_params(True), X[tr], y[tr])
    rows.append(mlp_row("magnitude pruning (baseline)", mp, ref, X, None, "sparse MACs, no guarantee", sparse=True))
    table("Track 1 results (deviation measured on ALL 4096 inputs)", rows)
    ROWS.append(("T1", rows))


# ============================================================================ Track 2
def hybrid_term(rng, n=12, m=96, h=64):
    """Authored as factorised layers (STRUCTURAL redundancy: two consecutive linear maps) with half of the
    units planted dead on the domain (SEMANTIC redundancy)."""
    Wa, ba = rng.normal(0, 1 / np.sqrt(n), (m, n)), np.zeros(m)
    Wb, bb = rng.normal(0, 1 / np.sqrt(m), (h, m)), rng.normal(0, .1, h)
    Wc, bc = rng.normal(0, 1 / np.sqrt(h), (h, h)), rng.normal(0, .1, h)
    bb[rng.random(h) < .5] = -1e3
    bc[rng.random(h) < .5] = -1e3
    Wo, bo = rng.normal(0, 1 / np.sqrt(h), (1, h)), np.zeros(1)
    L, A, R = lnn.Lin, lnn.App, lnn.Relu
    body = A(L(Wo, bo), A(R(h), A(L(Wc, bc), A(R(h), A(L(Wb, bb), A(L(Wa, ba), lnn.Var(0)))))))
    return lnn.Lam(lnn.vec(n), body)


def track2(seed=1, n=12):
    print("\n" + "=" * 100 + "\nTrack 2: hybrid network authored as a lambda term (structural + semantic redundancy)\n" + "=" * 100)
    rng = np.random.default_rng(seed)
    X = cao.boolean_cube(n)
    lo, hi = np.zeros(n), np.ones(n)
    term = hybrid_term(rng, n)
    lnn.infer(term)
    ref = lnn.run(term, X)[:, 0]
    rows = [term_row("original term", term, n, ref, X, note="12 -> 96 -> 64 -(ReLU)-> 64 -(ReLU)-> 1")]

    as_mlp = term_to_mlp(term)
    rows.append(dict(name="complexity-aware alone", params=0, macs=0, dev=None, agree=None, time=None,
                     note="N/A: consecutive linear layers are not in its MLP class"
                     if as_mlp is None else "applicable"))

    opt, steps, lg, t = lam_optimise(term)
    rows.append(term_row("lambda rewriting alone", opt, n, ref, X, t, f"rules {lg}"))

    net = term_to_mlp(opt)
    if net is not None:
        t0 = time.perf_counter()
        s, _ = cao.simplify(net, "exact", lo, hi, Xdom=X, verbose=False)
        rows.append(mlp_row("lambda -> complexity-aware (exact)", s, ref, X, time.perf_counter() - t0 + t,
                            f"sizes {s.sizes()}"))
        t0 = time.perf_counter()
        s2, _ = cao.simplify(net, "ibp", lo, hi, verbose=False)
        rows.append(mlp_row("lambda -> complexity-aware (IBP)", s2, ref, X, time.perf_counter() - t0 + t,
                            f"sizes {s2.sizes()}"))
    table("Track 2 results (the two methods are complementary; order matters)", rows)
    ROWS.append(("T2", rows))


# ============================================================================ Track 3
def track3(seed=0):
    print("\n" + "=" * 100 + "\nTrack 3: RNN on the language '#a%3==0 and no bb' (state-space redundancy)\n" + "=" * 100)
    rng = np.random.default_rng(seed)
    Xtr, Ytr = arc.gen_data(2000, 16, rng)
    Xs, Ys = arc.gen_data(800, 16, rng, p_avoid=0.8)
    Xl, Yl = arc.gen_data(500, 40, rng, p_avoid=0.9)
    teacher = arc.train(arc.RNN(64, np.random.default_rng(seed)), Xtr, Ytr, epochs=500)
    tl = teacher.run(Xl) > 0

    def row(name, params, macs, pred_l, t=None, note=""):
        return dict(name=name, params=params, macs=macs, dev=None, agree=float(np.mean(pred_l == tl)), time=t,
                    note=f"{note}acc40 vs truth {np.mean(pred_l == (Yl > 0.5)):.3f}")

    rows = [row("original RNN (64 units)", teacher.n_params(), teacher.macs_per_step(), tl, None, "")]
    t0 = time.perf_counter()
    D, _, T2, A2, q0, _ = are.choose_depth(teacher, Xs)
    t_dfa = time.perf_counter() - t0
    k = len(A2)
    rows.append(row(f"automata: minimal DFA ({k} states)", k * arc.ALPHA + k, 1, arc.dfa_run(T2, A2, q0, Xl), t_dfa,
                    f"D={D}; "))
    relu = are.dfa_to_relu_rnn(T2, A2, q0)
    rows.append(row("automata: DFA as sparse ReLU RNN", relu.n_params(), relu.macs_per_step(), relu.run(Xl) > 0, t_dfa,
                    "exact realisation; "))
    t0 = time.perf_counter()
    small = arc.train(arc.RNN(k, np.random.default_rng(1)), Xtr, Ytr, epochs=500)
    rows.append(row(f"baseline: small RNN from scratch ({k})", small.n_params(), small.macs_per_step(),
                    small.run(Xl) > 0, time.perf_counter() - t0, ""))
    pca = arc.train(arc.pca_compress(teacher, Xtr, k), Xtr, Ytr, epochs=150, lr=2e-3)
    rows.append(row(f"baseline: PCA + fine-tune ({k})", pca.n_params(), pca.macs_per_step(), pca.run(Xl) > 0, None, ""))
    rows.append(dict(name="complexity-aware / lambda", params=0, macs=0, dev=None, agree=None, time=None,
                     note="N/A: act on feed-forward weights, not on recurrent state"))
    table("Track 3 results (agree = agreement with the teacher on length-40 strings)", rows)
    ROWS.append(("T3", rows))


# ============================================================================ scorecard
def scorecard():
    print("\n" + "=" * 100 + "\nScorecard\n" + "=" * 100)

    def best(track, name):
        rows = dict(ROWS)[track]
        base = rows[0]
        r = next(r for r in rows if r["name"].startswith(name))
        return base["macs"] / max(r["macs"], 1), base["params"] / max(r["params"], 1), r["dev"], r["agree"]

    print(f"   {'method':<26}{'acts on':<28}{'redundancy it sees':<30}{'guarantee':<26}")
    for m in (("complexity-aware opt", "ReLU MLP + input domain", "dead/affine/duplicate neurons", "exact on domain (exact/IBP)"),
              ("lambda_nn rewriting", "any typed lambda-term net", "algebraic/structural (fusion,\nrank, idempotence)", "exact everywhere (sound)"),
              ("automata RNN->DFA", "recurrent state machines", "equivalent hidden states", "empirical (bounded D)")):
        a, b, c, d = m
        print(f"   {a:<26}{b:<28}{c.splitlines()[0]:<30}{d:<26}")
        if "\n" in c:
            print(f"   {'':<26}{'':<28}{c.splitlines()[1]:<30}")

    print("\n   Measured compression (MACs and params, original / compressed):")
    for label, track, name in (("T1 complexity-aware (exact)", "T1", "complexity-aware (exact"),
                               ("T1 complexity-aware -> lambda", "T1", "complexity-aware -> lambda"),
                               ("T1 lambda alone", "T1", "lambda rewriting alone"),
                               ("T2 lambda -> complexity-aware", "T2", "lambda -> complexity-aware (exact"),
                               ("T2 lambda alone", "T2", "lambda rewriting alone"),
                               ("T3 automata minimal DFA", "T3", "automata: minimal DFA")):
        try:
            m, p, dev, ag = best(track, name)
            dv = "n/a" if dev is None else f"{dev:.1e}"
            print(f"     {label:<34} MACs x{m:>7.1f}   params x{p:>6.1f}   max|Δ| {dv:<8} agreement {ag:.4f}")
        except StopIteration:
            pass
    print("\n   Reading guide: x-factors are not comparable ACROSS tracks (different tasks); compare methods"
          "\n   within a track, and note which ones are guaranteed exact vs empirically faithful.")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    for w in which:
        {"1": track1, "2": track2, "3": track3}[w]()
    scorecard()
