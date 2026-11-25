"""
Lambda calculus for neural architectures (pure NumPy)
=====================================================

Idea
----
Write a network as a simply-typed lambda term whose constants are neural primitives
(Lin(W,b), ReLU, Add). Then:

  * types are tensor shapes          -> ill-shaped architectures are rejected by type checking
  * depth is a Church numeral        -> `church(n) block` IS an n-layer network
  * beta-reduction                   -> compiles higher-order definitions into a concrete graph
  * Church-Rosser (confluence)       -> every reduction order reaches the same normal form
  * sound rewrite rules on the normal form (linear fusion, ReLU idempotence, eta, exact
    low-rank refactoring) -> function-preserving simplification, applied only when the COST
    MODEL (MACs) strictly improves.

Run:  python lambda_nn.py
"""
import random
import numpy as np


# ============================================================================ types
def vec(n):
    return ("vec", n)


def arr(a, b):
    return ("arr", a, b)


# ============================================================================ terms (de Bruijn)
class Var:
    def __init__(self, i): self.i = i


class Lam:
    def __init__(self, dom, body): self.dom, self.body = dom, body


class App:
    def __init__(self, f, a): self.f, self.a = f, a


class Lin:           # constant  vec(in) -> vec(out);  W: [out,in], b: [out] or None
    def __init__(self, W, b=None): self.W, self.b = W, b


class Relu:          # constant  vec(n) -> vec(n)
    def __init__(self, n): self.n = n


class Add:           # constant  vec(n) -> vec(n) -> vec(n)
    def __init__(self, n): self.n = n


def infer(t, ctx=()):
    """Type inference. Shapes are types, so a shape mismatch is a type error."""
    if isinstance(t, Var):
        return ctx[t.i]
    if isinstance(t, Lam):
        return arr(t.dom, infer(t.body, (t.dom,) + tuple(ctx)))
    if isinstance(t, App):
        tf, ta = infer(t.f, ctx), infer(t.a, ctx)
        if tf[0] != "arr" or tf[1] != ta:
            raise TypeError(f"cannot apply {tf} to {ta}")
        return tf[2]
    if isinstance(t, Lin):
        return arr(vec(t.W.shape[1]), vec(t.W.shape[0]))
    if isinstance(t, Relu):
        return arr(vec(t.n), vec(t.n))
    if isinstance(t, Add):
        return arr(vec(t.n), arr(vec(t.n), vec(t.n)))
    raise TypeError(t)


def show(t):
    if isinstance(t, Var): return f"#{t.i}"
    if isinstance(t, Lam): return f"(λ.{show(t.body)})"
    if isinstance(t, App): return f"({show(t.f)} {show(t.a)})"
    if isinstance(t, Lin): return f"L{t.W.shape[0]}x{t.W.shape[1]}"
    if isinstance(t, Relu): return "relu"
    if isinstance(t, Add): return "add"


def teq(a, b):
    """alpha-equivalence (de Bruijn => structural); constants compared by value."""
    if type(a) is not type(b): return False
    if isinstance(a, Var): return a.i == b.i
    if isinstance(a, Lam): return a.dom == b.dom and teq(a.body, b.body)
    if isinstance(a, App): return teq(a.f, b.f) and teq(a.a, b.a)
    if isinstance(a, Lin):
        same = lambda x, y: x is y or (x is not None and y is not None and x.shape == y.shape and np.array_equal(x, y))
        return same(a.W, b.W) and same(a.b, b.b)
    return a.n == b.n


# ============================================================================ evaluation + cost
class Ctr:
    macs = 0


def ev(t, env, ctr):
    if isinstance(t, Var): return env[t.i]
    if isinstance(t, Lam): return lambda v: ev(t.body, [v] + env, ctr)
    if isinstance(t, App): return ev(t.f, env, ctr)(ev(t.a, env, ctr))
    if isinstance(t, Lin):
        def f(x):
            ctr.macs += t.W.size
            y = x @ t.W.T
            return y if t.b is None else y + t.b
        return f
    if isinstance(t, Relu): return lambda x: np.maximum(x, 0)
    if isinstance(t, Add): return lambda a: (lambda b: a + b)


def run(net, X):
    return ev(net, [], Ctr())(X)


def macs(net, d):
    c = Ctr()
    ev(net, [], c)(np.zeros(d))
    return c.macs


def params(t, seen=None):
    """distinct parameter arrays (shared weights counted once)."""
    seen = {} if seen is None else seen
    if isinstance(t, Lam): params(t.body, seen)
    elif isinstance(t, App): params(t.f, seen); params(t.a, seen)
    elif isinstance(t, Lin):
        seen[id(t.W)] = t.W.size
        if t.b is not None: seen[id(t.b)] = t.b.size
    return sum(seen.values())


# ============================================================================ beta-reduction
def shift(t, d, c=0):
    if isinstance(t, Var): return Var(t.i + d) if t.i >= c else t
    if isinstance(t, Lam): return Lam(t.dom, shift(t.body, d, c + 1))
    if isinstance(t, App): return App(shift(t.f, d, c), shift(t.a, d, c))
    return t


def subst(t, j, s):
    if isinstance(t, Var): return s if t.i == j else t
    if isinstance(t, Lam): return Lam(t.dom, subst(t.body, j + 1, shift(s, 1)))
    if isinstance(t, App): return App(subst(t.f, j, s), subst(t.a, j, s))
    return t


def beta(lam, arg):
    return shift(subst(lam.body, 0, shift(arg, 1)), -1)


def redex_paths(t, p=()):
    out = []
    if isinstance(t, App):
        if isinstance(t.f, Lam): out.append(p)
        out += redex_paths(t.f, p + ("f",)) + redex_paths(t.a, p + ("a",))
    elif isinstance(t, Lam):
        out += redex_paths(t.body, p + ("b",))
    return out


def at(t, p):
    for k in p: t = {"f": lambda: t.f, "a": lambda: t.a, "b": lambda: t.body}[k]()
    return t


def replace(t, p, new):
    if not p: return new
    k = p[0]
    if k == "f": return App(replace(t.f, p[1:], new), t.a)
    if k == "a": return App(t.f, replace(t.a, p[1:], new))
    return Lam(t.dom, replace(t.body, p[1:], new))


def normalise(t, strategy="normal", rng=None, cap=50000):
    """Beta-reduce to normal form. strategy: normal (leftmost-outermost), applicative
    (leftmost-innermost) or random. Returns (term, #steps)."""
    steps = 0
    while steps < cap:
        P = redex_paths(t)
        if not P: break
        if strategy == "normal": p = P[0]
        elif strategy == "applicative": p = next(p for p in P if not any(q != p and q[:len(p)] == p for q in P))
        else: p = (rng or random).choice(P)
        r = at(t, p)
        t = replace(t, p, beta(r.f, r.a))
        steps += 1
    return t, steps


# ============================================================================ sound, cost-guided rewriting
def has_var(t, i):
    if isinstance(t, Var): return t.i == i
    if isinstance(t, Lam): return has_var(t.body, i + 1)
    if isinstance(t, App): return has_var(t.f, i) or has_var(t.a, i)
    return False


def lin_compose(l1, l2):
    """Lin1 . Lin2  as one Lin."""
    b = None
    if l1.b is not None or l2.b is not None:
        b = (l1.W @ l2.b if l2.b is not None else 0) + (l1.b if l1.b is not None else 0)
    return Lin(l1.W @ l2.W, b)


def chain(t):
    """peel nested Lin applications: Lin1(Lin2(...(base))) -> ([Lin1,Lin2,...], base)"""
    ls = []
    while isinstance(t, App) and isinstance(t.f, Lin):
        ls.append(t.f)
        t = t.a
    return ls, t


def collapse(ls):
    m = ls[-1]
    for l in reversed(ls[:-1]):
        m = lin_compose(l, m)
    return m


def try_rules(t, cost_aware, rank_refactor, log):
    # eta:  \x. f x  ->  f      (x not free in f)
    if isinstance(t, Lam) and isinstance(t.body, App) and isinstance(t.body.a, Var) and t.body.a.i == 0 \
            and not has_var(t.body.f, 0):
        log["eta"] += 1
        return shift(t.body.f, -1)
    if not isinstance(t, App):
        return None
    f, a = t.f, t.a
    # relu . relu = relu
    if isinstance(f, Relu) and isinstance(a, App) and isinstance(a.f, Relu):
        log["relu-idem"] += 1
        return a
    # Lin . Lin  (fuse) -- only if MACs strictly decrease
    if isinstance(f, Lin) and isinstance(a, App) and isinstance(a.f, Lin):
        ls, base = chain(t)
        old = sum(l.W.size for l in ls)
        new = ls[0].W.shape[0] * ls[-1].W.shape[1]
        if not cost_aware or new < old:
            log["lin-fuse"] += 1
            return App(collapse(ls), base)
        return None
    # add(chain1(t), chain2(t)) = (M1+M2) t   (linear maps of the SAME base distribute over add)
    if isinstance(f, App) and isinstance(f.f, Add):
        l1, b1 = chain(f.a)
        l2, b2 = chain(a)
        if (l1 or l2) and teq(b1, b2):
            d_in = (l1 or l2)[-1].W.shape[1]
            d_out = (l1 or l2)[0].W.shape[0]
            eye = Lin(np.eye(d_in))
            m1 = collapse(l1) if l1 else eye
            m2 = collapse(l2) if l2 else eye
            old = sum(l.W.size for l in l1 + l2)
            new = d_out * d_in
            if (not cost_aware or new < old) and m1.W.shape == m2.W.shape:
                b = None if m1.b is None and m2.b is None else (m1.b if m1.b is not None else 0) + (m2.b if m2.b is not None else 0)
                log["add-fuse"] += 1
                return App(Lin(m1.W + m2.W, b), b1)
    # exact low-rank refactor:  Lin(W) t  ->  Lin(U S) (Lin(Vt) t)   if rank*(out+in) < out*in
    if rank_refactor and isinstance(f, Lin):
        W = f.W
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        r = int((S > 1e-9 * max(S[0], 1e-300)).sum())
        if r * (sum(W.shape)) < W.size and r >= 1:
            A, B = U[:, :r] * S[:r], Vt[:r]
            if np.abs(A @ B - W).max() < 1e-8 * max(1.0, np.abs(W).max()):
                log["lowrank"] += 1
                return App(Lin(A, f.b), App(Lin(B), a))
    return None


def rewrite(t, cost_aware=True, rank_refactor=True, cap=200):
    log = {"eta": 0, "relu-idem": 0, "lin-fuse": 0, "add-fuse": 0, "lowrank": 0}

    def once(t):
        if isinstance(t, Lam): t = Lam(t.dom, once(t.body))
        elif isinstance(t, App): t = App(once(t.f), once(t.a))
        r = try_rules(t, cost_aware, rank_refactor, log)
        return r if r is not None else t

    for _ in range(cap):
        before = dict(log)
        t = once(t)
        if log == before:
            break
    return t, log


# ============================================================================ builders
def church(n, T):
    """n : (T->T) -> T -> T"""
    body = Var(0)
    for _ in range(n): body = App(Var(1), body)
    return Lam(arr(T, T), Lam(T, body))


def compose_term(A, B, C):
    """B_ : (B->C) -> (A->B) -> A -> C"""
    return Lam(arr(B, C), Lam(arr(A, B), Lam(A, App(Var(2), App(Var(1), Var(0))))))


def compose(f, g, A, B, C):
    return App(App(compose_term(A, B, C), f), g)


def ident(A):
    return Lam(A, Var(0))


# ============================================================================ experiments
def evaluate(name, net, d, rng, cost_aware=True, show_terms=False):
    infer(net)                                                     # type (= shape) check
    nf, steps = normalise(net)
    opt, log = rewrite(nf, cost_aware=cost_aware)
    assert infer(opt) == infer(net), "rewriting must preserve the type"
    X = rng.normal(size=(200, d))
    err = float(np.abs(run(net, X) - run(opt, X)).max())
    row = (name, steps, macs(net, d), macs(opt, d), params(net), params(opt), err, {k: v for k, v in log.items() if v})
    print(f"   {name:<34} β-steps={steps:>3}  MACs {row[2]:>6} -> {row[3]:>6}   params {row[4]:>6} -> {row[5]:>6}"
          f"   max|Δ|={err:.1e}   rules={row[7]}")
    if show_terms:
        print(f"      normal form : {show(nf)}\n      optimised   : {show(opt)}")
    return row


def main(seed=0):
    rng = np.random.default_rng(seed)
    print("== 1. Typing: shapes are types ==")
    try:
        infer(App(Lin(np.zeros((5, 4))), Var(0)), (vec(3),))
    except TypeError as e:
        print(f"   ill-shaped architecture rejected: {e}")

    print("\n== 2. Compile + optimise (sound rewriting, cost-guided) ==")
    d = 32
    T = vec(d)
    W = rng.normal(0, 1 / np.sqrt(d), (d, d))
    block = Lam(T, App(Lin(W, rng.normal(0, .1, d)), Var(0)))
    N1 = Lam(T, App(App(church(5, T), block), Var(0)))
    evaluate("N1 Church(5) deep-linear stack", N1, d, rng)

    d2, r = 64, 8
    T2 = vec(d2)
    W1, W2 = rng.normal(size=(r, d2)) / 8, rng.normal(size=(d2, r)) / 3
    N2 = Lam(T2, App(Lin(W2), App(Lin(W1), Var(0))))
    evaluate("N2 bottleneck 64-8-64 (cost-aware)", N2, d2, rng)
    evaluate("N2 bottleneck 64-8-64 (ALWAYS fuse)", N2, d2, rng, cost_aware=False)

    Wc = rng.normal(size=(d2, d2)) / 8
    Lc = Lin(Wc)
    f1 = compose(Relu(d2), compose(ident(T2), Relu(d2), T2, T2, T2), T2, T2, T2)
    N3 = Lam(T2, App(compose(f1, Lc, T2, T2, T2), Var(0)))
    evaluate("N3 relu∘id∘relu∘Lin (sloppy)", N3, d2, rng)

    A, B = rng.normal(size=(4, d2)) / 8, rng.normal(size=(d2, 4)) / 2
    N4 = Lam(T2, App(App(Add(d2), App(Lin(Wc), Var(0))), App(Lin(B), App(Lin(A), Var(0)))))
    evaluate("N4 LoRA: W x + B(A x)", N4, d2, rng)

    Wlr = rng.normal(size=(d2, 6)) @ rng.normal(size=(6, d2)) / 8
    N5 = Lam(T2, App(Relu(d2), App(Lin(Wlr, rng.normal(size=d2)), Var(0))))
    evaluate("N5 rank-6 64x64 layer", N5, d2, rng)

    h = 48
    Ws = [rng.normal(size=(h, d2)) / 8, rng.normal(size=(h, h)) / 7, rng.normal(size=(10, h)) / 7]
    N6 = Lam(T2, App(Lin(Ws[2]), App(Relu(h), App(Lin(Ws[1]), App(Relu(h), App(Lin(Ws[0]), Var(0)))))))
    evaluate("N6 ordinary ReLU MLP (control)", N6, d2, rng)

    print("\n== 3. Confluence (Church-Rosser) and the cost of beta ==")
    nf0, s_n = normalise(N1, "normal")
    _, s_a = normalise(N1, "applicative")
    forms = []
    for k in range(25):
        nf, _ = normalise(N1, "random", random.Random(k))
        forms.append(nf)
    uniq = [f for i, f in enumerate(forms) if not any(teq(f, g) for g in forms[:i])]
    print(f"   N1: normal order {s_n} steps, applicative {s_a} steps; "
          f"25 random reduction orders -> {len(uniq)} distinct normal form(s) (confluence)")
    # duplication: (\y. add y y) R   where R is k nested identity redexes
    k = 12
    R = Lin(np.eye(4))
    arg = App(R, Var(0))
    for _ in range(k): arg = App(ident(vec(4)), arg)
    dup = Lam(vec(4), App(Lam(vec(4), App(App(Add(4), Var(0)), Var(0))), arg))
    _, sn = normalise(dup, "normal")
    _, sa = normalise(dup, "applicative")
    print(f"   duplication: (λy. y+y) R with R = {k} nested redexes -> normal order {sn} steps, applicative {sa} steps"
          f"\n   (β is not unit cost: substitution copies the argument; sharing / let-bindings avoid this)")


if __name__ == "__main__":
    main()
