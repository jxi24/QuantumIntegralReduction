"""Symbolic-regression IBP reduction pipeline.

Originally specialized to a fixed 2D 6x8 IBP system; now refactored so the
core pipeline (sampling -> pseudoinverse reduction -> PySR pole removal
-> numerator fit -> validation) works for an arbitrary IBP matrix over an
arbitrary number of parameters.

Build an :class:`IBPProblem` describing the system and call
:func:`solve_coefficient` for one (hard, master) coefficient or
:func:`solve_problem` to sweep all of them.

The legacy 2D example is preserved in this module so existing tests in
``tests/test_ibp.py`` continue to import the same names.
"""
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pysr
import sympy


# =============================================================================
# Generic framework
# =============================================================================

@dataclass
class IBPProblem:
    """An IBP reduction problem.

    Parameters
    ----------
    ibp_matrix:
        Callable ``(*params) -> np.ndarray`` returning the IBP matrix at
        numeric parameter values. Columns index the integrals; rows are the
        IBP relations.
    masters:
        Column indices of the master integrals.
    var_names:
        Names of the parameters (one per positional argument of
        ``ibp_matrix``). These names are forwarded to PySR and used to build
        the sympy symbols.
    bounds:
        Sampling bounds ``[(low, high), ...]`` for each parameter.
    exact:
        Optional ``{(hard_index, master_index): sympy.Expr}`` of known
        closed-form coefficients used for validation.
    """

    ibp_matrix: Callable[..., np.ndarray]
    masters: Sequence[int]
    var_names: Sequence[str]
    bounds: Sequence[tuple]
    exact: Optional[Mapping[tuple, Any]] = None

    @property
    def syms(self):
        s = sympy.symbols(list(self.var_names))
        if isinstance(s, sympy.Symbol):
            return [s]
        return list(s)

    @property
    def n_vars(self):
        return len(self.var_names)


def reduction_map(A, masters, eps=1e-12):
    """Least-norm reduction of hard integrals onto masters via pseudoinverse.

    Returns ``(hard_indices, R)`` where ``R[k, j]`` is the coefficient with
    which the ``j``-th master appears in the reduction of ``hard_indices[k]``.
    """
    n = A.shape[1]
    hard = [i for i in range(n) if i not in masters]
    Ah = A[:, hard]
    Am = A[:, masters]
    Ah_pinv = np.linalg.pinv(Ah, rcond=eps)
    R = -Ah_pinv @ Am
    return hard, R


def lhs_sample(n_samples, bounds, rng=None):
    """Latin-hypercube sample inside an axis-aligned box."""
    if rng is None:
        rng = np.random.default_rng()
    dim = len(bounds)
    result = np.zeros((n_samples, dim))
    for i, (low, high) in enumerate(bounds):
        perm = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples
        result[:, i] = low + result[:, i] * (high - low)
    return result


def condition_number(A):
    """SVD-based condition number, returning ``inf`` for singular matrices."""
    s = np.linalg.svd(A, compute_uv=False)
    if s.min() <= 0:
        return np.inf
    return s.max() / s.min()


def collect_data(problem, hard_index, master_index, n_samples,
                 rng=None, cond_threshold=1e6):
    """Sample parameter space and tabulate one reduction coefficient.

    Returns ``(X, y)`` where ``X`` is a ``(k, n_vars)`` array of well
    conditioned points and ``y`` is the matching coefficient values.
    """
    samples = lhs_sample(n_samples, problem.bounds, rng)
    # Precompute the hard column indices once. The conditioning that
    # actually governs the pseudoinverse reduction is that of the hard
    # sub-block ``A[:, hard]`` -- the full matrix can be perfectly
    # well-conditioned even where the reduction has a pole (e.g. the
    # sunrise pseudothreshold ``4 m^2 = M^2``), and filtering on the
    # full matrix would silently let those bad points through.
    midpoint = [0.5 * (lo + hi) for lo, hi in problem.bounds]
    n_cols = problem.ibp_matrix(*midpoint).shape[1]
    masters_set = set(problem.masters)
    hard_cols = [i for i in range(n_cols) if i not in masters_set]
    X, y = [], []
    for pt in samples:
        A = problem.ibp_matrix(*pt)
        if condition_number(A[:, hard_cols]) > cond_threshold:
            continue
        hard, R = reduction_map(A, problem.masters)
        if hard_index not in hard:
            continue
        X.append(list(pt))
        y.append(R[hard.index(hard_index), master_index])
    return np.array(X), np.array(y)


def eval_expr(expr, syms, X):
    """Evaluate a sympy expression at the rows of an ``(n, n_vars)`` array."""
    f = sympy.lambdify(syms, expr, modules="numpy")
    cols = [X[:, i] for i in range(X.shape[1])]
    if len(syms) == 1:
        return f(cols[0])
    return f(*cols)


def round_poly(expr, tol=0.2):
    """Round float coefficients to integers using the best uniform scaling.

    Tries each |coefficient| as the reference scale; for each scaling, rounds
    every coefficient and returns the rescaled polynomial whose maximum
    rounding error is below ``tol``.
    """
    expanded = sympy.expand(expr)
    terms = sympy.Add.make_args(expanded)

    pairs = []
    for term in terms:
        c, m = term.as_coeff_Mul()
        pairs.append((float(c), m))

    if not pairs:
        return None

    nonzero_coeffs = [abs(c) for c, _ in pairs if abs(c) > 1e-10]
    if not nonzero_coeffs:
        return None

    best = None
    best_err = float('inf')

    for ref in nonzero_coeffs:
        scale = 1.0 / ref
        scaled = [(c * scale, m) for c, m in pairs]
        err = max(abs(s - round(s)) for s, _ in scaled)

        if err < tol and err < best_err:
            result_terms = []
            for s, m in scaled:
                r = int(round(s))
                if r != 0:
                    result_terms.append(sympy.Integer(r) * m)
            if result_terms:
                result = sympy.Add(*result_terms)
                if not result.is_number:
                    best = result
                    best_err = err

    return best


def extract_denom_candidates(model, syms=None):
    """Collect rounded irreducible denominator factors from a PySR run.

    ``syms`` may be omitted to default to the legacy 2D ``(d, x)`` symbols.
    """
    if syms is None:
        syms = SYMS
    candidates = []
    seen = set()

    for i in range(len(model.equations_)):
        try:
            expr = model.sympy(i)
            denom = sympy.denom(sympy.together(expr))
            if denom.is_number:
                continue

            rounded = round_poly(denom)
            if rounded is None:
                continue

            factored = sympy.factor(rounded)
            for f in factored.as_ordered_factors():
                if f.is_number:
                    continue
                f_expanded = sympy.expand(f)
                key = str(f_expanded)
                if key not in seen:
                    seen.add(key)
                    candidates.append(f_expanded)
        except Exception:
            continue

    return candidates


def pole_score(factor, X, y, n_check=30, syms=None):
    """Score ``factor`` as a denominator pole of ``y`` via log-correlation.

    A genuine ``1/f^k`` pole satisfies ``log|y| = -k log|f| + (other vars)``
    on the bulk of the dataset, so the Pearson correlation between
    ``log|y|`` and ``-log|f|`` is strongly positive even when ``y`` has
    *additional* poles that contaminate both extremes of the partition
    (which is exactly what defeated the previous near/far mean ratio).

    Spurious factors are filtered by two checks: the correlation must be
    positive, and ``f`` must actually approach zero somewhere in the
    sampled region -- a polynomial with no real roots in the box (e.g.
    ``m^2 + m + 1``) is monotone in its variable and would otherwise
    alias the correlation of the genuine factor it shadows.

    The score is in ``[-1, 1]``; ``n_check`` is retained for backward
    compatibility with older callers but is no longer used. ``syms``
    defaults to the legacy 2D ``(d, x)`` symbols.
    """
    del n_check  # kept for signature compatibility
    if syms is None:
        syms = SYMS
    fvals = np.asarray(eval_expr(factor, syms, X), dtype=float)
    if fvals.ndim == 0:
        return 0.0
    abs_f = np.abs(fvals)
    abs_y = np.abs(np.asarray(y, dtype=float))

    finite = np.isfinite(abs_f) & np.isfinite(abs_y)
    if finite.sum() < 20:
        return 0.0

    # Reject factors that never get close to zero in the dataset: only a
    # factor that actually vanishes somewhere can be a denominator pole.
    finite_f = abs_f[finite]
    median_f = float(np.median(finite_f))
    if median_f > 0 and finite_f.min() > 0.2 * median_f:
        return 0.0

    mask = finite & (abs_f > 1e-12) & (abs_y > 1e-12)
    if mask.sum() < 20:
        return 0.0

    log_f = np.log(abs_f[mask])
    log_y = np.log(abs_y[mask])
    log_f = log_f - log_f.mean()
    log_y = log_y - log_y.mean()
    den = float(np.sqrt((log_f ** 2).sum() * (log_y ** 2).sum()))
    if den < 1e-12:
        return 0.0
    return float(-(log_f * log_y).sum() / den)


def make_model(niterations=200, maxsize=25):
    return pysr.PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        maxsize=maxsize,
        model_selection='score',
        verbosity=1,
    )


def find_system_poles(problem, n_samples=500, niterations=100, rng=None):
    """Discover candidate poles by fitting the (pseudo-)determinant of ``Ah``.

    By Cramer's rule, every denominator factor of every reduction coefficient
    divides ``det(Ah)`` (or, for non-square hard blocks, the product of its
    singular values). Fitting that scalar once with PySR therefore yields a
    *superset* of candidate pole factors that can be reused across all
    (hard, master) coefficients of the same IBP system.

    Returns a list of unique, rounded, irreducible symbolic factors.
    """
    if rng is None:
        rng = np.random.default_rng(456)

    midpoint = [0.5 * (lo + hi) for lo, hi in problem.bounds]
    A0 = problem.ibp_matrix(*midpoint)
    n_cols = A0.shape[1]
    masters_set = set(problem.masters)
    hard_cols = [i for i in range(n_cols) if i not in masters_set]
    if len(hard_cols) == 0:
        return []

    samples = lhs_sample(n_samples, problem.bounds, rng)
    valid_X, y = [], []
    for pt in samples:
        A = problem.ibp_matrix(*pt)
        Ah = A[:, hard_cols]
        if Ah.shape[0] == Ah.shape[1]:
            val = np.linalg.det(Ah)
        else:
            # Pseudo-determinant: product of singular values. Vanishes
            # exactly where Ah loses rank, which is what we care about.
            s = np.linalg.svd(Ah, compute_uv=False)
            val = float(np.prod(s))
        if np.isfinite(val):
            y.append(val)
            valid_X.append(pt)

    if not valid_X:
        return []

    X = np.array(valid_X)
    y = np.array(y)

    print(f"Fitting system (pseudo-)determinant with {len(X)} points...")
    model = pysr.PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=25,
        verbosity=1,
    )
    model.fit(X, y, variable_names=list(problem.var_names))

    # Walk every equation in the Pareto front, factor each, and collect
    # unique non-constant irreducible factors after integer rounding.
    factors = []
    seen = set()
    for i in range(len(model.equations_)):
        try:
            expr = model.sympy(i)
            rounded = round_poly(sympy.expand(expr))
            if rounded is None:
                continue
            for f in sympy.factor(rounded).as_ordered_factors():
                if f.is_number:
                    continue
                f_expanded = sympy.expand(f)
                key = str(f_expanded)
                if key not in seen:
                    seen.add(key)
                    factors.append(f_expanded)
        except Exception:
            continue
    print(f"  Candidate system pole factors: {factors}")
    return factors


def solve_coefficient(problem, hard_index, master_index,
                      n_samples=1000, max_pole_iters=10,
                      pysr_pilot_iters=200, pysr_final_iters=500,
                      pole_score_threshold=0.25, rng=None,
                      system_poles=None):
    """Run pole removal + final fit for a single reduction coefficient.

    ``system_poles`` is an optional list of candidate denominator factors
    obtained from :func:`find_system_poles`. When solving many coefficients
    of the same problem, compute it *once* (e.g. via :func:`solve_problem`)
    and pass it in to avoid refitting the determinant per coefficient.
    If ``None``, it is computed lazily so single-coefficient calls still
    work standalone.

    Returns a dict with keys ``expr``, ``numerator``, ``denominator``, and
    (if ``problem.exact`` provides a reference) ``exact`` and ``match``.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    syms = problem.syms
    var_names = list(problem.var_names)

    if system_poles is None:
        print("\n" + "="*60)
        print("Global pole discovery (from Ah determinant)")
        system_poles = find_system_poles(problem, n_samples=n_samples//2, rng=rng)
    # Local working copy: we'll prune used factors from this list as the
    # iteration consumes them, without mutating the caller's list.
    remaining_system_poles = list(system_poles)
    print(f"  Available system pole candidates: {remaining_system_poles}")

    X, y_raw = collect_data(problem, hard_index, master_index,
                            n_samples, rng=rng)
    print(f"Initial dataset: {len(X)} good points (vars={var_names})")

    accumulated_denom = sympy.Integer(1)
    current_y = y_raw.copy()

    for it in range(max_pole_iters):
        print(f"\n{'='*60}")
        print(f"Pole removal iteration {it}")
        print(f"  Dataset: {len(X)} points; "
              f"accumulated denom: {sympy.factor(accumulated_denom)}")

        # Score the precomputed system-wide candidates first -- they are
        # essentially free compared to running PySR.
        scored_system = [
            (c, pole_score(c, X, current_y, syms=syms))
            for c in remaining_system_poles
        ]
        scored_system = [(c, s) for c, s in scored_system
                         if s > pole_score_threshold]

        best_factor = None
        best_score = -1.0

        if scored_system:
            scored_system.sort(key=lambda cs: -cs[1])
            best_factor, best_score = scored_system[0]
            print(f"  Found system pole candidate: {best_factor} "
                  f"(score={best_score:.2f})")

        # Only fall back to PySR pilot fitting if no precomputed candidate
        # is strongly supported by the data.
        if best_score < 0.8:
            model = make_model(niterations=pysr_pilot_iters)
            model.fit(X, current_y, variable_names=var_names)

            candidates = extract_denom_candidates(model, syms=syms)
            scored_pysr = [(c, pole_score(c, X, current_y, syms=syms))
                           for c in candidates]
            scored_pysr = [(c, s) for c, s in scored_pysr
                           if s > pole_score_threshold]

            if scored_pysr:
                scored_pysr.sort(key=lambda cs: -cs[1])
                p_factor, p_score = scored_pysr[0]
                if p_score > best_score:
                    best_factor, best_score = p_factor, p_score
                    print(f"  Found PySR pole candidate: {best_factor} "
                          f"(score={best_score:.2f})")

        if best_factor is None:
            print("  No more poles found")
            break

        current_y = current_y * eval_expr(best_factor, syms, X)
        accumulated_denom = accumulated_denom * best_factor
        print(f"  Accumulated denom: {sympy.factor(accumulated_denom)}")

        # Drop the consumed factor so we don't re-pick it next iteration.
        # (Higher-order poles will still be found via the PySR fallback.)
        remaining_system_poles = [
            p for p in remaining_system_poles if str(p) != str(best_factor)
        ]

    print(f"\n{'='*60}\nFinal numerator fit")
    final_model = make_model(niterations=pysr_final_iters)
    final_model.fit(X, current_y, variable_names=var_names)
    print(final_model)

    numerator = final_model.sympy()
    final_expr = sympy.simplify(numerator / accumulated_denom)

    print(f"\n{'='*60}")
    print(f"RESULT for (hard={hard_index}, master={master_index})")
    print(f"  Numerator (PySR): {numerator}")
    print(f"  Denominator:      {sympy.factor(accumulated_denom)}")
    print(f"  Result:           {final_expr}")

    result = {
        "hard_index": hard_index,
        "master_index": master_index,
        "expr": final_expr,
        "numerator": numerator,
        "denominator": accumulated_denom,
    }

    if problem.exact and (hard_index, master_index) in problem.exact:
        exact_expr = problem.exact[(hard_index, master_index)]
        diff = sympy.simplify(final_expr - exact_expr)
        print(f"  Exact:            {exact_expr}")
        print(f"  Difference:       {diff}")
        result["exact"] = exact_expr
        result["match"] = (diff == 0)

    return result


def solve_problem(problem, hard_indices=None, master_indices=None,
                  system_poles=None, **kwargs):
    """Solve every (hard, master) coefficient of an :class:`IBPProblem`.

    The candidate denominator factors of ``Ah`` are computed *once* (via
    :func:`find_system_poles`) and shared across every coefficient, since
    by Cramer's rule they form a superset of all per-coefficient poles.
    Pass ``system_poles`` explicitly to skip discovery (e.g. when reusing
    factors from a previous run or supplying analytic ones).

    Extra keyword arguments are forwarded to :func:`solve_coefficient`.
    """
    midpoint = [0.5 * (lo + hi) for lo, hi in problem.bounds]
    A0 = problem.ibp_matrix(*midpoint)
    n = A0.shape[1]
    all_hard = [i for i in range(n) if i not in problem.masters]
    if hard_indices is None:
        hard_indices = all_hard
    if master_indices is None:
        master_indices = list(range(len(problem.masters)))

    if system_poles is None:
        print("\n" + "#"*60)
        print("# Global pole discovery (shared across all coefficients)")
        print("#"*60)
        system_poles = find_system_poles(
            problem,
            n_samples=kwargs.get("n_samples", 1000) // 2,
        )

    results = {}
    for hi in hard_indices:
        for mi in master_indices:
            print(f"\n{'#'*60}")
            print(f"# Solving I[{hi}] coefficient of M[{mi}]")
            print(f"{'#'*60}")
            results[(hi, mi)] = solve_coefficient(
                problem, hi, mi, system_poles=system_poles, **kwargs
            )
    return results


# =============================================================================
# Legacy 2D example (preserved for backward compatibility with tests)
# =============================================================================

masters = [6, 7]
d_sym, x_sym = sympy.symbols("d x")
VAR_NAMES = ["d", "x"]
SYMS = [d_sym, x_sym]


def ibp_matrix(d, x):
    return np.array([
        [-x*(4 + x), 2 + (5 - d)*x, 2 + x, -4, 0, 0, 0, 0],
        [0, -x*(4 + x), 0, 0, -2, 2 + x, (3 - d)*x, 0],
        [0, 0, -x*(4 + x), 0, -2, 2 + x, (3 - d)*x, 0],
        [0, 0, 0, -2, 2 - d/2, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 1 - d/2],
    ], dtype=float)


def is_good_point(d, x, threshold=1e6):
    return condition_number(ibp_matrix(d, x)) < threshold


def generate_data(samples):
    """Legacy: generate ``(X, c1)`` for I1's coefficient of M1 in the 2D system."""
    X, y = [], []
    for d, x in samples:
        A = ibp_matrix(d, x)
        if not is_good_point(d, x):
            continue
        hard, R = reduction_map(A, masters)
        i1_index = hard.index(0)
        X.append([d, x])
        y.append(R[i1_index, 0])
    return np.array(X), np.array(y)


def eval_at_points(expr, X):
    """Legacy 2D evaluator (assumes the columns of ``X`` are ``(d, x)``)."""
    f = sympy.lambdify(SYMS, expr, modules="numpy")
    return f(X[:, 0], X[:, 1])


def make_2d_problem():
    """Build the legacy 2D IBP problem with its known closed-form answer."""
    exact = {
        (0, 0): (d_sym - 3) * ((d_sym - 6) * x_sym - 4)
                / (x_sym * (x_sym + 4) ** 2),
    }
    return IBPProblem(
        ibp_matrix=ibp_matrix,
        masters=masters,
        var_names=VAR_NAMES,
        bounds=[(2, 8), (-10, 10)],
        exact=exact,
    )


if __name__ == "__main__":
    problem = make_2d_problem()
    solve_coefficient(
        problem,
        hard_index=0,
        master_index=0,
        n_samples=1000,
        max_pole_iters=10,
        pysr_pilot_iters=200,
        pysr_final_iters=500,
    )
