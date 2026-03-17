import numpy as np
import pysr
import sympy

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


def reduction_map(A, masters, eps=1e-12):
    n = A.shape[1]
    hard = [i for i in range(n) if i not in masters]
    Ah = A[:, hard]
    Am = A[:, masters]
    Ah_pinv = np.linalg.pinv(Ah, rcond=eps)
    R = -Ah_pinv @ Am
    return hard, R


def lhs_sample(n_samples, bounds, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    dim = len(bounds)
    result = np.zeros((n_samples, dim))
    for i, (low, high) in enumerate(bounds):
        perm = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples
        result[:, i] = low + result[:, i] * (high - low)
    return result


def is_good_point(d, x, threshold=1e6):
    A = ibp_matrix(d, x)
    s = np.linalg.svd(A, compute_uv=False)
    cond = s.max() / s.min()
    return cond < threshold


def generate_data(samples):
    X = []
    y = []
    for d, x in samples:
        if not is_good_point(d, x):
            continue
        A = ibp_matrix(d, x)
        hard, R = reduction_map(A, masters)
        i1_index = hard.index(0)
        c1 = R[i1_index, 0]
        X.append([d, x])
        y.append(c1)
    return np.array(X), np.array(y)


def eval_at_points(expr, X):
    """Evaluate a sympy expression at data points."""
    f = sympy.lambdify(SYMS, expr, modules="numpy")
    return f(X[:, 0], X[:, 1])


def round_poly(expr, tol=0.2):
    """Round float coefficients to integers by finding the best scaling factor.

    Tries scaling by 1/|c| for each coefficient c, then rounds all coefficients.
    Returns the rounded polynomial with the smallest rounding error, or None.
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


def extract_denom_candidates(model):
    """Extract rounded, factored denominator candidates from all hall-of-fame expressions."""
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

            # Factor and collect irreducible symbolic factors
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


def pole_score(factor, X, y, n_check=30):
    """Score how well a factor explains a pole.

    Returns ratio of mean|y| near factor=0 vs far from factor=0.
    A ratio >> 1 indicates a real pole.
    """
    fvals = np.abs(eval_at_points(factor, X))
    n_check = min(n_check, len(y) // 5)
    if n_check < 5:
        return 0.0
    idx = np.argsort(fvals)
    near = idx[:n_check]
    far = idx[-n_check:]
    mean_near = np.mean(np.abs(y[near]))
    mean_far = np.mean(np.abs(y[far]))
    if mean_far < 1e-10:
        return 0.0
    return mean_near / mean_far


def make_model(niterations=200):
    return pysr.PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        maxsize=25,
        model_selection='best',
        verbosity=1,
    )


if __name__ == "__main__":
    bounds = [(2, 8), (-10, 10)]
    rng = np.random.default_rng(123)

    # Exact answer for validation
    exact_expr = (d_sym - 3) * ((d_sym - 6) * x_sym - 4) / (x_sym * (x_sym + 4)**2)

    # Generate initial data
    samples = lhs_sample(1000, bounds, rng)
    X, y_raw = generate_data(samples)
    print(f"Initial dataset: {len(X)} good points")

    accumulated_denom = sympy.Integer(1)
    current_y = y_raw.copy()

    for iteration in range(10):
        print(f"\n{'='*60}")
        print(f"Pole removal iteration {iteration}")
        print(f"  Dataset: {len(X)} points, accumulated denom: {accumulated_denom}")

        model = make_model(niterations=200)
        model.fit(X, current_y, variable_names=VAR_NAMES)
        print(model)

        # Extract denominator candidates from ALL hall-of-fame expressions
        candidates = extract_denom_candidates(model)
        print(f"  Denominator candidates: {candidates}")

        # Score and validate each candidate
        scored = [(c, pole_score(c, X, current_y)) for c in candidates]
        scored = [(c, s) for c, s in scored if s > 5.0]
        scored.sort(key=lambda x: -x[1])

        if not scored:
            print("  No valid poles found — stopping pole removal")
            break

        best_factor, best_score = scored[0]
        print(f"  Best pole factor: {best_factor} (score={best_score:.2f})")

        # Multiply out the pole
        current_y = current_y * eval_at_points(best_factor, X)
        accumulated_denom = accumulated_denom * best_factor
        print(f"  Accumulated denominator: {sympy.factor(accumulated_denom)}")

    # Final numerator fit with more iterations
    print(f"\n{'='*60}")
    print("Final numerator fit")
    final_model = make_model(niterations=500)
    final_model.fit(X, current_y, variable_names=VAR_NAMES)
    print(final_model)

    numerator = final_model.sympy()
    final_expr = sympy.simplify(numerator / accumulated_denom)
    print(f"\n{'='*60}")
    print("RESULT")
    print(f"  Numerator (PySR): {numerator}")
    print(f"  Denominator:      {sympy.factor(accumulated_denom)}")
    print(f"  Result:           {final_expr}")

    # Validation
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"  Exact: {exact_expr}")
    diff = sympy.simplify(final_expr - exact_expr)
    print(f"  Difference (simplified): {diff}")
    if diff == 0:
        print("  EXACT MATCH!")
    else:
        val_rng = np.random.default_rng(999)
        val_samples = lhs_sample(500, bounds, val_rng)
        val_X, val_y = generate_data(val_samples)
        exact_func = sympy.lambdify(SYMS, exact_expr, modules="numpy")
        found_func = sympy.lambdify(SYMS, final_expr, modules="numpy")
        exact_vals = exact_func(val_X[:, 0], val_X[:, 1])
        found_vals = found_func(val_X[:, 0], val_X[:, 1])
        print(f"  Max |exact - raw|:   {np.abs(exact_vals - val_y).max():.2e}")
        print(f"  Max |found - raw|:   {np.abs(found_vals - val_y).max():.2e}")
        print(f"  Max |found - exact|: {np.abs(found_vals - exact_vals).max():.2e}")
