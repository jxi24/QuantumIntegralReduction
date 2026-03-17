import numpy as np
import pysr
import sympy


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


def reduction_map(A, masters, eps=1e-10):
    n = A.shape[1]
    hard = [i for i in range(n) if i not in masters]
    Ah = A[:, hard]
    Am = A[:, masters]
    Ah_pinv = np.linalg.pinv(Ah, rcond=eps)
    R = -Ah_pinv @ Am
    return hard, R


def sample_good_points(ibp_func, n_samples, bounds, variable_names, cond_threshold=1e6, rng=None):
    """Sample points and filter by condition number."""
    raw = lhs_sample(n_samples, bounds, rng)
    good = []
    for row in raw:
        A = ibp_func(*row)
        s = np.linalg.svd(A, compute_uv=False)
        cond = s.max() / s.min()
        if cond < cond_threshold:
            good.append(row)
    return np.array(good)


def compute_targets(ibp_func, points, masters, hard_index, master_index):
    """Compute reduction coefficient at each sample point."""
    targets = np.zeros(len(points))
    for i, pt in enumerate(points):
        A = ibp_func(*pt)
        hard, R = reduction_map(A, masters)
        hi = hard.index(hard_index) if hard_index in hard else 0
        targets[i] = R[hi, master_index]
    return targets


def eval_sympy(expr, syms, points):
    """Evaluate a sympy expression at numpy points, handling single-variable case."""
    f = sympy.lambdify(syms, expr, modules="numpy")
    if len(syms) == 1:
        return f(points[:, 0])
    return f(*[points[:, i] for i in range(len(syms))])


def iterative_pole_removal(
    ibp_func,
    masters,
    hard_index,
    master_index,
    bounds,
    variable_names,
    n_pilot=100,
    n_final=1000,
    max_poles=10,
    pysr_pilot_iters=50,
    pysr_final_iters=500,
):
    """
    Iteratively discover and remove poles, then fit the polynomial numerator.

    Returns dict with keys: expr, numerator, denominator, validation
    """
    rng = np.random.default_rng(42)
    syms = sympy.symbols(variable_names)
    if not isinstance(syms, tuple):
        syms = [syms]
    else:
        syms = list(syms)

    # Step 1: pilot sample
    print(f"Sampling {n_pilot} pilot points...")
    pilot_points = sample_good_points(ibp_func, n_pilot * 3, bounds, variable_names, rng=rng)
    if len(pilot_points) > n_pilot:
        pilot_points = pilot_points[:n_pilot]
    print(f"  Got {len(pilot_points)} good points")

    # Step 2: compute targets
    targets = compute_targets(ibp_func, pilot_points, masters, hard_index, master_index)

    # Step 3: pole removal loop
    accumulated_denom = sympy.Integer(1)
    current_targets = targets.copy()

    for iteration in range(max_poles):
        print(f"\nPole removal iteration {iteration + 1}")

        model = pysr.PySRRegressor(
            niterations=pysr_pilot_iters,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            maxsize=20,
            model_selection="score",
            verbosity=1,
        )
        model.fit(pilot_points, current_targets, variable_names=variable_names)

        best_expr = model.sympy()
        # Extract structural denominator (not numeric coefficients)
        denom = sympy.denom(sympy.together(best_expr))
        # Factor and keep only symbolic factors (drop numeric content)
        symbolic_factors = [f for f in sympy.factor(denom).as_ordered_factors()
                            if not f.is_number]
        denom = sympy.Mul(*symbolic_factors) if symbolic_factors else sympy.Integer(1)
        print(f"  Best expression: {best_expr}")
        print(f"  Denominator: {denom}")

        # Check if denominator is constant (no more poles)
        if denom.is_number:
            print("  No poles found — stopping")
            break

        # Evaluate denominator at all points and multiply out
        denom_func = sympy.lambdify(syms, denom, modules="numpy")
        denom_vals = denom_func(*[pilot_points[:, i] for i in range(len(syms))])

        current_targets = current_targets * denom_vals
        accumulated_denom = accumulated_denom * denom
        print(f"  Accumulated denominator: {accumulated_denom}")

    # Step 4: final large run
    print(f"\nFinal run: sampling {n_final} points...")
    final_points = sample_good_points(ibp_func, n_final * 3, bounds, variable_names, rng=rng)
    if len(final_points) > n_final:
        final_points = final_points[:n_final]
    print(f"  Got {len(final_points)} good points")

    final_targets = compute_targets(ibp_func, final_points, masters, hard_index, master_index)

    # Multiply by accumulated denominator
    if accumulated_denom != sympy.Integer(1):
        denom_func = sympy.lambdify(syms, accumulated_denom, modules="numpy")
        denom_vals = denom_func(*[final_points[:, i] for i in range(len(syms))])
        final_targets = final_targets * denom_vals

    print(f"  Fitting numerator (pole-free)...")
    final_model = pysr.PySRRegressor(
        niterations=pysr_final_iters,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        maxsize=25,
        model_selection="score",
        verbosity=1,
    )
    final_model.fit(final_points, final_targets, variable_names=variable_names)

    numerator = final_model.sympy()
    print(f"  Numerator: {numerator}")
    print(f"  Denominator: {accumulated_denom}")

    final_expr = numerator / accumulated_denom
    simplified = sympy.simplify(final_expr)
    print(f"  Final expression: {simplified}")

    # Step 5: validation
    print("\nValidation:")
    val_points = sample_good_points(ibp_func, 200, bounds, variable_names, rng=np.random.default_rng(999))
    val_targets = compute_targets(ibp_func, val_points, masters, hard_index, master_index)

    expr_func = sympy.lambdify(syms, simplified, modules="numpy")
    val_pred = expr_func(*[val_points[:, i] for i in range(len(syms))])

    abs_err = np.abs(val_targets - val_pred)
    print(f"  Max absolute error:  {abs_err.max():.2e}")
    print(f"  Mean absolute error: {abs_err.mean():.2e}")

    # Verify IBP constraint A @ I ≈ 0
    ibp_residuals = []
    for pt in val_points[:20]:
        A = ibp_func(*pt)
        hard, R = reduction_map(A, masters)
        M = np.random.default_rng(0).standard_normal(len(masters))
        I = np.zeros(A.shape[1])
        I[[masters[j] for j in range(len(masters))]] = M
        for k, h in enumerate(hard):
            I[h] = R[k] @ M
        ibp_residuals.append(np.linalg.norm(A @ I))
    print(f"  IBP constraint ||A@I|| (max): {max(ibp_residuals):.2e}")

    return {
        "expr": simplified,
        "numerator": numerator,
        "denominator": accumulated_denom,
        "validation": {
            "max_error": float(abs_err.max()),
            "mean_error": float(abs_err.mean()),
            "ibp_max_residual": float(max(ibp_residuals)),
        },
    }


if __name__ == "__main__":
    # 1D test case from test.py
    def ibp_matrix_1d(d):
        return np.array([
            [1.0, d - 2.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, -1.0, 0.0, d - 4.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ])

    masters_1d = [3, 4]
    hard_integrals = [0, 1, 2]
    bounds_1d = [(2.5, 6.0)]
    variable_names_1d = ["d"]

    # Known exact solutions for reference
    def exact1(d):
        return -(d - 3) / (2 * d - 5), -((d - 3) * (d - 2)) / (2 * d - 5)

    def exact2(d):
        return -1 / (2 * d - 5), (d - 3) / (2 * d - 5)

    def exact3(d):
        return -1 + 1 / (2 * d - 5), -(3 * d - 8) / (2 * d - 5)

    exact_funcs = [exact1, exact2, exact3]

    for hi_idx, hi in enumerate(hard_integrals):
        for mi in range(2):
            master_name = f"M{mi + 1}"
            exact_c1, exact_c2 = exact_funcs[hi_idx](sympy.Symbol("d"))
            exact_coeff = [exact_c1, exact_c2][mi]
            print("=" * 60)
            print(f"Hard integral I{hi + 1}, coefficient of {master_name}")
            print(f"  Exact: {exact_coeff}")
            print("=" * 60)

            result = iterative_pole_removal(
                ibp_func=ibp_matrix_1d,
                masters=masters_1d,
                hard_index=hi,
                master_index=mi,
                bounds=bounds_1d,
                variable_names=variable_names_1d,
                n_pilot=100,
                n_final=500,
                pysr_pilot_iters=50,
                pysr_final_iters=200,
            )

            print(f"\n  Result: {result['expr']}")
            print(f"  Exact:  {exact_coeff}")
            print(f"  Match:  {sympy.simplify(result['expr'] - exact_coeff) == 0}")
            print()
