"""Tests for the generic IBP framework on a two-loop figure-eight vacuum
diagram. The reduction has known closed forms, so we can verify the
numerical pseudoinverse output without invoking PySR.
"""
import numpy as np
import pytest
import sympy

from test2 import (
    IBPProblem,
    collect_data,
    condition_number,
    eval_expr,
    reduction_map,
    solve_problem,
)
from two_loop_example import (
    HARD_LABELS,
    MASTERS,
    make_problem,
    two_loop_figure8_matrix,
)


# --- IBP matrix structure ---

class TestTwoLoopMatrix:
    def test_shape(self):
        A = two_loop_figure8_matrix(4.0, 1.0)
        assert A.shape == (3, 4)

    def test_well_conditioned(self):
        A = two_loop_figure8_matrix(4.0, 1.0)
        assert condition_number(A) < 1e3

    def test_full_row_rank(self):
        A = two_loop_figure8_matrix(4.7, 1.3)
        assert np.linalg.matrix_rank(A) == 3


# --- Reduction matches the closed form ---

class TestTwoLoopReduction:
    @pytest.mark.parametrize("d,m", [
        (3.0, 1.0),
        (4.0, 2.0),
        (5.5, 0.7),
        (3.7, 4.2),
    ])
    def test_reduction_against_closed_form(self, d, m):
        A = two_loop_figure8_matrix(d, m)
        hard, R = reduction_map(A, MASTERS)

        c_v21 = (d - 2) / (2 * m)
        c_v12 = (d - 2) / (2 * m)
        c_v22 = (d - 2) ** 2 / (4 * m ** 2)

        np.testing.assert_allclose(R[hard.index(0), 0], c_v21, atol=1e-10)
        np.testing.assert_allclose(R[hard.index(1), 0], c_v12, atol=1e-10)
        np.testing.assert_allclose(R[hard.index(2), 0], c_v22, atol=1e-10)

    def test_ibp_constraint_holds(self):
        d, m = 4.5, 1.3
        A = two_loop_figure8_matrix(d, m)
        hard, R = reduction_map(A, MASTERS)

        rng = np.random.default_rng(0)
        M = rng.standard_normal(1)
        I = np.zeros(4)
        I[3] = M[0]
        for k, h in enumerate(hard):
            I[h] = R[k] @ M

        residual = np.linalg.norm(A @ I)
        assert residual < 1e-10


# --- IBPProblem dataclass + collect_data ---

class TestIBPProblem:
    def test_construction(self):
        problem = make_problem()
        assert problem.n_vars == 2
        assert list(problem.var_names) == ["d", "m"]
        assert problem.masters == [3]

    def test_syms(self):
        problem = make_problem()
        syms = problem.syms
        assert len(syms) == 2
        assert syms[0].name == "d"
        assert syms[1].name == "m"

    def test_single_var_problem_returns_list(self):
        # Sanity check the IBPProblem.syms property in the 1-parameter case.
        problem = IBPProblem(
            ibp_matrix=lambda d: np.eye(2),
            masters=[1],
            var_names=["d"],
            bounds=[(2.0, 4.0)],
        )
        syms = problem.syms
        assert isinstance(syms, list)
        assert len(syms) == 1
        assert syms[0].name == "d"


class TestCollectData:
    def test_v21_targets_match_closed_form(self):
        problem = make_problem()
        rng = np.random.default_rng(7)
        X, y = collect_data(problem, hard_index=0, master_index=0,
                            n_samples=50, rng=rng)
        assert len(X) > 0
        assert X.shape[1] == 2
        for (d, m), c in zip(X, y):
            np.testing.assert_allclose(c, (d - 2) / (2 * m), atol=1e-10)

    def test_v22_targets_match_closed_form(self):
        problem = make_problem()
        rng = np.random.default_rng(11)
        X, y = collect_data(problem, hard_index=2, master_index=0,
                            n_samples=50, rng=rng)
        assert len(X) > 0
        for (d, m), c in zip(X, y):
            np.testing.assert_allclose(
                c, (d - 2) ** 2 / (4 * m ** 2), atol=1e-10
            )

    def test_filters_ill_conditioned(self):
        # An over-strict threshold should drop every point.
        problem = make_problem()
        rng = np.random.default_rng(0)
        X, y = collect_data(problem, hard_index=0, master_index=0,
                            n_samples=20, rng=rng, cond_threshold=1.0)
        assert len(X) == 0


# --- eval_expr (multivariate generic evaluator) ---

class TestEvalExpr:
    def test_two_var_polynomial(self):
        d, m = sympy.symbols("d m")
        X = np.array([[3.0, 1.0], [4.0, 2.0]])
        out = eval_expr((d - 2) / (2 * m), [d, m], X)
        np.testing.assert_allclose(out, [0.5, 0.5])

    def test_single_var(self):
        d = sympy.Symbol("d")
        X = np.array([[2.0], [4.0], [6.0]])
        out = eval_expr(d ** 2, [d], X)
        np.testing.assert_allclose(out, [4.0, 16.0, 36.0])


# --- Sanity check that solve_problem dispatches correctly (no PySR) ---

class TestSolveProblemDispatch:
    def test_default_indices(self, monkeypatch):
        """``solve_problem`` should iterate over every (hard, master) pair."""
        problem = make_problem()
        called = []

        def fake_solve_coefficient(p, hi, mi, **kwargs):
            called.append((hi, mi))
            return {"hard_index": hi, "master_index": mi, "expr": None}

        # Patch the symbol that ``solve_problem`` looks up at module level.
        import test2
        monkeypatch.setattr(test2, "solve_coefficient", fake_solve_coefficient)

        results = solve_problem(problem)
        assert set(results.keys()) == {(0, 0), (1, 0), (2, 0)}
        assert sorted(called) == [(0, 0), (1, 0), (2, 0)]


# --- Hard-integral labels ---

class TestLabels:
    def test_label_keys_cover_hard(self):
        assert set(HARD_LABELS.keys()) == {0, 1, 2}
