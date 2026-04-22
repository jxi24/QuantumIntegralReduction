"""Tests for the unequal-mass sunrise vacuum two-loop IBP system.

The 2x5 IBP matrix has a 2x2 hard sub-block whose inverse gives the
reductions of ``S(2,1,1)`` and ``S(1,1,2)`` onto three masters
``{S(1,1,1), V_m, X}``. The closed-form coefficients are nontrivial
rational functions of ``(d, m^2, M^2)`` with the sunrise pseudothreshold
``4 m^2 - M^2`` in the denominator.
"""
import numpy as np
import pytest
import sympy

from test2 import collect_data, condition_number, reduction_map, solve_problem
from two_loop_sunrise import (
    HARD_LABELS,
    MASTERS,
    make_problem,
    sunrise_ibp_matrix,
)


# --- Matrix structure ---

class TestSunriseMatrix:
    def test_shape(self):
        A = sunrise_ibp_matrix(4.0, 1.0, 0.7)
        assert A.shape == (2, 5)

    def test_full_row_rank(self):
        A = sunrise_ibp_matrix(4.0, 1.0, 0.7)
        assert np.linalg.matrix_rank(A) == 2

    def test_well_conditioned_generic_point(self):
        A = sunrise_ibp_matrix(4.0, 1.0, 0.7)
        assert condition_number(A) < 1e3

    def test_singular_at_pseudothreshold(self):
        # 4 m^2 - M^2 = 0 makes the hard 2x2 sub-block singular.
        m2, M2 = 1.0, 4.0
        A = sunrise_ibp_matrix(4.0, m2, M2)
        Ah = A[:, [0, 1]]
        assert abs(np.linalg.det(Ah)) < 1e-10


# --- Reduction matches the closed form ---

def _closed_form(d, m2, M2):
    """Returns the 2x3 matrix of expected reduction coefficients."""
    den = 4 * m2 - M2
    return np.array([
        [(d - 3) / den,
         (d - 2) / (2 * m2 * den),
         -(d - 2) / (2 * m2 * den)],
        [(2 * m2 - M2) * (d - 3) / (M2 * den),
         -(d - 2) / (M2 * den),
         (d - 2) / (M2 * den)],
    ])


class TestSunriseReduction:
    @pytest.mark.parametrize("d,m2,M2", [
        (3.0, 1.0, 0.5),
        (4.0, 1.5, 0.7),
        (5.5, 0.8, 1.2),
        (3.7, 2.0, 1.4),
    ])
    def test_reduction_against_closed_form(self, d, m2, M2):
        A = sunrise_ibp_matrix(d, m2, M2)
        hard, R = reduction_map(A, MASTERS)
        assert hard == [0, 1]
        expected = _closed_form(d, m2, M2)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    @pytest.mark.parametrize("d,m2,M2", [
        (3.5, 1.0, 0.7),
        (5.0, 1.5, 1.0),
        (4.2, 0.6, 0.4),
    ])
    def test_ibp_constraint(self, d, m2, M2):
        A = sunrise_ibp_matrix(d, m2, M2)
        hard, R = reduction_map(A, MASTERS)

        rng = np.random.default_rng(0)
        master_vals = rng.standard_normal(len(MASTERS))
        I = np.zeros(A.shape[1])
        for j, mcol in enumerate(MASTERS):
            I[mcol] = master_vals[j]
        for k, h in enumerate(hard):
            I[h] = R[k] @ master_vals

        assert np.linalg.norm(A @ I) < 1e-10

    def test_pseudothreshold_blowup(self):
        # Coefficients should diverge as we approach 4 m^2 = M^2.
        d, m2 = 4.0, 1.0
        near = sunrise_ibp_matrix(d, m2, 4.0 - 1e-3)
        far = sunrise_ibp_matrix(d, m2, 1.0)
        _, R_near = reduction_map(near, MASTERS)
        _, R_far = reduction_map(far, MASTERS)
        assert np.abs(R_near[0, 0]) > 100 * np.abs(R_far[0, 0])


# --- IBPProblem + collect_data on the sunrise ---

class TestSunriseProblem:
    def test_problem_metadata(self):
        problem = make_problem()
        assert problem.n_vars == 3
        assert list(problem.var_names) == ["d", "m2", "M2"]
        assert problem.masters == [2, 3, 4]
        assert set(problem.exact.keys()) == {
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        }

    def test_exact_matches_numerics(self):
        """The closed forms recorded in ``problem.exact`` should agree with
        the numerical pseudoinverse reduction at random points."""
        problem = make_problem()
        d_sym, m2_sym, M2_sym = problem.syms

        rng = np.random.default_rng(2026)
        for _ in range(10):
            d_v = rng.uniform(2.5, 6.0)
            m2_v = rng.uniform(0.6, 2.0)
            M2_v = rng.uniform(0.3, 1.4)
            A = sunrise_ibp_matrix(d_v, m2_v, M2_v)
            hard, R = reduction_map(A, MASTERS)
            for (hi, mi), expr in problem.exact.items():
                f = sympy.lambdify((d_sym, m2_sym, M2_sym), expr, "numpy")
                expected = float(f(d_v, m2_v, M2_v))
                np.testing.assert_allclose(
                    R[hard.index(hi), mi], expected, atol=1e-10
                )

    def test_collect_data_targets(self):
        problem = make_problem()
        rng = np.random.default_rng(7)
        X, y = collect_data(problem, hard_index=0, master_index=0,
                            n_samples=40, rng=rng)
        assert X.shape[1] == 3
        assert len(X) > 0
        for (d_v, m2_v, M2_v), c in zip(X, y):
            np.testing.assert_allclose(
                c, (d_v - 3) / (4 * m2_v - M2_v), atol=1e-10
            )


# --- Sanity check that solve_problem dispatches correctly ---

class TestSolveProblemDispatch:
    def test_dispatches_six_pairs(self, monkeypatch):
        problem = make_problem()
        called = []

        def fake_solve_coefficient(p, hi, mi, **kwargs):
            called.append((hi, mi))
            return {"hard_index": hi, "master_index": mi, "expr": None}

        import test2
        monkeypatch.setattr(test2, "solve_coefficient", fake_solve_coefficient)

        results = solve_problem(problem)
        assert set(results.keys()) == {
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        }
        assert sorted(called) == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ]


class TestLabels:
    def test_hard_labels_cover_hard(self):
        assert set(HARD_LABELS.keys()) == {0, 1}
