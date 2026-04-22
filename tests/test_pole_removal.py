"""Tests for pole_removal.py utility functions."""
import numpy as np
import pytest
import sympy

from pole_removal import (
    lhs_sample,
    reduction_map,
    sample_good_points,
    compute_targets,
    eval_sympy,
)


def ibp_matrix_1d(d):
    return np.array([
        [1.0, d - 2.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, -1.0, 0.0, d - 4.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
    ])


class TestSampleGoodPoints:
    def test_output_shape(self):
        pts = sample_good_points(ibp_matrix_1d, 100, [(2.5, 6.0)], ["d"],
                                 rng=np.random.default_rng(0))
        assert pts.ndim == 2
        assert pts.shape[1] == 1

    def test_filters_points(self):
        # All points should have good condition number
        pts = sample_good_points(ibp_matrix_1d, 200, [(2.5, 6.0)], ["d"],
                                 rng=np.random.default_rng(0))
        for row in pts:
            A = ibp_matrix_1d(*row)
            s = np.linalg.svd(A, compute_uv=False)
            cond = s.max() / s.min()
            assert cond < 1e6


class TestComputeTargets:
    def test_1d_known_values(self):
        """Test against exact I1 coefficients for 1D system."""
        masters_1d = [3, 4]
        d_vals = np.array([[3.5], [4.0], [5.0]])
        targets = compute_targets(ibp_matrix_1d, d_vals, masters_1d,
                                  hard_index=0, master_index=0)

        for i, d in enumerate([3.5, 4.0, 5.0]):
            expected = -(d - 3) / (2 * d - 5)
            np.testing.assert_allclose(targets[i], expected, atol=1e-10)

    def test_1d_second_master(self):
        masters_1d = [3, 4]
        d_vals = np.array([[4.0]])
        targets = compute_targets(ibp_matrix_1d, d_vals, masters_1d,
                                  hard_index=0, master_index=1)
        # Exact: -((d-3)*(d-2))/(2d-5)
        d = 4.0
        expected = -((d - 3) * (d - 2)) / (2 * d - 5)
        np.testing.assert_allclose(targets[0], expected, atol=1e-10)


class TestEvalSympy:
    def test_single_variable(self):
        d = sympy.Symbol("d")
        pts = np.array([[2.0], [3.0], [4.0]])
        result = eval_sympy(d**2, [d], pts)
        np.testing.assert_allclose(result, [4.0, 9.0, 16.0])

    def test_two_variables(self):
        d, x = sympy.symbols("d x")
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = eval_sympy(d + x, [d, x], pts)
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_rational_expression(self):
        d = sympy.Symbol("d")
        pts = np.array([[4.0]])
        expr = (d - 3) / (2 * d - 5)
        result = eval_sympy(expr, [d], pts)
        np.testing.assert_allclose(result, [1.0 / 3.0])


class TestIBPConstraint1D:
    """Integration test: verify IBP constraint A @ I = 0 for the 1D system."""

    @pytest.mark.parametrize("d", [3.1, 3.5, 4.0, 4.5, 5.0, 5.5])
    def test_constraint(self, d):
        masters_1d = [3, 4]
        A = ibp_matrix_1d(d)
        hard, R = reduction_map(A, masters_1d)

        rng = np.random.default_rng(0)
        M = rng.standard_normal(2)
        I = np.zeros(5)
        I[masters_1d[0]] = M[0]
        I[masters_1d[1]] = M[1]
        for k, h in enumerate(hard):
            I[h] = R[k] @ M

        assert np.linalg.norm(A @ I) < 1e-10

    @pytest.mark.parametrize("d", [3.5, 4.5, 5.5])
    def test_exact_solutions(self, d):
        """Verify all three exact solutions for the 1D system."""
        def exact1(d):
            return -(d - 3) / (2 * d - 5), -((d - 3) * (d - 2)) / (2 * d - 5)
        def exact2(d):
            return -1 / (2 * d - 5), (d - 3) / (2 * d - 5)
        def exact3(d):
            return -1 + 1 / (2 * d - 5), -(3 * d - 8) / (2 * d - 5)

        masters_1d = [3, 4]
        A = ibp_matrix_1d(d)
        hard, R = reduction_map(A, masters_1d)
        exact_funcs = [exact1, exact2, exact3]

        for i, hi in enumerate(hard):
            exact_c1, exact_c2 = exact_funcs[i](d)
            np.testing.assert_allclose(R[i, 0], exact_c1, atol=1e-10,
                                       err_msg=f"I{hi+1} coeff of M1 at d={d}")
            np.testing.assert_allclose(R[i, 1], exact_c2, atol=1e-10,
                                       err_msg=f"I{hi+1} coeff of M2 at d={d}")
