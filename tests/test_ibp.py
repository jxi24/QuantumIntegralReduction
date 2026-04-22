"""Tests for IBP matrix construction, reduction, and data generation (test2.py)."""
import numpy as np
import pytest
import sympy

from test2 import (
    ibp_matrix,
    reduction_map,
    lhs_sample,
    is_good_point,
    generate_data,
    eval_at_points,
    round_poly,
    pole_score,
    d_sym,
    x_sym,
    masters,
)


# --- IBP matrix ---

class TestIBPMatrix:
    def test_shape(self):
        A = ibp_matrix(4.0, 2.0)
        assert A.shape == (6, 8)

    def test_dtype(self):
        A = ibp_matrix(4.0, 2.0)
        assert A.dtype == float

    def test_known_entries(self):
        A = ibp_matrix(4.0, 1.0)
        # Row 0: [-x*(4+x), 2+(5-d)*x, 2+x, -4, 0, 0, 0, 0]
        # d=4, x=1: [-1*5, 2+1*1, 3, -4, 0, 0, 0, 0] = [-5, 3, 3, -4, 0, 0, 0, 0]
        np.testing.assert_allclose(A[0], [-5, 3, 3, -4, 0, 0, 0, 0])

    def test_row3_structure(self):
        # Row 3: [0, 0, 0, -2, 2-d/2, 0, 0, 0]
        A = ibp_matrix(6.0, 3.0)
        np.testing.assert_allclose(A[3], [0, 0, 0, -2, 2 - 3, 0, 0, 0])

    def test_row4_row5(self):
        # These rows don't depend on d or x
        A = ibp_matrix(5.0, 7.0)
        np.testing.assert_allclose(A[4], [0, 0, 0, 0, 1, -1, 0, 0])
        np.testing.assert_allclose(A[5], [0, 0, 0, 0, -1, 0, 0, 1 - 5.0/2])


# --- Reduction map ---

class TestReductionMap:
    def test_hard_indices(self):
        A = ibp_matrix(4.0, 2.0)
        hard, R = reduction_map(A, masters)
        assert hard == [0, 1, 2, 3, 4, 5]
        assert 6 not in hard and 7 not in hard

    def test_R_shape(self):
        A = ibp_matrix(4.0, 2.0)
        hard, R = reduction_map(A, masters)
        assert R.shape == (6, 2)  # 6 hard integrals, 2 masters

    def test_ibp_constraint_satisfied(self):
        """A @ I should be ~0 when I = R @ M."""
        d, x = 5.0, 3.0
        A = ibp_matrix(d, x)
        hard, R = reduction_map(A, masters)

        rng = np.random.default_rng(42)
        M = rng.standard_normal(2)
        I = np.zeros(8)
        I[masters[0]] = M[0]
        I[masters[1]] = M[1]
        for k, h in enumerate(hard):
            I[h] = R[k] @ M

        residual = np.linalg.norm(A @ I)
        assert residual < 1e-10

    def test_exact_coefficient_1d(self):
        """Check reduction against known exact answer for the 1D system."""
        d = 4.3
        A = np.array([
            [1.0, d - 2.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, -1.0, 0.0, d - 4.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ])
        masters_1d = [3, 4]
        hard, R = reduction_map(A, masters_1d, eps=1e-10)

        # Exact: I1 coefficient of M1 = -(d-3)/(2d-5)
        expected_c1 = -(d - 3) / (2 * d - 5)
        assert hard[0] == 0
        np.testing.assert_allclose(R[0, 0], expected_c1, atol=1e-10)

    def test_exact_coefficient_2d(self):
        """Check reduction against known exact answer for the 2D system."""
        d, x = 5.0, 3.0
        A = ibp_matrix(d, x)
        hard, R = reduction_map(A, masters)

        # Exact: I1 coeff of M1 = (d-3)*((d-6)*x - 4) / (x*(x+4)^2)
        expected = (d - 3) * ((d - 6) * x - 4) / (x * (x + 4)**2)
        i1_idx = hard.index(0)
        np.testing.assert_allclose(R[i1_idx, 0], expected, atol=1e-10)


# --- LHS sampling ---

class TestLHSSample:
    def test_shape(self):
        samples = lhs_sample(50, [(0, 1), (2, 5)], rng=np.random.default_rng(0))
        assert samples.shape == (50, 2)

    def test_bounds_respected(self):
        bounds = [(1.5, 3.5), (-2, 8)]
        samples = lhs_sample(200, bounds, rng=np.random.default_rng(0))
        assert samples[:, 0].min() >= bounds[0][0]
        assert samples[:, 0].max() <= bounds[0][1]
        assert samples[:, 1].min() >= bounds[1][0]
        assert samples[:, 1].max() <= bounds[1][1]

    def test_reproducible(self):
        s1 = lhs_sample(30, [(0, 1)], rng=np.random.default_rng(42))
        s2 = lhs_sample(30, [(0, 1)], rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_1d(self):
        samples = lhs_sample(100, [(0, 10)], rng=np.random.default_rng(0))
        assert samples.shape == (100, 1)


# --- is_good_point ---

class TestIsGoodPoint:
    def test_well_conditioned_point(self):
        assert is_good_point(5.0, 3.0)

    def test_near_singular_at_x0(self):
        # At x=0 the matrix loses rank
        assert not is_good_point(5.0, 0.0)

    def test_near_singular_at_xm4(self):
        # At x=-4 the matrix loses rank
        assert not is_good_point(5.0, -4.0)

    def test_custom_threshold(self):
        # Very strict threshold should reject more points
        assert not is_good_point(5.0, 3.0, threshold=1.0)


# --- generate_data ---

class TestGenerateData:
    def test_filters_bad_points(self):
        # Include x=0 which should be filtered out
        samples = np.array([[5.0, 0.0], [5.0, 3.0], [5.0, -4.0]])
        X, y = generate_data(samples)
        assert len(X) == 1  # only (5, 3) survives
        np.testing.assert_allclose(X[0], [5.0, 3.0])

    def test_output_matches_exact(self):
        samples = np.array([[5.0, 3.0]])
        X, y = generate_data(samples)
        expected = (5 - 3) * ((5 - 6) * 3 - 4) / (3 * (3 + 4)**2)
        np.testing.assert_allclose(y[0], expected, atol=1e-10)

    def test_empty_input(self):
        X, y = generate_data(np.empty((0, 2)))
        assert len(X) == 0
        assert len(y) == 0


# --- eval_at_points ---

class TestEvalAtPoints:
    def test_constant(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = eval_at_points(sympy.Integer(5), X)
        np.testing.assert_allclose(result, [5, 5])

    def test_linear(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        expr = d_sym + x_sym
        result = eval_at_points(expr, X)
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_rational(self):
        X = np.array([[2.0, 5.0]])
        expr = d_sym / x_sym
        result = eval_at_points(expr, X)
        np.testing.assert_allclose(result, [0.4])


# --- round_poly ---

class TestRoundPoly:
    def test_already_integer(self):
        d, x = sympy.symbols("d x")
        result = round_poly(x + 4)
        assert result is not None
        assert sympy.expand(result - (x + 4)) == 0

    def test_close_to_integer(self):
        d, x = sympy.symbols("d x")
        result = round_poly(x + 3.9924)
        assert result is not None
        assert sympy.expand(result - (x + 4)) == 0

    def test_scaled_expression(self):
        d, x = sympy.symbols("d x")
        # 2x + 8.0038 should round to 2(x+4) -> factor x+4
        result = round_poly(2 * x + 8.0038)
        assert result is not None
        assert sympy.expand(result - (x + 4)) == 0

    def test_quarter_scale(self):
        d, x = sympy.symbols("d x")
        # 0.25*x + 1 = (x+4)/4, should round to x+4
        result = round_poly(0.249880 * x + 1.0)
        assert result is not None
        assert sympy.expand(result - (x + 4)) == 0

    def test_non_roundable(self):
        d, x = sympy.symbols("d x")
        # Irrational-ish coefficients
        result = round_poly(x + d * (-0.505) + 7.935)
        assert result is None

    def test_pure_number_returns_none(self):
        result = round_poly(sympy.Float(3.14))
        assert result is None

    def test_multivariable(self):
        d, x = sympy.symbols("d x")
        result = round_poly(2.01 * d + 3.99 * x + 1.005)
        assert result is not None
        assert sympy.expand(result - (2 * d + 4 * x + 1)) == 0


# --- pole_score ---

class TestPoleScore:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        samples = lhs_sample(2000, [(2, 8), (-10, 10)], rng)
        X, y = generate_data(samples)
        return X, y

    def test_real_pole_x(self, sample_data):
        X, y = sample_data
        score = pole_score(x_sym, X, y)
        assert score > 0.35  # x=0 is a real pole

    def test_real_pole_x_plus_4(self, sample_data):
        X, y = sample_data
        score = pole_score(x_sym + 4, X, y)
        assert score > 0.35  # x=-4 is a real pole

    def test_fake_pole_d_minus_3(self, sample_data):
        X, y = sample_data
        score = pole_score(d_sym - 3, X, y)
        # d=3 is a numerator zero, not a denominator pole; the new
        # log-correlation metric returns a *negative* score for those.
        assert score < 0.2

    def test_fake_pole_arbitrary(self, sample_data):
        X, y = sample_data
        score = pole_score(2 * d_sym - x_sym, X, y)
        assert score < 0.2

    def test_polynomial_far_from_zero(self, sample_data):
        # ``x^2 + 200`` is bounded below by 200 across the whole box, so
        # it cannot be a denominator factor of ``y``; the
        # "min|f| / median|f|" filter should reject it outright.
        X, y = sample_data
        score = pole_score(x_sym ** 2 + 200, X, y)
        assert score == 0.0

    def test_too_few_points(self):
        X = np.array([[5.0, 3.0]])
        y = np.array([1.0])
        score = pole_score(x_sym, X, y)
        assert score == 0.0


# --- IBP constraint (integration test) ---

class TestIBPConstraint:
    """Verify that A @ I = 0 holds for multiple parameter values."""

    @pytest.mark.parametrize("d,x", [
        (3.0, 1.0),
        (4.5, 2.5),
        (7.0, -3.0),
        (2.5, 8.0),
        (6.0, -9.0),
    ])
    def test_constraint(self, d, x):
        if not is_good_point(d, x):
            pytest.skip("Point is ill-conditioned")
        A = ibp_matrix(d, x)
        hard, R = reduction_map(A, masters)

        rng = np.random.default_rng(0)
        M = rng.standard_normal(2)
        I = np.zeros(8)
        I[masters[0]] = M[0]
        I[masters[1]] = M[1]
        for k, h in enumerate(hard):
            I[h] = R[k] @ M

        assert np.linalg.norm(A @ I) < 1e-10


# --- Exact answer matches numerical reduction ---

class TestExactAnswer:
    """Verify the exact formula (d-3)((d-6)x-4)/(x(x+4)^2) matches numerics."""

    @pytest.mark.parametrize("d,x", [
        (3.5, 1.0),
        (5.0, 3.0),
        (4.0, 7.0),
        (7.0, -2.0),
        (2.5, 5.5),
        (6.0, -8.0),
        (3.0, 2.0),  # d=3 -> numerator is 0
    ])
    def test_matches_numerical(self, d, x):
        if not is_good_point(d, x):
            pytest.skip("Point is ill-conditioned")

        A = ibp_matrix(d, x)
        hard, R = reduction_map(A, masters)
        i1_idx = hard.index(0)
        numerical = R[i1_idx, 0]

        exact = (d - 3) * ((d - 6) * x - 4) / (x * (x + 4)**2)
        np.testing.assert_allclose(numerical, exact, atol=1e-10)
