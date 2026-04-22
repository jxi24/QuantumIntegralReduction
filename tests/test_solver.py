"""Tests for RationalSystemSolver (main.py)."""
import numpy as np
import pytest
from fractions import Fraction

from main import RationalSystemSolver


class TestRationalSystemSolver:
    def test_setup_valid(self):
        system = np.array([
            [Fraction(1), Fraction(2), Fraction(3)],
            [Fraction(4), Fraction(5), Fraction(6)],
        ])
        solver = RationalSystemSolver(prime=47)
        solver.setup(system)
        assert solver.system is not None
        assert solver.system.shape == (2, 3)

    def test_setup_integer_input(self):
        system = np.array([[1, 2, 3], [4, 5, 6]])
        solver = RationalSystemSolver(prime=47)
        solver.setup(system)
        assert solver.system[0, 0] == Fraction(1)

    def test_setup_wrong_shape_1d(self):
        with pytest.raises(ValueError, match="2D array"):
            RationalSystemSolver().setup(np.array([1, 2, 3]))

    def test_setup_wrong_shape_square(self):
        with pytest.raises(ValueError, match="Expected shape"):
            RationalSystemSolver().setup(np.array([[1, 2], [3, 4]]))

    def test_setup_float_rejected(self):
        system = np.array([[1.5, 2.0, 3.0], [4.0, 5.0, 6.0]])
        solver = RationalSystemSolver()
        with pytest.raises(TypeError):
            solver.setup(system)

    def test_solve_simple_system(self):
        # x + 2y = 5, 3x + y = 5 => x=1, y=2
        system = np.array([
            [Fraction(1), Fraction(2), Fraction(5)],
            [Fraction(3), Fraction(1), Fraction(5)],
        ])
        solver = RationalSystemSolver(prime=47)
        solver.setup(system)
        solutions = solver.solve(verbose=False)
        assert solutions[0] == Fraction(1)
        assert solutions[1] == Fraction(2)

    def test_solve_fractional_system(self):
        # (1/2)x + y = 1/2,  (1/4)x - (3/4)y = -1/2
        system = np.array([
            [Fraction(1, 2), Fraction(1), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(-3, 4), Fraction(-1, 2)],
        ])
        solver = RationalSystemSolver(prime=997)
        solver.setup(system)
        solutions = solver.solve(verbose=False)

        # Verify: substitute back
        lhs0 = Fraction(1, 2) * solutions[0] + solutions[1]
        lhs1 = Fraction(1, 4) * solutions[0] + Fraction(-3, 4) * solutions[1]
        assert lhs0 == Fraction(1, 2)
        assert lhs1 == Fraction(-1, 2)

    def test_solve_no_system(self):
        solver = RationalSystemSolver(prime=47)
        with pytest.raises(ValueError, match="No system setup"):
            solver.solve()

    def test_solve_no_prime(self):
        system = np.array([
            [Fraction(1), Fraction(2), Fraction(3)],
            [Fraction(4), Fraction(5), Fraction(6)],
        ])
        solver = RationalSystemSolver()
        solver.setup(system)
        with pytest.raises(ValueError, match="prime"):
            solver.solve()

    def test_method_chaining(self):
        system = np.array([
            [Fraction(1), Fraction(0), Fraction(7)],
            [Fraction(0), Fraction(1), Fraction(3)],
        ])
        solutions = RationalSystemSolver(prime=997).setup(system).solve()
        assert solutions[0] == Fraction(7)
        assert solutions[1] == Fraction(3)


class TestRationalReconstruction:
    def test_small_integer(self):
        solver = RationalSystemSolver(prime=97)
        solver.gf = None  # not needed for reconstruction
        # 5 mod 97 should reconstruct to 5/1
        num, den = solver._rational_reconstruction(5)
        assert num == 5
        assert den == 1

    def test_negative_integer(self):
        solver = RationalSystemSolver(prime=97)
        # -3 mod 97 = 94
        num, den = solver._rational_reconstruction(94)
        assert num == -3
        assert den == 1

    def test_simple_fraction(self):
        solver = RationalSystemSolver(prime=97)
        # 1/2 mod 97: 2^{-1} mod 97 = 49 (since 2*49=98≡1 mod 97)
        num, den = solver._rational_reconstruction(49)
        assert Fraction(num, den) == Fraction(1, 2)

    def test_fraction_one_third(self):
        solver = RationalSystemSolver(prime=97)
        # 1/3 mod 97: 3^{-1} mod 97
        inv3 = pow(3, -1, 97)
        num, den = solver._rational_reconstruction(inv3)
        assert Fraction(num, den) == Fraction(1, 3)
