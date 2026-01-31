import numpy as np
import galois
import matplotlib.pyplot as plt
from fractions import Fraction
from math import gcd
from functools import reduce


class RationalSystemSolver:
    def __init__(self, prime=None):
        self.prime = prime
        self.gf = None
        self.system = None
        self._finite_system = None

    def setup(self, system):
        """
        Setup the system of equations given in augmented matrix form
        with a shape of (n, n+1)

        Parameters:
        -----------
        system: array-like
            Shape (n, n+1) representing the system of equations
            Can contain: int or Fraction representations
        """

        if len(system.shape) != 2:
            raise ValueError("Input must be a 2D array")

        n_rows, n_cols = system.shape
        if n_cols != n_rows+1:
            raise ValueError(f"Expected shape (n, n+1), got ({n_rows}, {n_cols})")

        # Convert all entries to Fractions
        rational_system = np.empty((n_rows, n_cols), dtype=object)

        for i in range(n_rows):
            for j in range(n_cols):
                rational_system[i, j] = self._to_fraction(system[i, j])

        self.system = rational_system

        return self

    def _to_fraction(self, value):
        """ Convert to Fraction type """
        if isinstance(value, Fraction):
            return value
        elif isinstance(value, (int, np.integer)):
            return Fraction(value)
        else:
            raise TypeError(f"Expected Fraction or integer type, got {type(value)}")

    def _to_finite_field(self, frac):
        num = self.gf(frac.numerator % self.prime)
        den = self.gf(frac.denominator % self.prime)
        one = self.gf(1)

        if den == 0:
            raise ValueError(f"Denominator {frac.denominator} is divisible by prime {self.prime}")

        return num / den


    def solve(self, verbose=True, reconstruct=True):
        if self.system is None:
            raise ValueError("No system setup. Call setup() first.")
        n = self.system.shape[0]

        if self.prime is None:
            raise ValueError("Please set the prime to use.")
        self.gf = galois.GF(self.prime)

        # Convert equations to finite fields
        self._finite_system = self.gf(np.zeros_like(self.system))
        for i in range(n):
            for j in range(n+1):
                self._finite_system[i, j] = self._to_finite_field(self.system[i, j])
        print(self._finite_system)

        # Solve the system of equations and convert back to rational expression
        gf_solutions = np.linalg.solve(self._finite_system[:, :-1],
                                       self._finite_system[:, -1])
        solutions = np.zeros(n, dtype=object)
        for i in range(n):
            num, den = self._rational_reconstruction(gf_solutions[i])
            if num is not None:
                solutions[i] = Fraction(num, den)
            else:
                raise ValueError("Prime is too small to reconstruct solution")

        print(solutions)
        print(self.system[:, :-1] @ solutions)
        print(self.system[:, -1])
        return solutions

    def _rational_reconstruction(self, a, N=None):
        if N is None:
            N = int((self.prime / 2) ** 0.5)

        # Convert from galois field element to int
        if hasattr(a, 'item'):
            a = int(a.item())
        else:
            a = int(a) % self.prime

        # Extended Euclidean algorithm
        r_prev, r_curr = self.prime, a
        s_prev, s_curr = 1, 0
        t_prev, t_curr = 0, 1

        while r_curr > N:
            if r_curr == 0:
                break
            q = r_prev // r_curr
            r_prev, r_curr = r_curr, r_prev - q * r_curr
            s_prev, s_curr = s_curr, s_prev - q * s_curr
            t_prev, t_curr = t_curr, t_prev - q * t_curr

        # Check if valid
        if abs(t_curr) <= N and t_curr != 0:
            # Ensure positive denominator
            if t_curr < 0:
                r_curr, t_curr = -r_curr, -t_curr

            # Ensure in simplified form
            if gcd(abs(r_curr), abs(t_curr)) == 1:
                return r_curr, t_curr

        return None, None


def main():
    system = np.array([
        [Fraction(1, 2), Fraction(1, 1), Fraction(1, 2)],
        [Fraction(1, 4), Fraction(-3, 4), Fraction(-1, 2)]
    ])

    solver = RationalSystemSolver(47)
    solver.setup(system).solve()


if __name__ == "__main__":
    main()
