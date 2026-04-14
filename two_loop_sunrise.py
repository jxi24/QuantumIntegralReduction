"""Unequal-mass sunrise vacuum bubble: a non-trivial two-loop IBP system.

Two-loop sunrise diagram with three propagators carrying masses
``(m, m, M)`` and no external momentum:

    S(a, b, c) = int d^d k1 d^d k2 /
                 [(k1^2 - m^2)^a (k2^2 - m^2)^b ((k1+k2)^2 - M^2)^c]

The integrals ``S(2,1,1)`` and ``S(1,1,2)`` reduce, via two independent
IBP seeds at ``(1,1,1)``, to a basis of three master integrals:

    M0 = S(1,1,1)             (the sunrise master)
    M1 = V_m = T_m(1)^2       (squared one-loop tadpole, mass m)
    M2 = X   = T_m(1) T_M(1)  (mixed one-loop tadpole product)

The two seed IBPs are

    Eq1:  (d - 3) S(1,1,1) - 2 m^2 S(2,1,1) - M^2 S(1,1,2) = 0
    Eq2:  2 m^2 (d - 3) S(1,1,1) - 2 m^2 M^2 S(2,1,1)
            - 4 m^2 M^2 S(1,1,2) - (d - 2) V_m + (d - 2) X = 0

(the second is the d/dk1 * (k1 + k2) seed, multiplied through by ``2 m^2``
to clear fractions and after eliminating the tadpole sub-products
``S(2,1,0) = (d-2)/(2 m^2) V_m`` and ``S(2,0,1) = (d-2)/(2 m^2) X``).
Solving the 2x2 hard sub-block produces a denominator
``M^2 * (4 m^2 - M^2)``; the second factor is the famous sunrise
pseudothreshold ``4 m^2 = M^2``.

Throughout this module ``m2`` and ``M2`` denote the *squared* masses
``m^2`` and ``M^2`` (the IBP coefficients are polynomial in those).
"""
import numpy as np
import sympy

from test2 import IBPProblem, find_system_poles, solve_coefficient


# Basis order: I = [ S(2,1,1), S(1,1,2), S(1,1,1), V_m, X ]
# Master columns: 2, 3, 4
def sunrise_ibp_matrix(d, m2, M2):
    return np.array([
        [-2 * m2,       -M2,           d - 3,            0,         0     ],
        [-2 * m2 * M2,  -4 * m2 * M2,  2 * m2 * (d - 3), -(d - 2),  d - 2 ],
    ], dtype=float)


MASTERS = [2, 3, 4]
HARD_LABELS = {0: "S(2,1,1)", 1: "S(1,1,2)"}
MASTER_LABELS = {0: "S(1,1,1)", 1: "V_m", 2: "X"}


def make_problem():
    """Build the sunrise :class:`test2.IBPProblem` with closed-form answers."""
    d, m2, M2 = sympy.symbols("d m2 M2")
    den = 4 * m2 - M2
    exact = {
        # S(2,1,1) reductions
        (0, 0): (d - 3) / den,
        (0, 1): (d - 2) / (2 * m2 * den),
        (0, 2): -(d - 2) / (2 * m2 * den),
        # S(1,1,2) reductions
        (1, 0): (2 * m2 - M2) * (d - 3) / (M2 * den),
        (1, 1): -(d - 2) / (M2 * den),
        (1, 2): (d - 2) / (M2 * den),
    }
    return IBPProblem(
        ibp_matrix=sunrise_ibp_matrix,
        masters=MASTERS,
        var_names=["d", "m2", "M2"],
        # The pseudothreshold 4 m2 = M2 lies at M2 in [0.4, 10] for the
        # m2 range below. We deliberately let the box straddle it (as
        # well as the m2 = 0 pole) so that LHS samples land on both
        # sides of each singularity -- the conditioning filter inside
        # collect_data drops the points that are too close, but the
        # surviving "near-pole" points are exactly what pole_score
        # needs in order to discover the (4 m2 - M2) and m2 factors.
        bounds=[(2.5, 6.0), (0.1, 2.5), (0.3, 8.0)],
        exact=exact,
    )


if __name__ == "__main__":
    problem = make_problem()

    # Discover candidate poles from the (pseudo-)determinant of Ah once
    # and reuse the result for every coefficient. For the sunrise system
    # we expect to recover ``m2`` and ``(4 m2 - M2)`` from this single fit.
    print(f"\n{'#'*60}\n# Global pole discovery (shared)\n{'#'*60}")
    system_poles = find_system_poles(problem, n_samples=300)

    # Sweep the three coefficients of S(2,1,1) onto the masters as a
    # representative demo. Comment in ``hi=1`` to also do S(1,1,2).
    for hi in (0,):
        for mi in (0, 1, 2):
            print(f"\n{'#'*60}")
            print(f"# Reducing {HARD_LABELS[hi]} -> {MASTER_LABELS[mi]}")
            print(f"{'#'*60}")
            solve_coefficient(
                problem,
                hard_index=hi,
                master_index=mi,
                n_samples=600,
                max_pole_iters=5,
                pysr_pilot_iters=120,
                pysr_final_iters=300,
                system_poles=system_poles,
            )
