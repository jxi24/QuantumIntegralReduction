"""Symbolic-regression IBP reduction for a simple two-loop integral.

We pick the two-loop figure-eight vacuum bubble (two disconnected one-loop
tadpoles of equal mass), whose integral

    V(a, b) = int d^d k1 d^d k2 / [(k1^2 - m^2)^a * (k2^2 - m^2)^b]

factorises as ``T(a) * T(b)`` for one-loop tadpoles ``T(a)``. The standard
one-loop IBP recursion ``T(a+1) = (d - 2a) / (2 a m^2) * T(a)`` then gives
the closed-form reductions

    V(2,1) = (d - 2) / (2 m^2) * V(1,1)
    V(1,2) = (d - 2) / (2 m^2) * V(1,1)
    V(2,2) = (d - 2)^2 / (4 m^4) * V(1,1)

Three independent IBP relations -- one loop-1 derivation
``(d - 2a) V(a,b) = 2 a m^2 V(a+1,b)`` at ``(a,b)=(1,1)`` and ``(1,2)``
plus the loop-2 partner at ``(1,1)`` -- on the four-element basis
``[V(2,1), V(1,2), V(2,2), V(1,1)]`` give a 3x4 IBP matrix. We pick
``V(1,1)`` (column 3) as the master and feed the system to the generic
:class:`test2.IBPProblem` pipeline.
"""
import numpy as np
import sympy

from test2 import IBPProblem, find_system_poles, solve_coefficient


# Integral order: I = [V(2,1), V(1,2), V(2,2), V(1,1)]
# Master:         V(1,1) = column 3
# Parameter ``m`` is the propagator mass squared.
def two_loop_figure8_matrix(d, m):
    return np.array([
        [-2 * m,  0,      0,      d - 2],   # (d-2) V(1,1) - 2m V(2,1) = 0
        [0,       d - 2, -2 * m,  0    ],   # (d-2) V(1,2) - 2m V(2,2) = 0
        [0,      -2 * m,  0,      d - 2],   # (d-2) V(1,1) - 2m V(1,2) = 0
    ], dtype=float)


MASTERS = [3]
HARD_LABELS = {0: "V(2,1)", 1: "V(1,2)", 2: "V(2,2)"}


def make_problem():
    d, m = sympy.symbols("d m")
    exact = {
        (0, 0): (d - 2) / (2 * m),                # V(2,1)
        (1, 0): (d - 2) / (2 * m),                # V(1,2)
        (2, 0): (d - 2) ** 2 / (4 * m ** 2),      # V(2,2)
    }
    return IBPProblem(
        ibp_matrix=two_loop_figure8_matrix,
        masters=MASTERS,
        var_names=["d", "m"],
        bounds=[(2.5, 6.0), (0.5, 5.0)],
        exact=exact,
    )


if __name__ == "__main__":
    problem = make_problem()

    # Discover the shared denominator-pole candidates from det(Ah) once;
    # every coefficient of this problem reuses the same factor pool.
    print(f"\n{'#'*60}\n# Global pole discovery (shared)\n{'#'*60}")
    system_poles = find_system_poles(problem, n_samples=200)

    summary = []
    for hi in (0, 1, 2):
        print(f"\n{'#'*60}")
        print(f"# Reducing {HARD_LABELS[hi]} -> V(1,1)")
        print(f"{'#'*60}")
        result = solve_coefficient(
            problem,
            hard_index=hi,
            master_index=0,
            n_samples=400,
            max_pole_iters=5,
            pysr_pilot_iters=100,
            pysr_final_iters=300,
            system_poles=system_poles,
        )
        summary.append((HARD_LABELS[hi], result))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for label, result in summary:
        match = result.get("match", "n/a")
        print(f"  {label} = {result['expr']}    (matches exact: {match})")
