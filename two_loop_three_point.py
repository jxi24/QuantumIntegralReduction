"""Coupled two-loop three-point IBP test problem.

This is a *constructed* (rather than derived-from-Feynman-rules) IBP
system designed to stress the symbolic-regression pipeline on a problem
that is genuinely two-loop in flavour: four kinematic parameters, four
hard integrals coupled to two masters, and four distinct pole factors
that all have to be rediscovered before the numerator fits can succeed.
The construction principle is to write a lower-triangular hard
sub-block whose diagonal entries are exactly the pole factors we want
the framework to find, and to thread off-diagonal couplings between
rows so that successive reductions inherit and combine those factors.
This gives a system whose answers a human would not realistically write
down by inspection -- but which we can verify by hand via forward
substitution -- making it a controlled stand-in for an "unknown" two-
loop three-point reduction.

Parameters
----------
``d``     : spacetime dimension
``s``     : Mandelstam-like invariant carrying a (s - 4 m^2) threshold
``t``     : second kinematic invariant
``m2``    : an internal squared mass

Basis order
-----------
``I = [ h0, h1, h2, h3, M0, M1 ]``  (masters = [4, 5])

IBP relations (one row per equation, ``A . I = 0``)
---------------------------------------------------
::

    Row 0:  (s - 4 m2) h0                                  + (d-3) M0          = 0
    Row 1:  (d-2)      h0 + m2  h1                                  + (d-2) M1 = 0
    Row 2:                  (d-3) h1 + t   h2              - (d-4) M0          = 0
    Row 3:  (d-4)      h0                  + (s-t) h3                + (d-3) M1 = 0

Hand-derived exact reductions
-----------------------------
::

    h0 = -(d-3)/(s - 4 m2) * M0
    h1 = (d-2)(d-3) / (m2 (s - 4 m2)) * M0 - (d-2)/m2 * M1
    h2 = [(d-4) m2 (s - 4 m2) - (d-2)(d-3)^2]
         / (t * m2 * (s - 4 m2)) * M0
         + (d-2)(d-3) / (t * m2) * M1
    h3 = (d-4)(d-3) / ((s - t)(s - 4 m2)) * M0
         - (d-3) / (s - t) * M1

The hard sub-block is lower-triangular, so

    det(Ah) = (s - 4 m2) * m2 * t * (s - t)

i.e. ``find_system_poles`` should rediscover all four factors from a
single determinant fit and share them across every coefficient call.
``h2`` is the most demanding case: a three-factor denominator and a
degree-4 polynomial numerator that mixes ``d``, ``s``, and ``m2``.
"""
import numpy as np
import sympy

from test2 import IBPProblem, find_system_poles, solve_coefficient


# Basis order:   I = [h0, h1, h2, h3, M0, M1]
# Master columns: 4, 5
def three_point_ibp_matrix(d, s, t, m2):
    return np.array([
        # (s-4 m2) h0 + (d-3) M0 = 0
        [s - 4.0 * m2, 0.0,    0.0, 0.0,        d - 3.0, 0.0    ],
        # (d-2) h0 + m2 h1 + (d-2) M1 = 0
        [d - 2.0,      m2,     0.0, 0.0,        0.0,     d - 2.0],
        # (d-3) h1 + t h2 - (d-4) M0 = 0
        [0.0,          d - 3.0, t,  0.0,       -(d - 4.0), 0.0  ],
        # (d-4) h0 + (s-t) h3 + (d-3) M1 = 0
        [d - 4.0,      0.0,    0.0, s - t,      0.0,     d - 3.0],
    ], dtype=float)


MASTERS = [4, 5]
HARD_LABELS = {0: "h0", 1: "h1", 2: "h2", 3: "h3"}
MASTER_LABELS = {0: "M0", 1: "M1"}


def make_problem():
    """Build the coupled three-point IBP problem with hand-derived answers."""
    d, s, t, m2 = sympy.symbols("d s t m2")
    sm = s - 4 * m2          # s threshold factor
    exact = {
        # h0
        (0, 0): -(d - 3) / sm,
        (0, 1): sympy.Integer(0),
        # h1
        (1, 0):  (d - 2) * (d - 3) / (m2 * sm),
        (1, 1): -(d - 2) / m2,
        # h2 -- the demanding one
        (2, 0):  ((d - 4) * m2 * sm - (d - 2) * (d - 3) ** 2)
                 / (t * m2 * sm),
        (2, 1):  (d - 2) * (d - 3) / (t * m2),
        # h3
        (3, 0):  (d - 4) * (d - 3) / ((s - t) * sm),
        (3, 1): -(d - 3) / (s - t),
    }
    return IBPProblem(
        ibp_matrix=three_point_ibp_matrix,
        masters=MASTERS,
        var_names=["d", "s", "t", "m2"],
        # Bounds chosen so every pole factor actually vanishes (or comes
        # close to it) somewhere in the LHS-sampled box: ``s - 4 m2`` and
        # ``s - t`` straddle zero, and ``m2``/``t`` get small enough that
        # ``pole_score``'s "factor must approach zero" filter accepts
        # them. Without this, factors that never go near zero in the
        # dataset are silently rejected.
        bounds=[(2.5, 6.0), (0.3, 5.0), (0.2, 4.0), (0.05, 1.0)],
        exact=exact,
    )


if __name__ == "__main__":
    problem = make_problem()

    # Single global determinant fit -- expected to recover the four
    # factors {s - 4 m2, m2, t, s - t}. All four reductions then reuse
    # the same candidate pool instead of refitting per coefficient.
    print(f"\n{'#'*60}\n# Global pole discovery (shared)\n{'#'*60}")
    system_poles = find_system_poles(problem, n_samples=400, niterations=150)

    summary = []
    for hi in (0, 1, 2, 3):
        for mi in (0, 1):
            print(f"\n{'#'*60}")
            print(f"# Reducing {HARD_LABELS[hi]} -> {MASTER_LABELS[mi]}")
            print(f"{'#'*60}")
            result = solve_coefficient(
                problem,
                hard_index=hi,
                master_index=mi,
                n_samples=800,
                max_pole_iters=6,
                pysr_pilot_iters=150,
                pysr_final_iters=400,
                system_poles=system_poles,
            )
            summary.append((HARD_LABELS[hi], MASTER_LABELS[mi], result))

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for hl, ml, result in summary:
        match = result.get("match", "n/a")
        print(f"  {hl} -> {ml}: {result['expr']}    (matches exact: {match})")
