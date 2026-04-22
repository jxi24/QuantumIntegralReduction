"""End-to-end driver: auto-generated sunrise IBPs -> IBPProblem -> PySR.

Loads the sunrise family from ``examples/sunrise/*.yaml`` via
``kira_loader``, has ``ibp_generator`` build the IBP matrix, wraps it as
an ``IBPProblem``, and runs the existing symbolic-regression reduction
pipeline on the ``S(2,1,1) -> S(1,1,1)`` coefficient.

What works cleanly
------------------
With only top-sector seeds, the generator leaves sub-sector integrals
(like S(2,1,0), S(2,0,1), etc.) as "effective masters" -- there are no
seeds that reduce them. Pseudoinverse reductions of S(2,1,1) onto those
sub-sector proxies are therefore *min-norm* (not physical), and the
resulting coefficients are not the same rational functions as the
hand-coded 5-integral reduction. The ``S(1,1,1)`` master coefficient,
however, IS physical and IS the clean hand-coded answer::

    c_{S(2,1,1) -> S(1,1,1)}  =  (d - 3) / (4 m2 - M2)

We verify this end-to-end: the pseudoinverse produces it at arbitrary
kinematic points (numerical check at the top of this script), and PySR
then rediscovers the closed form from those samples. To get physical
reductions onto the other masters, one would additionally seed the
sub-sector IBPs -- the generator supports this (pass a different
``top_sector``) but the MVP driver only covers the top sector.
"""
import numpy as np
import sympy

from ibp_generator import build_ibp_system
from kira_loader import load_family, load_top_sector
from test2 import IBPProblem, find_system_poles, reduction_map, solve_coefficient


def build_problem(r_max: int = 4, d_max: int = 1):
    """Load the sunrise YAMLs, build the IBP matrix, wrap as an IBPProblem."""
    fam = load_family(
        "examples/sunrise/integralfamilies.yaml",
        "examples/sunrise/kinematics.yaml",
    )
    top = load_top_sector("examples/sunrise/integralfamilies.yaml")

    relations, basis, build_matrix = build_ibp_system(
        fam, r_max=r_max, s_max=0, d_max=d_max, top_sector=top
    )
    print(f"Generated {len(relations)} IBP relations over {len(basis)} integrals "
          f"for family '{fam.name}' (r_max={r_max}, d_max={d_max})")

    # Treat sub-sector integrals + the top-sector master S(1,1,1) as
    # masters. The top-sector-only seeds cannot reduce sub-sectors, so
    # they stand in as effective masters for the purposes of this
    # reduction.
    sub_sector_cols = [i for i, t in enumerate(basis) if any(x <= 0 for x in t)]
    idx_111 = basis.index((1, 1, 1))
    master_cols = sorted(sub_sector_cols + [idx_111])
    m_idx = {c: i for i, c in enumerate(master_cols)}

    # The one coefficient we can check against a known closed form: the
    # coefficient of I(1,1,1) in the S(2,1,1) reduction.
    d_sym, m2_sym, M2_sym = sympy.symbols("d m2 M2")
    hard_211 = basis.index((2, 1, 1))
    exact = {
        (hard_211, m_idx[idx_111]): (d_sym - 3) / (4 * m2_sym - M2_sym),
    }

    problem = IBPProblem(
        ibp_matrix=build_matrix,
        masters=master_cols,
        var_names=["d", "m2", "M2"],
        # Straddle the 4 m2 = M2 pseudothreshold and the m2 -> 0 pole
        # so pole discovery sees both singularities.
        bounds=[(2.5, 6.0), (0.3, 2.5), (0.3, 8.0)],
        exact=exact,
    )
    return problem, basis, master_cols


def numerical_sanity_check(problem, basis, master_cols):
    """Numerical reduction at four scattered points: confirm the
    I(1,1,1) coefficient equals (d-3)/(4 m2 - M2) exactly."""
    hard_211_col = basis.index((2, 1, 1))
    idx_111 = basis.index((1, 1, 1))
    m_idx_111 = master_cols.index(idx_111)

    print("\nNumerical reduction of S(2,1,1) onto I(1,1,1):")
    print(f"{'d':>5} {'m2':>6} {'M2':>6} | {'generator':>12} {'(d-3)/den':>12}")
    print("-" * 52)
    for d, m2, M2 in [(4.0, 1.3, 0.8), (5.0, 1.0, 2.0),
                      (3.5, 0.7, 1.4), (4.5, 2.0, 0.5)]:
        A = problem.ibp_matrix(d, m2, M2)
        hard, R = reduction_map(A, problem.masters)
        c = R[hard.index(hard_211_col), m_idx_111]
        expected = (d - 3) / (4 * m2 - M2)
        tag = "OK" if abs(c - expected) < 1e-10 else "MISMATCH"
        print(f"{d:>5.2f} {m2:>6.2f} {M2:>6.2f} | {c:>12.6f} {expected:>12.6f}   {tag}")


def main():
    problem, basis, master_cols = build_problem(r_max=4, d_max=1)
    numerical_sanity_check(problem, basis, master_cols)

    hard_211_col = basis.index((2, 1, 1))
    idx_111 = basis.index((1, 1, 1))
    m_idx_111 = master_cols.index(idx_111)

    print(f"\n{'#' * 60}")
    print("# Global pole discovery (shared across masters)")
    print("#" * 60)
    system_poles = find_system_poles(problem, n_samples=200, niterations=80)

    print(f"\n{'#' * 60}")
    print(f"# PySR reduction of S(2,1,1) -> I(1,1,1)")
    print(f"#   expected: (d - 3) / (4*m2 - M2)")
    print("#" * 60)
    result = solve_coefficient(
        problem,
        hard_index=hard_211_col,
        master_index=m_idx_111,
        n_samples=400,
        max_pole_iters=4,
        pysr_pilot_iters=100,
        pysr_final_iters=250,
        system_poles=system_poles,
    )

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    print(f"  PySR expr:    {result['expr']}")
    print(f"  Exact:        {result.get('exact')}")
    print(f"  Matches:      {result.get('match')}")


if __name__ == "__main__":
    main()
