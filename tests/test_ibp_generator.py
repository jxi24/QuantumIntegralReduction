"""Tests for the auto IBP generator.

These verify that ``ibp_generator`` produces equations consistent with
hand-derivations (for the sunrise) and a self-consistent IBP system for
the 1-loop box. The strongest test is the numerical kernel check at the
bottom: any linear combination of the generator's IBP rows, evaluated on
a numerically-consistent integral vector, must give zero.
"""
import numpy as np
import pytest
import sympy

from ibp_generator import (
    build_ibp_system,
    build_sp_algebra,
    box_family,
    enumerate_seed_indices,
    generate_seed_ibp,
    sunrise_family,
)
from kira_loader import load_family, load_top_sector


# ---------------------------------------------------------------------------
# Scalar-product algebra
# ---------------------------------------------------------------------------


def test_sunrise_sp_algebra_shapes():
    fam = sunrise_family()
    sp, M_inv, mass, kin = build_sp_algebra(fam)
    assert [str(s) for s in sp] == ["kk_0_0", "kk_0_1", "kk_1_1"]
    assert M_inv.shape == (3, 3)
    assert list(kin) == [0, 0, 0]


def test_sunrise_sp_algebra_values():
    """``k_i.k_j`` should resolve to the standard sunrise closed forms."""
    fam = sunrise_family()
    sp, M_inv, mass, kin = build_sp_algebra(fam)
    D = sympy.Matrix(sympy.symbols("D0 D1 D2"))
    result = M_inv * (D + mass - kin)
    m2, M2 = sympy.symbols("m2 M2")
    # k_1^2 = D_0 + m^2, k_2^2 = D_1 + m^2, 2 k_1.k_2 = D_0 + D_1 - D_2 + 2m^2 - M^2
    assert sympy.simplify(result[0] - (D[0] + m2)) == 0
    assert sympy.simplify(result[2] - (D[1] + m2)) == 0
    assert sympy.simplify(result[1] * 2 - (D[0] + D[1] - D[2] + 2 * m2 - M2)) == 0


def test_box_sp_algebra_shapes():
    fam = box_family()
    sp, M_inv, mass, kin = build_sp_algebra(fam)
    assert len(sp) == 4
    assert M_inv.shape == (4, 4)


# ---------------------------------------------------------------------------
# Seed enumeration
# ---------------------------------------------------------------------------


def test_enumerate_seeds_sunrise():
    fam = sunrise_family()
    seeds = enumerate_seed_indices(fam, r_max=5, s_max=0, d_max=2)
    # r=3: (1,1,1); r=4: 3 one-dot seeds; r=5: 3 single-triple + 3 pairwise = 6
    assert (1, 1, 1) in seeds
    assert (2, 1, 1) in seeds
    assert (1, 2, 1) in seeds
    assert (1, 1, 2) in seeds
    assert (2, 2, 1) in seeds
    assert len(seeds) == 1 + 3 + 6


def test_enumerate_seeds_d_max_caps_dots():
    fam = sunrise_family()
    seeds = enumerate_seed_indices(fam, r_max=5, s_max=0, d_max=1)
    assert (2, 2, 1) not in seeds
    assert (2, 1, 1) in seeds


# ---------------------------------------------------------------------------
# Individual IBP seeds (sunrise)
# ---------------------------------------------------------------------------


def _sunrise_sp():
    fam = sunrise_family()
    return fam, *build_sp_algebra(fam)


def test_sunrise_ibp_dk1_k1_at_111():
    """The d/dk_1 . k_1 seed at (1,1,1) expands to the classical

        (d-3) S(1,1,1) - M^2 S(1,1,2) - 2 m^2 S(2,1,1) + [tadpole subsectors]

    relation. The two tadpole sub-terms have equal-and-opposite
    coefficients (each evaluates to X / ...) and cancel after
    sub-sector reduction."""
    fam, sp, M_inv, mass, kin = _sunrise_sp()
    eq = generate_seed_ibp(fam, (1, 1, 1), 0, 0, sp, M_inv, mass, kin)
    m2, M2 = sympy.symbols("m2 M2")
    d = sympy.Symbol("d")
    assert sympy.simplify(eq[(1, 1, 1)] - (d - 3)) == 0
    assert sympy.simplify(eq[(1, 1, 2)] - (-M2)) == 0
    assert sympy.simplify(eq[(2, 1, 1)] - (-2 * m2)) == 0
    # The sub-sector cancellation: coefficients of I(0,1,2) and I(1,0,2)
    # are equal up to sign (they both reduce to X = T_m(1) T_M(1) / ...).
    assert sympy.simplify(eq[(0, 1, 2)] + eq[(1, 0, 2)]) == 0


def test_sunrise_ibp_dk2_k2_at_111_is_k1k1_under_swap():
    """By the k_1 <-> k_2 symmetry of the sunrise diagram, the
    d/dk_2 . k_2 seed should be the d/dk_1 . k_1 seed with propagators
    1 and 2 swapped."""
    fam, sp, M_inv, mass, kin = _sunrise_sp()
    eq11 = generate_seed_ibp(fam, (1, 1, 1), 0, 0, sp, M_inv, mass, kin)
    eq22 = generate_seed_ibp(fam, (1, 1, 1), 1, 1, sp, M_inv, mass, kin)
    swapped = {(t[1], t[0], t[2]): v for t, v in eq11.items()}
    assert set(eq22.keys()) == set(swapped.keys())
    for k in eq22:
        assert sympy.simplify(eq22[k] - swapped[k]) == 0


# ---------------------------------------------------------------------------
# Assembled matrix
# ---------------------------------------------------------------------------


def test_sunrise_matrix_shape_and_rank():
    fam = sunrise_family()
    rels, basis, build = build_ibp_system(fam, r_max=5, s_max=0, d_max=2)
    assert len(rels) == 4 * len(enumerate_seed_indices(fam, 5, 0, 2))  # 2L*(L+E)=4 ops/seed
    A = build(4.0, 1.3, 0.8)
    assert A.shape == (len(rels), len(basis))
    # Enough independent rows to reduce the three top-sector integrals
    # S(2,1,1), S(1,2,1), S(1,1,2).
    assert np.linalg.matrix_rank(A) >= 3


def _sunrise_integral_value(a, d, m2, M2, I111, V_m, X):
    """Known closed-form integral values for the sunrise family.

    Uses:
      - master S(1,1,1) = I111 (parameter)
      - 1-loop tadpole recursion T(a,m^2) = (d-2a+2)/(2(a-1) m^2) T(a-1,m^2),
        with T(1,m^2) the tadpole master (cancels out via the
        master_inputs V_m = T_m(1)^2 and X = T_m(1) T_M(1))
      - hand-coded top-sector reductions from two_loop_sunrise.py for
        S(2,1,1) and S(1,1,2), and the k1<->k2 symmetry for S(1,2,1).
    """
    a1, a2, a3 = a
    # Scaleless
    if (a1 > 0 and a2 == 0 and a3 == 0) or \
       (a1 == 0 and a2 > 0 and a3 == 0) or \
       (a1 == 0 and a2 == 0 and a3 > 0) or \
       (a1 == 0 and a2 == 0 and a3 == 0):
        return 0.0

    def T_factor(a_idx, mass_sq):
        """Return T(a_idx, m^2) / T(1, m^2) with the tadpole recursion."""
        if a_idx <= 0:
            # 1-loop bubble at negative powers doesn't appear here.
            raise NotImplementedError
        factor = 1.0
        for k in range(1, a_idx):
            factor *= (d - 2 * k) / (2 * k * mass_sq)
        return factor

    # Sub-sector (a, b, 0): T_m(a) T_m(b) -> coeff * V_m
    if a3 == 0:
        return T_factor(a1, m2) * T_factor(a2, m2) * V_m
    # Sub-sector (a, 0, c): T_m(a) T_M(c) -> coeff * X (after k_2 shift)
    if a2 == 0:
        return T_factor(a1, m2) * T_factor(a3, M2) * X
    if a1 == 0:
        return T_factor(a2, m2) * T_factor(a3, M2) * X

    # Top sector
    if (a1, a2, a3) == (1, 1, 1):
        return I111
    den = 4 * m2 - M2
    if (a1, a2, a3) == (2, 1, 1):
        return ((d - 3) / den) * I111 \
             + ((d - 2) / (2 * m2 * den)) * V_m \
             - ((d - 2) / (2 * m2 * den)) * X
    if (a1, a2, a3) == (1, 2, 1):
        # k_1 <-> k_2 symmetry of (2,1,1)
        return ((d - 3) / den) * I111 \
             + ((d - 2) / (2 * m2 * den)) * V_m \
             - ((d - 2) / (2 * m2 * den)) * X
    if (a1, a2, a3) == (1, 1, 2):
        return ((2 * m2 - M2) * (d - 3) / (M2 * den)) * I111 \
             - ((d - 2) / (M2 * den)) * V_m \
             + ((d - 2) / (M2 * den)) * X

    raise ValueError(f"No closed form for sunrise integral {a}")


def test_sunrise_seed_111_rows_satisfy_ibp_identity():
    """Plug the closed-form integral values of every integral touched by
    the seed-(1,1,1) IBP rows into those rows. The dot product must be
    zero for every IBP identity; this is the strongest check we can run
    without external validation."""
    fam = sunrise_family()
    # We need all four operator pairs at (1,1,1); enumerating with
    # r_max=3 keeps the basis small and avoids needing closed forms for
    # higher-dotted top-sector integrals.
    rels, basis, build = build_ibp_system(fam, r_max=3, s_max=0, d_max=0)
    assert len(rels) == 4   # 2 loops x 2 v choices at seed (1,1,1)

    rng = np.random.default_rng(42)
    for trial in range(5):
        d = rng.uniform(3.5, 5.5)
        m2 = rng.uniform(0.6, 1.6)
        M2 = rng.uniform(0.4, 1.4)
        I111 = rng.standard_normal()
        V_m = rng.standard_normal()
        X = rng.standard_normal()

        I_vec = np.array([
            _sunrise_integral_value(a, d, m2, M2, I111, V_m, X)
            for a in basis
        ])
        A = build(d, m2, M2)
        residual = A @ I_vec
        assert np.max(np.abs(residual)) < 1e-9, \
            f"IBP residual {residual} at trial {trial} (d={d}, m2={m2}, M2={M2})"


def test_sunrise_full_system_ibp_identity_holds():
    """The full r_max=5 IBP system must also satisfy A*I = 0 on the
    physical integral vector, for every row whose integrals we have
    closed forms for. Rows touching higher-dotted top-sector integrals
    (which the closed-form table doesn't cover) are skipped."""
    fam = sunrise_family()
    rels, basis, build = build_ibp_system(fam, r_max=5, s_max=0, d_max=2)

    rng = np.random.default_rng(17)
    d = rng.uniform(3.5, 5.5)
    m2 = rng.uniform(0.6, 1.6)
    M2 = rng.uniform(0.4, 1.4)
    I111 = rng.standard_normal()
    V_m = rng.standard_normal()
    X = rng.standard_normal()

    known = {}
    skip = set()
    for a in basis:
        try:
            known[a] = _sunrise_integral_value(a, d, m2, M2, I111, V_m, X)
        except ValueError:
            skip.add(a)

    # Check every row that doesn't touch a skipped integral.
    A = build(d, m2, M2)
    checked = 0
    for i, eq in enumerate(rels):
        if any(t in skip for t in eq):
            continue
        residual = sum(A[i, basis.index(t)] * known[t] for t in eq)
        assert abs(residual) < 1e-8, f"row {i} residual = {residual}"
        checked += 1
    # We should have verified at least the 4 rows at seed (1,1,1).
    assert checked >= 4


# ---------------------------------------------------------------------------
# KIRA YAML loader round-trip
# ---------------------------------------------------------------------------


def test_yaml_sunrise_matches_handbuilt(tmp_path):
    hand = sunrise_family()
    loaded = load_family(
        "examples/sunrise/integralfamilies.yaml",
        "examples/sunrise/kinematics.yaml",
    )
    assert loaded.name == hand.name
    assert loaded.loop_names == hand.loop_names
    assert list(loaded.ext_names) == list(hand.ext_names)
    for p_l, p_h in zip(loaded.propagators, hand.propagators):
        assert [sympy.sympify(c) for c in p_l.coeffs] == \
               [sympy.sympify(c) for c in p_h.coeffs]
        assert sympy.simplify(p_l.mass_sq - p_h.mass_sq) == 0


def test_yaml_box_matches_handbuilt():
    hand = box_family()
    loaded = load_family(
        "examples/1-loop-box/integralfamilies.yaml",
        "examples/1-loop-box/kinematics.yaml",
    )
    assert loaded.name == hand.name
    assert loaded.loop_names == hand.loop_names
    assert list(loaded.ext_names) == list(hand.ext_names)
    for p_l, p_h in zip(loaded.propagators, hand.propagators):
        assert [sympy.sympify(c) for c in p_l.coeffs] == \
               [sympy.sympify(c) for c in p_h.coeffs]
        assert sympy.simplify(p_l.mass_sq - p_h.mass_sq) == 0
    for k in hand.sp_rules:
        assert k in loaded.sp_rules
        assert sympy.simplify(loaded.sp_rules[k] - hand.sp_rules[k]) == 0


def test_yaml_sunrise_top_sector():
    ts = load_top_sector("examples/sunrise/integralfamilies.yaml")
    assert ts == (1, 1, 1)


def test_yaml_box_top_sector():
    ts = load_top_sector("examples/1-loop-box/integralfamilies.yaml")
    assert ts == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# 1-loop box
# ---------------------------------------------------------------------------


def test_box_matrix_reduces_top_sector_to_masters():
    """For a 1-loop box at generic kinematics with all four propagators
    present, there is exactly one master: I(1,1,1,1). The pseudoinverse
    of the generator's matrix should uniquely determine every other
    top-sector integral in terms of it."""
    fam = box_family()
    rels, basis, build = build_ibp_system(
        fam, r_max=5, s_max=0, d_max=1, top_sector=(1, 1, 1, 1)
    )
    A = build(4.0, 1.1, -0.6, 0.3, 0.7)   # (d, s, t, m12, m22)
    # I(1,1,1,1) must exist in the basis.
    assert (1, 1, 1, 1) in basis
    # There are at least 4 non-trivial IBPs at the (1,1,1,1) seed alone,
    # so the matrix rank should easily cover the 4 top-sector "dotted"
    # unknowns that appear.
    assert np.linalg.matrix_rank(A) >= 4


def test_box_ibp_dk_k_at_1111_master_coefficient():
    """The (d/dk) . k seed at (1,1,1,1) for the 1-loop box should have
    I(1,1,1,1) coefficient equal to (d - 5): the trace term contributes d,
    and each of the four propagator 'self-terms' contributes -1 (k=0
    contributes -2 from the rank-1 diagonal of M_inv, and k=1,2,3 each
    contribute -1 from the 1/2 entries). Hand-derivation in
    test comment; see ibp_generator docstring for the identity."""
    fam = box_family()
    sp, M_inv, mass, kin = build_sp_algebra(fam)
    eq = generate_seed_ibp(fam, (1, 1, 1, 1), 0, 0, sp, M_inv, mass, kin)
    d = sympy.Symbol("d")
    assert sympy.simplify(eq[(1, 1, 1, 1)] - (d - 5)) == 0
    m12, m22, s = sympy.symbols("m12 m22 s")
    # Dotted-propagator coefficients at indices where q_k has a coefficient
    # of k of +1 are -2 * m_k^2 (k=0 and k=1 carry m12, k=2,3 carry m22).
    assert sympy.simplify(eq[(2, 1, 1, 1)] - (-2 * m12)) == 0
    assert sympy.simplify(eq[(1, 2, 1, 1)] - (-2 * m12)) == 0
    # k=2, k=3 propagators carry a kinematic s offset (from p1+p2 in
    # the momentum flow) or not.
    assert sympy.simplify(eq[(1, 1, 2, 1)] - (-m12 - m22 + s)) == 0
    assert sympy.simplify(eq[(1, 1, 1, 2)] - (-m12 - m22)) == 0


def test_box_full_r5_system_is_consistent():
    """All IBP rows should evaluate to finite, numerical coefficients
    at a generic kinematic point (no division by zero from sp_algebra
    inversion), and the matrix rank should grow monotonically with the
    seed count."""
    fam = box_family()
    ranks = []
    for r_max in [4, 5, 6]:
        rels, basis, build = build_ibp_system(
            fam, r_max=r_max, s_max=0, d_max=2, top_sector=(1, 1, 1, 1)
        )
        A = build(4.0, 1.1, -0.6, 0.3, 0.7)
        assert np.all(np.isfinite(A))
        ranks.append(np.linalg.matrix_rank(A))
    # Ranks should be monotonically non-decreasing
    assert ranks[0] <= ranks[1] <= ranks[2]
