"""Automated IBP seed-equation generator.

Given a description of a Feynman integral family (loop momenta, external
momenta, propagators, kinematic invariants and their scalar-product
relations), this module produces:

    1.  A list of seed IBP equations. Each equation is a dict mapping
        an integral multi-index ``(a_1, ..., a_N)`` (propagator powers)
        to a sympy coefficient expression in ``(d, *kinematic_invariants)``.
    2.  ``integral_basis``: the unique integrals appearing across all
        equations, used as column labels for the matrix.
    3.  ``build_matrix(*kinematic_point)``: a callable that evaluates the
        symbolic coefficient matrix at a numeric kinematic point and
        returns a ``np.ndarray`` suitable for feeding to the existing
        ``test2.IBPProblem`` pipeline.

The generator works for any family whose number of propagators equals the
number of loop-loop + loop-external scalar products (a "complete" family);
irreducible-scalar-product (ISP) completion is out of scope for the MVP.

The core algorithm is the textbook IBP identity ::

    0 = d/dk_i^mu ( v_j^mu / prod_k D_k^{a_k} )
      = (d v_j^mu / dk_i^mu) / prod D_k^{a_k}
        - sum_k a_k (v_j . dD_k/dk_i) / ( D_k * prod_l D_l^{a_l} )

where ``v_j . dD_k/dk_i = 2 alpha_i^k (v_j . q_k)``. The scalar product
``v_j . q_k`` is rewritten in the propagator basis via the (invertible)
scalar-product matrix, producing shifted integrals.
"""
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import sympy


D_SYM = sympy.Symbol("d")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Propagator:
    """A propagator ``D_k = q_k^2 - m_k^2``.

    ``coeffs`` is a length ``L + E`` vector of sympy/numeric coefficients
    giving ``q_k = sum_i coeffs[i] k_i + sum_j coeffs[L+j] p_j``.
    ``mass_sq`` is a sympy expression in the kinematic symbols.
    """

    coeffs: Sequence
    mass_sq: sympy.Expr


@dataclass
class IntegralFamily:
    """Parsed description of an integral family."""

    name: str
    loop_names: Sequence[str]
    ext_names: Sequence[str]
    propagators: Sequence[Propagator]
    kin_symbols: Sequence[sympy.Symbol]
    # sp_rules[(i, j)] (with i <= j, 0-indexed) = sympy expr in kin_symbols
    # for ``p_i . p_j``. Required for every external-external pair.
    sp_rules: Mapping
    d_sym: sympy.Symbol = field(default=D_SYM)

    @property
    def n_loops(self) -> int:
        return len(self.loop_names)

    @property
    def n_externals(self) -> int:
        return len(self.ext_names)

    @property
    def n_propagators(self) -> int:
        return len(self.propagators)


# ---------------------------------------------------------------------------
# Scalar-product algebra
# ---------------------------------------------------------------------------


def _sp_ll(i: int, j: int) -> sympy.Symbol:
    i, j = sorted((i, j))
    return sympy.Symbol(f"kk_{i}_{j}")


def _sp_le(i: int, j: int) -> sympy.Symbol:
    return sympy.Symbol(f"kp_{i}_{j}")


def _momentum_dot(u: Sequence, v: Sequence, family: IntegralFamily) -> sympy.Expr:
    """Scalar product of two momenta given as coefficient vectors.

    External-external components are substituted via ``family.sp_rules`` so
    the result is an expression in loop-loop / loop-external SP unknowns
    plus kinematic constants only.
    """
    L, E = family.n_loops, family.n_externals
    total = sympy.Integer(0)
    for a in range(L + E):
        if u[a] == 0:
            continue
        for b in range(L + E):
            if v[b] == 0:
                continue
            c = sympy.sympify(u[a]) * sympy.sympify(v[b])
            if a < L and b < L:
                total += c * _sp_ll(a, b)
            elif a < L and b >= L:
                total += c * _sp_le(a, b - L)
            elif a >= L and b < L:
                total += c * _sp_le(b, a - L)
            else:
                i, j = sorted((a - L, b - L))
                if (i, j) not in family.sp_rules:
                    raise KeyError(
                        f"Missing sp_rule for external pair "
                        f"({family.ext_names[i]}, {family.ext_names[j]})"
                    )
                total += c * family.sp_rules[(i, j)]
    return sympy.expand(total)


def build_sp_algebra(family: IntegralFamily):
    """Return ``(sp_basis, M_inv, mass_vec, kin_vec)``.

    ``sp_basis`` lists the loop-loop and loop-external SP symbols, ordered
    canonically. ``M`` (implicit) satisfies
    ``q_k^2 = sum_s M[k,s] sp_basis[s] + kin_vec[k]`` so that the
    propagator ``D_k = q_k^2 - m_k^2`` gives
    ``sp = M_inv @ (D + mass_vec - kin_vec)``.
    """
    L, E = family.n_loops, family.n_externals
    N = family.n_propagators

    sp_basis: list[sympy.Symbol] = []
    for i in range(L):
        for j in range(i, L):
            sp_basis.append(_sp_ll(i, j))
    for i in range(L):
        for j in range(E):
            sp_basis.append(_sp_le(i, j))

    if len(sp_basis) != N:
        raise ValueError(
            f"Family '{family.name}' has {N} propagators but "
            f"{len(sp_basis)} loop-loop + loop-external SPs. The MVP "
            f"requires a complete basis (no ISPs to solve for)."
        )

    rows = []
    kin_consts = []
    for prop in family.propagators:
        q2 = _momentum_dot(prop.coeffs, prop.coeffs, family)
        row = [q2.coeff(sp) for sp in sp_basis]
        const_part = sympy.expand(
            q2 - sum(r * s for r, s in zip(row, sp_basis))
        )
        rows.append(row)
        kin_consts.append(const_part)

    M = sympy.Matrix(rows)
    if M.det() == 0:
        raise ValueError(
            f"SP-to-propagator matrix for '{family.name}' is singular."
        )
    M_inv = M.inv()
    mass_vec = sympy.Matrix([p.mass_sq for p in family.propagators])
    kin_vec = sympy.Matrix(kin_consts)
    return sp_basis, M_inv, mass_vec, kin_vec


# ---------------------------------------------------------------------------
# Seed IBP generation
# ---------------------------------------------------------------------------


def generate_seed_ibp(
    family: IntegralFamily,
    a: Sequence[int],
    i: int,
    j: int,
    sp_basis,
    M_inv,
    mass_vec,
    kin_vec,
) -> dict[tuple, sympy.Expr]:
    """Generate one IBP relation from the operator ``(d/dk_i) . v_j`` at
    base indices ``a``.

    ``i`` is a loop index in ``[0, L)``. ``j`` indexes the momentum ``v_j``
    in the combined loop+external list (``j < L`` -> ``k_j``; otherwise
    ``p_{j-L}``).
    """
    L, E, N = family.n_loops, family.n_externals, family.n_propagators
    result: dict[tuple, sympy.Expr] = {}

    def add(key, coeff):
        coeff = sympy.expand(coeff)
        if coeff == 0:
            return
        if key in result:
            result[key] = sympy.expand(result[key] + coeff)
        else:
            result[key] = coeff

    v_j = [0] * (L + E)
    v_j[j] = 1

    # Term 1: (d v_j^mu / d k_i^mu) / prod D^a.
    # v_j = k_i  -> trace gives d (spacetime dimension); v_j = k_l (l != i) -> 0;
    # v_j = p_m  -> 0.
    if j < L and j == i:
        add(tuple(a), family.d_sym)

    # Term 2: - sum_k a_k * 2 alpha_i^k (v_j . q_k) / D_k / prod D^a
    for k in range(N):
        alpha_ik = sympy.sympify(family.propagators[k].coeffs[i])
        if alpha_ik == 0 or a[k] == 0:
            continue
        q_k = family.propagators[k].coeffs
        dot = _momentum_dot(v_j, q_k, family)
        # Split into (coeffs over sp_basis) + kin_part
        sp_coeffs = sympy.Matrix([dot.coeff(sp) for sp in sp_basis])
        kin_part = sympy.expand(
            dot - sum(c * s for c, s in zip(sp_coeffs, sp_basis))
        )
        # sp = M_inv @ (D + mass - kin), so dot = sp_coeffs^T * sp_vec.
        row = sp_coeffs.T * M_inv       # 1 x N: coefficients on D_l
        const = (row * (mass_vec - kin_vec))[0, 0] + kin_part

        prefactor = -sympy.sympify(a[k]) * 2 * alpha_ik
        for l in range(N):
            c_l = sympy.expand(row[0, l])
            if c_l == 0:
                continue
            new_a = list(a)
            new_a[k] += 1
            new_a[l] -= 1
            add(tuple(new_a), prefactor * c_l)
        const = sympy.expand(const)
        if const != 0:
            new_a = list(a)
            new_a[k] += 1
            add(tuple(new_a), prefactor * const)

    return {k: v for k, v in result.items() if v != 0}


# ---------------------------------------------------------------------------
# Seed enumeration
# ---------------------------------------------------------------------------


def _compositions(total: int, n: int, min_val: int = 0) -> Iterable[tuple]:
    if n == 0:
        if total == 0:
            yield ()
        return
    for first in range(min_val, total - min_val * (n - 1) + 1):
        for rest in _compositions(total - first, n - 1, min_val):
            yield (first,) + rest


def enumerate_seed_indices(
    family: IntegralFamily,
    r_max: int,
    s_max: int,
    d_max: int,
    top_sector: Optional[Sequence[int]] = None,
) -> list[tuple]:
    """Enumerate integrand index tuples for a sector under KIRA-style cutoffs.

    ``top_sector[k] == 1`` means propagator ``k`` must have positive power
    (propagator present); others carry non-positive powers (numerators).
    ``r_max`` caps ``sum_{k in sector} a_k`` and ``s_max`` caps
    ``sum_{k not in sector} max(0, -a_k)``. ``d_max`` caps the number of
    dotted propagators (``a_k >= 2``) inside the sector.
    """
    N = family.n_propagators
    if top_sector is None:
        top_sector = tuple([1] * N)
    sector = [k for k in range(N) if top_sector[k] == 1]
    outside = [k for k in range(N) if top_sector[k] == 0]

    seeds: list[tuple] = []
    for r in range(len(sector), r_max + 1):
        for pos in _compositions(r, len(sector), min_val=1):
            if sum(1 for v in pos if v >= 2) > d_max:
                continue
            for s in range(0, s_max + 1):
                if not outside and s > 0:
                    continue
                for neg in _compositions(s, len(outside), min_val=0):
                    a = [0] * N
                    for p, v in zip(sector, pos):
                        a[p] = v
                    for p, v in zip(outside, neg):
                        a[p] = -v
                    seeds.append(tuple(a))
                if not outside:
                    break
    return seeds


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_ibp_system(
    family: IntegralFamily,
    r_max: int,
    s_max: int = 0,
    d_max: int = 2,
    top_sector: Optional[Sequence[int]] = None,
):
    """Build the full IBP system for a family under a cutoff.

    Returns
    -------
    relations: list of dicts ``{integral_tuple: coeff_expr}``. Each dict is
        a single IBP seed equation.
    integral_basis: sorted list of unique integral index tuples.
    build_matrix: callable ``(d_val, *kin_vals) -> np.ndarray`` of shape
        ``(len(relations), len(integral_basis))``.
    """
    sp_basis, M_inv, mass_vec, kin_vec = build_sp_algebra(family)
    seeds = enumerate_seed_indices(family, r_max, s_max, d_max, top_sector)

    L, E = family.n_loops, family.n_externals
    relations: list[dict] = []
    for a in seeds:
        for i in range(L):
            for j in range(L + E):
                eq = generate_seed_ibp(
                    family, a, i, j, sp_basis, M_inv, mass_vec, kin_vec
                )
                if eq:
                    relations.append(eq)

    all_ints: set[tuple] = set()
    for eq in relations:
        all_ints.update(eq.keys())
    integral_basis = sorted(all_ints, key=lambda t: (sum(t), t))
    col = {t: i for i, t in enumerate(integral_basis)}

    n_rel = len(relations)
    n_int = len(integral_basis)
    A_sym = sympy.zeros(n_rel, n_int)
    for r_idx, eq in enumerate(relations):
        for t, c in eq.items():
            A_sym[r_idx, col[t]] = c

    params = [family.d_sym, *family.kin_symbols]
    f = sympy.lambdify(params, A_sym, modules="numpy")

    def build_matrix(*kin_point) -> np.ndarray:
        if len(kin_point) != len(params):
            raise TypeError(
                f"build_matrix expects {len(params)} args "
                f"({[str(p) for p in params]}); got {len(kin_point)}."
            )
        return np.asarray(f(*kin_point), dtype=float)

    build_matrix.params = tuple(str(p) for p in params)
    build_matrix.integral_basis = integral_basis
    return relations, integral_basis, build_matrix


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def render_relation(eq: Mapping[tuple, sympy.Expr], name: str = "I") -> sympy.Expr:
    """Render a relation ``{tuple: coeff}`` as ``sum_t coeff_t * I[t]``."""
    I = sympy.IndexedBase(name)
    return sympy.Add(*(c * I[t] for t, c in eq.items()))


# ---------------------------------------------------------------------------
# Hand-built family factories (useful as ground truth in tests)
# ---------------------------------------------------------------------------


def sunrise_family() -> IntegralFamily:
    """2-loop unequal-mass sunrise: no externals, two masses ``m2, M2``."""
    m2, M2 = sympy.symbols("m2 M2")
    return IntegralFamily(
        name="sunrise",
        loop_names=["k1", "k2"],
        ext_names=[],
        propagators=[
            Propagator(coeffs=[1, 0], mass_sq=m2),
            Propagator(coeffs=[0, 1], mass_sq=m2),
            Propagator(coeffs=[1, -1], mass_sq=M2),
        ],
        kin_symbols=[m2, M2],
        sp_rules={},
    )


def box_family() -> IntegralFamily:
    """1-loop box with two internal masses ``m12, m22`` and massless externals.

    Basis of independent externals: ``p1, p2, p3`` (``p4 = -p1-p2-p3``).
    Invariants: ``s = (p1+p2)^2``, ``t = (p2+p3)^2``.
    """
    s, t, m12, m22 = sympy.symbols("s t m12 m22")
    return IntegralFamily(
        name="box",
        loop_names=["k"],
        ext_names=["p1", "p2", "p3"],
        propagators=[
            Propagator(coeffs=[1, 0, 0, 0], mass_sq=m12),
            Propagator(coeffs=[1, 1, 0, 0], mass_sq=m12),
            Propagator(coeffs=[1, 1, 1, 0], mass_sq=m22),
            Propagator(coeffs=[1, 1, 1, 1], mass_sq=m22),
        ],
        kin_symbols=[s, t, m12, m22],
        sp_rules={
            (0, 0): sympy.Integer(0),
            (1, 1): sympy.Integer(0),
            (2, 2): sympy.Integer(0),
            (0, 1): s / 2,
            (1, 2): t / 2,
            (0, 2): -(s + t) / 2,
        },
    )

if __name__ == '__main__':
    family = box_family()
    results = build_ibp_system(family, 2)
