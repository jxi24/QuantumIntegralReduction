"""YAML loader for KIRA-style integral-family descriptions.

Reads a pair of YAML files ::

    integralfamilies.yaml
    kinematics.yaml

matching the format used by the KIRA IBP reducer (a permissive subset of
it) and produces an :class:`ibp_generator.IntegralFamily` ready for
:func:`ibp_generator.build_ibp_system`.

Supported YAML keys
-------------------

``integralfamilies.yaml``::

    integralfamilies:
      - name: "sunrise"
        loop_momenta: [k1, k2]
        top_level_sectors: [b111]           # optional; decoded via _parse_sector
        propagators:
          - ["k1",      "m2"]
          - ["k2",      "m2"]
          - ["k1 - k2", "M2"]

``kinematics.yaml``::

    kinematics:
      incoming_momenta: [p1, p2]
      outgoing_momenta: [p3, p4]
      momentum_conservation: [p4, "-p1-p2-p3"]   # optional
      kinematic_invariants:
        - [s, 2]
        - [t, 2]
        - [m12, 2]
        - [m22, 2]
      scalarproduct_rules:
        - [[p1, p2], "s/2"]
        - [[p2, p3], "t/2"]
        - [[p1, p3], "-(s+t)/2"]
        - [[p1, p1], 0]
        - [[p2, p2], 0]
        - [[p3, p3], 0]
"""
from pathlib import Path
from typing import Union

import sympy
import yaml

from ibp_generator import IntegralFamily, Propagator


def _parse_expr(expr, local_syms) -> sympy.Expr:
    return sympy.sympify(sympy.S(expr, locals=local_syms))


def _resolve_externals(kinematics: dict):
    """Return ``(independent_external_names, conservation_subs)``.

    The independent set is ``incoming + outgoing`` minus any momentum
    eliminated by an entry in ``momentum_conservation``.
    """
    incoming = [str(m) for m in kinematics.get("incoming_momenta", []) or []]
    outgoing = [str(m) for m in kinematics.get("outgoing_momenta", []) or []]
    all_ext = incoming + outgoing

    conservation = kinematics.get("momentum_conservation") or []
    if conservation and isinstance(conservation[0], str):
        conservation = [conservation]

    syms = {m: sympy.Symbol(m) for m in all_ext}
    subs = {}
    for item in conservation:
        name, expr = item[0], item[1]
        subs[syms[name]] = _parse_expr(expr, syms)

    independent = [m for m in all_ext if syms[m] not in subs]
    return independent, subs, syms


def _momentum_coeffs(expr: sympy.Expr, basis_syms):
    """Return the coefficient list of ``expr`` (assumed linear) over
    the ordered ``basis_syms``."""
    expr = sympy.expand(expr)
    coeffs = []
    for s in basis_syms:
        c = expr.coeff(s)
        coeffs.append(c)
    remainder = sympy.expand(expr - sum(c * s for c, s in zip(coeffs, basis_syms)))
    if remainder != 0:
        raise ValueError(
            f"Momentum expression {expr} is not a pure linear combination "
            f"of {basis_syms}; leftover: {remainder}"
        )
    return coeffs


def _parse_sector(label) -> tuple:
    """Parse a KIRA sector label ``bNNNN`` (or an int/list) into a 0/1 tuple."""
    if isinstance(label, (list, tuple)):
        return tuple(int(b) for b in label)
    s = str(label)
    if s.startswith("b"):
        s = s[1:]
    return tuple(int(c) for c in s)


def load_family(
    families_path: Union[str, Path],
    kinematics_path: Union[str, Path],
    family_name: str = None,
) -> IntegralFamily:
    """Load a single :class:`IntegralFamily` from a pair of YAML files.

    If the families YAML lists several, ``family_name`` picks one; by
    default the first is returned.
    """
    with open(families_path) as f:
        fams = yaml.safe_load(f)["integralfamilies"]
    with open(kinematics_path) as f:
        kin = yaml.safe_load(f)["kinematics"]

    if family_name is None:
        entry = fams[0]
    else:
        entry = next(e for e in fams if e["name"] == family_name)

    loop_names = [str(m) for m in entry["loop_momenta"]]
    loop_syms = [sympy.Symbol(n) for n in loop_names]

    ext_names, cons_subs, ext_all_syms = _resolve_externals(kin)
    ext_syms = [ext_all_syms[n] for n in ext_names]

    kin_invariants = kin.get("kinematic_invariants") or []
    kin_symbols = [sympy.Symbol(str(e[0])) for e in kin_invariants]
    kin_locals = {str(s): s for s in kin_symbols}
    all_locals = {**kin_locals, **{n: s for n, s in zip(loop_names, loop_syms)}}
    all_locals.update(ext_all_syms)

    propagators = []
    for mom_expr, mass_expr in entry["propagators"]:
        mom = _parse_expr(mom_expr, all_locals)
        if cons_subs:
            mom = sympy.expand(mom.subs(cons_subs))
        basis_syms = loop_syms + ext_syms
        coeffs = _momentum_coeffs(mom, basis_syms)
        mass_sq = _parse_expr(mass_expr, kin_locals) if mass_expr != 0 else sympy.Integer(0)
        propagators.append(Propagator(coeffs=coeffs, mass_sq=mass_sq))

    sp_rules_raw = kin.get("scalarproduct_rules") or []
    sp_rules: dict = {}
    ext_index = {name: i for i, name in enumerate(ext_names)}
    for pair, val in sp_rules_raw:
        a, b = str(pair[0]), str(pair[1])
        a_expr = _parse_expr(a, ext_all_syms)
        b_expr = _parse_expr(b, ext_all_syms)
        if cons_subs:
            a_expr = sympy.expand(a_expr.subs(cons_subs))
            b_expr = sympy.expand(b_expr.subs(cons_subs))
        # For the MVP we only accept rules whose sides are single independent
        # external symbols; reduced forms come through via ``cons_subs`` when
        # the propagator expressions are expanded.
        a_coeffs = _momentum_coeffs(a_expr, ext_syms)
        b_coeffs = _momentum_coeffs(b_expr, ext_syms)
        ai = _single_nonzero(a_coeffs)
        bi = _single_nonzero(b_coeffs)
        if ai is None or bi is None:
            # Drop: non-trivial combinations are implied by linearity.
            continue
        i, j = sorted((ai, bi))
        rule_val = _parse_expr(val, kin_locals) if val != 0 else sympy.Integer(0)
        rule_val = rule_val * a_coeffs[ai] * b_coeffs[bi]
        sp_rules[(i, j)] = sympy.expand(rule_val)

    return IntegralFamily(
        name=str(entry["name"]),
        loop_names=loop_names,
        ext_names=ext_names,
        propagators=propagators,
        kin_symbols=kin_symbols,
        sp_rules=sp_rules,
    )


def _single_nonzero(coeffs):
    nz = [i for i, c in enumerate(coeffs) if c != 0]
    if len(nz) == 1 and coeffs[nz[0]] != 0:
        return nz[0]
    return None


def load_top_sector(families_path: Union[str, Path], family_name: str = None):
    """Return the first ``top_level_sectors`` entry as a 0/1 tuple, or ``None``."""
    with open(families_path) as f:
        fams = yaml.safe_load(f)["integralfamilies"]
    entry = fams[0] if family_name is None else next(
        e for e in fams if e["name"] == family_name
    )
    sectors = entry.get("top_level_sectors")
    if not sectors:
        return None
    return _parse_sector(sectors[0])
