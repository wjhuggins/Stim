"""Tests for K-pipe colouring (`color_z`) next to Y cubes.

The SAT/SMT encoding gives no colour variables for K-pipes (time-like pipes).
`color_z` fills them in afterwards: ColorKM[i][j][k] is the colour at the
minus end of the K-pipe (i,j,k)->(i,j,k+1), which sits at cube k. ColorKP is
the colour at its plus end, at cube k+1. -1 means "not coloured yet". A
colour is the orientation of the patch's X/Z boundaries at that end. When
KM != KP on one pipe the boundaries rotate along the pipe, which is a logical
Hadamard (ZXGridGraph and networkx_generator both read it this way).

The expected values come from the physics, not from running the code:

  Continuity. At an ordinary cube (not a Y cube, not a port) the patch runs
  straight through, so two K-pipes meeting at cube k have the same colour
  there: ColorKP[k-1] == ColorKM[k]. Colour must therefore propagate across
  every ordinary cube.

  Y cubes. A Y cube is a Y-basis termination. Its lower half measures the
  pipe below it in Y, and its upper half prepares |Y> for the pipe above it.
  A Y cube with K-pipes on both sides is two separate Y-tails that happen to
  be the bottom and top halves of one cube. Nothing connects them, so colour
  must never propagate across a Y cube. The colour at a Y end is not fixed
  by anything: H Y H = -Y, so flipping it only flips a sign (a Pauli frame).

The bug these tests pin down (present since the pass was written): the
from-below step in `propogate_Kcolor` checked `NodeY[i][j][k-1]` (the cube
at the far end of pipe k-1) instead of `NodeY[i][j][k]` (cube k, where pipes
k-1 and k actually meet). The from-above step already checked the shared
cube, `NodeY[i][j][k+1]`. Checking the wrong cube causes two failures:

  (a) Leak. With a Y cube at k and an ordinary cube at k-1, colour was copied
      across the Y cube from one tail to the other. Harmless for the graph,
      since that end is free anyway, but it contradicts the physics.
  (b) Blocked propagation. With an ordinary cube at k and a Y cube at k-1,
      propagation through cube k was refused. The loop in `color_kp_km`
      then falls back to `assign_Kcolor`, which may pick a different colour
      for pipe k, breaking continuity at cube k. LaSsynth's own translators
      read this as an extra Hadamard on the Y-tail. Because H Y H = -Y, that
      only changes a sign, so LaSsynth's own stabilizer check never saw it.
      Stricter consumers that require X/Z consistency at every ordinary cube
      reject the colouring.

All fixtures are 1x1xN columns, so only K-pipes exist.
"""

import copy

import pytest
import stimzx

from lassynth.lattice_surgery_synthesis import LatticeSurgerySolution
from lassynth.rewrite_passes import color_z as cz


def _k(values):
    """Wrap a per-k list as a 1x1xN array."""
    return [[list(values)]]


# ---------------------------------------------------------------------------
# One call of propogate_Kcolor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0, 1])
def test_colour_propagates_through_ordinary_cube_above_y(c):
    """Continuity, failure (b). Column, bottom to top:

        cube 0: Y (upper half prepares |Y>)
        pipe 0: 0 -> 1, the Y-tail; ColorKP[0] = c
        cube 1: ordinary cube, with K-pipes only
        pipe 1: 1 -> 2; ColorKM[1] unknown

    Cube 1 is ordinary, so the patch is continuous there, and one
    propagation pass must set ColorKM[1] = ColorKP[0] = c.

    Old code: at k=1 the from-below step checked NodeY[0] (the Y cube at the
    far end of pipe 0) and skipped the copy, leaving ColorKM[1] = -1.
    """
    n_k = 3
    exist_k = _k([1, 1, 0])
    node_y = _k([1, 0, 0])
    kp = _k([c, -1, -1])
    km = _k([-1, -1, -1])

    cz.propogate_Kcolor(1, 1, n_k, exist_k, kp, km, node_y)

    assert km[0][0][1] == c, (
        f"colour did not propagate through ordinary cube 1: "
        f"ColorKM[1]={km[0][0][1]}, expected {c}")


def test_colour_does_not_cross_two_half_y_cube():
    """Y cubes, failure (a). Column, bottom to top:

        pipe 0: 0 -> 1; ColorKP[0] = 1
        cube 1: Y cube with pipes on both sides = two separate Y-tails
                (lower half measures pipe 0, upper half prepares pipe 1)
        pipe 1: 1 -> 2; ColorKM[1] and ColorKP[1] unknown

    The two tails are not connected, so nothing may flow across cube 1.
    After one propagation pass ColorKM[1] must still be -1.

    Old code: at k=1 the from-below step checked NodeY[0] (ordinary, 0) and
    copied ColorKP[0] into ColorKM[1] across the Y cube.
    """
    n_k = 3
    exist_k = _k([1, 1, 0])
    node_y = _k([0, 1, 0])
    kp = _k([1, -1, -1])
    km = _k([-1, -1, -1])

    cz.propogate_Kcolor(1, 1, n_k, exist_k, kp, km, node_y)

    assert km[0][0][1] == -1, (
        f"colour crossed the two-half Y cube at k=1: ColorKM[1]={km[0][0][1]}, "
        "expected -1 (unknown)")


# ---------------------------------------------------------------------------
# Full colouring of a Y-prep column that ends in an output port.
# ---------------------------------------------------------------------------


def _y_prep_column(port_colour):
    """Column: Y prep at cube 0, ordinary cubes 1 and 2, output port cube 3.

        cube 3: output port cube (port on pipe 2, e='+')
        pipe 2: 2 -> 3; ColorKP[2] = port colour
        cube 2: ordinary
        pipe 1: 1 -> 2
        cube 1: ordinary (carries the Y-tail)
        pipe 0: 0 -> 1, the Y-tail
        cube 0: Y (upper half prepares |Y>)

    As a circuit, this prepares a Y eigenstate on one output qubit. There
    are no specified stabilizers (n_s = 0) because only the colouring and
    the derived ZX graph are under test here.
    """
    n_k = 4
    lasre = {
        "n_i": 1,
        "n_j": 1,
        "n_k": n_k,
        "n_p": 1,
        "n_s": 0,
        "ports": [{"i": 0, "j": 0, "k": 2, "d": "K", "e": "+", "c": port_colour}],
        "stabs": [],
        "NodeY": _k([1, 0, 0, 0]),
        "ExistI": _k([0] * n_k),
        "ExistJ": _k([0] * n_k),
        "ExistK": _k([1, 1, 1, 0]),
        "ColorI": _k([-1] * n_k),
        "ColorJ": _k([-1] * n_k),
    }
    # check_lasre validates the layout and fills in port_cubes.
    return LatticeSurgerySolution(lasre).lasre


def _colour_with_pinned_y_end(lasre, y_end_colour):
    """Run color_kp_km's own steps, but pin the free Y-end colour first.

    The colour at the Y end (ColorKM[0]) is a free choice (see "Y cubes"
    above), and the default code happens to pick 0. Every choice must still
    give a continuous colouring. The loop below is color_kp_km's loop,
    unchanged. Only the pin is added, between the port/IJ seeding and the
    loop.
    """
    lasre = copy.deepcopy(lasre)
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]
    kp = [[[-1] * n_k for _ in range(n_j)] for _ in range(n_i)]
    km = [[[-1] * n_k for _ in range(n_j)] for _ in range(n_i)]
    cz.color_ports(lasre["ports"], kp, km)
    cz.propogate_IJcolor(n_i, n_j, n_k, lasre["ExistI"], lasre["ExistJ"],
                         lasre["ExistK"], lasre["ColorI"], lasre["ColorJ"], kp,
                         km)
    km[0][0][0] = y_end_colour
    while cz.if_uncolorK(n_i, n_j, n_k, lasre["ExistK"], kp, km):
        if not cz.propogate_Kcolor(n_i, n_j, n_k, lasre["ExistK"], kp, km,
                                   lasre["NodeY"]):
            if not cz.assign_Kcolor(n_i, n_j, n_k, lasre["ExistK"], kp, km,
                                    lasre["NodeY"]):
                raise ValueError("Cannot assign color to all K-pipes")
    lasre["ColorKP"], lasre["ColorKM"] = kp, km
    return lasre


def _continuity_violations(lasre):
    """Ordinary cubes (not Y, not port) where the two K-pipes' colours differ."""
    bad = []
    port_cubes = {tuple(c) for c in lasre["port_cubes"]}
    ek, ny = lasre["ExistK"], lasre["NodeY"]
    kp, km = lasre["ColorKP"], lasre["ColorKM"]
    for i in range(lasre["n_i"]):
        for j in range(lasre["n_j"]):
            for k in range(1, lasre["n_k"]):
                if ny[i][j][k] or (i, j, k) in port_cubes:
                    continue
                if ek[i][j][k - 1] and ek[i][j][k]:
                    if kp[i][j][k - 1] != km[i][j][k]:
                        bad.append(((i, j, k), kp[i][j][k - 1], km[i][j][k]))
    return bad


_CASES = [("default", None)] + [(f"y_end={y}", y) for y in (0, 1)]


def _coloured(port_colour, y_end):
    lasre = _y_prep_column(port_colour)
    if y_end is None:
        return cz.color_z(copy.deepcopy(lasre))
    return _colour_with_pinned_y_end(lasre, y_end)


@pytest.mark.parametrize("port_colour", [0, 1])
@pytest.mark.parametrize("name,y_end", _CASES, ids=[c[0] for c in _CASES])
def test_any_y_end_choice_gives_continuous_colouring(name, y_end, port_colour):
    """Continuity + Y cubes, failure (b) end to end. Whatever the free Y-end
    colour is, the colouring must be continuous at ordinary cubes 1 and 2.

    Old code with y_end=1: ColorKP[0] becomes 1 (assign_Kcolor copies
    ColorKM[0] to ColorKP[0]). At cube 1 the from-below step checked NodeY[0]
    and refused to propagate, so the fallback in assign_Kcolor forced
    ColorKM[1] = 0, which differs from ColorKP[0] = 1 at ordinary cube 1.
    The default and y_end=0 cases passed only because the fallback's 0
    happened to match.
    """
    lasre = _coloured(port_colour, y_end)

    assert _continuity_violations(lasre) == []


def _unsigned_output_stabilizers(lasre):
    graph = LatticeSurgerySolution(copy.deepcopy(lasre)).to_networkx_graph()
    stabs = stimzx.zx_graph_to_external_stabilizers(graph)
    out = set()
    for s in stabs:
        assert len(s.input) == 0
        out.add(str(s.output)[1:])  # drop the sign character
    return out


@pytest.mark.parametrize("port_colour", [0, 1])
@pytest.mark.parametrize("name,y_end", _CASES, ids=[c[0] for c in _CASES])
def test_y_end_choice_changes_at_most_a_sign(name, y_end, port_colour):
    """Y cubes, regression guard. The column prepares a Y eigenstate, so the
    output stabilizer must be Y up to sign for every Y-end choice and port
    colour. Only the sign may depend on the choice (H Y H = -Y).

    This passed on the old code too: the extra Hadamard produced by failure
    (b) sits on a Y-tail, so it only flips the sign. It is here to make sure
    the fix does not change the unsigned stabilizer.
    """
    lasre = _coloured(port_colour, y_end)

    assert _unsigned_output_stabilizers(lasre) == {"Y"}


# ---------------------------------------------------------------------------
# assign_Kcolor: what to do when propagation is stuck.
#
# `color_kp_km` alternates two steps: propagate colours as far as they go
# (`propogate_Kcolor`), and when nothing more can be deduced, make one free
# choice (`assign_Kcolor`) and go back to propagating. `assign_Kcolor` has
# three rules, tried in order. Each makes exactly one assignment and returns
# True:
#
#   (1) A pipe with a known minus end and an unknown plus end: copy KM to KP.
#       Nothing fixes KP yet, so this puts no Hadamard on the pipe; any
#       Hadamard the chain needs lands where it meets a fixed colour, and a
#       Hadamard can slide along a chain of ordinary cubes without changing
#       the diagram.
#   (2) A pipe ending at a Y cube whose Y end is unknown: set it to 0. The
#       colour at a Y end is free up to a sign (H Y H = -Y).
#   (3) A pipe with a known plus end and an unknown minus end: copy KP to KM.
#
# If no rule applies, it returns False and `color_kp_km` raises ValueError.
#
# Why rule (3) exists, and why nothing else can be stuck. Suppose
# propagation and rules (1) and (2) all fail to apply, and take any pipe P
# with an unknown end.
#   * If KM is known, then KP is unknown and rule (1) applies. So P's KM is
#     unknown.
#   * Descend the column from P. Let Q be the current pipe (first P), with
#     KM unknown. Q's minus cube is not a Y cube (else rule (2) applies) and
#     has no I/J pipe (else `propogate_IJcolor` would have coloured Q's KM).
#     If a K-pipe arrives at it from below, that pipe's KP is unknown (else
#     from-below propagation applies) and so is its KM (else rule (1)
#     applies). Step down to it, now with both ends unknown, and repeat. The
#     column is finite, so the descent stops at a pipe Q whose minus cube
#     touches nothing but Q: a non-Y cube of degree 1.
#   * The SAT forbids a non-Y cube of degree 1 (`constraint_no_deg1`) except
#     at the two ends of a port pipe, so Q is a K port pipe. With e='-',
#     `color_ports` would have coloured Q's KM, so e='+': Q's plus cube is
#     the port cube and Q's KP is known. Every pipe the descent stepped onto
#     has both ends unknown, so Q is P itself.
#   * Hence every pipe with an unknown end is an output port pipe (e='+')
#     with KP known, KM unknown, and an inner cube of degree 1: a dangling
#     output port.
# The SAT allows a dangling port only when every specified stabilizer is the
# identity on it (the inner cube carries no correlation surface). That
# happens for partial specifications (fewer stabilizers than ports, or none).
# Rule (3) colours exactly that pipe, so `return False` is unreachable on SAT
# output. `remove_unconnected`, which runs first, removes whole components
# and does not change the degree of any cube it keeps.
# ---------------------------------------------------------------------------


def _dangling_port_column(e, c):
    """1x1x2 column with one K-pipe, pipe 0, which is a port pipe:

        e='+' (output port): cube 1 is the port cube, cube 0 the inner cube.
        e='-' (input port):  cube 0 is the port cube, cube 1 the inner cube.

    The inner cube has no other pipe (degree 1). The SAT allows this only
    for a port that every stabilizer leaves as the identity.
    """
    return {
        "n_i": 1,
        "n_j": 1,
        "n_k": 2,
        "ports": [{"i": 0, "j": 0, "k": 0, "d": "K", "e": e, "c": c}],
        "NodeY": _k([0, 0]),
        "ExistI": _k([0, 0]),
        "ExistJ": _k([0, 0]),
        "ExistK": _k([1, 0]),
        "ColorI": _k([-1, -1]),
        "ColorJ": _k([-1, -1]),
    }


@pytest.mark.parametrize("c", [0, 1])
def test_dangling_output_port_is_coloured_without_a_hadamard(c):
    """Rule (3). Output port (e='+') with colour c and a degree-1 inner cube.
    `color_ports` sets ColorKP[0] = c and nothing determines ColorKM[0].
    Rule (3) copies it: KM == KP == c, so the pipe carries no Hadamard.

    This matches what rule (1) already does for a dangling input port (next
    test). The two cases are mirror images and should be coloured alike.

    Old code: the last-resort loop forced ColorKM[0] = 0. For c = 1 that
    put a Hadamard (KM != KP) on a pipe that leads nowhere.
    """
    lasre = cz.color_z(_dangling_port_column("+", c))

    assert (lasre["ColorKM"][0][0][0], lasre["ColorKP"][0][0][0]) == (c, c)


@pytest.mark.parametrize("c", [0, 1])
def test_dangling_input_port_is_coloured_without_a_hadamard(c):
    """Rule (1). Input port (e='-') with colour c and a degree-1 inner cube.
    `color_ports` sets ColorKM[0] = c and rule (1) copies it to ColorKP[0].
    This passed on the old code too; it pins the behaviour that rule (3)
    mirrors.
    """
    lasre = cz.color_z(_dangling_port_column("-", c))

    assert (lasre["ColorKM"][0][0][0], lasre["ColorKP"][0][0][0]) == (c, c)


def _stuck_y_prep_columns(n_columns, c):
    """n_columns side-by-side copies of the Y-prep column, in the state that
    `color_kp_km` reaches before its first call to `assign_Kcolor`:

        cube 3: output port cube (port on pipe 2, e='+'); ColorKP[2] = c
        cube 2: ordinary
        cube 1: ordinary
        cube 0: Y (upper half prepares |Y>)

    Only ColorKP[2] is known. Propagation cannot move it down: pipe 1 would
    take its KP from ColorKM[2], which is unknown, and no propagation rule
    copies KP to KM within one pipe except at a Y cube. So rule (1) does not
    apply and rule (2) does, at the Y end of pipe 0 in every column.
    """
    n_k = 4
    exist_k = [[[1, 1, 1, 0]] for _ in range(n_columns)]
    node_y = [[[1, 0, 0, 0]] for _ in range(n_columns)]
    kp = [[[-1, -1, c, -1]] for _ in range(n_columns)]
    km = [[[-1, -1, -1, -1]] for _ in range(n_columns)]
    return n_k, exist_k, node_y, kp, km


def _assigned(before, after):
    return [(name, i, k)
            for name in ("KP", "KM")
            for i, (col_b, col_a) in enumerate(zip(before[name], after[name]))
            for k, (b, a) in enumerate(zip(col_b[0], col_a[0]))
            if b != a]


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("c", [0, 1])
def test_y_rule_makes_exactly_one_assignment(n_columns, c):
    """Rule (2) must make one free choice and hand control back to
    propagation, like the other rules.

    One call of `assign_Kcolor` in the state above must assign only
    ColorKM[0] = 0 in the first column (the free Y end) and return True.
    Propagation then carries that colour up through the ordinary cubes 1 and
    2, and pipe 2 (KM = 0, KP = c) holds the Hadamard when c = 1.

    Old code: rule (2) ended with `break`, which leaves only the innermost
    (k) loop. The scan went on into the next column (with two columns, it
    set ColorKM[0] = 0 there as well), and then fell through to the
    last-resort loop, which forced ColorKP[0] = 0 in the first column. That
    is 2 assignments with one column and 3 with two, made without
    propagating in between. Here the extra assignments happen to match what
    propagation would have produced. But with rule (3) in place of the
    last-resort loop, the fall-through would instead copy KP into KM on the
    port pipe 2 (ColorKM[2] = c), and the Hadamard would move from pipe 2
    to pipe 1.
    """
    n_k, exist_k, node_y, kp, km = _stuck_y_prep_columns(n_columns, c)
    before = {"KP": copy.deepcopy(kp), "KM": copy.deepcopy(km)}

    assert cz.assign_Kcolor(n_columns, 1, n_k, exist_k, kp, km, node_y)

    assert _assigned(before, {"KP": kp, "KM": km}) == [("KM", 0, 0)]
    assert km[0][0][0] == 0
