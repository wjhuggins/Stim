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
