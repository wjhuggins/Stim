"""Tests for LatticeSurgerySAT's handling of given values and its decoder.

Three entry points read or write the per-cube arrays (NodeY, ExistI/J/K,
ColorI/J) and the per-stabilizer arrays (CorrIJ, ..., CorrKJ):

  plugin_arrs(data)    constrains variables to values given as whole arrays
                       (-1 means "not given"). Used for `given_arrs` and to
                       load a kissat assignment back into z3.
  plugin_vals(list)    constrains single variables, each given as
                       {"array", "indices", "value"}. Used for `given_vals`.
  get_result()         reads the z3 model back out as a LaSRe dict.

plugin_arrs and get_result loop `for s in range(n_s)` and handle the
per-cube arrays inside that loop on the `s == 0` pass. With no stabilizers
(n_s == 0) the loop body never runs, so the per-cube arrays were silently
dropped: plugin_arrs ignored them and get_result returned all -1. LaSsynth
accepts n_s == 0 (check_lasre only prints "no stabilizer!"), so the decoder
must work for it. The fix keeps one pass when n_s == 0 and only skips the
Corr arrays, which keeps the order of the constraints, and so the solver's
input, unchanged when n_s > 0.

plugin_vals checked `len(data["indices"] != 3)`, i.e. the length of a bool,
which raises TypeError for every input, so `given_vals` could not be used
at all. The intended check is `len(data["indices"]) != 3` (4 for Corr).
"""

import pytest

from lassynth.sat_synthesis.lattice_surgery_sat import LatticeSurgerySAT


def _identity_spec(stabilizers):
    """1x1x3 box with an input port at the bottom and an output port at the
    top of the same column. With stabilizers ["XX", "ZZ"] the solution is a
    straight pipe (the identity). The port pipes are K(0,0,0) for the input
    port (port cube 0, e='-') and K(0,0,1) for the output port (port cube 2,
    e='+')."""
    return {
        "max_i": 1,
        "max_j": 1,
        "max_k": 3,
        "ports": [
            {"location": [0, 0, 0], "direction": "+K",
             "z_basis_direction": "J"},
            {"location": [0, 0, 2], "direction": "-K",
             "z_basis_direction": "J"},
        ],
        "stabilizers": stabilizers,
    }


_PORT_PIPES = [(0, 0, 0), (0, 0, 1)]
_CELL_ARRAYS = ["NodeY", "ExistI", "ExistJ", "ExistK"]


def _all_values(arr):
    return {v for plane in arr for row in plane for v in row}


@pytest.mark.parametrize("stabilizers", [[], ["XX", "ZZ"]],
                         ids=["n_s=0", "n_s=2"])
def test_get_result_reads_cell_arrays(stabilizers):
    """get_result must return the model's per-cube arrays for any n_s.

    Port pipes always exist (constraint_port), so ExistK is 1 on both port
    pipes, and every per-cube entry is 0 or 1, never -1.

    Old code, n_s == 0: every per-cube entry came back -1, although z3's
    model has values for them.
    """
    sat = LatticeSurgerySAT(input_dict=_identity_spec(stabilizers))
    assert sat.check_z3(print_progress=False)

    result = sat.get_result()

    assert [result["ExistK"][i][j][k] for (i, j, k) in _PORT_PIPES] == [1, 1]
    for arr in _CELL_ARRAYS:
        assert _all_values(result[arr]) <= {0, 1}, arr


def _given_exist_k(value_at_input_port_pipe):
    arr = [[[-1, -1, -1]]]
    arr[0][0][0] = value_at_input_port_pipe
    return arr


@pytest.mark.parametrize("stabilizers", [[], ["XX", "ZZ"]],
                         ids=["n_s=0", "n_s=2"])
def test_plugin_arrs_constrains_cell_arrays(stabilizers):
    """plugin_arrs must add the given per-cube values for any n_s.

    Giving ExistK = 0 on a port pipe contradicts constraint_port, so the
    model must become UNSAT. That shows the value was actually plugged in.

    Old code, n_s == 0: the given array was dropped and the model stayed
    SAT.
    """
    sat = LatticeSurgerySAT(input_dict=_identity_spec(stabilizers))
    sat.plugin_arrs({"ExistK": _given_exist_k(0)})

    assert not sat.check_z3(print_progress=False)


@pytest.mark.parametrize("make_indices", [list, tuple],
                         ids=["list", "tuple"])
def test_given_vals_constrain_cell_variable(make_indices):
    """given_vals must be accepted and applied.

    ExistK = 0 on a port pipe contradicts constraint_port, so the model is
    UNSAT. ExistK = 1 there agrees with it, so the model stays SAT.

    Old code: TypeError("object of type 'bool' has no len()") for any
    given_vals entry.
    """
    spec = _identity_spec(["XX", "ZZ"])
    indices = make_indices(_PORT_PIPES[0])

    contradicting = LatticeSurgerySAT(
        input_dict=spec,
        given_vals=[{"array": "ExistK", "indices": indices, "value": 0}])
    agreeing = LatticeSurgerySAT(
        input_dict=spec,
        given_vals=[{"array": "ExistK", "indices": indices, "value": 1}])

    assert not contradicting.check_z3(print_progress=False)
    assert agreeing.check_z3(print_progress=False)


def test_given_vals_accept_corr_indices():
    """Corr arrays take four indices (s, i, j, k). Old code: TypeError."""
    LatticeSurgerySAT(
        input_dict=_identity_spec(["XX", "ZZ"]),
        given_vals=[{"array": "CorrKI", "indices": [0, 0, 0, 1],
                     "value": 0}])


@pytest.mark.parametrize("array,indices", [("ExistK", [0, 0]),
                                           ("CorrKI", [0, 0, 0])])
def test_given_vals_reject_wrong_index_count(array, indices):
    """A wrong number of indices is a ValueError naming the array.

    Old code: TypeError for every input, so this message was unreachable.
    """
    with pytest.raises(ValueError, match=f"indices for {array}"):
        LatticeSurgerySAT(
            input_dict=_identity_spec(["XX", "ZZ"]),
            given_vals=[{"array": array, "indices": indices, "value": 0}])
