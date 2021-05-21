"""
"""
import numpy as np
from ..spin_evolution import _get_norm, _get_u_norm, lgspin_vs_t
from ..spin_evolution import SPIN_PARAM_BOUNDS, DEFAULT_SPIN_PARAMS
from ..spin_evolution import get_bounded_params, get_unbounded_params


def test_bounded_norm():
    u_norm_arr = np.linspace(-5, 5, 500)
    norm_arr = _get_norm(u_norm_arr)
    assert np.all(norm_arr >= SPIN_PARAM_BOUNDS["spin_norm"][0])
    u_norm_arr2 = _get_u_norm(norm_arr)
    assert np.allclose(u_norm_arr, u_norm_arr2, atol=0.1)


def test_bounded_params():
    p = np.array(list(DEFAULT_SPIN_PARAMS.values()))
    u_p = get_unbounded_params(p)
    p2 = get_bounded_params(u_p)
    assert np.allclose(p, p2, atol=0.01)


def test_unbounded_params():
    n_test = 100
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 4)
        p = get_bounded_params(up)
        up2 = get_unbounded_params(p)
        assert np.allclose(up, up2, atol=0.01)


def test_lgspin_properly_bounded():
    n_test = 100
    tarr = np.linspace(0, 13.8, 100)
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 4)
        p = get_bounded_params(up)
        lgspin = lgspin_vs_t(tarr, *p)
        assert np.all(lgspin >= SPIN_PARAM_BOUNDS["spin_norm"][0])
