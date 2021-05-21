"""Model for the evolution of spin of individual dark matter halos.
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp


DEFAULT_SPIN_PARAMS = OrderedDict(
    spin_norm=-1.5, spin_amp=0.1, spin_tau=2.0, spin_phase=jnp.pi
)
SPIN_PARAM_BOUNDS = OrderedDict(
    spin_norm=(-2.25, 0.0),
    spin_tau=(1.0, 25.0),
    spin_phase=(0.0, 2 * jnp.pi),
)
_X0, _K = 0.0, 0.1


@jjit
def lgspin_vs_t(t, norm, amp, tau, phase):
    return norm + amp * jnp.cos(t / tau + phase)


@jjit
def u_lgspin_vs_t(t, u_norm, u_amp, u_tau, u_phase):
    u_params = u_norm, u_amp, u_tau, u_phase
    params = get_bounded_params(u_params)
    return lgspin_vs_t(t, *params)


@jjit
def get_bounded_params(u_params):
    u_norm, u_amp, u_tau, u_phase = u_params
    norm = _get_norm(u_norm)
    amp = _get_amp(u_amp, norm)
    tau = _get_tau(u_tau)
    phase = _get_phase(u_phase)
    return jnp.array((norm, amp, tau, phase))


@jjit
def get_unbounded_params(p):
    norm, amp, tau, phase = p
    u_norm = _get_u_norm(norm)
    u_amp = _get_u_amp(amp, norm)
    u_tau = _get_u_tau(tau)
    u_phase = _get_u_phase(phase)
    return jnp.array((u_norm, u_amp, u_tau, u_phase))


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _softplus(x):
    return jnp.log(1 + jnp.exp(x))


@jjit
def _inverse_softplus(s):
    return jnp.log(jnp.exp(s) - 1.0)


@jjit
def _get_norm(u_norm):
    return _sigmoid(u_norm, _X0, _K, *SPIN_PARAM_BOUNDS["spin_norm"])


@jjit
def _get_u_norm(norm):
    return _inverse_sigmoid(norm, _X0, _K, *SPIN_PARAM_BOUNDS["spin_norm"])


@jjit
def _get_amp(u_amp, norm):
    amp_max = norm - SPIN_PARAM_BOUNDS["spin_norm"][0]
    return _sigmoid(u_amp, _X0, _K, 0.0, amp_max)


@jjit
def _get_u_amp(amp, norm):
    amp_max = norm - SPIN_PARAM_BOUNDS["spin_norm"][0]
    return _inverse_sigmoid(amp, _X0, _K, 0.0, amp_max)


@jjit
def _get_tau(u_tau):
    return _softplus(u_tau) + SPIN_PARAM_BOUNDS["spin_tau"][0]


@jjit
def _get_u_tau(tau):
    return _inverse_softplus(tau - SPIN_PARAM_BOUNDS["spin_tau"][0])


@jjit
def _get_phase(u_phase):
    return _sigmoid(u_phase, _X0, _K, 0.0, 2 * jnp.pi)


@jjit
def _get_u_phase(phase):
    return _inverse_sigmoid(phase, _X0, _K, 0.0, 2 * jnp.pi)
