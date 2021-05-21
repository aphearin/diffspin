"""Helper functions for fitting NFW concentration histories of individual halos."""
import numpy as np
from scipy.optimize import curve_fit
from jax import jit as jjit
from jax import vmap as jvmap
from jax import numpy as jnp
from jax.experimental import optimizers as jax_opt
from jax import value_and_grad, grad
from .spin_evolution import u_lgspin_vs_t
from .spin_evolution import DEFAULT_SPIN_PARAMS
from .spin_evolution import get_bounded_params, get_unbounded_params
import warnings

T_FIT_MIN = 2.0


_a = (0, None, None, None, None)
_jac_func = jjit(jvmap(grad(u_lgspin_vs_t, argnums=(1, 2, 3, 4)), in_axes=_a))


def fit_lgspin(t_sim, spin_sim, log_mah_sim, lgm_min, n_step=300):
    u_p0, loss_data = get_loss_data(t_sim, spin_sim, log_mah_sim, lgm_min)
    t, lgsp, msk = loss_data

    if len(lgsp) < 10:
        method = -1
        p_best = np.nan
        loss = np.nan
        return p_best, loss, method, loss_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            u_p = curve_fit(u_lgspin_vs_t, t, lgsp, p0=u_p0, jac=jac_lgspin)[0]
            method = 0
            p_best = get_bounded_params(u_p)
            loss = log_spin_mse_loss(u_p, loss_data)
        except RuntimeError:
            res = jax_adam_wrapper(log_spin_mse_loss_and_grads, u_p0, loss_data, n_step)
            u_p = res[0]
            if ~np.all(np.isfinite(u_p)):
                method = -1
                p_best = np.nan
                loss = np.nan
            else:
                method = 1
                p_best = get_bounded_params(u_p)
                loss = log_spin_mse_loss(u_p, loss_data)
    return p_best, loss, method, loss_data


def jac_lgspin(t, u_lgs, u_a, u_tau, u_phase):
    return np.array(_jac_func(t, u_lgs, u_a, u_tau, u_phase)).T


@jjit
def log_spin_mse_loss(u_params, loss_data):
    """MSE loss function for fitting individual halo growth."""
    t_target, log_conc_target, msk = loss_data
    log_conc_pred = u_lgspin_vs_t(t_target, *u_params)
    log_conc_loss = _mse(log_conc_pred, log_conc_target)
    return log_conc_loss


@jjit
def log_spin_mse_loss_and_grads(u_params, loss_data):
    """MSE loss and grad function for fitting individual halo growth."""
    return value_and_grad(log_spin_mse_loss, argnums=0)(u_params, loss_data)


def get_target_data(t_sim, spin_sim, log_mah_sim, lgm_min, t_fit_min=T_FIT_MIN):
    """"""
    msk = log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min
    msk &= spin_sim > 0

    t_target = t_sim[msk]
    log_spin_target = np.log10(spin_sim[msk])
    return t_target, log_spin_target, msk


def get_loss_data(t_sim, spin_sim, log_mah_sim, lgm_min, t_fit_min=T_FIT_MIN):
    t_target, log_spin_target, msk = get_target_data(
        t_sim,
        spin_sim,
        log_mah_sim,
        lgm_min,
        t_fit_min,
    )
    loss_data = (t_target, log_spin_target, msk)

    p_init = np.array(list(DEFAULT_SPIN_PARAMS.values())).astype("f4")
    u_p_init = get_unbounded_params(p_init)

    return u_p_init, loss_data


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


def jax_adam_wrapper(
    loss_and_grad_func,
    params_init,
    loss_data,
    n_step,
    step_size=0.2,
    tol=-float("inf"),
):
    loss_arr = np.zeros(n_step).astype("f4") - 1.0
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)

    best_loss = float("inf")
    for istep in range(n_step):
        p = jnp.array(get_params(opt_state))
        loss, grads = loss_and_grad_func(p, loss_data)

        nanmsk = ~np.isfinite(loss)
        nanmsk &= ~np.all(np.isfinite(grads))
        if nanmsk:
            best_fit_params = np.nan
            best_loss = np.nan
            break

        loss_arr[istep] = loss
        if loss < best_loss:
            best_fit_params = p
            best_loss = loss
        if loss < tol:
            loss_arr[istep:] = best_loss
            break
        opt_state = opt_update(istep, grads, opt_state)

    return best_fit_params, best_loss, loss_arr


def get_outline(halo_id, p_best, loss, method):
    """Return the string storing fitting results that will be written to disk"""
    _d = np.array(p_best).astype("f4")
    data_out = (halo_id, method, *_d, float(loss))
    outprefix = str(halo_id) + " " + str(method) + " "
    outdata = " ".join(["{:.5e}".format(x) for x in data_out[2:]])
    return outprefix + outdata + "\n"


def get_outline_bad_fit(halo_id, p_best, loss, method):
    norm, amp, tau, phase = -1.0, -1.0, -1.0, -1.0
    _d = np.array((norm, amp, tau, phase)).astype("f4")
    loss_best = -1.0
    method = -1
    data_out = (halo_id, method, *_d, float(loss_best))
    outprefix = str(halo_id) + " " + str(method) + " "
    outdata = " ".join(["{:.5e}".format(x) for x in data_out[2:]])
    return outprefix + outdata + "\n"


def get_header():
    return "# halo_id method spin_norm spin_amp spin_tau spin_phase spin_loss\n"
