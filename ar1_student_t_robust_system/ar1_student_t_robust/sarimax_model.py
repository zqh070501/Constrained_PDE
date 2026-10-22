import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import N, mu, q, r, x0


def sarimax_result(phi, z_vals):
    # Centering removes the nonzero long-run mean from the SARIMAX state.
    y_centered = z_vals - mu
    model = SARIMAX(
        y_centered,
        order=(1, 0, 0),
        trend="n",
        measurement_error=True,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    a1 = phi * (x0 - mu)
    P1 = q**2
    model.ssm.initialize_known(np.array([a1]), np.array([[P1]]))
    model.loglikelihood_burn = 0
    model.ssm.loglikelihood_burn = 0

    params = np.array([phi, r**2, q**2])
    return model.smooth(params, transformed=True)


def sarimax_filter_nll(phi, z_vals):
    return -float(sarimax_result(phi, z_vals).llf)


def sarimax_smoothed_state(phi, z_vals):
    res = sarimax_result(phi, z_vals)
    return np.asarray(res.smoothed_state[0]) + mu


def smooth_state_nll(phi, z_vals):
    xs = sarimax_smoothed_state(phi, z_vals)
    B, b, J, y, c = global_system(phi, z_vals)

    process_resid = B @ xs - b
    measurement_resid = z_vals - xs
    Fhat = (
        0.5 * (process_resid @ process_resid) / q**2
        + 0.5 * (measurement_resid @ measurement_resid) / r**2
    )

    sign, logdetJ = np.linalg.slogdet(J)
    if sign <= 0:
        return np.inf

    return (
        Fhat + 0.5 * logdetJ
        + N * np.log(q) + N * np.log(r)
        + 0.5 * N * np.log(2.0 * np.pi)
    )


def global_system(phi, z_vals):
    from .global_model import global_system
    return global_system(phi, z_vals)
