import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from .config import MC, N, mu, q, r, x0, eps_prior, eps_post
from .global_model import analytic_nll, global_system


def prior_samples(phi, eps_values):
    """Samples from the unconstrained AR(1) prior with fixed x0."""
    X = np.empty((MC, N))
    prev = np.full(MC, x0)
    for t in range(N):
        prev = mu + phi * (prev - mu) + q * eps_values[:, t]
        X[:, t] = prev
    return X


def posterior_samples(phi, z_vals, eps_values):
    """x|z,phi ~ N(J^{-1}y, J^{-1})."""
    _, _, J, y, _ = global_system(phi, z_vals)
    mean = np.linalg.solve(J, y)
    L = np.linalg.cholesky(J)
    delta = solve_triangular(L.T, eps_values.T, lower=False).T
    return mean + delta


def smoothed_orthant_probability(phi, z_vals):
    X = posterior_samples(phi, z_vals, eps_post)
    count = np.count_nonzero(np.all(X >= 0.0, axis=1))
    return (count + 0.5) / (MC + 1.0)


def prior_orthant_probability(phi):
    X = prior_samples(phi, eps_prior)
    count = np.count_nonzero(np.all(X >= 0.0, axis=1))
    return (count + 0.5) / (MC + 1.0)


def constrained_nll(phi, z_vals):
    p_post = smoothed_orthant_probability(phi, z_vals)
    p_prior = prior_orthant_probability(phi)
    return analytic_nll(phi, z_vals) - np.log(p_post) + np.log(p_prior)


def constrained_smooth_state(phi, z_vals):
    _, _, J, y, _ = global_system(phi, z_vals)
    x_unc = np.linalg.solve(J, y)

    def obj(x):
        return 0.5 * x @ J @ x - y @ x

    def grad(x):
        return J @ x - y

    res = minimize(
        obj,
        np.maximum(x_unc, 0.0),
        jac=grad,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * N,
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 2000},
    )
    if not res.success:
        print("warning:", res.message)
    return res.x
