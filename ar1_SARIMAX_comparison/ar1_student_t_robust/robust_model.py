import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import N, mu, q, r, x0, nu, ADD_OUTLIERS
from .global_model import global_system


def ar1_B_b(phi):
    B = np.eye(N)
    B[np.arange(1, N), np.arange(N - 1)] = -phi
    b = np.full(N, (1.0 - phi) * mu)
    b[0] += phi * x0
    return B, b


def student_measurement_nll(residual, nu_value=nu):
    return (
        gammaln(nu_value / 2)
        - gammaln((nu_value + 1) / 2)
        + 0.5 * np.log(nu_value * np.pi)
        + np.log(r)
        + 0.5 * (nu_value + 1) * np.log1p(residual**2 / (nu_value * r**2))
    )


def robust_mode(phi, data, nu_value=nu):
    B, b = ar1_B_b(phi)
    J_process = B.T @ B / q**2
    y_process = B.T @ b / q**2
    J_gauss = J_process + np.eye(N) / r**2
    x_init = np.linalg.solve(J_gauss, y_process + data / r**2)

    def value_grad(x):
        e = data - x
        process = B @ x - b
        value = 0.5 * (process @ process) / q**2
        value += np.sum(student_measurement_nll(e, nu_value))
        grad = J_process @ x - y_process
        grad -= (nu_value + 1) * e / (nu_value * r**2 + e**2)
        return value, grad

    result = minimize(
        value_grad,
        x_init,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        print("Robust mode warning:", result.message)
    return result.x, result.fun


def robust_laplace_nll(phi, data, nu_value=nu):
    x_map, F_map = robust_mode(phi, data, nu_value)
    B, _ = ar1_B_b(phi)
    e = data - x_map
    a = nu_value * r**2
    rho_second = (nu_value + 1) * (a - e**2) / (a + e**2)**2
    H = (B.T @ B) / q**2 + np.diag(rho_second)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        return np.inf
    return F_map + N * np.log(q) + 0.5 * logdet


def gaussian_nll_for_data(phi, data):
    B, b = ar1_B_b(phi)
    J = B.T @ B / q**2 + np.eye(N) / r**2
    y = B.T @ b / q**2 + data / r**2
    c = (b @ b) / q**2 + (data @ data) / r**2
    xhat = np.linalg.solve(J, y)
    sign, logdet = np.linalg.slogdet(J)
    return (
        0.5 * (c - y @ xhat)
        + 0.5 * logdet
        + N * np.log(q)
        + N * np.log(r)
        + 0.5 * N * np.log(2 * np.pi)
    ) if sign > 0 else np.inf
