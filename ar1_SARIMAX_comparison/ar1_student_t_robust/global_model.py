import numpy as np

from .config import N, mu, q, r, x0


def global_system(phi, z):
    B = np.eye(N)
    B[np.arange(1, N), np.arange(N - 1)] = -phi
    b = np.full(N, (1.0 - phi) * mu)
    b[0] += phi * x0
    J = (B.T @ B) / q**2 + np.eye(N) / r**2
    y = (B.T @ b) / q**2 + z / r**2
    c = (b @ b) / q**2 + (z @ z) / r**2
    return B, b, J, y, c


def analytic_nll(phi, z):
    B, b, J, y, c = global_system(phi, z)
    xhat = np.linalg.solve(J, y)
    sign, logdetJ = np.linalg.slogdet(J)
    if sign <= 0:
        return np.inf
    return (
        0.5 * (c - y @ xhat)
        + 0.5 * logdetJ
        + N * np.log(q)
        + N * np.log(r)
        + 0.5 * N * np.log(2.0 * np.pi)
    )


def global_smooth_state(phi, z):
    _, _, J, y, _ = global_system(phi, z)
    return np.linalg.solve(J, y)
