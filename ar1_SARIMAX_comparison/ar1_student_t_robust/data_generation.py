import numpy as np

from .config import N, mu, phi_true, q, r, x0, rng


def draw_positive_trajectory(rng_state):
    """Generate from the Gaussian AR(1) trajectory conditioned on x_t >= 0."""
    attempts = 0
    while True:
        attempts += 1
        x = np.empty(N)
        prev = x0
        for t in range(N):
            prev = mu + phi_true * (prev - mu) + q * rng_state.normal()
            x[t] = prev
        if np.all(x >= 0.0):
            return x, attempts
