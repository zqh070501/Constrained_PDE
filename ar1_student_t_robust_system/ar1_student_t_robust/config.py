import numpy as np

N = 80
mu = 0.30
phi_true = 0.85
q = 0.10
r = 0.15
x0 = mu

MC = 6000
rng = np.random.default_rng(7)
rng_mc = np.random.default_rng(12345)

# Common random numbers for Monte Carlo approximations
# These are initialized once and reused to match the notebook behavior.
eps_prior = rng_mc.standard_normal((MC, N))
eps_post = rng_mc.standard_normal((MC, N))

# Student-t robust extension settings
nu = 5.0
ADD_OUTLIERS = True

bounds = (0.05, 0.98)
opts = {"xatol": 1e-8}
