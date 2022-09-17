import numpy as np
from math import sqrt, floor
from scipy.stats import norm


# https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html

def brownian(x0, n, dt, delta):
    """
    X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)

    :param x0: numpy array Wiener process input
    :param n: Number of steps to take
    :param dt: Time step
    :param delta: "Speed" of Wiener process progression
    :return: numpy array of n steps of Wiener process applied to x0
    """
    x0 = np.asarray(x0)

    # For each x0, generate n samples from normal distribution
    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    out = np.empty(r.shape)
    np.cumsum(r, axis=1, out=out)

    return out


class PoissonDistribution:
    def __init__(self, high, num_discrete, num_steps, low=0.):
        self.inputs = np.linspace(low, high, num=num_discrete)
        og_distribution = np.array([1 / num_discrete for _ in range(num_discrete)])
        self.num_discrete = num_discrete

        if num_steps > 0:
            t = 60  # seconds
            dt = t / num_steps
            delta = 0.5 / (num_discrete * num_steps)

            self.brownian_motion = brownian(og_distribution, num_steps, dt, delta)
            self.distribution = np.abs(self.brownian_motion[:, -1])
            self.distribution /= np.linalg.norm(self.distribution)
        else:
            self.distribution = og_distribution

        self.num_updates = 0

    def mean(self):
        """
        Calculates the mean of the Bayesian-estimated distribution

        :return: mean (float)
        """
        return np.sum(self.inputs * self.distribution)

    def update_distribution(self, time, n_units):
        """
        Does a Bayesian update of the distribution

        :param time: (fixed) elapsed time in seconds
        :param n_units: number of "units" processed in elapsed time. Unit is defined by constants.py e.g. bit, MByte
        """
        n_units = floor(n_units)
        if n_units == 0:
            return
        rate_time = time * self.inputs
        intermediates = self.distribution * np.power(rate_time, n_units) \
            / np.math.factorial(n_units)
        intermediates *= np.exp(-rate_time)
        self.distribution = intermediates / np.sum(intermediates)
