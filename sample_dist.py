import numpy as np
import scipy.interpolate

import numpy as np
from scipy import interpolate


class SampleDistribution:

    def __init__(self, x, y, bounds_error=False, fill_value=0):
        """ """
        self.zero_flag = False

        # compute cumulative distribution
        # with range in 0 and 1
        c = np.cumsum(y)
        c = np.concatenate([[0], c[:-1]])
        if c[-1] > 0:
            c /= c[-1]
        else:
            self.zero_flag = True

        # store interpolation function
        if not self.zero_flag:
            self.func = interpolate.interp1d(x, y,
                                            kind='previous',
                                            bounds_error=bounds_error,
                                            fill_value=fill_value)

            # construct interpolation function of inverse
            self.cumu_inv_func = interpolate.interp1d(c, x)

    def __call__(self, x):
        """Evaluate the function"""
        if self.zero_flag:
            return x * 0.
        return self.func(x)

    def sample(self, n):
        """Draw samples from the distribution"""
        if self.zero_flag:
            raise ValueError("Cannot draw samples from zero function")
        u = np.random.uniform(0, 1, n)
        x = self.cumu_inv_func(u)
        return x
