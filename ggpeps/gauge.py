import numpy as np


class ZNGauge:
    """Implements a Z_N gauge group, under a convenient representation in which values are chosen on the unit circle
    and given in multiples of 2pi/N. 
    """

    def __init__(self, n):
        self.n = n

    def get_random_gauge_value(self,rng_state):
        return rng_state.randint(0,self.n) * 2 * np.pi / self.n

    def get_neutral_gauge_value(self):
        return 0

    def get_possible_gauge_values(self):
        prefactor = 2.*np.pi / self.n
        dest = np.zeros(self.n)
        for i in range(self.n):
            dest[i] = i*prefactor
        return dest

    def get_increment(self):
        return 2.*np.pi / self.n
