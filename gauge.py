import numpy as np


class ZNGauge:
    def __init__(self, n):
        self.n = n

    def get_random_gauge_value(self):
        return np.random.rand() * 2 * np.pi / self.n

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
