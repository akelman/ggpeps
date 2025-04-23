import numpy as np


class ZNGauge:
    """Implements a Z_N gauge group, under a convenient representation in which values are chosen on the unit circle
    and given in multiples of 2pi/N.
    """

    def __init__(self, n: int):
        self.n = n
        self.rep_dim = 1  # dimension of the representation

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> float:
        return rng_state.randint(0, self.n) * 2 * np.pi / self.n

    def get_neutral_gauge_value(self) -> float:
        return 0.0

    def get_possible_gauge_values(self) -> np.ndarray:
        prefactor = 2.0 * np.pi / self.n
        dest = np.zeros(self.n)
        for i in range(self.n):
            dest[i] = i * prefactor
        return dest

    def get_increment(self) -> float:
        return 2.0 * np.pi / self.n


class D2nGauge:
    """Implements a D_2n gauge group, under a real 2D representation of rotation and reflection matrices.
    The representaion for a group element of the form:
        D(p,q=0)=        [
                    [np.cos(2 * np.pi * p / self.n), -np.sin(2 * np.pi * p / self.n)],
                    [np.sin(2 * np.pi * p / self.n), np.cos(2 * np.pi * p / self.n)],
                ]
        D(p,q=1)=                 [
                    [np.cos(2 * np.pi * p / self.n), np.sin(2 * np.pi * p / self.n)],
                    [np.sin(2 * np.pi * p / self.n), -np.cos(2 * np.pi * p / self.n)],
                ]


    """

    def __init__(self, n: int):
        self.n = n
        self.rep_dim = 2

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> float:
        p = rng_state.randint(0, self.n)
        q = rng_state.randint(0, 2)
        return self.get_representaion(p, q)

    def get_representaion(self, p, q):
        """Get a real 2D representaion of the group"""
        prefactor = 2.0 * np.pi / self.n
        prefactor_times_p = p * prefactor
        if q % 2:  # we work in a convention of q=0 mod2
            representaion = np.array(
                [
                    [np.cos(prefactor_times_p), -np.sin(prefactor_times_p)],
                    [np.sin(prefactor_times_p), np.cos(prefactor_times_p)],
                ]
            )
        else:  # if q=1 mod 2
            representaion = np.array(
                [
                    [np.cos(prefactor_times_p), np.sin(prefactor_times_p)],
                    [np.sin(prefactor_times_p), -np.cos(prefactor_times_p)],
                ]
            )
        return representaion

    def get_neutral_gauge_value(self) -> np.array:
        return np.identity(2)

    def get_possible_gauge_values(self) -> np.ndarray:
        dest = np.array(
            [self.get_representaion(p, q) for q in range(2) for p in range(self.n)]
        )
        return dest
