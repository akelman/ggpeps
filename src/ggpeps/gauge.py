import numpy as np


class ZNGauge:
    """Implements a Z_N gauge group, under a convenient representation in which values are chosen on the unit circle
    and given in multiples of 2pi/N.
    """

    def __init__(self, n: int):
        self.n = n
        self.rep_dim = 1  # dimension of the representation

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> float:
        theta = rng_state.randint(0, self.n) * 2 * np.pi / self.n
        return self.get_representation(theta)

    def get_representation(self, theta: float) -> np.ndarray:
        """Get a representation of the group element"""
        return np.array([[np.exp(1.0j * theta)]], dtype=np.complex64)

    def get_neutral_gauge_value(self) -> float:
        return np.array([[1.0]], dtype=np.complex64)

    def get_possible_gauge_values(self) -> np.ndarray:
        prefactor = 2.0 * np.pi / self.n
        dest = []
        for i in range(self.n):
            dest.append(self.get_representation(i * prefactor))
        return np.array(dest, dtype=np.complex64)

    def get_increment(self) -> float:
        return 2.0 * np.pi / self.n

    def get_angle(self, g) -> float:
        """Get the angle, theta, for a group elemnt g=[[exp(1j*theta)]]"""
        return np.angle(g[0][0])


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
        if self.n == 6:
            self.forbidden_transitions = [  # we define it with set since order doesn't matter
                set(self.get_representation(0, 0), self.get_representation(0, 1)),
                set(self.get_representation(1, 0), self.get_representation(1, 1)),
                set(self.get_representation(1, 0), self.get_representation(2, 1)),
                set(self.get_representation(2, 0), self.get_representation(1, 1)),
                set(self.get_representation(2, 0), self.get_representation(2, 1)),
            ]  # Contains all the forbidden transitions for updating the gamma matrix, i.e., the update matrix of this transitions is singualr.
            # TODO: not sure if this should be here or in the system config, since it is not clear yet whether this list depends on number of copies or how we define the projectors.

    def get_nonsingular_path(self, g_old, g_new):
        """Get the non singular update gauge field path between two gauge values.
        If the transition between the two gauge fields yields a singular update we
        return a path containing middle steps, such that we don't run into singular update matrices.
        """
        # TODO: not sure if this should be here or in the system config, since it is not clear yet whether this list depends on number of copies or how we define the projectors.
        dest = []
        p_0_q_0 = self.get_neutral_gauge_value()
        p_0_q_1 = self.get_representation(0, 1)
        p_1_q_0 = self.get_representation(1, 0)
        if set(g_old, g_new) in self.forbidden:
            if np.allclose(g_old, p_0_q_0) or np.allclose(g_old, p_0_q_1):
                dest.append(p_1_q_0)
            else:
                dest.append(p_0_q_0)
        return dest

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> float:
        p = rng_state.randint(0, self.n)
        q = rng_state.randint(0, 2)
        return self.get_representation(p, q)

    def get_representation(self, p, q):
        """Get a real 2D representaion of the group"""
        prefactor = 2.0 * np.pi / self.n
        prefactor_times_p = p * prefactor
        if q % 2 == 0:  # we work in a convention of q=0 mod2
            representation = np.array(
                [
                    [np.cos(prefactor_times_p), -np.sin(prefactor_times_p)],
                    [np.sin(prefactor_times_p), np.cos(prefactor_times_p)],
                ],
                dtype=np.complex64,
            )
        else:  # if q=1 mod 2
            representation = np.array(
                [
                    [np.cos(prefactor_times_p), np.sin(prefactor_times_p)],
                    [np.sin(prefactor_times_p), -np.cos(prefactor_times_p)],
                ],
                dtype=np.complex64,
            )
        return representation

    def get_neutral_gauge_value(self) -> np.array:
        return np.identity(2, dtype=np.complex64)

    def get_possible_gauge_values(self) -> np.ndarray:
        dest = np.array(
            [self.get_representation(p, q) for q in range(2) for p in range(self.n)],
            dtype=np.complex64,
        )
        return dest
