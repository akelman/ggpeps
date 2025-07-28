import numpy as np


class ZNGauge:
    """Implement a Z_N gauge group with elements on the unit circle.

    Elements are represented as 1x1 complex matrices corresponding to
    unit-modulus complex numbers spaced at intervals of 2π/N around the circle.
    """

    def __init__(self, n: int) -> None:
        """Initialize a Z_N gauge group.

        Args:
            n (int): Order of the group (i.e., the number of discrete elements).
        """
        self.n = n
        self.rep_dim = 1  # Each group element is represented as a 1×1 matrix.
        self.forbidden_transitions: list = []  # List of forbidden transitions - Empty for Z_N gauge group.

    def get_nonsingular_path(self, g_old: np.ndarray, g_new: np.ndarray) -> list[np.ndarray]:
        """Return an empty path since Z_N has no singular transitions.

        In Z_N, all gauge values lie on the unit circle and transitions
        between any two elements are always nonsingular.

        Args:
            g_old (np.ndarray): Current gauge value (1x1 complex matrix).
            g_new (np.ndarray): Target gauge value (1x1 complex matrix).

        Returns:
            list[np.ndarray]: Empty list; no intermediate steps are required.
        """
        return []

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> np.ndarray:
        """
        Generate a random Z_N group element as a 1x1 complex matrix.

        Each element g of Z_N is represented as exp(i * theta), where theta = (2 * pi * k) / N,
        and k is an integer in the range [0, N - 1].

        Args:
            rng_state (np.random.RandomState): A NumPy random number generator used to ensure
                reproducibility of the random selection. Using the same seed will produce
                the same sequence of random values.

        Returns:
            np.ndarray: 1x1 matrix representing a randomly selected Z_N group element.
        """
        k = rng_state.randint(0, self.n)
        theta = (2 * np.pi * k) / self.n
        return self.get_representation(theta)

    def get_representation(self, theta: float) -> np.ndarray:
        """
        Return the matrix representation of a Z_N group element given an angle theta.

        The element g = exp(i * theta) is returned as a 1x1 complex NumPy array for consistency
        with other (possibly non-Abelian) gauge groups that use higher-dimensional representations.

        Args:
            theta (float): Angle parameter for the group element, typically (2 * pi * k) / N.

        Returns:
            np.ndarray: 1x1 complex matrix representing the group element.
        """
        return np.array([[np.exp(1.0j * theta)]])

    def get_neutral_gauge_value(self) -> np.ndarray:
        """
        Return the identity element of the Z_N gauge group.

        The identity is represented as a 1x1 complex matrix: [[1 + 0j]].

        Returns:
            np.ndarray: 1x1 matrix representing the identity element.
        """
        return np.array([[1.0 + 0.0j]])

    def get_possible_gauge_values(self) -> np.ndarray:
        """
        Generate all Z_N group elements as 1x1 complex matrices.

        Each element is represented as [[exp(i * theta)]], where
        theta = (2 * pi * k) / N for k in [0, N - 1].

        Returns:
            np.ndarray: Array of shape (N, 1, 1), where each element is a 1x1 matrix
                        representing a Z_N group element.
        """
        prefactor = 2.0 * np.pi / self.n
        dest = []
        for k in range(self.n):
            dest.append(self.get_representation(k * prefactor))
        return np.array(dest)

    def get_increment(self) -> float:
        """
        Return the angular separation between adjacent Z_N group elements.

        Each element in Z_N is represented as exp(i * theta), where theta = (2 * pi * k) / N.
        The difference in angle between consecutive elements is 2 * pi / N.

        Note:
            In Z_N, multiplying an element g(k) by exp(2 * pi * i / N) yields the next element: g(k + 1).

        Returns:
            float: Angular increment between neighboring group elements.
        """
        return 2.0 * np.pi / self.n

    def get_angle(self, g) -> float:
        """
        Return the angle theta for a Z_N group element g = [[exp(i * theta)]].

        Extract the complex phase of a 1x1 matrix representing a Z_N group element.

        Args:
            g (np.ndarray): 1x1 matrix representing a Z_N group element.

        Returns:
            float: Angle theta in radians.
        """
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
        self.forbidden_transitions = [
            (self.get_representation(p0, 0), self.get_representation(p1, 1))
            for p0 in range(self.n)
            for p1 in range(self.n)
            if not (p0 == 0 and p1 == 0)
        ]
        self.transition_pair = (
            self.get_representation(0, 0),
            self.get_representation(0, 1),
        )  # (0,0) -> (0,1) is not singular, but we treat it as a special case in the update_gauge_ind method.
        # Contains all the forbidden transitions for updating the gamma matrix, i.e., the update matrix of this transitions is singualr. These are pairs that change under reflection.
        # Note that we don't include here the transition (0,0) -> (0,1) since we turn it into a non singular transition in the update_gauge_ind method.

    def get_nonsingular_path(self, g_old, g_new):
        """Get the non singular update gauge field path between two gauge values.
        If the transition between the two gauge fields yields a singular update we
        return a path containing middle steps, such that we don't run into singular update matrices.
        """
        p_0_q_0 = self.get_neutral_gauge_value()
        p_0_q_1 = self.get_representation(0, 1)
        q_old = self.get_reflection_index(g_old)
        q_new = self.get_reflection_index(g_new)
        if q_old == 0 and q_new == 1:
            # we need to go through the representation (0,1)
            dest = [p_0_q_1]
            if not np.allclose(g_old, p_0_q_0):  # we need to first go to (0,0)
                dest = [p_0_q_0] + dest
        elif q_old == 1 and q_new == 0:
            # we need to go through the representation (0,0)
            dest = [p_0_q_0]
            if not np.allclose(g_old, p_0_q_1):  # we need to first go to (0,1)
                dest = [p_0_q_1] + dest
        else:  # this is not a forbidden transition
            # we can go directly from g_old to g_new
            dest = []

        return dest
        dest = []
        p_0_q_0 = self.get_neutral_gauge_value()
        p_0_q_1 = self.get_representation(0, 1)
        p_1_q_0 = self.get_representation(1, 0)
        if np.allclose(g_old, p_0_q_0) or np.allclose(g_old, p_0_q_1):
            dest.append(p_1_q_0)
        else:
            dest.append(p_0_q_0)
        return dest

    def get_reflection_index(self, g):
        """Get the reflection index of a gauge value"""
        det = np.linalg.det(g)
        if np.isclose(det, 1.0):  # rotation - not reflection
            return 0
        elif np.isclose(det, -1):
            return 1
        else:
            raise ValueError("Gauge value not in D2n group")

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> float:
        p = rng_state.randint(0, self.n)
        q = rng_state.randint(0, 2)
        return self.get_representation(p, q)

    def get_representation(self, p, q):
        """Get a real 2D representaion of the group"""
        prefactor = 2.0 * np.pi / self.n
        prefactor_times_p = p * prefactor
        if q % 2 == 0:  # we work in a convention of q=0 mod 2
            representation = np.array(
                [
                    [np.cos(prefactor_times_p), -np.sin(prefactor_times_p)],
                    [np.sin(prefactor_times_p), np.cos(prefactor_times_p)],
                ],
            )
        else:  # if q=1 mod 2
            representation = np.array(
                [
                    [np.cos(prefactor_times_p), np.sin(prefactor_times_p)],
                    [np.sin(prefactor_times_p), -np.cos(prefactor_times_p)],
                ],
            )
        return representation

    def get_neutral_gauge_value(self) -> np.ndarray:
        return np.identity(2)

    def get_possible_gauge_values(self) -> np.ndarray:
        dest = np.array(
            [self.get_representation(p, q) for q in range(2) for p in range(self.n)],
        )
        return dest


if __name__ == "__main__":
    print("Z_2 Gauge Group Elements:")
    Z_2 = ZNGauge(2)
    for k in range(Z_2.n):
        theta = (2 * np.pi * k) / Z_2.n
        print(f"Z_2 element {k}: {Z_2.get_representation(theta)}")

    print("Z_4 Gauge Group Elements:")
    Z_4 = ZNGauge(4)
    for k in range(Z_4.n):
        theta = (2 * np.pi * k) / Z_4.n
        print(f"Z_4 element {k}: {Z_4.get_representation(theta)}")
