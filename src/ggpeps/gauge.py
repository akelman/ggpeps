import logging
from abc import ABC, abstractmethod

import numpy as np

import ggpeps

logger = logging.getLogger(ggpeps.LOGGER_NAME)


class GaugeGroup(ABC):
    """Abstract base class for gauge groups.

    Defines the interface that all gauge group implementations must provide.
    This class cannot be instantiated directly.

    Subclasses must implement all abstract methods and set the following attributes
    in their ``__init__``:

    Attributes:
        n (int): Primary order parameter of the group (e.g. N for Z_N, half-order for D_2n).
        rep_dim (int): Dimension of the fundamental matrix representation.
        group_order (int): Total number of group elements.
        el_mult_factor (float): Uniform prefactor for the electric energy operator.
        el_offset (int): Offset used to keep the electric energy non-negative.
        group_elements_for_el_energy (tuple[np.ndarray, ...]): Group elements that contribute
            non-zero weight to the electric energy.
    """

    n: int
    rep_dim: int
    group_order: int
    el_mult_factor: float
    el_offset: int
    group_elements_for_el_energy: tuple[np.ndarray, ...]

    @abstractmethod
    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> np.ndarray:
        """Return a randomly selected group element as a matrix.

        Args:
            rng_state (np.random.RandomState): Random number generator.

        Returns:
            np.ndarray: Matrix representing a randomly selected group element.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def get_irrep(self, irrep_label: int, group_element: np.ndarray) -> np.ndarray:
        """Return the irreducible representation matrix for a given irrep label and group element.

        Args:
            irrep_label (int): Label of the irrep.
            group_element (np.ndarray): Matrix representing a group element.

        Returns:
            np.ndarray: Matrix representing the irrep of the group element.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def get_possible_irrep_labels(self) -> list:
        """Return all irrep labels for this gauge group.

        Returns:
            list[int]: List of all irrep labels.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def get_electric_energy_factor(self, irrep_label: int) -> float:
        """Return the electric energy factor for a given irrep label.

        Args:
            irrep_label (int): Label of the irrep.

        Returns:
            float: Electric energy factor for the specified irrep.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    def get_irrep_character(self, group_element: np.ndarray, irrep_label: int) -> complex:
        """Return the character of the irrep evaluated at the given group element.

        Args:
            group_element (np.ndarray): Matrix representing the group element.
            irrep_label (int): Label of the irrep.

        Returns:
            complex: Character of the irrep evaluated at group_element.
        """
        return np.trace(self.get_irrep(irrep_label, group_element))

    @abstractmethod
    def get_irrep_dimension(self, irrep_label: int) -> int:
        """Return the dimension of the specified irrep.

        Args:
            irrep_label (int): Label of the irrep.

        Returns:
            int: Dimension of the specified irrep.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def get_neutral_gauge_value(self) -> np.ndarray:
        """Return the identity element of the gauge group as a matrix.

        Returns:
            np.ndarray: Matrix representing the identity element.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def get_possible_gauge_values(self) -> np.ndarray:
        """Return all group elements as an array of matrices.

        Returns:
            np.ndarray: Array of shape (group_order, rep_dim, rep_dim) containing all group elements.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def get_group_elements_and_factors_for_el_energy(self) -> tuple[float, tuple[np.ndarray, ...]]:
        """Compute group elements and the uniform coefficient for the electric energy term.

        Returns:
            float: The uniform coefficient for the electric energy.
            tuple[np.ndarray, ...]: Group elements that contribute to the electric energy.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")


class ZNGauge(GaugeGroup):
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
        self.group_order = n
        self.el_mult_factor, self.group_elements_for_el_energy = self.get_group_elements_and_factors_for_el_energy()

        self.el_offset = 2
        # The maximum value of the electric energy for a specific link.
        # To impose el_energy >= 0 in system:
        # el_energy = self.cfg.g_el *
        # (self.cfg.el_offset * nlinks - self.cfg.gaugemgr.el_mult_factor * self.el_energy_op)

        self.mag_offset = 1
        # The maximum value of the magnetic energy for a specific plaquette.
        # To impose mag_energy >= 0 in system:
        # mag_energy = (
        #   self.cfg.g_mag * 2 * (nplaq * self.cfg.gaugemgr.mag_offset - self.mag_energy_op)
        # )

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

    def get_irrep(self, irrep_label: int, group_element: np.ndarray) -> np.ndarray:
        """
        Return the irreducible representation matrix for a given irrep label.

        In Z_N, each irrep is one-dimensional and corresponds to a phase factor
        exp(i * (2 * pi * j * m) / N), where m is the group element index and
        j is the irrep label.

        Args:
            irrep_label (int): Label of the irrep, an integer in [0, N - 1].
            group_element (np.ndarray): 1x1 matrix representing a Z_N group element.
        Returns:
            np.ndarray: 1x1 complex matrix representing the irrep of the group element.
        """
        theta = self.get_angle(group_element)
        return np.array([[np.exp(1.0j * theta * irrep_label)]])

    def get_possible_irrep_labels(self) -> list:
        """
        Generate all irrep labels representations of the Z_N group.

        Each irrep is one-dimensional and corresponds to a phase factor
        exp(i * (2 * pi * j * m) / N), where j is the irrep label.

        Returns:
            list[int]: List of irrep labels from 0 to N - 1.
        """
        return list(range(self.n))

    def get_electric_energy_factor(self, irrep_label: int) -> float:
        """
        Return the electric energy factor for a given irrep label in Z_N.

        The electric energy factor is defined as:
            f(j) = 2 * cos(2 * pi * j / N)

        Args:
            irrep_label (int): Label of the irrep, an integer in [0, N - 1].

        Returns:
            float: Electric energy factor for the specified irrep.
        """
        increment = self.get_increment()
        angle = irrep_label * increment
        return 2.0 * np.cos(angle)

    def get_irrep_dimension(self, irrep_label: int) -> int:
        """
        Return the dimension of the specified irrep in Z_N.

        All irreps of Z_N are one-dimensional.

        Args:
            irrep_label (int): Label of the irrep, an integer in [0, N - 1].
        Returns:
            int: Dimension of the specified irrep (always 1 for Z_N).
        """
        return 1

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

    def get_angle(self, g: np.ndarray) -> float:
        """
        Return the angle theta for a Z_N group element g = [[exp(i * theta)]].

        Extract the complex phase of a 1x1 matrix representing a Z_N group element.

        Args:
            g (np.ndarray): 1x1 matrix representing a Z_N group element.

        Returns:
            float: Angle theta in radians.
        """
        return np.angle(g[0][0])

    def get_group_elements_and_factors_for_el_energy(self) -> tuple[float, tuple[np.ndarray, ...]]:
        """
        Compute group elements and coefficients for the electric energy term.
        We use a hardcoded version instead of the one appearing in D2n gauge,
        because we will get get_representation(increment) and get_representation(-increment)
        which can be compensated by the factor 2 appearing here
        and taking the real part, which is done in system and in config.

        Returns:
            float: The coeefficient for the electric energy.
                    In theory there's a coeefficient for each group element,
                    but it could be shown that they should be equal.
            tuple: group elements for which the electric energy term should be calculated
        """

        increment = self.get_increment()
        return 2.0, tuple([self.get_representation(increment)])


class D2nGauge(GaugeGroup):
    """
    Implement the D_2n gauge group using a real 2D matrix representation.

    The D_2n group is the dihedral group of order 2n, representing the symmetries
    (rotations and reflections) of a regular n-gon. Each element is identified by a pair (p, q):
        - p in {0, 1, ..., n-1} represents rotation by angle theta = 2 * pi * p / n
        - q in {0, 1}, where q = 0 denotes rotation and q = 1 denotes reflection

    The group elements are represented as 2x2 real orthogonal matrices:
        - D(p, 0) = [[ cos(theta), -sin(theta)],
                     [ sin(theta),  cos(theta)]]
        - D(p, 1) = [[ cos(theta),  sin(theta)],
                     [ sin(theta), -cos(theta)]]
    """

    def __init__(self, n: int) -> None:
        """
        Initialize the D_2n gauge group with a real 2D matrix representation.

        Args:
            n (int): Number of rotations in the dihedral group. The full group has 2n elements,
                    including n rotations and n reflections.

        Attributes:
            n (int): The number of rotations in the group (half the total number of elements).
            rep_dim (int): The dimension of the matrix representation (2 for D_2n).
        """
        self.n = n
        self.rep_dim = 2
        self.group_order = 2 * n
        factors_for_el_energy, group_elements_for_el_energy = self.get_group_elements_and_factors_for_el_energy()
        self.group_elements_for_el_energy: tuple[np.ndarray, ...] = group_elements_for_el_energy
        self.el_mult_factor = factors_for_el_energy

        if self.n == 3:
            self.el_offset = 3
        elif self.n == 4:
            self.el_offset = 6
        # The maximum value of the electric energy for a specific link.
        # To impose el_energy >= 0 in system:
        # el_energy = self.cfg.g_el *
        # (self.cfg.el_offset * nlinks - self.cfg.gaugemgr.el_mult_factor * self.el_energy_op)

        self.mag_offset = 2
        # The maximum value of the magnetic energy for a specific plaquette.
        # To impose mag_energy >= 0 in system:
        # mag_energy = (
        #   self.cfg.g_mag * 2 * (nplaq * self.cfg.gaugemgr.mag_offset - self.mag_energy_op)
        # )

    def get_representation(self, p: int, q: int) -> np.ndarray:
        """
        Return the 2x2 real matrix for the group element (p, q).

        Args:
            p (int): Rotation index.
            q (int): Reflection index (0 for rotation, 1 for reflection).

        Returns:
            np.ndarray: 2x2 real orthogonal matrix.
        """
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

    def get_irrep(self, irrep_label: int, group_element: np.ndarray) -> np.ndarray:
        """
        Return the irreducible representation matrix for a given irrep label.

        The irreps are labelled as follows:
        j=0: D^j(p,q)=1
        j=1 (\bar{0}): D^j(p,q)=(-1)^q

        if n is even:
         j=2 (1): D^j(p,q)=(-1)^p
         j=3 (\bar{1}): D^j(p,q)=(-1)^q * (-1)^p
         j=4,...,(n+4)/2 (2_1,...2_{(n-2)/2}) (overall (n-2)/2 2D irreps):
                    D^j(p,q)= [[cos(2π(j-3)p/n), -sin(2π(j-3)p/n)],
                                [sin(2π(j-3)p/n),  cos(2π(j-3)p/n)]] for q=0
                    D^j(p,q)= [[cos(2π(j-3)p/n),  sin(2π(j-3)p/n)],
                                [sin(2π(j-3)p/n), -cos(2π(j-3)p/n)]] for q=1
        if n is odd:
         j=2,...,(n+1)/2 (2_1,...2_{(n-1)/2}) (overall (n-1)/2 2D irreps):
                    D^j(p,q)= [[cos(2π(j-2)p/n), -sin(2π(j-2)p/n)],
                                [sin(2π(j-1)p/n),  cos(2π(j-1)p/n)]] for q=0
                    D^j(p,q)= [[cos(2π(j-1)p/n),  sin(2π(j-1)p/n)],
                                [sin(2π(j-1)p/n), -cos(2π(j-1)p/n)]] for q=1

        Args:
            irrep_label (int): Label of the irrep.
            group_element (np.ndarray): matrix representing a D_2N group element.
        Returns:
            np.ndarray: matrix representing the irrep of the group element.
        """
        q = self.get_reflection_index(group_element)
        p = self.get_rotation_index(group_element)
        if irrep_label == 0:
            return np.array([[1]])
        elif irrep_label == 1:
            return np.array([[(-1) ** q]])
        if self.n % 2 == 0:  # n is even
            if irrep_label == 2:
                return np.array([[(-1) ** p]])
            elif irrep_label == 3:
                return np.array([[(-1) ** q * (-1) ** p]])
            else:  # 2D irreps
                j_2d = irrep_label - 4 + 1
                theta = (2.0 * np.pi * j_2d * p) / self.n
                if q == 0:
                    return np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ],
                    )
                else:  # q==1
                    return np.array(
                        [
                            [np.cos(theta), np.sin(theta)],
                            [np.sin(theta), -np.cos(theta)],
                        ],
                    )
        else:  # n is odd
            if irrep_label >= 2:  # 2D irreps
                j_2d = irrep_label - 2 + 1
                theta = (2.0 * np.pi * j_2d * p) / self.n
                if q == 0:
                    return np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ],
                    )
                else:  # q==1
                    return np.array(
                        [
                            [np.cos(theta), np.sin(theta)],
                            [np.sin(theta), -np.cos(theta)],
                        ],
                    )
            else:
                raise ValueError("Invalid irrep label for D2n group")

    def get_possible_irrep_labels(self) -> list:
        """
        Generate all irrep labels representations of the D_2N group.
        The irreps are labelled as follows:
        j=0: D^j(p,q)=1
        j=1 (\bar{0}): D^j(p,q)=(-1)^q

        if n is even:
         j=2 (1): D^j(p,q)=(-1)^p
         j=3 (\bar{1}): D^j(p,q)=(-1)^q * (-1)^p
         j=4,...,(n+4)/2 (2_1,...2_{(n-2)/2}) (overall (n-2)/2 2D irreps):
                    D^j(p,q)= [[cos(2π(j-3)p/n), -sin(2π(j-3)p/n)],
                                [sin(2π(j-3)p/n),  cos(2π(j-3)p/n)]] for q=0
                    D^j(p,q)= [[cos(2π(j-3)p/n),  sin(2π(j-3)p/n)],
                                [sin(2π(j-3)p/n), -cos(2π(j-3)p/n)]] for q=1
        if n is odd:
         j=2,...,(n+1)/2 (2_1,...2_{(n-1)/2}) (overall (n-1)/2 2D irreps):
                    D^j(p,q)= [[cos(2π(j-2)p/n), -sin(2π(j-2)p/n)],
                                [sin(2π(j-1)p/n),  cos(2π(j-1)p/n)]] for q=0
                    D^j(p,q)= [[cos(2π(j-1)p/n),  sin(2π(j-1)p/n)],
                                [sin(2π(j-1)p/n), -cos(2π(j-1)p/n)]] for q=1

        Returns:
            list[int]: List of irrep labels from 0 to N - 1.
        """
        if self.n % 2 == 0:
            return list(range((self.n + 6) // 2))
        return list(range((self.n + 3) // 2))

    def get_electric_energy_factor(self, irrep_label: int) -> float:
        """
        Return the electric energy factor for a given irrep label in Z_N.

        The electric energy factor is defined as:
            f(j) = 2 * cos(2 * pi * j / N)

        Args:
            irrep_label (int): Label of the irrep, an integer in [0, N - 1].

        Returns:
            float: Electric energy factor for the specified irrep.
        """
        if self.n == 3:
            if irrep_label == 0:
                return -3.0
            elif irrep_label == 1:
                return 3.0
            else:
                return 0.0
        elif self.n == 4:
            if irrep_label == 0:
                return -6.0
            elif irrep_label == 1:
                return 2.0
            elif irrep_label == 2:
                return 2.0
            elif irrep_label == 3:
                return 2.0
            else:
                return 0.0
        else:
            raise NotImplementedError("Electric energy factors not implemented for D2n with n>4")

    def get_irrep_dimension(self, irrep_label: int) -> int:
        """
        Return the dimension of the specified irrep in D_2N.

        All irreps of Z_N are one-dimensional.

        Args:
            irrep_label (int): Label of the irrep.
        Returns:
            int: Dimension of the specified irrep.
        """
        if self.n % 2 == 0:
            if irrep_label in [0, 1, 2, 3]:
                return 1
            else:
                return 2
        else:
            if irrep_label in [0, 1]:
                return 1
            else:
                return 2

    def get_reflection_index(self, g: np.ndarray) -> int:
        """
        Determine whether the given gauge element is a rotation or reflection.

        Uses the determinant of the matrix to distinguish:
            - det ≈ +1 → rotation (q = 0)
            - det ≈ -1 → reflection (q = 1)

        Args:
            g (np.ndarray): A 2x2 matrix representing a D_2n group element.

        Returns:
            int: 0 if rotation, 1 if reflection.

        Raises:
            ValueError: If g is not a valid D_2n matrix (det not close to ±1).
        """
        det = np.linalg.det(g)
        if np.isclose(det, 1.0):  # rotation - not reflection
            return 0
        elif np.isclose(det, -1):
            return 1
        else:
            raise ValueError("Gauge value not in D2n group")

    def get_rotation_index(self, g: np.ndarray) -> int:
        """
        Extract the rotation index p from the D_2n group element matrix.

        The rotation index p is determined from the angle theta of the rotation part of the matrix.
        Args:
            g (np.ndarray): A 2x2 matrix representing a D_2n group element
            (has to be the faithful fundamental representation of get_representation).
        Returns:
            int: Rotation index p in {0, 1, ..., n-1}.
        """
        cos_theta = g[0, 0]
        sin_theta = g[1, 0]
        theta = np.atan2(sin_theta, cos_theta)
        p = int(round((theta * self.n) / (2.0 * np.pi))) % self.n
        return p

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> np.ndarray:
        """
        Generate a random D_2n group element as a 2x2 real matrix.

        Each element is defined by a pair (p, q) where:
            - p in {0, ..., n-1} is the rotation index
            - q in {0, 1} is the reflection index

        Args:
            rng_state (np.random.RandomState): Random number generator used for reproducibility.

        Returns:
            np.ndarray: 2x2 matrix representing a randomly selected D_2n group element.
        """
        p = rng_state.randint(0, self.n)
        q = rng_state.randint(0, 2)
        return self.get_representation(p, q)

    def get_neutral_gauge_value(self) -> np.ndarray:
        """
        Return the identity element of the D_2n group as a 2x2 identity matrix.

        This corresponds to the group element (p = 0, q = 0), representing no rotation and no reflection.

        Returns:
            np.ndarray: 2x2 identity matrix.
        """
        return np.identity(2)

    def get_possible_gauge_values(self) -> np.ndarray:
        """
        Generate all group elements of D_2n in matrix representation.

        Returns:
            np.ndarray: Array of shape (2n, 2, 2), containing all 2x2 matrices corresponding
                        to the elements of the D_2n group.
        """
        dest = np.array([self.get_representation(p, q) for q in range(2) for p in range(self.n)])
        return dest

    def get_group_elements_and_factors_for_el_energy(self):
        """
        Compute group elements and coefficients for the electric energy term.

        Returns:
            float: The coeefficient for the electric energy.
                    In theory there's a coeefficient for each group element,
                    but it could be shown that they should be equal.
            tuple: group elements for which the electric energy term should be calculated

        """
        irreps = self.get_possible_irrep_labels()
        group_elements = self.get_possible_gauge_values()
        group_for_el_energy = []
        factor_for_el = []
        for h in group_elements:
            pref = 0.0
            h_inv = np.conjugate(np.transpose(h))
            for irrep in irreps:
                irrep_character = self.get_irrep_character(h_inv, irrep)
                electric_energy_factor = self.get_electric_energy_factor(irrep)
                dim_irrep = self.get_irrep_dimension(irrep)
                pref += electric_energy_factor * irrep_character * dim_irrep
            pref = -pref / self.group_order
            # The minus is a convention to fit
            # el_energy = self.cfg.g_el * (2 * nlinks - self.cfg.gaugemgr.el_mult_factor * self.el_energy_op)
            # in system_base.

            if abs(pref) > 10e-5:
                factor_for_el.append(pref)
                group_for_el_energy.append(h)
                first_factor = factor_for_el[0]
        if factor_for_el:
            tol = 10e-5
            if not np.allclose(factor_for_el, first_factor, atol=tol):
                raise ValueError(
                    f"Electric energy factors are not uniform within tolerance {tol}."
                    f"These values should be equal for discrete groups."
                    f"Values found: {factor_for_el}"
                )
        return factor_for_el[0], tuple(group_for_el_energy)


class Z2RepGauge2D(GaugeGroup):
    """Z_2 gauge group represented by 2x2 matrices that mix colors.

    Group elements: {+1, -1}. Representation matrices:
        D(+1) = I_2
        D(-1) = sigma_x = [[0, 1], [1, 0]]   # swaps the two colors

    The two irreps of Z_2 are still 1D (Z_2 is abelian) — trivial and sign — so the
    electric-energy bookkeeping (irrep characters, dimensions, factors) is identical
    to the standard ZNGauge(2). What changes is the FAITHFUL representation used by
    the gauging operator U_h: it is now a 2x2 matrix that mixes the two color blocks,
    exactly like the D6 2D irrep. This class exists to test whether the D6 electric-
    energy discrepancy is specific to D6 or appears for any 2D-rep gauging.
    """

    SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])

    def __init__(self) -> None:
        logger.warning(
            "Z2RepGauge2D (the 2D-representation Z2 gauge group) was created for "
            "debugging purposes only and has not been tested."
        )
        self.n = 2
        self.rep_dim = 2
        self.group_order = 2
        self.el_mult_factor, self.group_elements_for_el_energy = self.get_group_elements_and_factors_for_el_energy()
        self.el_offset = 2
        self.mag_offset = 2

    # ---- group structure ---------------------------------------------------
    def get_representation(self, sign: int) -> np.ndarray:
        """sign in {+1, -1} -> 2x2 matrix."""
        return np.eye(2) if sign > 0 else self.SIGMA_X.copy()

    def get_neutral_gauge_value(self) -> np.ndarray:
        return np.eye(2)

    def get_possible_gauge_values(self) -> np.ndarray:
        return np.array([np.eye(2), self.SIGMA_X])

    def get_random_gauge_value(self, rng_state: np.random.RandomState) -> np.ndarray:
        k = rng_state.randint(0, 2)
        return np.eye(2) if k == 0 else self.SIGMA_X.copy()

    # ---- irrep bookkeeping (1D irreps, same as ZNGauge(2)) -----------------
    def _sign_of(self, group_element: np.ndarray) -> int:
        return 1 if np.allclose(group_element, np.eye(2)) else -1

    def get_irrep(self, irrep_label: int, group_element: np.ndarray) -> np.ndarray:
        s = self._sign_of(group_element)
        if irrep_label == 0:  # trivial irrep
            return np.array([[1.0 + 0.0j]])
        # sign irrep
        return np.array([[float(s) + 0.0j]])

    def get_irrep_character(self, group_element: np.ndarray, irrep_label: int) -> complex:
        return complex(np.trace(self.get_irrep(irrep_label, group_element)))

    def get_irrep_dimension(self, irrep_label: int) -> int:
        return 1

    def get_possible_irrep_labels(self) -> list:
        return [0, 1]

    def get_electric_energy_factor(self, irrep_label: int) -> float:
        # f_j = 2 cos(2*pi*j / N), N=2
        return 2.0 if irrep_label == 0 else 0.0

    def get_group_elements_and_factors_for_el_energy(self):
        """Return (el_mult_factor, group elements used for the electric energy).

        For Z_2 the single non-neutral element gives pref(h) = 2 (same as ZNGauge(2)).
        Using el_mult_factor=2.0 keeps the el_energy formula
            el_energy = g_el * (el_offset * nlinks - el_mult_factor * el_energy_op)
        consistent with the standard Z_2 convention.
        """
        return 2.0, tuple([self.SIGMA_X.copy()])
