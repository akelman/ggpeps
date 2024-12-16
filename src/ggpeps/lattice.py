import sys
from enum import IntEnum

import logging
import numpy as np

import ggpeps

logger = logging.getLogger(ggpeps.LOGGER_NAME)


class Direction(IntEnum):
    """Enum to capture the direction of a link"""

    X = 0
    Y = 1
    Z = 2

    def __str__(self):
        return self.name


class Lattice2D:
    """
    Handler of a square lattice of size nx x ny

    Args:
        nx (int): Extend of the lattice in x direction (given in number of vertices)
        ny (int): Extend of the lattice in y direction (given in number of vertices)
    """

    dim = 2

    def __init__(self, nx: int, ny: int, gf_num_of_rows: int = -1):
        self.nx = nx
        self.ny = ny
        self.nlinks = 2 * nx * ny
        self.nplaquettes = nx * ny
        self.size = nx * ny  # number of sites
        self.ntreelinks = nx * ny - 1
        self.ncomptreelinks = (
            nx * ny + 1
        )  # number of links not in the tree - complementary tree links

        # We trust the user not to modify these
        if gf_num_of_rows == -1:  # If we gauge_fix over a maximal tree
            self.maximal_tree = self.generate_tree()
        else:  # We fix a specific number of rows
            self.maximal_tree = self.generate_tree(gf_num_of_rows)

        self.comp_tree = self.generate_tree_complement()

    def __str__(self):
        """Generate a string representation of the lattice.
        For a given index the coordinate representation and the adjoint links are printed: <ind>,(x_ind,y_ind): x_link, y_link

        Returns:
            str: String representation of the lattice
        """
        dest = ""
        for ind in range(self.nplaquettes):
            x, y = self.ind2coord(ind)
            x_link = self.coord2ind_dir((x, y), Direction.X)
            y_link = self.coord2ind_dir((x, y), Direction.Y)
            dest += f"{ind:02d}, ({x:02d},{y:02d}): {x_link:02d},{y_link:02d}\n"
        return dest

    def ind2coord(self, ind: int) -> tuple:
        """Conversion method from integer to lattice coordinate (a tuple).

        Args:
            ind (int): Index of a site

        Returns:
            tuple: Tuple of integers (x,y)
        """
        return (ind % self.nx, ind // self.nx)

    def coord2ind(self, coord: tuple) -> int:
        """Conversion method from coordinate tuples to lattice index (integer).

        Args:
            coord (tuple): Tuple of integers (x,y)

        Returns:
            int: Lattice index of the site
        """
        x, y = coord
        return self.nx * y + x

    def ind2coord_dir(self, ind: int) -> tuple:
        """Conversion method to convert a link index into a coordinate tuple with a direction ((x,y),dir)

        Args:
            ind (int): link index

        Returns:
            tuple: coordinate of the adjacent vertex and the link direction ((x,y),dir)
        """
        dir = Direction(ind // (self.nx * self.ny))
        if dir == Direction.X:
            return (
                (
                    (ind % (self.nx * self.ny)) % self.nx,
                    (ind % (self.nx * self.ny)) // self.nx,
                ),
                dir,
            )
        elif dir == Direction.Y:
            return (
                (
                    (ind % (self.nx * self.ny)) // self.ny,
                    (ind % (self.nx * self.ny)) % self.ny,
                ),
                dir,
            )
        else:
            logger.error("ind2coord_dir: There are only X and Y as directions")
            return None

    def coord2ind_dir(self, coord: tuple, dir: Direction) -> int:
        """Conversion method from a coordinate tuple (x,y) and a direction to a link index.
        If one wishes to get a link in the 'negative direction', simply change coord so that travelling in the positive direction gives the correct link.
        The function can handle negative coordinates (assumes periodic boundary conditions).

        Args:
            coord (tuple): Coordinate tuple (x,y)
            dir (Direction): Direction of the link

        Returns:
            int: link index
        """
        x, y = coord

        # Handle case where coordinates have wrapped around the boundary in the negative direction
        while x < 0:
            x += self.nx
        while y < 0:
            y += self.ny

        if dir == Direction.X:
            return self.nx * self.ny * dir.value + self.nx * y + x
        elif dir == Direction.Y:
            return self.nx * self.ny * dir.value + self.ny * x + y
        else:
            logger.error(
                "coord2ind_dir: There are only X and Y as directions", file=sys.stderr
            )
            return None

    def get_neighbor(self, coord: tuple, orient: Direction) -> tuple:
        """Get the next coordinate tuple in a given direction (wraps around periodic boundary conditions)

        Args:
            coord (tuple): (x,y) coordinates of the original point
            orient (Direction): direction of the desired neighbor

        Returns:
            tuple: (x,y) coordinate of the next point
        """
        # We assume periodic boundary conditions
        x, y = coord
        if orient == Direction.X:
            xn = (x + 1) % self.nx
            yn = y
        elif orient == Direction.Y:
            xn = x
            yn = (y + 1) % self.ny
        return (xn, yn)

    def generate_polyakov_loop(
        self, coord: tuple, dir: Direction, use_indices: bool = True
    ) -> list:
        """Generate a Polyakov loop, a loop around the full system.
        We only need one point so start from and a direction.
        The loop is returned in the format [(link_id,bool),...,(link_id,bool)].
        The <link_id> can be either a tuple of coordinates or an integer id of a link (depending on use_indices).
        The bool in the tuples returned by this function signifies the orientation.
        "True" means flip gauge field, "False" means no flip.

        Args:
            coord (tuple): Coordinate to start from
            dir (Direction): Direction of the loop (x or y)
            use_indices (bool, optional): Return the loop in terms of link indices. Defaults to True.

        Returns:
            list: Links on the Polyakov loop
        """
        x, y = coord
        dest = []
        if dir == Direction.X:
            # Build polyakov loop in X direction
            for i in range(self.nx):
                coord_link = ((i, y), dir)
                dest.append((coord_link, False))
        elif dir == Direction.Y:
            # Build polyakov loop in Y direction
            for i in range(self.ny):
                coord_link = ((x, i), dir)
                dest.append((coord_link, False))
        else:
            logger.error("generate_polyakov_loop: There are only X and Y as directions")
            return None
        if use_indices:
            # Transform the coordinates to indices
            dest = [(self.coord2ind_dir(*coorddir), conj) for (coorddir, conj) in dest]
        return dest

    def generate_wilson_loop(
        self, coord: tuple, size: tuple, use_indices: bool = True
    ) -> list:
        """Generate a Wilson loop with bottom left corner at coord and an extend specified by the tuple size.
        This method is aware of the periodic boundary conditions of the lattice.
        The loop is returned in the format [(link_id,bool),...,(link_id,bool)].
        The <link_id> can be either a tuple of coordinates or an integer id of a link (depending on use_indices).
        The bool in the tuples returned by this function signifies the orientation.
        "True" means flip gauge field, "False" means no flip.

        Args:
            coord (tuple): bottom left corner (x,y) of the Wilson loop
            size (tuple): extend in (x,y)
            use_indices (bool, optional): Use link indices instead of coordinate representation. Defaults to True.

        Returns:
            list: List of tuples of the form (link_id,<bool>)
        """
        ext_x, ext_y = size
        x, y = coord
        dest = []
        for i in range(ext_x):
            coord_link = ((x + i) % self.nx, y)
            dest.append(((coord_link, Direction.X), False))
        for i in range(ext_y):
            coord_link = ((x + ext_x) % self.nx, (y + i) % self.ny)
            dest.append(((coord_link, Direction.Y), False))
        for i in range(ext_x):
            coord_link = ((x + ext_x - i - 1) % self.nx, (y + ext_y) % self.ny)
            dest.append(((coord_link, Direction.X), True))
        for i in range(ext_y):
            coord_link = (x, (y + ext_y - i - 1) % self.ny)
            dest.append(((coord_link, Direction.Y), True))
        if use_indices:
            # Transform the coordinates to indices
            dest = [(self.coord2ind_dir(*coorddir), conj) for (coorddir, conj) in dest]
        return dest

    def generate_allowed_loop_dimensions(self, include_all: bool = False) -> list:
        """Generate all rectangular loop dimensions.
        The max loop size (that we care about) is half the lattice size because of the periodic boundary conditions
        This function is separated out from generate_all_wilson_loops() so that it can be used in managing observables.

        Args:
            include_all (bool): True if loops should include those that are rotations of each other (e.g. 2x1 and 1x2 loops). Defaults to False.

        Returns:
            list: A list of tuples will allowed loop sizes.
        """
        sizes = []
        max_x = (
            self.nx // 2
        )  # due to periodic boundary conditions, loops should only go up to half the system size
        max_y = self.ny // 2

        for size_x in range(1, max_x + 1):
            for size_y in range(1, max_y + 1):
                if include_all or size_y <= size_x:
                    sizes.append((size_x, size_y))

        return sizes

    def generate_all_wilson_loops(
        self, coord: tuple, sizes: list = [], use_indices: bool = True
    ) -> list:
        """Generate all rectangular Wilson loops with bottom left corner at coord, up to a size determined by the lattice size.
        This method is aware of the periodic boundary conditions of the lattice.
        Each loop is returned in the format [(link_id,bool),...,(link_id,bool)].
        The <link_id> can be either a tuple of coordinates or an integer id of a link (depending on use_indices).
        The bool in the tuples returned by this function signifies the orientation: "True" means flip gauge field, "False" means no flip.

        Args:
            coord (tuple): bottom left corner (x,y) of the Wilson loop
            sizes (list): list of tuples (size_x, size_y) of desired loop sizes. If sizes is empty, all allowed loops will be generated. Defaults to an empty list.
            use_indices (bool, optional): Use link indices instead of coordinate representation. Defaults to True.

        Returns:
            list of lists: List of list of tuples of the form (link_id,<bool>). Each element is a complete Wilson loop.
        """
        loops = []

        if len(sizes) == 0:
            sizes = self.generate_allowed_loop_dimensions()

        for size in sizes:
            loop = self.generate_wilson_loop(coord, size, use_indices)
            loops.append(loop)

        return loops

    def generate_tree(self, num_of_rows: int = None):
        """Generate a tree on the lattice.
        This allows all values on the tree to be fixed to the identity when gauge_fixing
        (no integration is needed over links on the tree).
        This method is built for a lattice with periodic boundary conditions.

        The particular tree returned by this function includes all the horizontal links in the first num_of_rows rows but the last one on each row.

        If num_of_rows is not given then a maximal tree containing all the rows but the last link
        and all the vertical links but the last one on the first column.

        Args:
            num_of_rows (int, optional): Number of rows to fix. Defaults to None.

        Returns:
            list: List of link-indices in the tree
        """
        tree = []
        if (
            num_of_rows is None or num_of_rows > self.ny
        ):  # If number of rows to fix is not given or larger than lattice size we generate a maximal tree
            num_of_rows = self.ny

            tree += [
                self.coord2ind_dir((0, y), Direction(1)) for y in range(self.ny - 1)
            ]

        # add horizontal links, except for the last
        tree += [
            self.coord2ind_dir((x, y), Direction(0))
            for y in range(num_of_rows)
            for x in range(self.nx - 1)
        ]

        return tree

    def generate_tree_complement(self):
        """Generate the list of links complementary to a maximal tree.
        This method is built for a lattice with periodic boundary conditions.

        Returns:
            list: List of links which are not in the maximal tree
        """
        fixed_links_ind = [i for i in range(self.nlinks) if i not in self.maximal_tree]
        return fixed_links_ind


class Lattice3D:
    """
    Handler of a square lattice of size nx x ny x nz.
    The functions are very similar to the Lattice2D class.
    For more documentation on the methods, refer to the Lattice2D class.
    """

    dim = 3

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nlinks = 3 * nx * ny * nz
        self.size = nx * ny * nz  # number of sites

    def __str__(self):
        arr = np.arange(self.get_size())
        arr = np.reshape(arr, (self.nx, self.ny))
        return str(arr)

    def get_size(self):
        """Returns number of sites on the lattice"""
        return self.nx * self.ny * self.nz

    def ind2coord(self, ind):
        x = ind % self.nx
        y = (ind % (self.nx * self.ny)) // self.nx
        z = ind // (self.nx * self.ny)
        return (x, y, z)

    def coord2ind(self, coord):
        x, y, z = coord
        return z * self.nx * self.ny + self.nx * y + x

    def ind2coord_dir(self, ind):
        ind_coord = ind % (self.nx * self.ny * self.nz)
        x = ind_coord % self.nx
        y = (ind_coord % (self.nx * self.ny)) // self.nx
        z = ind_coord // (self.nx * self.ny)
        dir = Direction(ind // (self.nx * self.ny * self.nz))
        return (x, y, z), dir

    def coord2ind_dir(self, coord, dir):
        x, y, z = coord
        return (
            self.nx * self.ny * self.nz * dir.value
            + self.nx * self.ny * z
            + self.nx * y
            + x
        )


class PermutationBuilderGMS2DU1:
    """Build a permutation matrix for gamma_maj_sys for the U1 parametrization
     The default mode-order of the elementary T matrix is {p,l,r,u,d}.
     We double this T matrix to accomodate for positive and negative modes.
     By building the Dirac covariance matrix from it and transforming it to a Majorana matrix, we obtain a mode-order of {p_1,p_2,l+_1, l+_2, l-_1, l-_2, r+_1, r+_2, r-_1, r-_2, d+_1, d+_2, d-_1, d-_2, u+_1, u+_2, u-_1, u-_2}.
     The covariance matrix of the projectors is chosen such that modes from adherent vertices can be modified together, i.e. it is ordered by links.
     The order of the links is first along x and then along y. In the case of a 2x2 lattice, it reads

     |         |
    "5"       "7"
     |         |
     2 --"2"-- 3 --"3"--
     |         |
    "4"       "6"
     |         |
     0 --"0"-- 1 --"1"--

     The vertex indices are written as <number>, the link indices are written as "<number>".

     Before the transformation the covariance matrix of the gamma_maj_sys has the order (taking the 2x2 system as an example for concreteness)
     { l1_0, r1_0, d1_0, u1_0, l2_0, r2_0, d2_0, u2_0,
       l1_1, r1_1, d1_1, u1_1, l2_1, r2_1, d2_1, u2_1,
       l1_2, r1_2, d1_2, u1_2, l2_2, r2_2, d2_2, u2_2,
       l1_3, r1_3, d1_3, u1_3, l2_3, r2_3, d2_3, u2_3 }
     The formatting of the modes above follows the structure {ldru}<copy>_<site_index>.
     Gamma_in for two copies has the order
     { l1_1, r1_0, l2_1, r2_0, l1_0, r1_1, l2_0, r2_1,
       l1_3, r1_2, l2_3, r2_2, l1_2, r1_3, l2_2, r2_3,
       d1_2, u1_0, d2_2, u2_0, d1_0, u1_2, d2_0, u2_2,
       d1_3, u1_1, d2_3, u2_1, d1_1, d1_3, d2_1, d2_3 }.
     This class provides the necessary permutation matrix to change from one basis to the other.
    """

    def __init__(self, lat: Lattice2D, nmodes_per_link: int):
        self.lattice = lat
        self.nmodes_per_link = nmodes_per_link
        self.ncopies = 1

    def perm(self):
        size = self.lattice.size
        nx = self.lattice.nx
        ny = self.lattice.ny
        offset_physical_modes = 2 * size
        nmodes = 16
        mat_size = offset_physical_modes + nmodes * size
        dest = np.zeros((mat_size, mat_size))
        # The physical modes are not permuted at all, so we insert an identity matrix
        dest[range(2 * size), range(2 * size)] = 1
        # Now we have to permute the virtual modes
        # Permutation of the l,r modes.
        # We order them row-wise: Example for a 4x4 system
        # l(0,1), r(0,0), l(0,2), r(0,1), l(0,3), r(0,2), l(0,0), r(0,3), l(1,1),
        # r(1,0), l(1,2), r(1,1)....
        for y in range(ny):
            for x in range(nx):
                # Treatment of  the left and right modes
                offset_site = offset_physical_modes + nmodes * x + y * nx * nmodes
                l_j_perm = offset_site
                l_i_perm = offset_physical_modes + y * 2 * nx * 4 + (x - 1) * 8
                r_j_perm = offset_site + 4
                r_i_perm = offset_physical_modes + y * 2 * nx * 4 + x * 8 + 4
                if x == 0:
                    # We have to add the periodic boundary condition here for the left mode
                    l_i_perm = offset_physical_modes + y * 2 * nx * 4 + (nx - 1) * 8
                dest[range(l_i_perm, l_i_perm + 4), range(l_j_perm, l_j_perm + 4)] = 1
                dest[range(r_i_perm, r_i_perm + 4), range(r_j_perm, r_j_perm + 4)] = 1

                # Treatment of the down and up modes
                d_j_perm = offset_site + 8
                d_i_perm = offset_physical_modes + 8 * size + (y - 1) * nx * 8 + x * 8
                u_j_perm = offset_site + 12
                u_i_perm = offset_physical_modes + 8 * size + y * nx * 8 + 8 * x + 4
                if y == 0:
                    # We have to add the periodic boundary condition here for the down mode
                    d_i_perm = (
                        offset_physical_modes + 8 * size + (ny - 1) * nx * 8 + 8 * x
                    )
                dest[range(d_i_perm, d_i_perm + 4), range(d_j_perm, d_j_perm + 4)] = 1
                dest[range(u_i_perm, u_i_perm + 4), range(u_j_perm, u_j_perm + 4)] = 1
        return dest


if __name__ == "__main__":
    print("Lattice 2d, 3x2")
    lat_3x2 = Lattice2D(4, 5)
    print(lat_3x2)
    wilson_loop = lat_3x2.generate_wilson_loop((0, 0), (1, 1))
    print(wilson_loop)
    lst = lat_3x2.generate_tree()
    print(lst)
    print([lat_3x2.ind2coord_dir(ind) for ind in lst])
    print(len(lst))
