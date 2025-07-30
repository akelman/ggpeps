from typing import Optional
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

    def __str__(self) -> str:
        return self.name


class Lattice2D:
    """
    Handler for a 2D square lattice and its gauge-fixing utilities.

    This class represents a 2D square lattice with periodic boundary conditions.
    All methods in this class respect these boundary conditions.

    It manages lattice geometry (sites, links, and plaquettes) and provides
    helper methods for coordinate conversions, neighbor lookups, and generation
    of Wilson loops, Polyakov loops, and gauge-fixing trees.
    """

    dim = 2

    def __init__(self, nx: int, ny: int, gf_num_of_rows: Optional[int] = 0) -> None:
        """
        Initialize a 2D lattice and set its gauge-fixing tree.

        Args:
            nx (int): Lattice size in the x direction (number of vertices).
            ny (int): Lattice size in the y direction (number of vertices).
            gf_num_of_rows (int, optional): Number of rows to fix during gauge fixing.
                - 0 (default): No gauge fixing.
                - -1: Fix a maximal tree.
                - -2: Fix a chess tree.

        Notes:
            In this codebase, lattice sites and links follow the convention illustrated below.
            For example, a 2x2 lattice representation (sites and links), where the bottom-left
            corner is (0,0), is shown as:

            |         |
           "5"       "7"
            |         |
            2 --"2"-- 3 --"3"--
            |         |
           "4"       "6"
            |         |
            0 --"0"-- 1 --"1"--

        """
        self.nx = nx
        self.ny = ny
        self.nlinks = 2 * nx * ny
        self.nplaquettes = nx * ny
        self.size = nx * ny  # number of sites/plaquettes

        # Sublattice factors: 1 on the even sublattice, -1 on the odd sublattice
        self.sublattice_factors = [(-1) ** np.sum(self.ind2coord(site)) for site in range(self.size)]

        # We trust the user not to modify these
        if gf_num_of_rows == -2:  # we fix a chess tree
            self.fixed_tree = self.generate_chess_tree()
        else:
            if gf_num_of_rows == -1:  # If we gauge_fix over a maximal tree
                gf_num_of_rows = None  # We fix a maximal tree
            self.fixed_tree = self.generate_tree(gf_num_of_rows)

        self.comp_tree = self.generate_tree_complement()

    def __str__(self) -> str:
        """
        Generate a string representation of the lattice.
        For each site index, print its coordinates and the indices of its adjacent
        X and Y links in the format:
            <site_index>, (x, y): x_link, y_link

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

    def ind2coord(self, ind: int) -> tuple[int, int]:
        """
        Convert a site index to lattice coordinates.

        Args:
            ind (int): Site index.

        Returns:
            tuple[int, int]: (x, y) coordinates of the site.
        """
        return (ind % self.nx, ind // self.nx)

    def coord2ind(self, coord: tuple[int, int]) -> int:
        """
        Convert lattice coordinates to a site index.

        Args:
            coord (tuple[int, int]): (x, y) coordinates of the site.

        Returns:
            int: Lattice site index.
        """
        x, y = coord
        return self.nx * y + x

    def ind2coord_dir(self, ind: int) -> tuple[tuple[int, int], Direction]:
        """
        Convert a link index to its lattice coordinates and direction.

        Args:
            ind (int): Link index.

        Returns:
            tuple[tuple[int, int], Direction]: ((x, y), Direction) of the link.
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
            raise ValueError("ind2coord_dir: Only X and Y are valid directions.")

    def coord2ind_dir(self, coord: tuple[int, int], dir: Direction) -> int:
        """
        Convert lattice coordinates and a direction to a link index.

        If a link in the negative direction is needed, adjust `coord` so that moving
        in the positive direction yields the desired link. The method handles negative
        coordinates, and assumes periodic boundary conditions.

        Args:
            coord (tuple[int, int]): (x, y) coordinates of the link.
            dir (Direction): Direction of the link (Direction.X or Direction.Y).

        Returns:
            int: Link index.
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
            raise ValueError("coord2ind_dir: Only X and Y are valid directions.")

    def links_coord_to_index(
        self, links: list[tuple[tuple[tuple[int, int], Direction], bool]]
    ) -> list[tuple[int, bool]]:
        """
        Convert a list of coordinate-based links into index-based links.

        Args:
            links (list[tuple[((x, y), Direction), bool]]):
                Links in coordinate form, where each element is
                (((x, y), direction), orientation).

        Returns:
            list[tuple[int, bool]]: Links in index form, where each element is
            (link_index, orientation).
        """
        return [(self.coord2ind_dir(coord, direction), conj) for (coord, direction), conj in links]

    def get_neighbor(self, coord: tuple[int, int], direction: Direction, orientation: bool = True) -> tuple[int, int]:
        """
        Get the neighboring plaquette or site coordinates in a given direction.

        The method respects periodic boundary conditions.

        Args:
            coord (tuple[int, int]): (x, y) coordinates of the current plaquette or site.
            direction (Direction): Direction to move (Direction.X or Direction.Y).
            orientation (bool, optional):
                True  -> positive direction (right for X, up for Y).
                False -> negative direction (left for X, down for Y). Defaults to True.

        Returns:
            tuple[int, int]: (x, y) coordinates of the neighbor.
        """
        x, y = coord
        if direction == Direction.X:
            if orientation:  # Move right
                return ((x + 1) % self.nx, y)
            else:  # Move left
                return ((x - 1) % self.nx, y)
        elif direction == Direction.Y:
            if orientation:  # Move up
                return (x, (y + 1) % self.ny)
            else:  # Move down
                return (x, (y - 1) % self.ny)
        else:
            raise ValueError("get_neighbor: Only X and Y directions are supported")

    def get_path_endpoints(self, path: list[tuple[int, bool]]) -> tuple[int, int]:
        """
        Get the lattice site endpoints of a path.

        The start will be one of the sites adjacent to the first link, determined by whether the link
        is conjugated or not. The end will be one of the sites adjacent to the last link, also determined
        by whether the link is conjugated or not.

        Args:
            path (list[tuple[int, bool]]):
                List of links in index form. Each link is:
                    - (link_id, conj)

        Returns:
            tuple[int, int]:
                Start and end sites as lattice site indices.
        """
        if path == []:
            raise ValueError("There are no start/end points for an empty path.")

        start_link, end_link = path[0][0], path[-1][0]
        is_start_link_conj, is_end_link_conj = path[0][1], path[-1][1]

        start_site_coord, start_site_dir = self.ind2coord_dir(start_link)
        end_site_coord, end_site_dir = self.ind2coord_dir(end_link)

        if is_start_link_conj:
            start_site_coord = self.get_neighbor(start_site_coord, start_site_dir)

        if not is_end_link_conj:
            end_site_coord = self.get_neighbor(end_site_coord, end_site_dir)

        start_site_index = self.coord2ind(start_site_coord)
        end_site_index = self.coord2ind(end_site_coord)
        return (start_site_index, end_site_index)

    def generate_polyakov_loop(self, coord: tuple[int, int], dir: Direction) -> list[tuple[int, bool]]:
        """
        Generate a Polyakov loop around the full system in the positive direction.

        The loop starts from a given point and proceeds entirely in the positive
        direction. Each link orientation is always False for Polyakov loops (that is,
        no links are conjugated).

        Args:
            coord (tuple[int, int]): Starting site coordinates (x, y).
            dir (Direction): Loop direction (Direction.X or Direction.Y).

        Returns:
            list[tuple[int, bool]]: Links forming the Polyakov loop, each as
            (link_index, orientation).
        """
        x, y = coord
        dest: list[tuple[tuple[tuple[int, int], Direction], bool]] = []

        if dir == Direction.X:
            for i in range(self.nx):
                dest.append((((i, y), dir), False))
        elif dir == Direction.Y:
            for i in range(self.ny):
                dest.append((((x, i), dir), False))
        else:
            raise ValueError("generate_polyakov_loop:  Only X and Y are valid directions.")

        return self.links_coord_to_index(dest)

    def generate_wilson_loop(self, coord: tuple[int, int], size: tuple[int, int]) -> list[tuple[int, bool]]:
        """
        Generate a Wilson loop with a given size and bottom-left starting point.

        The loop follows the rectangle edges in counter-clockwise order and
        respects periodic boundary conditions.

        Args:
            coord (tuple[int, int]): Bottom-left corner of the loop (x, y).
            size (tuple[int, int]): Loop size as (extent_x, extent_y).

        Returns:
            list[tuple[int, bool]]: Links forming the Wilson loop, each as
            (link_index, orientation).
        """
        ext_x, ext_y = size
        x, y = coord
        dest = []

        # Bottom edge (left -> right, natural orientation)
        coord_edge = (x, y)
        for _ in range(ext_x):
            dest.append(((coord_edge, Direction.X), False))
            coord_edge = self.get_neighbor(coord_edge, Direction.X, orientation=True)

        # Right edge (bottom -> top, natural orientation)
        for _ in range(ext_y):
            dest.append(((coord_edge, Direction.Y), False))
            coord_edge = self.get_neighbor(coord_edge, Direction.Y, orientation=True)

        # Top edge (right -> left, reversed orientation)
        for _ in range(ext_x):
            coord_edge = self.get_neighbor(coord_edge, Direction.X, orientation=False)
            dest.append(((coord_edge, Direction.X), True))

        # Left edge (top -> bottom, reversed orientation)
        for _ in range(ext_y):
            coord_edge = self.get_neighbor(coord_edge, Direction.Y, orientation=False)
            dest.append(((coord_edge, Direction.Y), True))

        return self.links_coord_to_index(dest)

    def generate_L_string(self, coord: tuple[int, int], size: tuple[int, int]) -> list[tuple[int, bool]]:
        """
        Generate an L-shaped path starting from a bottom-left corner.

        The path extends rightward, then upward, respecting periodic
        boundary conditions. The path only moves in the positive direction,
        so no conjugation (flip) occurs.

        Args:
            coord (tuple[int, int]): Bottom-left corner of the path (x, y).
            size (tuple[int, int]): Path extents as (extent_x, extent_y).

        Returns:
            list[tuple[int, bool]]: Links forming the L-shaped path, each as
            (link_index, orientation).
        """
        ext_x, ext_y = size
        dest = []

        # Horizontal segment (left -> right, natural orientation)
        coord_edge = coord
        for _ in range(ext_x):
            dest.append(((coord_edge, Direction.X), False))
            coord_edge = self.get_neighbor(coord_edge, Direction.X, orientation=True)

        # Vertical segment (bottom -> top, natural orientation)
        for _ in range(ext_y):
            dest.append(((coord_edge, Direction.Y), False))
            coord_edge = self.get_neighbor(coord_edge, Direction.Y, orientation=True)

        return self.links_coord_to_index(dest)

    def generate_allowed_loop_dimensions(self, include_all: bool = False) -> list[tuple[int, int]]:
        """
        Generate all allowed rectangular loop dimensions.

        The maximum loop size is limited to half the lattice size due to periodic
        boundary conditions.

        Args:
            include_all (bool, optional): If True, include rotated loops
                (e.g., both 2x1 and 1x2). Defaults to False.

        Returns:
            list[tuple[int, int]]: Allowed loop sizes as (extent_x, extent_y).
        """
        # TODO: Check/Test why are we limited to half the system size, and not the full system size.
        sizes = []
        max_x = self.nx // 2  # due to periodic boundary conditions, loops should only go up to half the system size
        max_y = self.ny // 2

        for size_x in range(1, max_x + 1):
            for size_y in range(1, max_y + 1):
                if include_all or size_y <= size_x:
                    sizes.append((size_x, size_y))

        return sizes

    def generate_all_wilson_loops(
        self, coord: tuple[int, int], sizes: list[tuple[int, int]] = []
    ) -> list[list[tuple[int, bool]]]:
        """
        Generate all allowed rectangular Wilson loops from a starting point.

        If no sizes are provided, generate all allowed loop sizes.
        The method respects periodic boundary conditions.

        Args:
            coord (tuple[int, int]): Bottom-left corner of the Wilson loops (x, y).
            sizes (list[tuple[int, int]], optional): Loop sizes as (extent_x, extent_y).
                If empty, all allowed loop sizes are generated.

        Returns:
            list[list[tuple[int, bool]]]: A list of Wilson loops, each represented as
            a list of (link_index, orientation) tuples.
        """
        loops = []

        if len(sizes) == 0:
            sizes = self.generate_allowed_loop_dimensions()

        for size in sizes:
            loop = self.generate_wilson_loop(coord, size)
            loops.append(loop)

        return loops

    def generate_tree(self, num_of_rows: Optional[int] = None) -> list[int]:
        """
        Generate a gauge-fixing tree on the lattice.

        Links on the tree are fixed to the identity during gauge fixing.
        The method respects periodic boundary conditions.

        Behavior:
            - If num_of_rows is provided: Includes all horizontal links in the first
            `num_of_rows` rows (except the last link in each row).
            - If num_of_rows is None: Generates a maximal tree containing all rows
            (except the last link) and all vertical links in the first column
            (except the last one).

        Args:
            num_of_rows (int | None, optional): Number of rows to fix. Defaults to None.

        Returns:
            list[int]: Link indices forming the tree.
        """
        tree = []
        if (
            num_of_rows is None or num_of_rows > self.ny
        ):  # If number of rows to fix is not given or larger than lattice size we generate a maximal tree
            num_of_rows = self.ny

            tree += [self.coord2ind_dir((0, y), Direction(1)) for y in range(self.ny - 1)]

        # add horizontal links, except for the last
        tree += [self.coord2ind_dir((x, y), Direction(0)) for y in range(num_of_rows) for x in range(self.nx - 1)]

        return tree

    def generate_chess_tree(self) -> list[int]:
        """
        Generate a chess tree on the lattice.

        The tree consists of all X-direction links originating from even sites only.
        Links on the tree are fixed to the identity during gauge fixing.
        The method respects periodic boundary conditions.

        Returns:
            list[int]: Link indices forming the chess tree.
        """
        tree = []
        for y in range(self.ny):
            for x in range(self.nx):
                if (x + y) % 2 == 0:
                    tree.append(self.coord2ind_dir((x, y), Direction(0)))
        return tree

    def generate_tree_complement(self) -> list[int]:
        """
        Generate the list of links complementary to a maximal tree.

        The method returns all links not included in the gauge-fixing tree and
        respects periodic boundary conditions.

        Returns:
            list[int]: Link indices not part of the maximal tree.
        """
        fixed_links_ind = [i for i in range(self.nlinks) if i not in self.fixed_tree]
        return fixed_links_ind


class Lattice3D:
    """
    Handler of a square lattice of size nx x ny x nz.
    The functions are very similar to the Lattice2D class.
    For more documentation on the methods, refer to the Lattice2D class.
    """

    dim = 3

    def __init__(self, nx: int, ny: int, nz: int) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nlinks = 3 * nx * ny * nz
        self.size = nx * ny * nz  # number of sites

    def __str__(self) -> str:
        arr = np.arange(self.get_size())
        arr = np.reshape(arr, (self.nx, self.ny))
        return str(arr)

    def get_size(self) -> int:
        return self.nx * self.ny * self.nz

    def ind2coord(self, ind: int) -> tuple[int, int, int]:
        x = ind % self.nx
        y = (ind % (self.nx * self.ny)) // self.nx
        z = ind // (self.nx * self.ny)
        return (x, y, z)

    def coord2ind(self, coord: tuple[int, int, int]) -> int:
        x, y, z = coord
        return z * self.nx * self.ny + self.nx * y + x

    def ind2coord_dir(self, ind: int) -> tuple[tuple[int, int, int], Direction]:
        ind_coord = ind % (self.nx * self.ny * self.nz)
        x = ind_coord % self.nx
        y = (ind_coord % (self.nx * self.ny)) // self.nx
        z = ind_coord // (self.nx * self.ny)
        dir = Direction(ind // (self.nx * self.ny * self.nz))
        return (x, y, z), dir

    def coord2ind_dir(self, coord: tuple[int, int, int], dir: Direction) -> int:
        x, y, z = coord
        return self.nx * self.ny * self.nz * dir.value + self.nx * self.ny * z + self.nx * y + x


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

    def __init__(self, lat: Lattice2D, nmodes_per_link: int) -> None:
        """
        Initialize the permutation builder.

        Args:
            lat (Lattice2D): Lattice on which to build the permutation.
            nmodes_per_link (int): Number of modes per link.
        """
        self.lattice = lat
        self.nmodes_per_link = nmodes_per_link
        self.ncopies = 1

    def perm(self) -> np.ndarray:
        """
        Build the permutation matrix for transforming gamma_maj_sys to Gamma_in.

        The physical modes remain unchanged (identity mapping), while virtual modes
        are permuted according to the lattice structure and periodic boundary
        conditions.

        Returns:
            np.ndarray: Permutation matrix.
        """
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
                    d_i_perm = offset_physical_modes + 8 * size + (ny - 1) * nx * 8 + 8 * x
                dest[range(d_i_perm, d_i_perm + 4), range(d_j_perm, d_j_perm + 4)] = 1
                dest[range(u_i_perm, u_i_perm + 4), range(u_j_perm, u_j_perm + 4)] = 1
        return dest


if __name__ == "__main__":
    # print("Lattice 2d, 3x2")
    # lat_3x2 = Lattice2D(4, 5)
    # print(lat_3x2)
    # wilson_loop = lat_3x2.generate_wilson_loop((0, 0), (1, 1))
    # print(wilson_loop)
    # lst = lat_3x2.generate_tree()
    # print(lst)
    # print([lat_3x2.ind2coord_dir(ind) for ind in lst])
    # print(len(lst))
    # print("Lattice 2d, 3x3")
    # lat_3x3 = Lattice2D(2, 2)
    # print(lat_3x3)

    # def ind2coord_dir(self, ind: int) -> tuple:
    lat3x3 = Lattice2D(3, 3)
    wilson_loop1 = lat3x3.generate_wilson_loop((0, 0), (2, 2))
    print(wilson_loop1)
