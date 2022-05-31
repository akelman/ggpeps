import numpy as np
from enum import Enum
import sys
import logging
from scipy.linalg import block_diag


class Direction(Enum):
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

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self.nlinks = 2*nx*ny
        self.nplaquettes = nx*ny
        self.size = nx*ny # number of sites

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
            dest += ("{:02d}, ({:02d},{:02d}): {:02d},{:02d}\n".format(ind,
                     x, y, x_link, y_link))
        return dest

    def ind2coord(self, ind: int) -> tuple:
        """Conversion method from integer to lattice coordinate (a tuple).

        Args:
            ind (int): Index of a site

        Returns:
            tuple: Tuple of integers (x,y)
        """
        return (ind % self.nx, ind//self.nx)

    def coord2ind(self, coord: tuple) -> int:
        """Conversion method from coordinate tupels to lattice index (integer).

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
            return (((ind%(self.nx*self.ny)) % self.nx, (ind%(self.nx*self.ny))//self.nx),dir)
        elif dir == Direction.Y:
            return (((ind%(self.nx*self.ny)) // self.ny, (ind%(self.nx*self.ny)) % self.ny),dir)
        else:
            logging.error("ind2coord_dir: There are only X and Y as directions")
            return None

    def coord2ind_dir(self, coord: tuple, dir: Direction) -> int:
        """Conversion method from a coordinate tuple (x,y) and a direction to a link index.

        Args:
            coord (tuple): Coordinate tuple (x,y)
            dir (Direction): Direction of the link

        Returns:
            int: link index
        """
        x, y = coord
        if dir == Direction.X:
            return self.nx*self.ny*dir.value + self.nx * y + x
        elif dir == Direction.Y:
            return self.nx*self.ny*dir.value + self.ny * x + y
        else:
            logging.error(
                "coord2ind_dir: There are only X and Y as directions", file=sys.stderr)
            return None

    def get_neighbor(self, coord: tuple, orient: Direction) -> tuple:
        """Get the next coordinate tuple in a given direction (wraps around periodic boundary conditions)

        Args:
            coord (tuple): (x,y) coordinates of the original point
            orient (Direction): _description_

        Returns:
            tuple: (x,y) coordinate of the next point
        """
        # We assume periodic boundary conditions
        x, y = coord
        if dir == Direction.X:
            xn = (x+orient.value+self.nx) % self.nx
            yn = y
        elif dir == Direction.Y:
            xn = x
            yn = (y+orient.value+self.ny) % self.ny
        return (xn,yn)

    def generate_polyakov_loop(self, coord: tuple, dir: Direction, use_indices=True) -> list:
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
            #Build polyakov loop in X direction
            for i in range(self.nx):
                coord_link = ((i, y), dir)
                dest.append((coord_link, False))
        elif dir == Direction.Y:
            #Build polyakov loop in Y direction
            for i in range(self.ny):
                coord_link = ((x, i), dir)
                dest.append((coord_link, False))
        else:
            logging.error(
                "generate_polyakov_loop: There are only X and Y as directions")
            return None
        if use_indices:
            #Transform the coordinates to indices
            dest = [(self.coord2ind_dir(*coorddir), conj)
                    for (coorddir, conj) in dest]
        return dest

    def generate_wilson_loop(self, coord: tuple, size: tuple, use_indices=True) -> list:
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
        dest=[]
        for i in range(ext_x):
            coord_link = ((x+i) % self.nx, y)
            dest.append(((coord_link, Direction.X), False))
        for i in range(ext_y):
            coord_link = ((x+ext_x) % self.nx, (y+i) % self.ny)
            dest.append(((coord_link, Direction.Y), False))
        for i in range(ext_x):
            coord_link = ((x+ext_x-i-1) % self.nx, (y+ext_y) % self.ny)
            dest.append(((coord_link, Direction.X), True))
        for i in range(ext_y):
            coord_link = (x, (y+ext_y-i-1) % self.ny)
            dest.append(((coord_link, Direction.Y), True))
        if use_indices:
            #Transform the coordinates to indices
            dest=[(self.coord2ind_dir(*coorddir),conj) for (coorddir,conj) in dest]
        return dest


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
        self.nlinks = 3*nx*ny*nz
        self.size = nx*ny*nz # number of sites

    def __str__(self):
        arr = np.arange(self.get_size())
        arr = np.reshape(arr, (self.nx, self.ny))
        return str(arr)

    def get_size(self):
        """Returns number of sites on the lattice"""
        return self.nx * self.ny * self.nz

    def ind2coord(self,ind):
        x = ind % self.nx
        y = (ind % (self.nx * self.ny))//self.nx
        z = ind // (self.nx * self.ny)
        return (x, y, z)

    def coord2ind(self, coord):
        x, y, z = coord
        return z * self.nx * self.ny + self.nx * y + x

    def ind2coord_dir(self,ind):
        ind_coord = ind % (self.nx * self.ny * self.nz)
        x = ind_coord % self.nx
        y = (ind_coord % (self.nx * self.ny))//self.nx
        z = ind_coord // (self.nx * self.ny)
        dir = Direction(ind // (self.nx * self.ny * self.nz))
        return (x, y, z), dir

    def coord2ind_dir(self, coord, dir):
        x, y, z = coord
        return self.nx * self.ny * self.nz * dir.value + self.nx*self.ny*z + self.nx * y + x


class PermutationBuilderGMS2D:
    """Build a permutation matrix for gamma_maj_sys
    The default mode-order of the T matrix is {p,l,r,d,u}.
    By building the Dirac covariance matrix from it and transforming it to a Majorana matrix, we obtain a mode-order of {p,l,r,d,u,p_dag,l_dat,r_dag,u_dag,d_dag}.
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
    {l_0, r_0, d_0, u_0, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2, l_3, r_3, d_3, u_3}.
    The numbers in the basis above are the vertex indices.
    Gamma_in has the order
    {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.
    This class provides the necessary permutation matrix to change from one basis to the other.
    """
    def __init__(self, lat, nmodes_per_link):
        self.lattice = lat
        self.nmodes_per_link = nmodes_per_link

    def _perm_lr(self):
        #Number of Majoranas per link (2 per mode)
        maj_per_link = 2*self.nmodes_per_link
        # left and right Majoranas: 2 * maj_per_link
        single_block = np.eye(maj_per_link)
        empty_3x1= np.zeros((maj_per_link,3*maj_per_link))
        building_block = np.block([[empty_3x1,single_block],[single_block,empty_3x1]])
        matrix_body_no_pad = np.kron(np.eye(self.lattice.nx-1),building_block)
        # Pad the matrix body by the remaining block distributed left and right
        matrix_body = np.pad(matrix_body_no_pad, [[0, 0], [maj_per_link, 3 * maj_per_link]])
        # Prepare the block for the periodic boundary conditions
        # The factor 4 in the second argument is the coordination number of the lattice
        block_pbc = np.zeros((2*maj_per_link, 4*maj_per_link*self.lattice.nx))
        # Set the first left mode of the row
        block_pbc[:maj_per_link, :maj_per_link] = single_block
        # Set the last right mode of the row
        block_pbc[-maj_per_link:, -3*maj_per_link:-2*maj_per_link] = single_block
        dest = np.block([[matrix_body],[block_pbc]])
        return dest

    def _perm_du(self):
        #Number of Majoranas per link (2 per mode)
        maj_per_link = 2*self.nmodes_per_link
        # left and right Majoranas: 2 * maj_per_link
        single_block = np.eye(maj_per_link)
        empty_3x1= np.zeros((maj_per_link,3*maj_per_link))
        empty_nx = np.zeros((maj_per_link,(self.lattice.nx-1)*maj_per_link*4))
        building_block = np.block([[empty_nx, empty_3x1,single_block],[single_block,empty_nx,empty_3x1]])
        # pad left for all the modes in a row
        matrix_body_no_pad = np.kron(np.eye(self.lattice.ny-1),building_block)
        # Pad the matrix body by one block on the top and the bottom
        matrix_body = np.pad(matrix_body_no_pad, [[0, 0], [3 * maj_per_link, maj_per_link]])
        # Prepare the block for the periodic boundary conditions
        # We want the first left and the first right mode of the row as padding on the left
        block_pbc = np.zeros((2*maj_per_link,(self.lattice.nx*(self.lattice.ny-1)+1)*maj_per_link*4))
        block_pbc[:maj_per_link, 2*maj_per_link:3*maj_per_link] = single_block
        block_pbc[-maj_per_link:, -maj_per_link:] = single_block
        dest = np.block([[matrix_body],[block_pbc]])
        return dest

    def perm(self):
        perm_lr=self._perm_lr()
        perm_du=self._perm_du()
        #Permutation of the lr modes
        top_perm=np.kron(np.eye(self.lattice.ny),perm_lr)
        #Permutation for the ud modes
        m_du,n_du=perm_du.shape
        maj_per_link=2*self.nmodes_per_link
        bottom_perm=np.zeros_like(top_perm)
        for y in range(self.lattice.nx):
            offset = 4*y*maj_per_link  # One vertex has 4 links to other vertices
            bottom_perm[y*m_du:y*m_du+m_du,offset:offset+n_du]=perm_du
        # Add the physical modes to the matrix. They do not get permuted.
        # We assume that there is one physical mode per site
        phys_block = np.eye(self.lattice.nx*self.lattice.ny*2)
        perm_virt = np.block([[top_perm], [bottom_perm]])
        dest = block_diag(phys_block,perm_virt)
        return dest

class PermutationBuilderGMS2D2C:
    """Build a permutation matrix for gamma_maj_sys with 2 copies
    The default mode-order of the T matrix is {p,l1,r1,d1,u1,l2,r2,d2,u2}.
    By building the Dirac covariance matrix from it and transforming it to a Majorana matrix, we obtain a mode-order of {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dag,r2_dag,u2_dag,d2_dag}.
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
    def __init__(self, lat, nmodes_per_link):
        self.lattice = lat
        self.nmodes_per_link = nmodes_per_link
        self.ncopies = 2

    def _perm_lr(self):
        #We are building this matrix top to bottom
        #Number of Majoranas per link (2 per mode)
        maj_per_link = 2*self.nmodes_per_link
        # left and right Majoranas: 2 * maj_per_link
        single_block = np.eye(maj_per_link)
        empty_1x3 = np.zeros((maj_per_link, 3 * maj_per_link))
        empty_2x4 = np.zeros((2 * maj_per_link, 4 * maj_per_link))
        empty_1x4 = np.zeros((maj_per_link, 4 * maj_per_link))
        # TODO: Change for more than 2 copies
        building_block_single_copy = np.block([[empty_1x3,empty_1x4,single_block],[single_block,empty_1x4,empty_1x3]])
        # Duplicate and shift for the second copy
        # TODO: Change for more than 2 copies
        building_block_no_pad = np.block(
            [[building_block_single_copy, empty_2x4],
             [empty_2x4, building_block_single_copy]])
        # Pad the matrix body by the remaining block distributed left and right
        building_block = np.pad(building_block_no_pad, [[0, 0], [maj_per_link, 3 * maj_per_link]])
        #Shift the building block to the other pairs of sites
        m_bb, n_bb = building_block.shape
        matrix_body = np.zeros(((self.lattice.nx-1)*self.ncopies*2*maj_per_link,self.ncopies*self.lattice.nx*4*maj_per_link))
        for x in range(self.lattice.nx-1):
            offset = self.ncopies * x * 4 * maj_per_link  # One vertex has 4 links to other vertices
            matrix_body[x*m_bb:x*m_bb+m_bb,offset:offset+n_bb]=building_block

        # Prepare the block for the periodic boundary conditions
        # The factor 4 in the second argument is the coordination number of the lattice
        block_pbc = np.zeros((2*maj_per_link, 4*maj_per_link*(self.ncopies*self.lattice.nx-1)))
        # Set the first left mode of the row
        block_pbc[:maj_per_link, :maj_per_link] = single_block
        # Set the last right mode of the row
        block_pbc[-maj_per_link:, -3*maj_per_link:-2*maj_per_link] = single_block
        # Duplicate and shift for the second copy
        # TODO: Change for more than 2 copies
        block_pbc = np.block([[block_pbc, empty_2x4], [empty_2x4, block_pbc]])
        dest = np.block([[matrix_body],[block_pbc]])
        return dest

    def _perm_du(self):
        #We are building this matrix top to bottom.
        #Number of Majoranas per link (2 per mode)
        maj_per_link = 2*self.nmodes_per_link
        # left and right Majoranas: 2 * maj_per_link
        single_block = np.eye(maj_per_link)
        empty_3x1 = np.zeros((maj_per_link, 3 * maj_per_link))
        empty_2x4 = np.zeros((2 * maj_per_link, 4 * maj_per_link))
        empty_nx = np.zeros((maj_per_link,(self.ncopies*self.lattice.nx-1)*maj_per_link*4))
        building_block_single_copy = np.block([[empty_nx, empty_3x1,single_block],[single_block,empty_nx,empty_3x1]])
        #Duplicate for the second copy
        building_block_no_pad = np.block(
            [[building_block_single_copy, empty_2x4],
             [empty_2x4, building_block_single_copy]])
        # Pad the matrix body by one block on the top and the bottom
        building_block = np.pad(building_block_no_pad, [[0, 0], [3 * maj_per_link, maj_per_link]])
        m_bb, n_bb = building_block.shape
        #Shift the building block to the other pairs of sites
        matrix_body = np.zeros(((self.lattice.ny-1)*self.ncopies*2*maj_per_link,(self.ncopies*(self.lattice.ny-1)*self.lattice.nx+2)*4*maj_per_link))
        for y in range(self.lattice.ny-1):
            offset = self.ncopies * y * self.lattice.nx * 4 * maj_per_link  # One vertex has 4 links to other vertices
            matrix_body[y*m_bb:y*m_bb+m_bb,offset:offset+n_bb]=building_block
        # Prepare the block for the periodic boundary conditions
        # We want the first left and the first right mode of the row as padding on the left
        # TODO: Check whether the -3 is ncopy dependent
        block_pbc = np.zeros((2*maj_per_link,((self.ncopies*(self.lattice.ny-1)*self.lattice.nx+2)-1)*4*maj_per_link))
        block_pbc[:maj_per_link, 2*maj_per_link:3*maj_per_link] = single_block
        block_pbc[-maj_per_link:, -maj_per_link:] = single_block
        # Duplicate and shift for the second copy
        # TODO: Change for more than 2 copies
        block_pbc = np.block([[block_pbc, empty_2x4], [empty_2x4, block_pbc]])
        dest = np.block([[matrix_body],[block_pbc]])
        return dest

    def perm(self):
        perm_lr=self._perm_lr()
        perm_du=self._perm_du()
        #Permutation of the lr modes
        top_perm=np.kron(np.eye(self.lattice.ny),perm_lr)
        #Permutation for the ud modes
        m_du,n_du=perm_du.shape
        maj_per_link=2*self.nmodes_per_link
        bottom_perm=np.zeros_like(top_perm)
        for y in range(self.lattice.nx):
            offset = 4 * y * maj_per_link * self.ncopies  # One vertex has 4 links to other vertices
            bottom_perm[y*m_du:y*m_du+m_du,offset:offset+n_du]=perm_du
        # Add the physical modes to the matrix. They do not get permuted.
        # We assume that there is one physical mode per site
        phys_block = np.eye(self.lattice.nx*self.lattice.ny*2)
        perm_virt = np.block([[top_perm], [bottom_perm]])
        dest = block_diag(phys_block,perm_virt)
        return dest

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
        dest[range(2*size), range(2*size)] = 1
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
                dest[range(l_i_perm,l_i_perm+4),range(l_j_perm,l_j_perm+4)]=1
                dest[range(r_i_perm,r_i_perm+4),range(r_j_perm,r_j_perm+4)]=1

                # Treatment of the down and up modes
                d_j_perm = offset_site + 8
                d_i_perm = offset_physical_modes + 8 * size + (y - 1) * nx * 8 + x * 8
                u_j_perm = offset_site + 12
                u_i_perm = offset_physical_modes + 8 * size + y * nx * 8 + 8 * x + 4
                if y == 0:
                    # We have to add the periodic boundary condition here for the down mode
                    d_i_perm = offset_physical_modes + 8 * size + (ny - 1) * nx * 8 + 8 * x;
                dest[range(d_i_perm,d_i_perm+4),range(d_j_perm,d_j_perm+4)]=1
                dest[range(u_i_perm,u_i_perm+4),range(u_j_perm,u_j_perm+4)]=1
        return dest


if __name__ == "__main__":
    print("Lattice 2d, 3x2")
    lat_3x2 = Lattice2D(3, 2)
    print(lat_3x2)
    wilson_loop = lat_3x2.generate_wilson_loop((0,0), (1,1))
    print(wilson_loop)