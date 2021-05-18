import itertools
import numpy as np
from enum import Enum
import utils
import os
import sys
import logging
from scipy.linalg import block_diag


class Direction(Enum):
    X = 0
    Y = 1
    Z = 2

    def __str__(self):
        return self.name

class Lattice2D:
    """
    Handler of a square lattice of size nx x ny
    """

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.nlinks = 2*nx*ny
        self.nplaquettes = nx*ny
        self.size=nx*ny

    def __str__(self):
        dest=""
        for ind in range(self.nplaquettes):
            x,y=self.ind2coord(ind)
            x_ind=self.coord2ind_dir((x,y),Direction.X)
            y_ind=self.coord2ind_dir((x,y),Direction.Y)
            dest+=("{:02d}, ({:02d},{:02d}): {:02d},{:02d}\n".format(ind,x,y,x_ind,y_ind))
        return dest

    def ind2coord(self,ind):
        return (ind % self.nx, ind//self.nx)

    def coord2ind(self, coord):
        x, y = coord
        return self.nx * y + x

    def ind2coord_dir(self,ind):
        dir=Direction(ind//(self.nx*self.ny))
        if dir == Direction.X:
            return (((ind%(self.nx*self.ny)) % self.nx, (ind%(self.nx*self.ny))//self.nx),dir)
        elif dir == Direction.Y:
            return (((ind%(self.nx*self.ny)) // self.ny, (ind%(self.nx*self.ny)) % self.ny),dir)
        else:
            print("ind2coord_dir: There are only X and Y as directions",file=sys.stderr)
            return None

    def coord2ind_dir(self, coord, dir):
        x, y = coord
        if dir == Direction.X:
            return self.nx*self.ny*dir.value + self.nx * y + x
        elif dir == Direction.Y:
            return self.nx*self.ny*dir.value + self.ny * x + y
        else:
            print("coord2ind_dir: There are only X and Y as directions",file=sys.stderr)
            return None
    
    def get_neighbor(self,coord,orient):
        # We assume periodic boundary conditions 
        x, y = coord
        if dir==Direction.X:
            xn = (x+orient.value+self.nx) % self.nx
            yn=y 
        elif dir==Direction.Y:
            xn=x
            yn = (y+orient.value+self.ny) % self.ny
        return (xn,yn)

    def generate_polyakov_loop(self,coord,dir,use_indices=True):
        x, y = coord
        dest=[]
        if dir==Direction.X:
            #Build polyakov loop in X direction
            for i in range(self.nx):
                coord_link = ((i, y), dir)
                dest.append((coord_link,False))
        elif dir==Direction.Y:
            #Build polyakov loop in Y direction
            for i in range(self.ny):
                coord_link = ((x, i), dir)
                dest.append((coord_link, False))
        else:
            print("generate_polyakov_loop: There are only X and Y as directions",file=sys.stderr)
            return None
        if use_indices:
            #Transform the coordinates to indices
            dest=[(self.coord2ind_dir(*coorddir),conj) for (coorddir,conj) in dest]
        return dest

    def generate_wilson_loop(self,coord,size,use_indices=True):
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
    Handler of a square lattice of size nx x ny x nz
    """

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nlinks = 3*nx*ny*nz

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
    """
    def __init__(self, lat, nmodes_per_link):
        self.lattice = lat
        self.nmodes_per_link = nmodes_per_link

    def _perm_lr(self):
        #Number of Majoranas per link (2 per mode)
        maj_per_link = 2*self.nmodes_per_link
        # left and right Majoranas: 2 * maj_per_link
        single_block = np.eye(maj_per_link)
        double_block = np.eye(2*maj_per_link)
        # pad right for the down and up modes
        mode_block = np.hstack([double_block, np.zeros((2*maj_per_link, 2*maj_per_link))])
        # This is the main part for all modes in the chain (w/o periodic boundary conditions)
        matrix_body = np.kron(np.eye(self.lattice.nx - 1), mode_block)
        # Pad the matrix body by one block on the top and the bottom
        matrix_body = np.pad(matrix_body,[[maj_per_link,maj_per_link],[0,0]])
        # Prepare the block for the periodic boundary conditions
        # We want the first left and the first right mode of the row as padding on the left
        block_pbc = np.zeros((2*self.lattice.nx*maj_per_link, 4*maj_per_link))
        block_pbc[:maj_per_link, maj_per_link:2*maj_per_link] = single_block
        block_pbc[-maj_per_link:, :maj_per_link] = single_block
        dest = np.block([block_pbc, matrix_body])
        return dest

    def _perm_du(self):
        #Number of Majoranas per link (2 per mode)
        maj_per_link = 2*self.nmodes_per_link
        # left and right Majoranas: 2 * maj_per_link
        single_block = np.eye(maj_per_link)
        double_block = np.eye(2*maj_per_link)
        # pad left for all the modes in a row 
        mode_block = np.hstack([np.zeros((2*maj_per_link, (4*self.lattice.nx-2)*maj_per_link)),double_block])
        matrix_body = np.kron(np.eye(self.lattice.ny-1), mode_block)
        # Pad the matrix body by one block on the top and the bottom
        matrix_body = np.pad(matrix_body,[[maj_per_link,maj_per_link],[0,0]])
        # Prepare the block for the periodic boundary conditions
        # We want the first left and the first right mode of the row as padding on the left
        block_pbc = np.zeros((2*self.lattice.ny*maj_per_link, 4*maj_per_link))
        block_pbc[:maj_per_link, -maj_per_link:] = single_block
        block_pbc[-maj_per_link:, -2*maj_per_link:-maj_per_link] = single_block
        dest = np.block([block_pbc, matrix_body])
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


if __name__ == "__main__":
    print("Lattice 2d, 3x2")
    lat_3x2=Lattice2D(3,2)
    print(lat_3x2)
