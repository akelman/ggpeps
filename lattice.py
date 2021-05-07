import itertools
import numpy as np
from enum import Enum
import utils
import os
import sys
import logging


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

    def __str__(self):
        arr = np.arange(self.get_size())
        arr = np.reshape(arr, (self.nx, self.ny))
        return str(arr)

    def get_size(self):
        """Returns number of sites on the lattice"""
        return self.nx*self.ny


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
