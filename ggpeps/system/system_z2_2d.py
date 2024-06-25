import logging
import sympy
#import numpy as np
from ggpeps import xnp as np
from scipy.linalg import block_diag
from pfapack import pfaffian as pf
from warnings import warn # Used for deprecation warnings

import ggpeps
from ggpeps import utils
from ggpeps import lattice as lat
from ggpeps.lattice import Direction
from ggpeps.modearray import generate_permutation_matrix
from .system_base import Config2DBase, System2DBase
from .system_base import calculate_lognorm, calculate_lognormvec, extract_partial_covmats, calculate_lognorm_inc

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################

class Z2System2DConfig(Config2DBase):
    _nparams = 6
    ncopy = 1
    nvirtmodes_vertex = 4 # We have one virtual mode per direction (1 mode x 4 directions)
    nvirtmodes_link = 2 # We have two virtual modes per link (l/r or u/d)

    def __init__(self, lattice, g_el, g_mag, g_int, g_mass, nlayer=1):
        #The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        super().__init__(lattice, g_el, g_mag, g_int, g_mass, nlayer)

    def make_pure_gauge(self):
        #The order of the parameters is [tr,yr,zr,ti,yi,zi] ({r,i} referring to the real/imaginary components)
        for ind in range(self.nlayer):
            # t real
            self.paramvec[ind, 0] = 0
            # t imag
            self.paramvec[ind, 3] = 0


class Z2System2D(System2DBase):
    """ Single copy (referring to the number of virtual modes on the links) of the Z2 GGPEPS ansatz

        Some general notes about conventions:

        Order of the paramvec: [tr,yr,zr,ti,yi,zi]   # We split the real and the imaginary part of the parameters into independent variables
        Mode order of T: {p,l,r,d,u}
        Mode Order of gamma_dirac:  {p,l,r,d,u,p_dag,l_dag,r_dag,d_dag,u_dag}.
        Mode Order of gamma_maj: {p_1,p_2,l_1,l_2,r_1,r_2,d_1,d_2,u_1,u_2}.
        The subscript indices are Majorana mode indices here.
    Args:
        System2DBase ([type]): [description]
    """
    def __init__(self, cfg: Z2System2DConfig):
        super().__init__(cfg)

        # constants used in the calculation of the electric energy
        prefactors = [[1, -1, 1.j, 1.j]]
        indices = [[(2,0), (3,1), (0,1), (2,3)]]
        idxarr = self.get_pfaffian_arrays(indices, prefactors)
        self.idxarr_vec = [idxarr]*self.cfg.nlayer
        self.el_overall_factors = [-1j/4]*self.cfg.nlayer # this arises due to normalization and the i^(# of modes/2) in the expression Tr[i^# * rho * (modes)]

    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the covariance matrices analytically.

        This method overwrites an abstract method in System2DBase.

        Returns:
            list: List of all analytic symbols
        """
        tr = sympy.Symbol("tr", real=True)
        yr = sympy.Symbol("yr", real=True)
        zr = sympy.Symbol("zr", real=True)
        ti = sympy.Symbol("ti", real=True)
        yi = sympy.Symbol("yi", real=True)
        zi = sympy.Symbol("zi", real=True)
        return [tr,yr,zr, ti, yi, zi]


    @property
    def tmat_symb(self):
        """Definition of the symbolic T matrix.
        The definition of T here is a result of an analytic consideration of global symmetries like rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the covariance matrices analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code. 
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l, r, d, u

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone for errors.

        This method overwrites an abstract method in System2DBase.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """
        [tr, yr, zr, ti, yi, zi] = self.symbolvec
        t = tr+1.j*ti
        y = yr+1.j*yi
        z = zr+1.j*zi
        tmat_symb=sympy.Matrix([[0, -1.j * t, 1.j * t, t, -t],
                            [1.j * t, 0, 1.j * y, z, 1.j * z],
                            [-1.j * t, -1.j * y, 0, -1.j * z, -z],
                            [-t, -z, 1.j * z, 0, -y], 
                            [t, -1.j * z, z, y, 0]])
        return tmat_symb


    def initialize_gamma_in_sys(self):
        """ 
        The mode-order in gamma_in_sys is dictated by the numbering of the links on the lattice.
        The numbering guarantees that we split the vertical from the horizontal links for easier gauging.

            |         |
            "5"       "7"
            |         |
            2 --"2"-- 3 --"3"--
            |         |
            "4"       "6"
            |         |
            0 --"0"-- 1 --"1"--

        The vertex indices are written as <number>, the link indices are written as "<number>". 

        For a 2x2 system, gamma_in has the order {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.
        The modes are named as <mode letter>_<vertex site>. Each constitent in the list above labels two Majorana modes.

        This method overwrites an abstract method in System2DBase.
        """

        size = self.cfg.lattice.size # number of sites

        # Initialize gamma_in_sys for the full system 
        id = np.eye(size) 
        neutral_gauge_X = np.kron( id, self.gamma_gauge_neutral[0][Direction.X] ) # just use the first gamma_gauge_neutral, since they're shared by all layers
        neutral_gauge_Y = np.kron( id, self.gamma_gauge_neutral[0][Direction.Y] )
        gamma_in_sys = block_diag(neutral_gauge_X, neutral_gauge_Y) # for the 3D case, simply add in the Z covariance matrix as well

        # Initialize all the trackers of inverses and determinants
        diffvec = [
            mat_d_inv - gamma_in_sys for mat_d_inv in self.mat_d_inv_vec
        ]
        wi_gamma_in_vec = [utils.WoodburyInverter(diff) for diff in diffvec]
        wi_gamma_out_vec = [
            utils.WoodburyInverter(mat_d - gamma_in_sys)
            for mat_d in self.mat_d_vec
        ]
        incdet_vec = [utils.IncLogAbsDeterminant(diff) for diff in diffvec]

        # Initialize the modified gamma_in_sys for the full system (and trackers)
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        gamma_in_sys_mod = gamma_in_sys[single_link_offset:, single_link_offset:]
        diffvec_mod = [
            mat_d_inv - gamma_in_sys_mod for mat_d_inv in self.mat_d_mod_inv_vec
        ]
        wi_gamma_in_mod_vec = [utils.WoodburyInverter(diff) for diff in diffvec_mod]
        wi_gamma_out_mod_vec = [
            utils.WoodburyInverter(mat_d - gamma_in_sys_mod)
            for mat_d in self.mat_d_mod_vec
        ]
        incdet_mod_vec = [utils.IncLogAbsDeterminant(diff) for diff in diffvec_mod]

        # Though for this ansatz gamma_in_sys does not vary between layers, it is convenient to have gamma_in_sys_vec available as a vector with length = nlayers
        # for general methods in system base
        gamma_in_sys_vec = [gamma_in_sys]*self.cfg.nlayer

        return gamma_in_sys_vec, (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec), (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec)


    def _generate_gamma_gauge_neutral_dict(self):
        """This matrix is the covariance matrix of the ungauged projectors.
        The mode order is {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2}, where the underscore notation explicitly denotes Majorana modes and not sites.
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Returns:
            np.ndarray: Covariance matrix of the ungauged projector on a single link
        """
        dest={}
        dest[Direction.X] = np.real_if_close(1.j*np.kron(utils.pauliy, utils.paulix)) # this just happens to be a convenient way to generate the covariance matrix that was calculated by hand
        dest[Direction.Y] = np.real_if_close(np.kron(1.j*utils.pauliy, utils.pauliz))
        return [dest]*self.cfg.nlayer

    #Gauging

    def generate_rotmat(self, theta, coord, dir):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.
        The mode order is (as for gamma_in_neutral) {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2}, depending on whether the link is vertical or horizontal.

        This method overwrites an abstract method in System2DBase.

        Args:
            theta (float): Angle of rotation
            coord (tuple): (x,y) coordinate on the lattice

        Returns:
            np.ndarray: Rotation matrix for gamma_in_neutral
        """
        # TODO: Do we want to stagger here?
        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        rot_right = np.array([[np.cos(theta), np.sin(theta)],
                              [-np.sin(theta), np.cos(theta)]])
        # We have only one left mode => 2 Majorana modes
        rot_left = np.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        dest = block_diag(rot_left, rot_right)
        return dest


    def update_gauge_ind(self, link_ind: int, theta: float):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be prohibitively expensive.

        This method overwrites an abstract method in System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (float): New gauge field value
        """
        # Update the gaugefield
        self._gaugefieldvec[link_ind] = theta
        # There are two directions per vertex and two Majoranas per link
        ind_mat = 4 * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)
        gamma_neutral_gauge = self.gamma_gauge_neutral[0][dir]  # calling every time (rather than storing) will cause some innefficiency 
                                                                # just use the first gamma_gauge_neutral, since they're shared by all layers
        gamma_in_subst = rotmat @ gamma_neutral_gauge @ np.transpose(
            rotmat)

        update = self.calculate_update_gamma_in(ind_mat, gamma_in_subst)
        # Update the determinant
        mat_inv_vec = [
            wi_gamma_in.inv() for wi_gamma_in in self.wi_gamma_in_vec
        ]
        detval_vec = [
            incdet.update_index(mat_inv, update, ind_mat, ind_mat)
            for mat_inv, incdet in zip(mat_inv_vec, self.incdet_vec)
        ]
        # Update the modified determinant
        offset = 2* self.cfg.nvirtmodes_link
        if ind_mat - offset >=0:
            for wi, incdet in zip(self.wi_gamma_in_mod_vec,self.incdet_mod_vec):
                mat_inv = wi.inv()
                incdet.update_index(mat_inv, update, ind_mat-offset, ind_mat-offset)
        # Update the weight
        self.weight = 0.5 * np.sum(detval_vec)
        # Update the matrix inversion
        [ wi_gamma_in.update_index(update, ind_mat, ind_mat) for wi_gamma_in in self.wi_gamma_in_vec ]
        [ wi_gamma_out.update_index(update, ind_mat, ind_mat) for wi_gamma_out in self.wi_gamma_out_vec ]

        if ind_mat - offset >= 0:
            # We do not update the matrix if the first link is updated (it is just not there)
            [ wi_gamma_in_mod.update_index(update, ind_mat-offset, ind_mat-offset) for wi_gamma_in_mod in self.wi_gamma_in_mod_vec ]
            [ wi_gamma_out_mod.update_index(update, ind_mat-offset, ind_mat-offset) for wi_gamma_out_mod in self.wi_gamma_out_mod_vec ]
        # Substitute in the array
        self.gamma_in_sys[ind_mat:ind_mat + 4,
                          ind_mat:ind_mat + 4] = gamma_in_subst
        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()


    # Calculating the norm

    def _compute_mass_energy_op_vec_and_grad(self, use_trans_inv=True):
        energies = [0]*self.cfg.nlayer
        gradients = [ [0]*self.cfg.nparams_per_layer for k in range(self.cfg.nlayer) ]
        return np.array(energies), np.array(gradients)
        # This function is not implemented yet! 
        # (and it can't be, because the ansatz doesn't have the required parameterization).
        # We return zeros just to not break the interface.
        raise NotImplementedError("The mass energy is not implemented yet for the 2 copy case.")

    def _compute_mag_energy_op(self, use_trans_inv=True):
        """Computation of the magnetic energy operator (w/o shift).
        This operator is diagonal in the gauge field (group element) basis and can thus be computed easily.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant computation method. Defaults to True.

        Returns:
            float: magnetic energy w/o shift for a single plaquette
        """
        if use_trans_inv:
            # Evaluate one plaquette and multiply by number of plaquettes
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop(
                (0, 0), (1, 1))
            mag_energy_bare = np.real(
                self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            logger.error("compute_mag_energy: not implemented yet")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")
            mag_energy_bare = None
        return mag_energy_bare
    
    def _compute_int_energy_op_vec_and_grad(self):
        energies = [0]*self.cfg.nlayer
        gradients = [ [0]*self.cfg.nparams_per_layer for k in range(self.cfg.nlayer) ]
        return np.array(energies), np.array(gradients)
        # This function is not implemented yet! 
        # (and it can't be, because the ansatz doesn't have the required parameterization).
        # We return zeros just to not break the interface.
        raise NotImplementedError("The interaction energy is not implemented yet for the 2 copy case.")
        