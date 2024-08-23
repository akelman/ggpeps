import sympy
import logging
from pfapack import pfaffian as pf
from typing import List

import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.global_funcs import *

from .system_base import Config2DBase, System2DBase
from .system_base import get_pfaffian_arrays
from .system_base import calculate_lognorm_inc

#from ggpeps.system.global_funcs import update_gauge_ind

logger = logging.getLogger(ggpeps.LOGGER_NAME)

###################### Z2System2D ##########################

class Z2System2D_G2C_F2C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 2 copies of virtual fermions on the links per layer.
    Each layer can either be pure-gauge (in which case the t-params are zeroed out), 
    or fermionic (in which case the y,z-params are zeroed out).

    Some general notes about conventions:
    
    Order of the paramvec: [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di].
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2}.
    """

    _nparams = 20
    ncopy = 2
    nvirtmodes_vertex = 8
    nvirtmodes_link = 4

    def __init__(self, lattice, g_el, g_mag,  g_int, g_mass, g_chem, num_pg_layer=1, num_fermionic_layer=1):
        super().__init__(lattice, g_el, g_mag, g_int, g_mass, g_chem, num_pg_layer, num_fermionic_layer)

        # Constants used in the calculation of the electric energy
        prefactors = [[1, -1, 1.j, 1.j], [1, -1, 1.j, 1.j]]
        indices_layer_pg = [[(2,4), (3,5), (4,5), (2,3)], [(6,0), (7,1), (0,1), (6,7)]]
        indices_layer_fermionic = [[(2,0), (3,1), (0,1), (2,3)], [(6,4), (7,5), (4,5), (6,7)]]
        idxarr_lay_pg = get_pfaffian_arrays(indices_layer_pg, prefactors)
        idxarr_lay_fermionic = get_pfaffian_arrays(indices_layer_fermionic, prefactors) 
        self.idxarr_vec = [idxarr_lay_pg]*self.num_pg_layer + [idxarr_lay_fermionic]*self.num_fermionic_layer
        self.el_overall_factors = [-1/16]*self.nlayer # this arises due to normalization and the i^(# of modes/2) in the expression Tr[i^# * rho * (modes)]

    def make_pure_gauge(self):
        """Make the ansatz pure gauge by setting t-params to zero.

        This function is obsolete for this ansatz, and is kept for compatibility reasons.
        """
        t_indices = [0,3,10,13] # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.nlayer):
            for t_ind in t_indices:
                coord = (layer_ind, t_ind)
                self.paramvec[coord] = 0
    
    def enforce_parameter_conditions(self, mat):
        """Enforce conditions on parameters on each layer to get the required behaviour for the ansatz.
        """
        # The order of the parameters (for each layer) is [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di]

        zeroed_params = [] # we'll save the indices of the zeroed parameters

        t_indices = [0,3,10,13] # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.num_pg_layer):
            for t_ind in t_indices:
                coord = (layer_ind, t_ind)
                if isinstance(mat, np.ndarray): # TODO: handle jax better
                    mat[coord] = 0
                else:
                    mat.at[coord].set(0)
                zeroed_params.append(coord)
        
        zero_for_fermionic_layer = [3,13,1,2,4,5,11,12,14,15] # index of t2r, t2i, y1r, z1r, y2r, z2r, y1i, z1i, y2i, z2i in symbolvec
        for layer_ind in range(self.num_pg_layer, self.nlayer):
            for ind in zero_for_fermionic_layer:
                coord = (layer_ind, ind)
                if isinstance(mat, np.ndarray):
                    mat[coord] = 0
                else:
                    mat.at[coord].set(0)
                zeroed_params.append(coord)
        
        # save zeroed params
        self.zeroed_params = zeroed_params
        return

    def _create_symbolvec(self) -> List[sympy.Symbol]:
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the covariance matrices analytically.

        Returns:
            list: List of all analytic symbols
        """
        t1r = sympy.Symbol("t1r", real=True)
        y1r = sympy.Symbol("y1r", real=True)
        z1r = sympy.Symbol("z1r", real=True)
        t2r = sympy.Symbol("t2r", real=True)
        y2r = sympy.Symbol("y2r", real=True)
        z2r = sympy.Symbol("z2r", real=True)
        ar  = sympy.Symbol("ar", real=True)
        br  = sympy.Symbol("br", real=True)
        cr  = sympy.Symbol("cr", real=True)
        dr  = sympy.Symbol("dr", real=True)

        t1i = sympy.Symbol("t1i", real=True)
        y1i = sympy.Symbol("y1i", real=True)
        z1i = sympy.Symbol("z1i", real=True)
        t2i = sympy.Symbol("t2i", real=True)
        y2i = sympy.Symbol("y2i", real=True)
        z2i = sympy.Symbol("z2i", real=True)
        ai  = sympy.Symbol("ai", real=True)
        bi  = sympy.Symbol("bi", real=True)
        ci  = sympy.Symbol("ci", real=True)
        di  = sympy.Symbol("di", real=True)
        return [t1r, y1r, z1r, t2r, y2r, z2r, ar, br, cr, dr, t1i, y1i, z1i, t2i, y2i, z2i, ai, bi, ci, di]

    @property
    def tmat_symb(self):
        """Definition of the symbolic T matrix.
        The definition of T here is a result of an analytic consideration of global symmetries like rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the covariance matrices analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code. 
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone to errors.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """
        [t1r, y1r, z1r, t2r, y2r, z2r, ar, br, cr, dr, t1i, y1i,
            z1i, t2i, y2i, z2i, ai, bi, ci, di] = self.symbolvec
        t1 = t1r+1.j*t1i
        y1 = y1r+1.j*y1i
        z1 = z1r+1.j*z1i
        t2 = t2r+1.j*t2i
        y2 = y2r+1.j*y2i
        z2 = z2r+1.j*z2i
        a = ar+1.j*ai
        b = br+1.j*bi
        c = cr+1.j*ci
        d = dr+1.j*di
        tmat_symb=sympy.Matrix([
            [0, -1.j*t1, 1.j*t1, t1, -t1, -1.j*t2, 1.j*t2, t2, -t2],
            [1.j*t1, 0, 1.j*y1, z1, 1.j*z1, -1.j*a, -1.j*c, -1.j*b, -1.j*d],
            [-1.j*t1, -1.j*y1, 0, -1.j*z1, -z1, 1.j*c, 1.j*a, 1.j*d, 1.j*b],
            [-t1, -z1, 1.j*z1, 0, -y1, d, b, a, c],
            [t1, -1.j*z1, z1, y1, 0, -b, -d, -c, -a],
            [1.j*t2, 1.j*a, -1.j*c, -d, b, 0, 1.j*y2, z2, 1.j*z2],
            [-1.j*t2, 1.j*c, -1.j*a, -b, d, -1.j*y2, 0, -1.j*z2, -z2],
            [-t2, 1.j*b, -1.j*d, -a, c, -z2, 1.j*z2, 0, -y2],
            [t2, 1.j*d, -1.j*b, -c, a, -1.j*z2, z2, y2, 0]
            ])
        return tmat_symb
    
    def generate_gamma_gauge_neutral_dict(self):
        """Generate the covariance matrix of the ungauged projectors.
        The mode order is {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2, d2_1, d2_2, u2_1, u2_2}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy. 
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This function returns two different covariance matrices for ungauged projectors:
        In the first, modes of copy 1 are coupled to modes of copy 2. 
        In the second, the projectors don't mix copies.
        The first option is used for the pure-gauge layer, the second for the fermionic layer.

        This method overwrites an abstract method in System2DBase.

        Returns:
            List[xnp.ndarray]: Covariance matrices of the ungauged projector on a single link
        """
        
        # 2 if for 2D lattice
        dest_mixed = [0]*2 # mixes copies
        dest_unmixed = [0]*2 # does not mix copies 
        
        # We want to give the projectors for the pure gauge part, which mix copies
        dest_mixed[Direction.X] = np.real_if_close(1.j*np.kron(utils.paulix,np.kron(utils.pauliy, utils.paulix)))
        dest_mixed[Direction.Y] = np.real_if_close(1.j*np.kron(utils.paulix,np.kron(utils.pauliy, utils.pauliz)))

        # We want to give the projectors for the fermionic part which don't mix copies (so as to preserve global U(1) symmetry)
        dest_unmixed[Direction.X] = np.array([ [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                                                [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.]])

        dest_unmixed[Direction.Y] = np.array([ [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0., -1.,  0., -0.,  0.,  0.],
                                                [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0., -0.,  0., -1.],
                                                [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]])

        return np.array([dest_mixed]*self.num_pg_layer + [dest_unmixed]*self.num_fermionic_layer)


class Z2System2D(System2DBase):
    """ 2D Z2 system GGPEPS ansatz with physical fermions.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec in the configs.
        We split the real and the imaginary part of the parameters into independent variables.
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3... and so on}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3... and so on}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2... and so on}.
    """

    def __init__(self, cfg):
        """Constructor of a Z2System2D system, with any number of virtual fermions per site per link 
        (provided a valid config is given).

        Args:
            cfg (Config2DBase): Configuration containing all system-related parameters
        """
        super().__init__(cfg)

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

        For a 2x2 system with 1 copy (one virtual fermions per site per link), gamma_in has the order 
        {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.

        For a 2x2 system with 2 copies (two virtual fermions per site per link), gamma_in has the order 
        { l1_1, r2_0, l1_1, r2_0, l1_0, r2_1, l1_0, r2_1,  
          l1_3, r2_2, l1_3, r2_2, l1_2, r2_3, l1_2, r2_3,  
          d1_2, u2_0, d1_2, u2_0, d1_0, u2_2, d1_0, u2_2,  
          d1_3, u2_1, d1_3, u2_1, d1_1, d2_3, d1_1, d2_3 }.

        The naming convention here is <mode letter><number of copy>_<vertex index>.
        (<number of copy> is ommitted for the 1 copy case).
        Each constituent in the lists above refers to two Majorana modes.

        This method overwrites an abstract method in System2DBase.
        """

        # Initialize empty lists
        gamma_in_sys_vec = []
        wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec = [], [], []
        wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec = [], [], []

        # Initialize gamma_in_sys for the full system (and trackers)
        size = self.cfg.lattice.size # number of sites
        id = xnp.eye(size) 

        # TODO: vectorize!
        for layer in range(self.cfg.nlayer):
            neutral_gauge_X = xnp.kron( id, self.gamma_gauge_neutral_vec[layer][Direction.X] )
            neutral_gauge_Y = xnp.kron( id, self.gamma_gauge_neutral_vec[layer][Direction.Y] )
            gamma_in_sys = xscipy.linalg.block_diag(neutral_gauge_X, neutral_gauge_Y)
            gamma_in_sys_vec.append(gamma_in_sys)

            wi_gamma_in_vec.append( utils.WoodburyInverter(self.mat_d_inv_vec[layer] - gamma_in_sys) )
            wi_gamma_out_vec.append( utils.WoodburyInverter(self.mat_d_vec[layer] - gamma_in_sys) )
            incdet_vec.append( utils.IncLogAbsDeterminant(self.mat_d_inv_vec[layer] - gamma_in_sys) )

            # Initialize the modified gamma_in_sys for the full system (and trackers)
            single_link_offset = 2 * self.cfg.nvirtmodes_link
            gamma_in_sys_mod = gamma_in_sys[single_link_offset:, single_link_offset:]
            wi_gamma_in_mod_vec.append( utils.WoodburyInverter(self.mat_d_mod_inv_vec[layer] - gamma_in_sys_mod) )
            wi_gamma_out_mod_vec.append( utils.WoodburyInverter(self.mat_d_mod_vec[layer] - gamma_in_sys_mod) )
            incdet_mod_vec.append( utils.IncLogAbsDeterminant(self.mat_d_mod_inv_vec[layer] - gamma_in_sys_mod) )

        return xnp.array(gamma_in_sys_vec), (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec), (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec)


    #Gauging

    def generate_rotmat(self, theta: float, coord: tuple, dir: Direction):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.

        The mode order is (as for gamma_in_neutral):
            1 copy: {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2},
            2 copies: {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2}, 
        depending on whether the link is vertical or horizontal.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy. 

        For pure gauge layers, modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        For fermionic layers, the projectors don't mix copies to ensure the U(1) symmetry is obeyed.

        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Args:
            theta (float): Angle of rotation
            coord (tuple): (x,y) coordinate on the lattice
            dir (lattice.Direction): direction of the link

        Returns:
            xnp.ndarray: Rotation matrix for gamma_in_neutral
        """
        # Gauging might be different depending on sublattice or link direction, but for this system it is the same
        if dir == Direction.X and (-1)**(coord[0] + coord[1]) == -1:
            #theta += xnp.pi 
            pass

        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        rot_right = xnp.array([[xnp.cos(theta), xnp.sin(theta)],
                               [-xnp.sin(theta), xnp.cos(theta)]])
        # We have only one left mode => 2 Majorana modes
        rot_left = xnp.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(rot_left, rot_right)
        rotmat = xnp.kron( xnp.eye(self.cfg.ncopy), dest)
        return rotmat

    # TODO: fix for JAX - DONE, except for stuff in utils
    def update_gauge_ind(self, link_ind, theta):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors 
        and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be prohibitively expensive.
    
        This method overwrites an abstract method in System2DBase.
    
        Args:
            link_ind (int): Link index to be updated
            theta (float): New gauge field value
        """
        # Update the gaugefield
        if ggpeps.PREFERRED_BACKEND == 'jax':
            self._gaugefieldvec = self._gaugefieldvec.at[link_ind].set(theta)
        else:
            self._gaugefieldvec[link_ind] = theta
        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)
    
        update_vec = []
        for layer in range(self.cfg.nlayer):
            gamma_neutral_gauge = self.gamma_gauge_neutral_vec[layer][dir]
            gamma_in_subst = rotmat @ gamma_neutral_gauge @ xnp.transpose(rotmat)
            update_vec.append( self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys=self.gamma_in_sys_vec[layer]) )
    
            # Substitute in the array
            if ggpeps.PREFERRED_BACKEND == 'jax':
                # TODO: should not modify "private" variable - make a setter?
                self._gamma_in_sys_vec = self.gamma_in_sys_vec.at[layer, ind_mat:ind_mat + rotmat.shape[0],
                                                ind_mat:ind_mat + rotmat.shape[1]].set(gamma_in_subst)
            else:
                self.gamma_in_sys_vec[layer][ind_mat:ind_mat + rotmat.shape[0],
                                         ind_mat:ind_mat + rotmat.shape[1]] = gamma_in_subst
    
        # Update the determinant
        mat_inv_vec = [
            wi_gamma_in.inv() for wi_gamma_in in self.wi_gamma_in_vec
        ]
        detval_vec = np.array([
            incdet.update_index(mat_inv, update, ind_mat, ind_mat)
            for mat_inv, update, incdet in zip(mat_inv_vec, update_vec, self.incdet_vec)
        ])
        # Update the modified determinant
        offset = 2 * self.cfg.nvirtmodes_link
        if ind_mat - offset >= 0:
            for wi, update, incdet in zip(self.wi_gamma_in_mod_vec, update_vec, self.incdet_mod_vec):
                mat_inv = wi.inv()
                incdet.update_index(mat_inv, update, ind_mat-offset, ind_mat-offset)
        # Update the weight
        self.weight = 0.5 * np.sum(detval_vec)
        # Update the matrix inversion
        [ wi_gamma_in.update_index(update, ind_mat, ind_mat) for wi_gamma_in, update in zip(self.wi_gamma_in_vec, update_vec) ]
        [ wi_gamma_out.update_index(update, ind_mat, ind_mat) for wi_gamma_out, update in zip(self.wi_gamma_out_vec, update_vec) ]
    
        if ind_mat - offset >= 0:
            # We do not update the matrix if the first link is updated (it is just not there)
            [ wi_gamma_in_mod.update_index(update, ind_mat-offset, ind_mat-offset) for wi_gamma_in_mod, update in zip(self.wi_gamma_in_mod_vec, update_vec) ]
            [ wi_gamma_out_mod.update_index(update, ind_mat-offset, ind_mat-offset) for wi_gamma_out_mod, update in zip(self.wi_gamma_out_mod_vec, update_vec) ]
    
        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()


    # def update_gauge_ind(self, link_ind, theta):
    #    update_gauge_ind(self, link_ind, theta)

    # Observables
    def _compute_mass_energy_op_vec_and_grad(self, use_trans_inv:bool=True):
        """Compute the mass term of the Hamiltonian for a single site.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to True.

        Returns:
            tuple: Tuple of (mass energy for a single site, gradients)
        """
        if not use_trans_inv:
            raise NotImplementedError("Translation invariance must be set to True.")

        mass_energy_op = [0]*self.cfg.num_pg_layer 
        gradients = [[0]*len(self.symbolvec)]*self.cfg.num_pg_layer

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            # only the fermionic layers directly contribute to the mass
            
            # Calculation prelimaries
            covmat = self.compute_ferm_cov(layer_ind)
            layer_mass_energy = 0.0
            layer_grads = [0]*len(self.symbolvec)
            
            # Calculate mass term
            # Since the system is translationally invariant, we could just calculate it for one site and multiply by nsites instead
            for site_ind in range(0, 2*self.cfg.lattice.size, 2):
                layer_mass_energy += 0.5 * (1 + covmat[site_ind+1, site_ind] ) # TODO: fix for JAX - NOT NEEDED

                for symbol_ind, symbol in enumerate(self.symbolvec):
                    if (layer_ind, symbol_ind) not in self.cfg.zeroed_params:
                        # the derivative calculation is relatively compuationally expensive (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero

                        d_gamma_out = self.d_gamma_out_symbolvec(layer_ind)[symbol_ind]
                        layer_grads[symbol_ind] += 0.5 * d_gamma_out[site_ind+1, site_ind] 

                    # further terms of the derivative are included higher up in the computation stack 
                    # because computing them requires knowing various expectation values, which are not available here

            mass_energy_op.append(xnp.asarray(layer_mass_energy))
            gradients.append(xnp.asarray(layer_grads))

        mass_energy_op = xnp.asarray(mass_energy_op)
        gradients = np.asarray(gradients)

        self.cfg.enforce_parameter_conditions(gradients)

        # When computing the electric energy, we have to weigh the gradients of each layer with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # However, here, because the mass term only acts on the fermionic layers, we simply multiply the mass_energy and grads by the norm of the first layer 
        # (this is handled higher up in the computation stack).

        return mass_energy_op, xnp.array(gradients)

    def _compute_el_energy_op_vec(self, use_trans_inv:bool=True):
        """Computation of the electric energy.
        Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.
        These are saved at the end of the function.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of electric energies for a single link
        """
        if not use_trans_inv:
            # Evaluate every link of the system
            logger.error("compute_el_energy: The non-translational invariant case is not implemented yet.")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")

        lognormvec_default = self.calculate_lognormvec_inc(all_factors=True)
        # This is the usual norm without any modifications
        lognorm_default = xnp.sum(lognormvec_default)
        # Number of fermions = # of sites
        # Since we have 2 copies, we get 8 virtual fermions per site
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        # We have to cut one link from gamma_in_sys as well
        gamma_in_sys_mod_vec = self.gamma_in_sys_mod_vec
        dest = []

        # Indices and prefactors for building the required Pfaffians
        overall_factors = self.cfg.el_overall_factors
        idxarrs = self.cfg.idxarr_vec

        # TODO: vectorize!
        for layerind in range(self.cfg.nlayer):

            # We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
            mat_a = self.mat_a_mod_vec[layerind] # dim: 2*nsites (for majorana) + 8 (= 4 virtual modes per link x2 for majorana)
            mat_b = self.mat_b_mod_vec[layerind]
            diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()

            gamma_in_sys_mod = gamma_in_sys_mod_vec[layerind]

            idxarr = idxarrs[layerind]
            overall_factor = overall_factors[layerind]

            ###################### Calculation of <P> ########################
            covmat_out = mat_a + \
                mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
            size = covmat_out.shape[1]
            covmat_out_virt = slice_matrix(covmat_out, size - single_link_offset, size, size - single_link_offset, size)
                                # covmat_out[-single_link_offset:, -single_link_offset:] # TODO: fix for JAX - DONE

            # The library pfapack is rather picky about the anti-symmetrization (to 1e-14)
            covmat_out_virt = utils.anti_symmetrize(covmat_out_virt)
            # For the modified norm, we still have to take into account the other contributions from the unmodified parts
            norm_mod = calculate_lognorm_inc(
                [self.incdet_mod_vec[layerind]],
                [self.det_mat_d_mod_vec[layerind]],
                gamma_in_sys_mod.shape[0],
                all_factors=True)
            norm_mod += xnp.sum(utils.select_except(lognormvec_default, layerind))
            # The matrix elements yield only the real part of <P>
            # If we use the log formulation, we can calculate the log of single terms.

            # Instead of writing down all the terms explicitly, we build tuples of the prefactors and the indices of the covariance matrix.
            # Then, we compute all terms in a list comprehension.
            pfarr = []
            pfvals = [] # without the prefactor
            for prefactor,ind in idxarr:
                ind = xnp.asarray(ind)
                pfaval = pf.pfaffian(covmat_out_virt[xnp.ix_(ind,ind)]) # TODO: fix for JAX - NOT NEEDED, jxnp.ix_ should work
                pfarr.append(prefactor * pfaval)
                pfvals.append(pfaval)
            el_energy_full = overall_factor * xnp.sum(xnp.array(pfarr))
            
            el_energy_layer = xnp.real(el_energy_full) * xnp.exp(norm_mod - lognorm_default)
            dest.append(el_energy_layer)
            
            # Save intermediate calculations for use in gradient calculation
            intermediate = self._electric_energy_intermediate_vals 
            intermediate.covmat_out_virt_vec.append(covmat_out_virt)
            intermediate.norm_mod_vec.append(norm_mod)
            intermediate.lognorm_default_vec.append(lognorm_default)
            intermediate.pfaffian_vec.append(pfvals)
        
        return xnp.asarray(dest)
    
    def _compute_el_grad_vec(self, use_trans_inv:bool=True):
        """Computation of the electric energy gradients.
        We start by calculating the electric energies, since these are needed for evaluating the gradients.
        Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of gradients for the full system
        """

        if not use_trans_inv:
            # Evaluate every link of the system
            logger.error("compute_el_energy: The non-translational invariant case is not implemented yet.")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")
        
        res = compute_el_grad_vec(self)
        return res

    def _compute_mag_energy_op(self, use_trans_inv:bool=True):
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
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop((0, 0), (1, 1))
            mag_energy_bare = xnp.real(self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            logger.error("compute_mag_energy: non-translational invariant case not implemented yet")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")
            mag_energy_bare = None
        return mag_energy_bare
    
    def _compute_int_energy_op_vec_and_grad(self):
        """Calculate the energy and energy gradient due to the interaction of the physical fermions with the gauge fields.
        Note: this function assumes that U = U^dagger, which is valid only for Z2. For other groups, the calculation will not be as simple.

        Returns:
            tuple: Tuple of (interaction energy for a single link, gradients)
        """

        int_energy_op = [0]*self.cfg.num_pg_layer 
        gradients = [[0]*len(self.symbolvec)]*self.cfg.num_pg_layer

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            layer_int_energy = 0.0
            covmat = self.compute_ferm_cov(layer_ind)
            layer_gradients = [0]*len(self.symbolvec)
            
            for site_ind in range(self.cfg.lattice.size): 
                coord = self.cfg.lattice.ind2coord(site_ind)
                site_ind_cov = 2 * site_ind # this is the index to use when accessing elements of the covariance matrix, which has 2 Majorana modes per site

                # Horizontal link
                ind_field_hor = self.cfg.lattice.coord2ind_dir(coord, Direction.X) # index of the horizontal link
                neighborX_coord = self.cfg.lattice.get_neighbor(coord, Direction.X) # coordinates of neighboring site
                neighborX_ind = 2 * self.cfg.lattice.coord2ind(neighborX_coord) # index of neighboring site, factor of 2 is due to Majorana modes (2 per site)
                gaugefield_hor = self.gaugefieldvec[ind_field_hor]
                cos_factor_hor = xnp.cos(gaugefield_hor) # simple way to get U from gauge value
                hor_link_energy = 0.5 * (covmat[site_ind_cov, neighborX_ind] - covmat[site_ind_cov+1, neighborX_ind+1]) # TODO: fix for JAX - NOT NEEDED
                layer_int_energy += hor_link_energy * cos_factor_hor

                # Vertical link
                ind_field_vert = self.cfg.lattice.coord2ind_dir(coord, Direction.Y)
                neighborY_coord = self.cfg.lattice.get_neighbor(coord, Direction.Y)
                neighborY_ind = 2 * self.cfg.lattice.coord2ind(neighborY_coord)
                gaugefield_vert = self.gaugefieldvec[ind_field_vert]
                cos_factor_vert = xnp.cos(gaugefield_vert)
                vert_link_energy = 0.5 * (covmat[site_ind_cov, neighborY_ind+1] + covmat[site_ind_cov+1, neighborY_ind])
                layer_int_energy -= vert_link_energy * cos_factor_vert

                # Calculate derivatives
                for symbol_ind, symbol in enumerate(self.symbolvec):
                    if (layer_ind, symbol_ind) not in self.cfg.zeroed_params:
                        # the derivative calculation is relatively compuationally expensive (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero

                        d_gamma_out = self.d_gamma_out_symbolvec(layer_ind)[symbol_ind]
                        grad = 0.5 * cos_factor_hor * (d_gamma_out[site_ind_cov, neighborX_ind] - d_gamma_out[site_ind_cov+1, neighborX_ind+1])
                        grad += - 0.5 * cos_factor_vert * (d_gamma_out[site_ind_cov, neighborY_ind+1] + d_gamma_out[site_ind_cov+1, neighborY_ind])
                        layer_gradients[symbol_ind] += grad
                    
            int_energy_op.append(layer_int_energy)
            gradients.append(layer_gradients)
        
        int_energy_op = xnp.asarray(int_energy_op)
        gradients = np.asarray(gradients) 

        self.cfg.enforce_parameter_conditions(gradients)
    
        # When computing the electric energy, we have to weigh the gradients of each layer with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # However, here (just as in the mass case), because the interaction term only acts on the fermionic layers, we simply multiply the int_energy and grads by the norm of the first layer 
        # (this is handled higher up in the computation stack).

        return int_energy_op, xnp.array(gradients)

    def _compute_chem_energy_op_vec_and_grad(self):
        """Calculate the chemical potential energy operator and its gradient.
        """

        chem_energy_op = [0] * self.cfg.num_pg_layer 
        gradients = [[0]*len(self.symbolvec)] * self.cfg.num_pg_layer

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            # only the fermionic layers directly contribute to the chemical potential 
            
            # Calculation prelimaries
            covmat = self.compute_ferm_cov(layer_ind)
            layer_chem_energy = 0.0
            layer_grads = [0]*len(self.symbolvec)
            
            # Calculate mass term
            # Since the system is translationally invariant, we could just calculate it for one site and multiply by nsites instead
            for site_ind in range(0, 2*self.cfg.lattice.size, 2):
                site_factor = (-1)**(site_ind) # even or odd sublattice
                layer_chem_energy += 0.5 * site_factor * (1 + covmat[site_ind+1, site_ind] )

                for symbol_ind, symbol in enumerate(self.symbolvec):
                    if (layer_ind, symbol_ind) not in self.cfg.zeroed_params:
                        # the derivative calculation is relatively compuationally expensive (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero

                        d_gamma_out = self.d_gamma_out_symbolvec(layer_ind)[symbol_ind]
                        layer_grads[symbol_ind] += 0.5 * site_factor * d_gamma_out[site_ind+1, site_ind] 

                    # further terms of the derivative are included higher up in the computation stack 
                    # because computing them requires knowing various expectation values, which are not available here

            chem_energy_op.append(np.asarray(layer_chem_energy))
            gradients.append(np.asarray(layer_grads))

        chem_energy_op = np.asarray(chem_energy_op)
        gradients = np.asarray(gradients)

        self.cfg.enforce_parameter_conditions(gradients)

        return chem_energy_op, gradients