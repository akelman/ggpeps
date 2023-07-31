import logging

import sympy
import numpy as np
from pfapack import pfaffian as pf
from scipy.linalg import block_diag

from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.modearray import generate_permutation_matrix

from .system_base import Config2DBase, System2DBase
from .system_base import calculate_lognorm_inc, compute_grad_over_norm, extract_partial_covmats


###################### Z2System2D ##########################

class Z2System2D_8C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 8 copies of virtual fermions on the links.
    More details about the mode order and the parameters can be found in the documentation of `Z2System2D2C`.
    """

    _nparams = 52 #36
    ncopy = 4
    nvirtmodes_vertex = 16
    nvirtmodes_link = 8

    def __init__(self, lattice, g_el, g_mag,  g_int, g_mass, nlayer=2):
        #if nlayer != 2:
        #    raise ValueError("When including physical fermions, 2 layers is required.")
        super().__init__(lattice, g_el, g_mag, g_int, g_mass, nlayer)
        self.num_pg_layers = self.nlayer - 1

    def make_pure_gauge(self):
        raise NotImplementedError("Haven't yet implemented parameter conditions for pure gauge.")
        reproduce_2C1L = False # zero out the second layer (used for physical fermions); this should reproduce the 2 copy, 1 layer ansatz
        if reproduce_2C1L:
            for ind in range(self._nparams):
                self.paramvec[1, ind] = 0 
        zeroed_params = [
                        # Set 1st layer (type I) t params to 0
                        (0,0),  # t1r
                        (0,3),  # t2r
                        (0,10), # t1i
                        (0,13), # t2i 
                        # Set 2nd layer (type II) t params to 0
                        (1, 0), # t1r
                        (1,3),  # t2r
                        (1, 10),# t1i
                        (1,13), # t2i
                        ]
        for coord in zeroed_params:
            self.paramvec[coord] = 0
    
    def enforce_parameter_conditions(self, mat):
        """Enforce conditions on parameters on each layer to get the required behaviour for the ansatz.
        """
        
        fli = self.nlayer - 1 # = self.num_pg_layers = fermionic_layer_ind

        # Set pure gauge conditions layer (type I) 
        for layer in range(fli):
            ind = 0

            copies = [1,3,5,7] # copies which couple to physical modes
            for cop in copies:
                for com in ['r', 'i']: # real or imaginary
                    mat[layer, ind] = 0
                    ind += 1

        # fermionic layer
        ind = 0
        copies = [1,3,5,7] # copies which couple to physical modes
        for cop in copies:
            for com in ['r', 'i']: 
                ind += 1 # don't zero out t params

        on_diag_symbols = []
        copies = [1,2,3,4] # copies which couple to themselves
        for cop in copies:
            for l in ['z', 'y']:
                for com in ['r', 'i']:
                    mat[fli, ind] = 0
                    ind += 1

        return


class Z2System2D_8C(System2DBase):
    """ 2 copy version of the Z2 system GGPEPS ansatz with multiple type of virtual fermions

    Some general notes about conventions:

    Order of the paramvec: [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di]
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2}.
    """

    def __init__(self, cfg: Z2System2D_8C_Config):
        """Constructor of a Z2System2D_8C system.
        We call only the constructor of the super class, since we do not have any class-specific setup.

        Args:
            cfg (Z2System2D_8C_Config): Configuration containing all system-related parameters
        """
        super().__init__(cfg)

        prefactors = [[1, -1, 1.j, 1.j], [1, -1, 1.j, 1.j], [1, -1, 1.j, 1.j], [1, -1, 1.j, 1.j]]
        indices_layer1 = [[(2,4), (3,5), (4,5), (2,3)], [(6,0), (7,1), (0,1), (6,7)], [(10,12), (11,13), (12,13), (10,11)], [(14,8), (15,9), (8,9), (14,15)]]
        indices_layer2 = [[(2,0), (3,1), (0,1), (2,3)], [(6,4), (7,5), (4,5), (6,7)], [(10,8), (11,9), (8,9), (10,11)], [(14,12), (15,13), (12,13), (14,15)]]
        idxarr_lay1 = self.get_pfaffian_arrays(indices_layer1, prefactors) # pure gauge layers
        idxarr_lay2 = self.get_pfaffian_arrays(indices_layer2, prefactors) # fermionic layers
        self.idxarr_vec = [idxarr_lay1]*(self.cfg.num_pg_layers) + [idxarr_lay2]


    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the covariance matrices analytically.

        Returns:
            list: List of all analytic symbols
        """

        phy_virt_symbols = [] # for coupling physical and virtual modes
        copies = [1,3,5,7] # copies which couple to physical modes
        for cop in copies:
            for com in ['r', 'i']: # real or imaginary
                symbol = sympy.Symbol(f"t{cop}{com}", real=True)
                phy_virt_symbols.append(symbol)
 
        on_diag_symbols = []
        copies = [1,2,3,4] # copies which couple to themselves (if not zeroed out in enforce_parameter_conditions)
        for cop in copies:
            for l in ['z', 'y']:
                for com in ['r', 'i']:
                    symbol = sympy.Symbol(f"{l}{cop}{com}", real=True)
                    on_diag_symbols.append(symbol)

        off_diag_symbols = [] # off-diagonal blocks
        copies_odd = [1,3,5,7]
        copies_even = [2,4,6,8]
        for r in copies_odd:
            for c in copies_even:
                for l in ['p', 'q', 'r', 's']:
                    for com in ['r', 'i']: 
                        symbol = sympy.Symbol(f"{l}{r}{c}{com}", real=True)
                        off_diag_symbols.append(symbol)

        return phy_virt_symbols + on_diag_symbols + off_diag_symbols


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
        The permutation matrices are prone for errors.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """

        # Build dictionary of parameters
        all_params = {}
        ind = 0

        copies = [1,3,5,7] # copies which couple to physical modes
        for cop in copies:
            all_params[f"t{cop}"] = self.symbolvec[ind] + 1.j*self.symbolvec[ind+1]
            ind += 2
 
        copies = [1,2,3,4] # copies which couple to themselves (if not zeroed out in enforce_parameter_conditions)
        for cop in copies:
            for l in ['z', 'y']:
                all_params[f"{l}{cop}"] = self.symbolvec[ind] + 1.j*self.symbolvec[ind+1]
                ind += 2

        copies_odd = [1,3,5,7]
        copies_even = [2,4,6,8]
        for r in copies_odd:
            for c in copies_even:
                for l in ['p', 'q', 'r', 's']:
                    all_params[f"{l}{r}{c}"] = self.symbolvec[ind] + 1.j*self.symbolvec[ind+1]
                    ind += 2

        # Extract params as variables for convenience
        z1 = all_params['z1']
        z2 = all_params['z2']
        y1 = all_params['y1']
        y2 = all_params['y2']
        z3 = all_params['z3']
        z4 = all_params['z4']
        y3 = all_params['y3']
        y4 = all_params['y4']

        p12, q12, r12, s12 = all_params['p12'], all_params['q12'], all_params['r12'], all_params['s12']
        p14, q14, r14, s14 = all_params['p14'], all_params['q14'], all_params['r14'], all_params['s14']
        p16, q16, r16, s16 = all_params['p16'], all_params['q16'], all_params['r16'], all_params['s16']
        p18, q18, r18, s18 = all_params['p18'], all_params['q18'], all_params['r18'], all_params['s18']

        p32, q32, r32, s32 = all_params['p32'], all_params['q32'], all_params['r32'], all_params['s32']
        p34, q34, r34, s34 = all_params['p34'], all_params['q34'], all_params['r34'], all_params['s34']
        p36, q36, r36, s36 = all_params['p36'], all_params['q36'], all_params['r36'], all_params['s36']
        p38, q38, r38, s38 = all_params['p38'], all_params['q38'], all_params['r38'], all_params['s38']

        p52, q52, r52, s52 = all_params['p52'], all_params['q52'], all_params['r52'], all_params['s52']
        p54, q54, r54, s54 = all_params['p54'], all_params['q54'], all_params['r54'], all_params['s54']
        p56, q56, r56, s56 = all_params['p56'], all_params['q56'], all_params['r56'], all_params['s56']
        p58, q58, r58, s58 = all_params['p58'], all_params['q58'], all_params['r58'], all_params['s58']

        p72, q72, r72, s72 = all_params['p72'], all_params['q72'], all_params['r72'], all_params['s72']
        p74, q74, r74, s74 = all_params['p74'], all_params['q74'], all_params['r74'], all_params['s74']
        p76, q76, r76, s76 = all_params['p76'], all_params['q76'], all_params['r76'], all_params['s76']
        p78, q78, r78, s78 = all_params['p78'], all_params['q78'], all_params['r78'], all_params['s78']

        # Block matrices that appear many times in the T matrix
        Block_1 = sympy.Matrix([-1.j*all_params['t1'], 1.j*all_params['t1'], all_params['t1'], -all_params['t1'], 0,0,0,0]) # this is a column matrix
        Block_2a = sympy.Matrix([
            [0,         1.j*y1, z1,         1.j*z1],
            [-1.j*y1,   0,      -1.j*z1,    -z1],
            [-z1,       1.j*z1, 0,          -y1],
            [-1.j*z1,   z1,     y1,         0],
            ])
        Block_2b = sympy.Matrix([
            [-1.j*p12,   -1.j*r12,  -1.j*q12,  -1.j*s12],
            [1.j*r12,    1.j*p12,   1.j*s12,   1.j*q12],
            [s12,        q12,       p12,       r12],
            [-q12,       -s12,      -r12,      -p12],
            ])
        zeros_4 = sympy.zeros(4)

        # first row
        M00 = sympy.zeros(1)
        M01 = -Block_1.T # copies 1,2
        M02 = -Block_1.subs([(all_params['t1'], all_params['t3'])]).T # copies 3,4
        M03 = -Block_1.subs([(all_params['t1'], all_params['t5'])]).T
        M04 = -Block_1.subs([(all_params['t1'], all_params['t7'])]).T

        # second row
        M10 = -M01.T
        M11 = sympy.Matrix( sympy.BlockMatrix([[Block_2a, Block_2b], [-Block_2b.T, Block_2a.subs([(z1, z2), (y1, y2)]) ]]) )
        M12 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b.subs([(p12, p14), (q12,q14), (r12, r14), (s12,s14)])], [Block_2b.subs([(p12,p32), (q12,q32), (r12, r32), (s12,s32)]), zeros_4]]) )
        M13 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b.subs([(p12, p16), (q12,q16), (r12, r16), (s12,s16)])], [Block_2b.subs([(p12,p52), (q12,q52), (r12, r52), (s12,s52)]), zeros_4]]) )
        M14 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b.subs([(p12, p18), (q12,q18), (r12, r18), (s12,s18)])], [Block_2b.subs([(p12,p72), (q12,q72), (r12, r72), (s12,s72)]), zeros_4]]) )

        # third row
        M20 = -M02.T
        M21 = -M12.T
        Block_2b_22 = Block_2b.subs([(p12, p34), (q12,q34), (r12, r34), (s12,s34)])
        M22 = sympy.Matrix(sympy.BlockMatrix([[Block_2a.subs([(z1,z3), (y1,y3)]), Block_2b_22], [-Block_2b_22.T, Block_2a.subs([(z1,z4), (y1,y4)])]]) ) #z3,y3 should go here
        M23 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b.subs([(p12, p36), (q12,q36), (r12, r36), (s12,s36)])], [Block_2b.subs([(p12,p54), (q12,q54), (r12, r54), (s14,s54)]), zeros_4]]) )
        M24 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b.subs([(p12, p38), (q12,q38), (r12, r38), (s12,s38)])], [Block_2b.subs([(p12,p74), (q12,q74), (r12, r74), (s14,s74)]), zeros_4]]) )

        # fourth row
        M30 = -M03.T
        M31 = -M13.T
        M32 = -M23.T
        Block_2b_33 = Block_2b.subs([(p12, p56), (q12,q56), (r12, r56), (s12,s56)])
        M33 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b_33], [-Block_2b_33.T, zeros_4]]) )
        M34 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b.subs([(p12, p58), (q12,q58), (r12, r58), (s12,s58)])], [Block_2b.subs([(p12,p76), (q12,q76), (r12, r76), (s14,s76)]), zeros_4]]) )

        # fifth row
        M40 = -M04.T
        M41 = -M14.T
        M42 = -M24.T
        M43 = -M34.T
        Block_2b_44 = Block_2b.subs([(p12, p78), (q12,q78), (r12, r78), (s12,s78)])
        M44 = sympy.Matrix(sympy.BlockMatrix([[zeros_4, Block_2b_44], [-Block_2b_44.T, zeros_4]]) )

        # Full T matrix
        tmat_symb = sympy.Matrix( sympy.BlockMatrix([[M00, M01, M02, M03, M04], [M10, M11, M12, M13, M14], [M20, M21, M22, M23, M24], [M30, M31, M32, M33, M34], [M40, M41, M42, M43, M44]]) )

        return tmat_symb


    def _expand_gamma_maj_to_system(self,covmat):
        """Expand the covariance matrix in Majorana modes to the full system.
        In order to obtain a structure that is convenient for further computations,
            (A    B)
            (-B^T D)
        we have to reorder the modes of the single-vertex matrix with respect to the full matrix.
        The biggest part of the permutation-matrix generation is done in PermutationBuilderGMS2D2C.
        The GMS stands for Gamma Majorana System, 2D for 2 dimensions, 2C for 2 copies.

        This method overwrites an abstract method in System2DBase.

        Args:
            covmat (np.ndarray): 2D covariance matrix of a single site

        Returns:
            np.ndarray: 2D covariance matrix of the full system
        """
        # Build permutation matrix to convert modes from site order to link order
        modes_link_order = self.get_link_based_mode_order()
        modes_site_order = self.get_site_based_mode_order()
        mat_perm_links = generate_permutation_matrix( modes_site_order, modes_link_order) # be careful with the convention of the permutation matrix vs its transpose; this way works with the code below.
        sites_perm = np.eye( 2 * self.cfg.lattice.nx * self.cfg.lattice.ny ) # total number of physical fermionic majorana modes on all the sites together
        mat_perm = block_diag(sites_perm, mat_perm_links)

        nsites = self.cfg.lattice.size
        id = np.eye(nsites)
        # Extract the parts of the covariance matrix
        amat = covmat[:2, :2] # assumes 1 fermion per site (two majorana modes)
        bmat = covmat[:2, 2:]
        dmat = covmat[2:, 2:]
        # Expand them
        amat_sys = np.kron(id, amat)
        bmat_sys = np.kron(id, bmat)
        dmat_sys = np.kron(id, dmat)
        # Reassemble them in the correct order
        mat_sys_unordered = np.block(
            [[amat_sys, bmat_sys], [-np.transpose(bmat_sys), dmat_sys]])
        dest = np.transpose(mat_perm) @ mat_sys_unordered @ mat_perm
        return dest


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

        For a 2x2 system, gamma_in has the order 
        { l1_1, r2_0, l1_1, r2_0, l1_0, r2_1, l1_0, r2_1,  
          l1_3, r2_2, l1_3, r2_2, l1_2, r2_3, l1_2, r2_3,  
          d1_2, u2_0, d1_2, u2_0, d1_0, u2_2, d1_0, u2_2,  
          d1_3, u2_1, d1_3, u2_1, d1_1, d2_3, d1_1, d2_3 }.

        The naming convention here is <mode letter><number of copy>_<vertex index>.
        Each constituent in the list above refers to two Majorana modes.

        This method overwrites an abstract method in System2DBase.
        """

        # Initialize empty lists
        gamma_in_sys_vec = []
        wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec = [], [], []
        wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec = [], [], []

        # Initialize gamma_in_sys for the full system (and trackers)
        size = self.cfg.lattice.size # number of sites
        id = np.eye(size) 

        for layer in range(self.cfg.nlayer):
            neutral_gauge_X = np.kron( id, self.gamma_gauge_neutral[layer][Direction.X] )
            neutral_gauge_Y = np.kron( id, self.gamma_gauge_neutral[layer][Direction.Y] )
            gamma_in_sys = block_diag(neutral_gauge_X, neutral_gauge_Y)
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

        return gamma_in_sys_vec, (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec), (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec)

    def _generate_gamma_gauge_neutral_dict(self):
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
            List[np.ndarray]: Covariance matrices of the ungauged projector on a single link
        """
        
        dest_mixed = {} # mixes copies
        dest_unmixed = {} # does not mix copies 

        zeros_8 = np.zeros((8,8))
        
        # We want to give the projectors for the pure gauge part, which mix copies
        mixed_X = np.real_if_close(1.j*np.kron(utils.paulix,np.kron(utils.pauliy, utils.paulix)))
        mixed_Y = np.real_if_close(1.j*np.kron(utils.paulix,np.kron(utils.pauliy, utils.pauliz)))

        dest_mixed[Direction.X] = np.block([ [mixed_X, zeros_8], [zeros_8, mixed_X] ])
        dest_mixed[Direction.Y] = np.block([ [mixed_Y, zeros_8], [zeros_8, mixed_Y] ])

        # We want to give the projectors for the fermionic part which don't mix copies (so as to preserve global U(1) symmetry)
        unmixed_X = np.array([  [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                                                [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.]])

        unmixed_Y = np.array([  [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0., -1.,  0., -0.,  0.,  0.],
                                                [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0., -0.,  0., -1.],
                                                [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                                                [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]])
        
        dest_unmixed[Direction.X] = np.block([ [unmixed_X, zeros_8], [zeros_8, unmixed_X] ])
        dest_unmixed[Direction.Y] = np.block([ [unmixed_Y, zeros_8], [zeros_8, unmixed_Y] ])
        
        return [dest_mixed]*(self.cfg.nlayer -1) + [dest_unmixed]

    #Gauging

    def generate_rotmat(self, theta: float, coord: tuple, dir: Direction):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.
        The mode order is (as for gamma_in_neutral) {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2}, depending on whether the link is vertical or horizontal.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy. 
        Modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Args:
            theta (float): Angle of rotation
            coord (tuple): (x,y) coordinate on the lattice
            dir (lattice.Direction): direction of the link

        Returns:
            np.ndarray: Rotation matrix for gamma_in_neutral
        """
        # Gauging might be different depending on sublattice or link direction
        if dir == Direction.X and (-1)**(coord[0] + coord[1]) == -1:
            #theta += np.pi 
            pass

        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        rot_right = np.array([[np.cos(theta), np.sin(theta)],
                              [-np.sin(theta), np.cos(theta)]])
        # We have only one left mode => 2 Majorana modes
        rot_left = np.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = block_diag(rot_left, rot_right, rot_left, rot_right)

        zeros_8 = np.zeros((8,8))
        rotmat = np.block( [[dest, zeros_8], [zeros_8, dest]])
        return rotmat


    def update_gauge_ind(self, link_ind, theta):
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
        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)

        update_vec = []
        for layer in range(self.cfg.nlayer):
            gamma_neutral_gauge = self.gamma_gauge_neutral[layer][dir]
            gamma_in_subst = rotmat @ gamma_neutral_gauge @ np.transpose(rotmat)
            update_vec.append( self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys=self.gamma_in_sys_vec[layer]) )

            # Substitute in the array
            self.gamma_in_sys_vec[layer][ind_mat:ind_mat + rotmat.shape[0],
                                         ind_mat:ind_mat + rotmat.shape[1]] = gamma_in_subst

        # Update the determinant
        mat_inv_vec = [
            wi_gamma_in.inv() for wi_gamma_in in self.wi_gamma_in_vec
        ]
        detval_vec = [
            incdet.update_index(mat_inv, update, ind_mat, ind_mat)
            for mat_inv, update, incdet in zip(mat_inv_vec, update_vec, self.incdet_vec)
        ]
        # Update the modified determinant
        offset = 2* self.cfg.nvirtmodes_link
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
        #if self.cfg.nlayer != 2:
        #    raise NotImplementedError("Two layers must be used with physical fermions.")

        mass_energy_op = [1]*(self.cfg.num_pg_layers) # Really the mass energy for the first layer is zero, but later we take the product of all layers, so we put a 1 here
        gradients = [[0]*len(self.symbolvec)]*(self.cfg.num_pg_layers)

        # Calculation prelimaries
        layer_ind = self.cfg.nlayer - 1 # = num_pg_layers; only the "fermionic" layer directly contributes to the mass
        covmat = self.compute_ferm_cov(layer_ind)
        layer_mass_energy = 0.0
        layer_grads = [0]*len(self.symbolvec)
        
        # Calculate mass term
        # Since the system is translationally invariant, we could just calculate it for one site and multiply by nsites instead
        for site_ind in range(0, 2*self.cfg.lattice.size, 2):
            layer_mass_energy += 0.5 * (1 + covmat[site_ind+1, site_ind] )

            # Update gradients
            for symbol_ind, symbol in enumerate(self.symbolvec):
                d_gamma_out = self.d_gamma_out_symbolvec(layer_ind)[symbol_ind]
                layer_grads[symbol_ind] += 0.5 * d_gamma_out[site_ind+1, site_ind] 

                # further terms of the derivative are included higher up in the computation stack 
                # because computing them requires knowing various expectation values, which are not available here

        mass_energy_op.append(np.asarray(layer_mass_energy))
        gradients.append(np.asarray(layer_grads))

        mass_energy_op = np.asarray(mass_energy_op)
        gradients = np.asarray(gradients)

        self.cfg.enforce_parameter_conditions(gradients)

        # When computing the electric energy, we have to weigh the gradients of each layer with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # However, here, because the mass term only acts on the second layer, we simply multiply the mass_energy and grads by the norm of the first layer 
        # (this is handled higher up in the computation stack).

        return mass_energy_op, gradients

    def _compute_el_energy_op_vec_and_grad(self, use_trans_inv:bool=True):
        """Computation of the electric energy and the electric gradient in a single method.
        Since many operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            tuple: Tuple of (list of electric energies for a single link, list of gradients for the full system)
        """
        if not use_trans_inv:
            # Evaluate every link of the system
            logging.error("compute_el_energy: The non-translational invariant case is not implemented yet.")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")

        lognormvec_default = self.calculate_lognormvec_inc(all_factors=True)
        # This is the usual norm without any modifications
        lognorm_default = np.sum(lognormvec_default)
        # Number of fermions = # of sites
        # Since we have 2 copies, we get 8 virtual fermions per site
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        offset = 2 * self.cfg.lattice.size + single_link_offset
        # We have to cut one link from gamma_in_sys as well
        gamma_in_sys_mod_vec = self.gamma_in_sys_mod_vec
        nlinks = self.cfg.lattice.nlinks
        dest = []
        dest_grad = []

        # Indices and prefactors for building the required Pfaffians
        overall_factors = [1/256]*(self.cfg.nlayer) # this arises due to normalization and the i^(# of modes/2) in the expression Tr[1^# * rho * (modes)]
        idxarrs = self.idxarr_vec

        for layerind in range(self.cfg.nlayer):
            layer_derivative = []
            
            #We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
            mat_a = self.mat_a_mod_vec[layerind] # dim: 2*nsites (for majorana) + 8 (= 4 virtual modes per link x2 for majorana)
            mat_b = self.mat_b_mod_vec[layerind]
            diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()
            diff_d_inv_gamma_inv = self.wi_gamma_in_mod_vec[layerind].inv()

            gamma_in_sys_mod = gamma_in_sys_mod_vec[layerind]

            idxarr = idxarrs[layerind]
            overall_factor = overall_factors[layerind]

            ###################### Calculation of <P> ########################
            covmat_out = mat_a + \
                mat_b @ diff_d_gamma_inv @ np.transpose(mat_b)
            covmat_out_virt = covmat_out[-single_link_offset:, -
                                        single_link_offset:]

            # The library pfapack is rather picky about the anti-symmetrization (to 1e-14)
            covmat_out_virt = utils.anti_symmetrize(covmat_out_virt)
            # For the modified norm, we still have to take into account the other contributions from the unmodified parts
            norm_mod = calculate_lognorm_inc(
                [self.incdet_mod_vec[layerind]],
                [self.det_mat_d_mod_vec[layerind]],
                gamma_in_sys_mod.shape[0],
                all_factors=True)
            norm_mod += np.sum(utils.select_except(lognormvec_default, layerind))
            # The matrix elements yield only the real part of <P>
            # If we use the log formulation, we can calculate the log of single terms.

            # Instead of writing down all the terms explicitly, we build tuples of the prefactors and the indices of the covariance matrix.
            # Then, we compute all terms in a list comprehension.
            pfarr = [prefactor * pf.pfaffian(covmat_out_virt[np.ix_(ind,ind)]) for prefactor,ind in idxarr]
            el_energy_full = overall_factor * np.sum(pfarr)
            
            el_energy_layer = np.real(el_energy_full) * np.exp(norm_mod - lognorm_default)
            dest.append(el_energy_layer)

            ###################### Calculation of the derivative ########################
            for symbol in self.symbolvec:
                deriv_gamma_maj_sys = self.gamma_maj_sys_deriv_vec(symbol)[layerind]
                d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats(deriv_gamma_maj_sys, offset)
                d_gamma_out = d_mat_a + \
                        d_mat_b @ diff_d_gamma_inv @ np.transpose(mat_b) \
                        + mat_b @ diff_d_gamma_inv @ np.transpose(d_mat_b) \
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ np.transpose(mat_b)
                # The virtual mode is the last link on the bottom right of the covariance matrix
                d_covmat_out_virt = d_gamma_out[-single_link_offset:, -single_link_offset:]
                # Summand with derivative of the covariance matrix
                # We re-use the list comprehension from above to use the indices
                deriv_pfarr = [prefactor * utils.derivative_pfaffian(covmat_out_virt[np.ix_(ind,ind)], d_covmat_out_virt[np.ix_(ind,ind)]) for prefactor,ind in idxarr]
                d_el_energy = overall_factor * np.real(np.sum(deriv_pfarr)) * np.exp(norm_mod - lognorm_default)
                                
                # Summand with derivative of norms
                trace_def = self.compute_grad_over_norm(symbol, layerind)
                trace_mod = compute_grad_over_norm(gamma_in_sys_mod, diff_d_inv_gamma_inv, d_mat_d, self.mat_d_mod_inv_vec[layerind])
                # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                d_el_energy += dest[layerind] * (trace_mod - trace_def)
                # Scale to system size
                d_el_energy *= nlinks
                layer_derivative.append(np.real(d_el_energy))
            dest_grad.append(layer_derivative)
        
        dest = np.asarray(dest)
        dest_grad = np.asarray(dest_grad)

        # We have to weigh the different layers with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative
        if self.cfg.nlayer > 1:
            for i in range(self.cfg.nlayer):
                prod_other_layers = utils.multiply_except(dest, i)
                dest_grad[i] *= prod_other_layers
        
        self.cfg.enforce_parameter_conditions(dest_grad)

        return dest, dest_grad


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
            mag_energy_bare = np.real(self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            logging.error("compute_mag_energy: not implemented yet")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")
            mag_energy_bare = None
        return mag_energy_bare
    
    def _compute_int_energy_op_vec_and_grad(self):
        """Calculate the energy and energy gradient due to the interaction of the physical fermions with the gauge fields.
        Note: this function assumes that U = U^dagger, which is valid only for Z2. For other groups, the calculation will not be as simple.

        Returns:
            tuple: Tuple of (interaction energy for a single link, gradients)
        """

        int_energy_op = [1]*(self.cfg.num_pg_layers) # Really the interaction energy for the pure gauge layers is zero, but later we take the product of all layers, so we put a 1 here
        gradients = [[0]*len(self.symbolvec)]*(self.cfg.num_pg_layers)

        #for layer_ind in range(self.cfg.nlayer):
        layer_ind = self.cfg.nlayer - 1 # = num_pg_layers; only the "fermionic" layer contributes
        layer_int_energy = 0.0
        covmat = self.compute_ferm_cov(layer_ind)
        
        for site_ind in range(self.cfg.lattice.size): 
            coord = self.cfg.lattice.ind2coord(site_ind)
            site_ind_cov = 2 * site_ind # this is the index to use when accessing elements of the covariance matrix, which has 2 Majorana modes per site
            sublattice_factor = 1 #(-1)**(coord[0] + coord[1]) # the odd sublattice gets a minus sign because of the particle-hole transformation

            # Horizontal link
            ind_field_hor = self.cfg.lattice.coord2ind_dir(coord, Direction.X) # index of the horizontal link
            neighborX_coord = self.cfg.lattice.get_neighbor(coord, Direction.X) # coordinates of neighboring site
            neighborX_ind = 2 * self.cfg.lattice.coord2ind(neighborX_coord) # index of neighboring site, factor of 2 is due to Majorana modes (2 per site)
            gaugefield_hor = self.gaugefieldvec[ind_field_hor]
            cos_factor_hor = np.cos(gaugefield_hor) # simple way to get U from gauge value
            hor_link_energy = 0.5 * (covmat[site_ind_cov, neighborX_ind] - covmat[site_ind_cov+1, neighborX_ind+1])
            layer_int_energy += sublattice_factor * hor_link_energy * cos_factor_hor

            # Vertical link
            ind_field_vert = self.cfg.lattice.coord2ind_dir(coord, Direction.Y)
            neighborY_coord = self.cfg.lattice.get_neighbor(coord, Direction.Y)
            neighborY_ind = 2 * self.cfg.lattice.coord2ind(neighborY_coord)
            gaugefield_vert = self.gaugefieldvec[ind_field_vert]
            cos_factor_vert = np.cos(gaugefield_vert)
            vert_link_energy = 0.5 * (covmat[site_ind_cov, neighborY_ind+1] + covmat[site_ind_cov+1, neighborY_ind])
            layer_int_energy -= vert_link_energy * cos_factor_vert

            # Calculate derivatives
            layer_gradients = []
            for symbol_ind, symbol in enumerate(self.symbolvec):
                d_gamma_out = self.d_gamma_out_symbolvec(layer_ind)[symbol_ind]
                
                grad = 0.5 * sublattice_factor * cos_factor_hor * (d_gamma_out[site_ind_cov, neighborX_ind] - d_gamma_out[site_ind_cov+1, neighborX_ind+1])
                grad += - 0.5 * cos_factor_vert * (d_gamma_out[site_ind_cov, neighborY_ind+1] + d_gamma_out[site_ind_cov+1, neighborY_ind])
                layer_gradients.append(grad)
        
        int_energy_op.append(layer_int_energy)
        gradients.append(layer_gradients)
        
        int_energy_op = np.asarray(int_energy_op)
        gradients = np.asarray(gradients) 

        self.cfg.enforce_parameter_conditions(gradients)
    
        # When computing the electric energy, we have to weigh the gradients of each layer with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # However, here (just as in the mass case), because the interaction term only acts on the second layer, we simply multiply the mass_energy and grads by the norm of the first layer 
        # (this is handled higher up in the computation stack).

        return int_energy_op, gradients


