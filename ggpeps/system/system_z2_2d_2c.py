import numpy as np
import logging
import sympy
from ggpeps import lattice as lat
import sympy
from scipy.linalg import block_diag
from ggpeps import utils
from .system_base import Z2System2DBase, Config2DBase
from .system_base import calculate_lognorm_inc, compute_grad_over_norm, extract_partial_covmats
from pfapack import pfaffian as pf

###################### Z2System2D ##########################


class Z2System2D2CConfig(Config2DBase):
    """Configuration of the Z2 system in 2D with 2 copies of virtual fermions on the links.
    More details about the mode order and the parameters can be found in the documentation of `Z2System2D2C`.
    """

    _nparams = 10
    ncopy = 2
    nvirtmodes_vertex = 8 # We have two virtual modes per direction (4 directions x 2 modes)
    nvirtmodes_link = 4 #Number of virtual modes per link (2 copies and l/r or u/d)

    def __init__(self, lattice, g2, g_gm, g_mag, nlayer=1):
        #The parameters have the following order: [[t1,y1,z1,t2,y2,z2,a,b,c,d],[..next layer..],....]
        super().__init__(lattice, g2, g_gm, g_mag, nlayer)

    def make_pure_gauge(self):
        #The order of the parameters is [t1,y1,z1,t2,y2,z2,a,b,c,d]
        for ind in range(self.nlayer):
            self.paramvec[ind, 0] = 0 # Set t1 to 0
            self.paramvec[ind, 3] = 0 # Set t2 to 0


class Z2System2D2C(Z2System2DBase):
    """ 2 copy version of the Z2 system GGPEPS ansatz

    Some general notes about conventions:

    Order of the paramvec: [t1,y1,z1,t2,y2,z2,a,b,c,d]
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2}.
    """

    def __init__(self, cfg: Z2System2D2CConfig):
        """Constructor of a Z2System2D2C system.
        We call only the constructor of the super class, since we do not have any class-specific setup.

        Args:
            cfg (Z2System2D2CConfig): Configuration containing all system-related parameters
        """
        super().__init__(cfg)


    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the covariance matrices analytically.

        Returns:
            list: List of all analytic symbols
        """
        t1 = sympy.Symbol("t1", real=True)
        y1 = sympy.Symbol("y1", real=True)
        z1 = sympy.Symbol("z1", real=True)
        t2 = sympy.Symbol("t2", real=True)
        y2 = sympy.Symbol("y2", real=True)
        z2 = sympy.Symbol("z2", real=True)
        a  = sympy.Symbol("a", real=True)
        b  = sympy.Symbol("b", real=True)
        c  = sympy.Symbol("c", real=True)
        d  = sympy.Symbol("d", real=True)
        return [t1, y1, z1, t2, y2, z2, a, b, c, d]


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
        [t1, y1, z1, t2, y2, z2, a, b, c, d]=self.symbolvec
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


    def _expand_gamma_maj_to_system(self,covmat):
        """Expand the covariance matrix in Majorana modes to the full system.
        In order to obtain a structure that is convenient for further computations,
            (A    B)
            (-B^T D)
        we have to reorder the modes of the single-vertex matrix with respect to the full matrix.
        The biggest part of the permutation-matrix generation is done in PermutationBuilderGMS2D2C.
        The GMS stands for Gamma Majorana System, 2D for 2 dimensions, 2C for 2 copies.

        This method overwrites an abstract method in Z2System2DBase.

        Args:
            covmat (np.ndarray): 2D covariance matrix of a single site

        Returns:
            np.ndarray: 2D covariance matrix of the full system
        """
        #TODO: Why are we calling this function with 1 mode per link? There are two copies and thus two modes per link around...
        permbuilder = lat.PermutationBuilderGMS2D2C(self.cfg.lattice, nmodes_per_link=1)
        mat_perm = permbuilder.perm()
        nsites=self.cfg.lattice.size
        id = np.eye(nsites)
        # Extract the parts of the covariance matrix
        amat = covmat[:2, :2]
        bmat = covmat[:2, 2:]
        dmat = covmat[2:, 2:]
        #Expand them
        amat_sys = np.kron(id, amat)
        bmat_sys = np.kron(id, bmat)
        dmat_sys = np.kron(id, dmat)
        #Reassemble them in the correct order
        mat_sys_unordered= np.block(
            [[amat_sys, bmat_sys], [-np.transpose(bmat_sys), dmat_sys]])
        dest=mat_perm@mat_sys_unordered@np.transpose(mat_perm)
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

        This method overwrites an abstract method in Z2System2DBase.

        """
        nlinks = self.cfg.lattice.nlinks
        id = np.eye(nlinks)
        neutral_gauge = self.gamma_neutral_gauge

        # Initialize gamma_in_sys for the full system (and trackers)
        gamma_in_sys = np.kron(id, neutral_gauge)
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

        return gamma_in_sys, (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec), (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec)


    def generate_gamma_gauge_neutral(self):
        """Generate the the covariance matrix of the ungauged projectors.
        The morde order is {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy. 
        Modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in Z2System2DBase.

        Returns:
            np.ndarray: Covariance matrix of the ungauged projector on a single link
        """
        return np.real_if_close(1.j*np.kron(utils.paulix,np.kron(utils.pauliy, utils.paulix)))

    #Gauging

    def generate_rotmat(self,theta,coord):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.
        The mode order is (as for gamma_in_neutral) {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2}, depending on whether the link is vertical or horizontal.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy. 
        Modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in Z2System2DBase.

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
        # We rotate the different copies in the SAME way.
        dest = block_diag(rot_left, rot_right, rot_left, rot_right)
        return dest


    def update_gauge_ind(self, link_ind, theta):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be prohibitively expensive.

        This method overwrites an abstract method in Z2System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (float): New gauge field value
        """
        # Update the gaugefield
        self._gaugefieldvec[link_ind] = theta
        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord(link_ind)
        rotmat = self.generate_rotmat(theta, coord)
        gamma_in_subst = rotmat @ self.gamma_neutral_gauge @ np.transpose(rotmat)
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
        self.gamma_in_sys[ind_mat:ind_mat + rotmat.shape[0],
                          ind_mat:ind_mat + rotmat.shape[1]] = gamma_in_subst
        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()


    # Observables

    def _compute_el_energy_op_vec_and_grad(self, use_trans_inv=True):
        """Computation of the electric energy and the electric gradient in a single method.
        Since many operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

        This method overwrites an abstract method in Z2System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            tuple: Tuple of (list of electric energies for a single link, list of gradients for the full system)
        """
        if use_trans_inv:
            lognormvec_default = self.calculate_lognormvec_inc(all_factors=True)
            # This is the usual norm without any modifications
            lognorm_default = np.sum(lognormvec_default)
            # Number of fermions = # of sites
            # Since we have 2 copies, we get 8 virtual fermions per site
            single_link_offset = 2 * self.cfg.nvirtmodes_link
            offset = 2 * self.cfg.lattice.size + single_link_offset
            # We have to cut one link from gamma_in_sys as well
            gamma_in_sys_mod = self.gamma_in_sys_mod
            nlinks = self.cfg.lattice.nlinks
            dest = []
            dest_grad = []

            for layerind in range(self.cfg.nlayer):
                layer_derivative=[]
                #We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
                mat_a = self.mat_a_mod_vec[layerind]
                mat_b = self.mat_b_mod_vec[layerind]
                diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()
                diff_d_inv_gamma_inv = self.wi_gamma_in_mod_vec[layerind].inv()

                ###################### Calculation of <P> ########################
                covmat_out = mat_a + \
                    mat_b @ diff_d_gamma_inv @ np.transpose(mat_b)
                covmat_out_virt = covmat_out[-single_link_offset:, -
                                            single_link_offset:]

                # The library pfapack is rather picky about the anti-symmtrization (to 1e-14)
                covmat_out_virt = utils.anti_symmetrize(covmat_out_virt)
                # For the modified norm, we still have to take into account the other contributions from the unmodified parts
                norm_mod = calculate_lognorm_inc(
                    [self.incdet_mod_vec[layerind]],
                    [self.det_mat_d_mod_vec[layerind]],
                    gamma_in_sys_mod.shape[0],
                    all_factors=True)
                norm_mod += np.sum(utils.select_except(lognormvec_default,layerind))
                # The matrix elements yield only the real part of <P>
                # If we use the log formulation, we can calculate the log of single terms.

                # Instead of writing down all the terms explicitly, we build tuples of the prefactors and the indices of the covariance matrix.
                # Then, we compute all terms in a list comprehension.
                idxarr= [ (   1, [0,2,4,6]), (  -1, [1,2,4,7]), (-1.j, [0,1,2,4]), (-1.j,[2,4,6,7]),
                          (  -1, [0,3,5,6]), (   1, [1,3,5,7]), ( 1.j, [0,1,3,5]), ( 1.j,[3,5,6,7]),
                          ( 1.j, [0,4,5,6]), (-1.j, [1,4,5,7]), (   1, [0,1,4,5]), (   1,[4,5,6,7]),
                          ( 1.j, [0,2,3,6]), (-1.j, [1,2,3,7]), (   1, [0,1,2,3]), (   1,[2,3,6,7])]
                pfarr = [prefactor * pf.pfaffian(covmat_out_virt[np.ix_(ind,ind)]) for prefactor,ind in idxarr]
                el_energy_full = 1/16 * np.sum(pfarr)
                
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
                    d_el_energy = 1/16 * np.sum([prefactor * utils.derivative_pfaffian(covmat_out_virt[np.ix_(ind,ind)],d_covmat_out_virt[np.ix_(ind,ind)]) for prefactor,ind in idxarr])
                                  
                    # Summand with derivative of norms
                    trace_def = self.compute_grad_over_norm(symbol, layerind)
                    trace_mod = compute_grad_over_norm(gamma_in_sys_mod, diff_d_inv_gamma_inv, d_mat_d, self.mat_d_mod_inv_vec[layerind])
                    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                    d_el_energy += dest[layerind] * (trace_mod - trace_def)
                    # Scale to system size
                    d_el_energy *= nlinks
                    layer_derivative.append(np.real(d_el_energy))
                dest_grad.append(layer_derivative)
            # We have to weight the different layers with the electric energy operator expectation of the other layers.
            # They act as a prefactor in the derivative
            dest = np.asarray(dest)
            dest_grad = np.asarray(dest_grad)
            if self.cfg.nlayer > 1:
                for i in range(self.cfg.nlayer):
                    prod_other_layers = utils.multiply_except(dest, i)
                    dest_grad[i] *= prod_other_layers
        else:
            # Evaluate every link of the system
            logging.error("compute_el_energy: not implemented yet")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")
            dest = np.asarray([None]*self.cfg.nlayer)
            dest_grad = np.asarray([[None]*len(self.symbolvec)]*self.cfg.nlayer)
        return dest, dest_grad


    def _compute_mag_energy_op(self, use_trans_inv=True):
        """Computation of the magnetic energy operator (w/o shift).
        This operator is diagonal in the gauge field (group element) basis and can thus be computed easily.

        This method overwrites an abstract method in Z2System2DBase.

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
            logging.error("compute_mag_energy: not implemented yet")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")
            mag_energy_bare = None
        return mag_energy_bare