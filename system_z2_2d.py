import numpy as np
import logging
import sys
from scipy.linalg.misc import norm
import sympy
import lattice as lat
import logging
import sympy
from scipy.linalg import block_diag
import utils
from system_base import Z2System2DConfigBase, Z2System2DBase
from system_base import calculate_lognorm, compute_grad_over_norm, calculate_lognormvec, extract_partial_covmats, calculate_lognormvec_inc,calculate_lognorm_inc


###################### Z2System2D ##########################


class Z2System2DConfig(Z2System2DConfigBase):
    _nparams = 3
    ncopy = 1
    nvirtmodes_vertex = 4 # We have one virtual mode per direction
    nvirtmodes_link = 2

    def __init__(self, lattice, g2, g_gm, g_mag, nlayer=1):
        #The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        super().__init__(lattice, g2, g_gm, g_mag,nlayer)

class Z2System2D(Z2System2DBase):
    def __init__(self, cfg: Z2System2DConfig):
        super().__init__(cfg)

        # Parameter dependent quantities

        # In the analytical part, we will keep the mode ordering {p,r,u,l,d} because the transformation matrices are easier to cope with.
        # In the numerics, however, we will stick with {p,l,r,d,u}
        self._tmat_vec = None #Order of the paramvec: [t,y,z]
        self._gamma_dirac_vec = None # Mode Order:  {p,l,r,d,u,p_dag,l_dag,r_dag,u_dag,d_dag}.
        self._gamma_maj_vec = None # Mode Order: {p_1,p_2,l_1,l_2,r_1,r_2,d_1,d_2,u_1,u_2}.
        self._gamma_maj_sys_vec = None


    @property
    def symbolvec(self):
        t = sympy.Symbol("t", real=True)
        y = sympy.Symbol("y", real=True)
        z = sympy.Symbol("z", real=True)
        return [t,y,z]

    @property
    def tmat_symb(self):
        [t, y, z] = self.symbolvec
        tmat_symb=sympy.Matrix([[0, -1.j * t, 1.j * t, t, -t],
                            [1.j * t, 0, 1.j * y, z, 1.j * z],
                            [-1.j * t, -1.j * y, 0, -1.j * z, -z],
                            [-t, -z, 1.j * z, 0, -y], [t, -1.j * z, z, y, 0]])
        return tmat_symb


    def _expand_gamma_maj_to_system(self,covmat):
        permbuilder = lat.PermutationBuilderGMS2D(self.cfg.lattice, nmodes_per_link=1)
        mat_perm = permbuilder.perm()
        nsites = self.cfg.lattice.size
        id = np.eye(nsites)
        # Extract the parts of the covariance matrix
        # The 2 is the number of physical fermionic Majorana modes
        amat, bmat, dmat = extract_partial_covmats(covmat, 2)
        #Expand them
        amat_sys = np.kron(id, amat)
        bmat_sys = np.kron(id, bmat)
        dmat_sys = np.kron(id, dmat)
        #Reassemble them in the correct order
        mat_sys_unordered= np.block(
            [[amat_sys, bmat_sys], [-np.transpose(bmat_sys), dmat_sys]])
        #utils.show_matrix(mat_perm,"mat_perm")
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

        For a 2x2 system, gamma_in has the order {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.
        The modes are named as <mode letter>_<vertex site>. Each constitent in the list above labels two Majorana modes.
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
        """This matrix is the covariance matrix of the ungauged projectors.
        The morde order is {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2}, where the underscore notation explicitly denotes Majorana modes and not sites.
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        Returns:
            np.ndarray: Covariance matrix of the ungauged projector on a single link
        """
        return np.real_if_close(1.j*np.kron(utils.pauliy, utils.paulix))

    #Gauging

    def generate_rotmat(self,theta):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.
        The mode order is (as for gamma_in_neutral) {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2}, depending on whether the link is vertical or horizontal.

        Args:
            theta (float): Angle of rotation

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


    def update_gauge_ind(self, ind, theta):
        # Update the gaugefield
        self._gaugefieldvec[ind] = theta
        # There are two directions per vertex and two Majoranas per link
        ind_mat = 4 * ind
        rotmat = self.generate_rotmat(theta)
        gamma_in_subst = rotmat @ self.gamma_neutral_gauge @ np.transpose(
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


    def calculate_weight_attempt(self, link_ind, theta, all_factors=False):
        # There are two directions per vertex and two Majoranas per link
        ind_mat = 4 * link_ind
        rotmat = self.generate_rotmat(theta)
        gamma_in_subst = rotmat @ self.gamma_neutral_gauge @ np.transpose(rotmat)
        update = self.calculate_update_gamma_in(ind_mat, gamma_in_subst)
        return self.update_lognorm_inc(ind_mat, update, all_factors)


    # Calculating the norm


    def _compute_el_energy_op_vec(self, use_trans_inv=True):
        if use_trans_inv:
            gamma_in_sys = self.gamma_in_sys
            normvec_default = calculate_lognormvec(self.gamma_in_sys,
                                                self.mat_d_vec,all_factors=True)
            # This is the usual norm without any modifications
            norm_default = np.sum(normvec_default)
            # Number of fermions = # of sites
            single_site_offset = 4
            offset = 2 * self.cfg.lattice.size + single_site_offset
            # We have to cut one link from gamma_in_sys as well
            gamma_in_sys_tilde = gamma_in_sys[single_site_offset:,
                                            single_site_offset:]
            el_energy_bare=[]
            for layerind in range(self.cfg.nlayer):
                #We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
                gamma_maj_sys = self.gamma_maj_sys_vec[layerind]
                mat_a, mat_b, mat_d = extract_partial_covmats(gamma_maj_sys, offset)
                #utils.show_matrix(mat_d)
                covmat_out = mat_a + \
                    mat_b @ np.linalg.inv(mat_d -
                                        gamma_in_sys_tilde) @ np.transpose(mat_b)
                covmat_out_virt = covmat_out[-single_site_offset:, -
                                            single_site_offset:]
                # For the modified norm, we still have to take into account the other contributions from the unmodified parts
                # utils.show_matrix(covmat_out_virt)
                norm_mod = calculate_lognorm(gamma_in_sys_tilde, [mat_d],all_factors=True)
                norm_mod += np.sum(utils.select_except(normvec_default,layerind))
                # The matrix elements yield only the real part of <P>
                el_energy_layer = 0.25*(
                    covmat_out_virt[0, 1] +
                    covmat_out_virt[2, 3]) * np.exp(norm_mod - norm_default)
                el_energy_bare.append(el_energy_layer)
        else:
            # Evaluate every link of the system
            logging.error("compute_el_energy: not implemented yet")
            el_energy_bare = [None]*self.cfg.nlayer
        return np.asarray(el_energy_bare)


    def _compute_el_energy_op_grad_vec(self):
        """The electric energy depends explicitly on the parameters of the Ansatz. 
        Thus, we have to build the explicit derivative of the electric energy with respect to the parameters.

        Args:
            var (str): Name of the variable (t,y,z)

        Returns:
            list: Matrix of the gradients of the electric energy wrt [[t1,y1,z1],[t2,y2,z2],...]
        """
        single_site_offset = 4
        offset = 2 * self.cfg.lattice.size + single_site_offset
        nlinks = self.cfg.lattice.nlinks
        dest = []
        # We have to cut one link from gamma_in_sys as well
        gamma_in_sys_tilde = self.gamma_in_sys[single_site_offset:,
                                               single_site_offset:]
        normvec_default = calculate_lognormvec(self.gamma_in_sys,
                                               self.mat_d_vec,all_factors=True)

        # This is the usual norm without any modifications
        norm_default = np.sum(normvec_default)
        for layerind in range(self.cfg.nlayer):
            layer_derivative=[]
            # The matrices must be re-extracted here since we slice at different positions than usually
            # The offset is changed such that one virtual link is attributed to the physical part
            _, mat_b, mat_d = extract_partial_covmats(self.gamma_maj_sys_vec[layerind], offset)
            #TODO: We could also track this inverse
            diff_d_gamma_inv = np.linalg.inv(mat_d - gamma_in_sys_tilde)
            #TODO: This inverse can be calculated once and stored afterwards
            mat_d_inv = np.linalg.inv(mat_d)
            diff_d_inv_gamma_inv = np.linalg.inv(mat_d_inv - gamma_in_sys_tilde)

            # For the modified norm, we still have to take into account the other contributions from the unmodified parts
            lognorm_mod = calculate_lognorm(gamma_in_sys_tilde, [mat_d],all_factors=True)
            lognorm_mod += np.sum(utils.select_except(normvec_default,layerind))
            for symbol in self.symbolvec:
                deriv_gamma_maj_sys = self.gamma_maj_sys_deriv_vec(symbol)[layerind]
                d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats(deriv_gamma_maj_sys, offset)
                gamma_out = d_mat_a + \
                        d_mat_b @ diff_d_gamma_inv @ np.transpose(mat_b) \
                        + mat_b @ diff_d_gamma_inv @ np.transpose(d_mat_b) \
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ np.transpose(mat_b)
                # The virtual mode is the last link on the bottom right of the covariance matrix
                covmat_out_virt = gamma_out[-single_site_offset:,
                                            -single_site_offset:]
                # Summand with derivative of the covariance matrix
                d_el_energy = 0.25 * (
                    covmat_out_virt[0, 1] +
                    covmat_out_virt[2, 3]) * np.exp(lognorm_mod - norm_default)
                # Summand with derivative of norms
                trace_def = self.compute_grad_over_norm(symbol, layerind)
                trace_mod = compute_grad_over_norm(gamma_in_sys_tilde, diff_d_inv_gamma_inv, d_mat_d, mat_d_inv)
                d_el_energy += self.el_energy_op_vec[layerind] * (trace_mod - trace_def)
                # Scale to system size
                d_el_energy *= nlinks
                layer_derivative.append(d_el_energy)
            dest.append(layer_derivative)
        # We have to weight the different layers with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative
        dest = np.asarray(dest)
        if self.cfg.nlayer > 1:
            for i in range(self.cfg.nlayer):
                prod_other_layers = utils.multiply_except(self.el_energy_op_vec, i)
                dest[i] *= prod_other_layers
        return dest

    def _compute_el_energy_op_vec_and_grad(self, use_trans_inv=True):
        if use_trans_inv:
            lognormvec_default_inc = self.calculate_lognormvec_inc(all_factors=True)
            # This is the usual norm without any modifications
            lognorm_default = np.sum(lognormvec_default_inc)
            # Number of fermions = # of sites
            # Since we have 1 copy, we get 2 virtual fermions per link, leading to 2 * 2 Majorana modes
            single_link_offset = 2 * self.cfg.nvirtmodes_link
            offset = 2 * self.cfg.lattice.size + single_link_offset
            # We have to cut one link from gamma_in_sys as well
            gamma_in_sys_mod = self.gamma_in_sys_mod
            nlinks = self.cfg.lattice.nlinks
            dest = []
            dest_grad = []

            for layerind in range(self.cfg.nlayer):
                layer_derivative=[]
                # We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
                # The shifted matrices are extracted at the initalization
                # The offset is changed such that one virtual link is attributed to the physical part
                mat_a = self.mat_a_mod_vec[layerind]
                mat_b = self.mat_b_mod_vec[layerind]
                diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()
                diff_d_inv_gamma_inv = self.wi_gamma_in_mod_vec[layerind].inv()

                ###################### Calculation of <P> ########################
                covmat_out = mat_a + mat_b @ self.wi_gamma_out_mod_vec[layerind].inv() @ np.transpose(mat_b)
                covmat_out_virt = covmat_out[-single_link_offset:, -
                                            single_link_offset:]
                # For the modified norm, we still have to take into account the other contributions from the unmodified parts
                norm_mod = calculate_lognorm_inc(
                    [self.incdet_mod_vec[layerind]],
                    [self.det_mat_d_mod_vec[layerind]],
                    gamma_in_sys_mod.shape[0],
                    all_factors=True)
                #norm_mod = calculate_lognorm(gamma_in_sys_mod, [mat_d],
                                             #all_factors=True)
                norm_mod += np.sum(utils.select_except(lognormvec_default_inc,layerind))
                # The matrix elements yield only the real part of <P>
                #el_energy_layer = 0.25*( covmat_out_virt[0, 1] + covmat_out_virt[2, 3] + 1.j*covmat_out_virt[0,2] - 1.j*covmat_out_virt[0,3]) * np.exp(norm_mod - lognorm_default)
                el_energy_layer = 0.25*( covmat_out_virt[0, 1] + covmat_out_virt[2, 3]) * np.exp(norm_mod - lognorm_default)
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
                    d_covmat_out_virt = d_gamma_out[-single_link_offset:,
                                                -single_link_offset:]
                    # Summand with derivative of the covariance matrix
                    d_el_energy = 0.25 * ( d_covmat_out_virt[0, 1] + d_covmat_out_virt[2, 3]) * np.exp(norm_mod - lognorm_default)
                    # Summand with derivative of norms
                    trace_def = self.compute_grad_over_norm(symbol, layerind)
                    trace_mod = compute_grad_over_norm(gamma_in_sys_mod, diff_d_inv_gamma_inv, d_mat_d, self.mat_d_mod_inv_vec[layerind])
                    d_el_energy += dest[layerind] * (trace_mod - trace_def)
                    # Scale to system size
                    d_el_energy *= nlinks
                    layer_derivative.append(d_el_energy)
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
            dest = np.asarray([None]*self.cfg.nlayer)
            dest_grad = np.asarray([[None]*len(self.symbolvec)]*self.cfg.nlayer)
        return dest, dest_grad

    def _compute_mag_energy_op(self, use_trans_inv=True):
        if use_trans_inv:
            # Evaluate one plaquette and multiply by number of plaquettes
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop(
                (0, 0), (1, 1))
            mag_energy_bare = np.real(
                self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            logging.error("compute_mag_energy: not implemented yet")
            mag_energy_bare = None
        return mag_energy_bare