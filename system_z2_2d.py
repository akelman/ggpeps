import numpy as np
import logging
import sys
import sympy
import lattice as lat
import gauge
import logging
import sympy
from scipy.linalg import block_diag
import utils

################## Utility Functions ######################

def extract_partial_covmats(mat,corner):
    """Extract the partial covariance matrices from a gaussian mapping

    Args:
        mat (np.ndarray): Full covariance matrix
        corner (int): Index of the top left element of the bottom right matrix

    Returns:
        tuple: Matrices (A,B,D)
    """
    mat_a = mat[:corner, :corner]
    mat_b = mat[:corner, corner:]
    mat_d = mat[corner:, corner:]
    return mat_a, mat_b, mat_d

###################### Z2System2D ##########################


class Z2System2DConfig:
    def __init__(self, params, lattice, g2, g_gm, g_mag):
        #The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        if not self.check_params(params):
            logging.error("The set of parameters is not consistent.")
            sys.exit(1)
        self.paramvec = np.asarray(params)
        self.nlayer = self.paramvec.shape[0]
        self.lattice = lattice

        #Parameters of the Hamiltonian
        self.g2 = g2
        self.g_el = g2/2
        if g_mag is None:
            self.g_mag = 1./(2*g2)
        else:
            self.g_mag = g_mag
        self.g_gm = g_gm

    def check_params(self,params):
        """Check the consistency of the input parameters.
        All arrays must have the same length.

        Args:
            params (list or np.ndarray): two dimensional array of input parameters
        """
        lenvec = np.asarray([len(x) for x in params])
        #We know that we need 3 parameters for each layer
        return np.all(lenvec == 3)

    def nvarparams(self):
        return 3*self.nlayer

class Z2System2D:
    def __init__(self, cfg: Z2System2DConfig):
        self.cfg = cfg

        # Parameter dependent quantities
        self._tmat_vec = None
        self._gamma_dirac_vec = None
        self._gamma_maj_vec = None
        self._gamma_maj_sys_vec = None
        self._mat_a_vec = None
        self._mat_b_vec = None
        self._mat_d_vec = None
        self._mat_d_inv_vec = None

        # Management of the gaugefields
        self.gamma_neutral_gauge = self.generate_gamma_gauge_neutral()
        self._gamma_in_sys = None
        self._gaugefieldvec = np.zeros(self.cfg.lattice.nlinks)
        self.gaugemgr = gauge.ZNGauge(2)

        # Observables
        self._energy = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._mag_energy_op = None

        # Weight
        self._weight = None

        # Gradients
        self._gamma_maj_sys_deriv_y_vec = None
        self._gamma_maj_sys_deriv_z_vec = None
        self._gamma_maj_sys_deriv_t_vec = None
        self._el_energy_op_grad_vec = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in_vec = None   #Tracks (D^-1 - gammain)^-1
        self._wi_gamma_out_vec = None  #Tracks (D - gammain)^-1
        self._incdet_vec = None        #Tracks det(D^-1 - gammain)

    def initialize(self):
        """Initialization function. 
        This is a good spot to copy essential data from the configuration.
        """
        return None

    def _compute_tmat(self, paramvec):
        #Order of the paramvec: [t,y,z]
        t = paramvec[0]
        y = paramvec[1]
        z = paramvec[2]
        tmat_eval = self.tmat_symb.evalf(subs={
            self.symbolvec[0]: t,
            self.symbolvec[1]: y,
            self.symbolvec[2]: z
        })
        return np.asarray(tmat_eval).astype(complex)

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

    def compute_tmat_deriv(self,symb):
        tmat_symb = self.tmat_symb
        return np.asarray(sympy.diff(tmat_symb, symb)).astype(complex)

    @property
    def tmat_vec(self):
        """
        Generate the T-matrix vector(single virtual fermion on the link).
        he mode ordering of this matrix is {p,l,r,d,u}.
        Analytically, this mode order is not advantageous, 
        but is makes the reshuffling of the modes easier for gamma_in and M_D in the covariance matrix.

        In the analytical part, we will keep the mode ordering {p,r,u,l,d} because the transformation matrices are easier to cope with.
        In the numerics, however, we will stick with {p,l,r,d,u}

        Returns:
            [np.array]: parameter matrix T
        """
        if self._tmat_vec is None:
            self._tmat_vec = [
                self._compute_tmat(params) for params in self.cfg.paramvec
            ]
        return self._tmat_vec

    def compute_gamma_dirac_deriv(self, symb, layerind):
        deriv_t=self.compute_tmat_deriv(symb)
        tmat=self.tmat_vec[layerind]
        tmatc=np.conjugate(tmat)
        idttinv_minus = np.linalg.inv(np.eye(deriv_t.shape[0]) - tmat @ tmatc)
        idtt_plus = np.eye(deriv_t.shape[0]) + tmat @ tmatc
        d_idtt_minus = -(deriv_t @ tmatc + tmat @ np.conjugate(deriv_t))
        d_idtt_plus = -d_idtt_minus
        d_lt = idttinv_minus @ d_idtt_minus @ idttinv_minus @ tmat - idttinv_minus @ deriv_t
        d_rt = 0.5 * idttinv_minus @ d_idtt_plus @ idttinv_minus @ idtt_plus + 0.5 * idttinv_minus @ d_idtt_plus
        d_lb = -np.conjugate(d_rt)
        d_rb = -np.conjugate(d_lt)
        return 1.j*np.block([[d_lt, d_rt], [d_lb, d_rb]])

    @property
    def gamma_dirac_vec(self):
        """Return the vector of covariance matrices in dirac modes.
        The mode order of this matrix is {p,l,r,d,u,p_dag,l_dag,r_dag,u_dag,d_dag}.

        Returns:
            [np.array]: Vector of covariance matrices in Dirac modes
        """
        if self._gamma_dirac_vec is None:
            self._gamma_dirac_vec = [
                utils.tmat_to_covariance_matrix(tmat) for tmat in self.tmat_vec
            ]
        return self._gamma_dirac_vec

    @property
    def gamma_maj_vec(self):
        """Return the covariance matrix in Majorana modes.
        The mode order of this matrix is {p_1,p_2,l_1,l_2,r_1,r_2,d_1,d_2,u_1,u_2}.
        The definition of Majorana modes used is
            \\gamma_1=c+c^\\dagger
            \\gamma_2=i(c-c^\\dagger)

        Returns:
            [np.array]: Covariance matrix in Majorana modes
        """
        if self._gamma_maj_vec is None:
            # We know that the gamma dirac matrices have all the same shape
            m, _ = self.gamma_dirac_vec[-1].shape
            smat = utils.generate_smat(m)
            self._gamma_maj_vec = [
                np.real(smat @ gamma_dirac @ np.transpose(smat))
                for gamma_dirac in self.gamma_dirac_vec
            ]
        return self._gamma_maj_vec

    def compute_gamma_maj_deriv(self,symb,layerind):
        gamma_dirac_deriv = self.compute_gamma_dirac_deriv(symb,layerind)
        m, _ = gamma_dirac_deriv.shape
        smat = utils.generate_smat(m)
        return np.real(smat@gamma_dirac_deriv@np.transpose(smat))


    def _expand_gamma_maj_to_system(self,covmat):
        permbuilder = lat.PermutationBuilderGMS2D(self.cfg.lattice, nmodes_per_link=1)
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

    @property
    def gamma_maj_sys_vec(self):
        """Return the covariance matrix of the full system in Majorana modes.
        The mode order is changed to fit the mode order of gamma_in.
        See documentation of gamma_in for details.

        Returns:
            [np.array]: Covariance matrix of the full system
        """
        if self._gamma_maj_sys_vec is None:
            self._gamma_maj_sys_vec = [
                self._expand_gamma_maj_to_system(gamma_maj)
                for gamma_maj in self.gamma_maj_vec
            ]
        return self._gamma_maj_sys_vec

    def initialize_gamma_in_sys(self):
        nlinks = self.cfg.lattice.nlinks
        id = np.eye(nlinks)
        neutral_gauge = self.gamma_neutral_gauge
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
        return gamma_in_sys, wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec

    @property
    def gamma_in_sys(self):
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

        Returns:
            np.ndarray: Covariance matrix of the projectors (full-system size)
        """
        if self._gamma_in_sys is None:
            self._gamma_in_sys, self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = self.initialize_gamma_in_sys()
        return self._gamma_in_sys

    @property
    def incdet_vec(self):
        if self._incdet_vec is None:
            self._gamma_in_sys, self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = self.initialize_gamma_in_sys()
        return self._incdet_vec

    @property
    def wi_gamma_in_vec(self):
        if self._wi_gamma_in_vec is None:
            self._gamma_in_sys, self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = self.initialize_gamma_in_sys()
        return self._wi_gamma_in_vec

    @property
    def wi_gamma_out_vec(self):
        if self._wi_gamma_out_vec is None:
            self._gamma_in_sys, self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = self.initialize_gamma_in_sys()
        return self._wi_gamma_out_vec

    def _exract_partial_covmatvec(self):
        #We are assuming one physical mode per site
        nsites = self.cfg.lattice.size
        mat_a_vec = []
        mat_b_vec = []
        mat_d_vec = []
        for ind in range(self.cfg.nlayer):
            mat_a, mat_b, mat_d = extract_partial_covmats(
                self.gamma_maj_sys_vec[ind], 2 * nsites)
            mat_a_vec.append(mat_a)
            mat_b_vec.append(mat_b)
            mat_d_vec.append(mat_d)
        return mat_a_vec, mat_b_vec, mat_d_vec

    @property
    def mat_a_vec(self):
        """Extract the matrix for physical-physical correlations.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.

        Returns:
            [np.array]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_vec is None:
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec()
        return self._mat_a_vec

    @property
    def mat_b_vec(self):
        """Extract the matrix for physical-virtual correlations.

        Returns:
            [np.array]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_vec is None:
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec()
        return self._mat_b_vec

    @property
    def mat_d_vec(self):
        """Extract the matrix for virtual-virtual correlations.

        Returns:
            [np.array]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_vec is None:
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec()
        return self._mat_d_vec

    @property
    def mat_d_inv_vec(self):
        if self._mat_d_inv_vec is None:
            self._mat_d_inv_vec = [
                np.linalg.inv(mat_d) for mat_d in self.mat_d_vec
            ]
        return self._mat_d_inv_vec

    @property
    def gaugefieldvec(self):
        return self._gaugefieldvec

    @gaugefieldvec.setter
    def gaugefieldvec(self,val):
        print(
            "Do not set the gaugefieldvec explicitly. Use 'update_gauge_ind'.", file=sys.stderr)

    @property
    def weight(self):
        if self._weight is None:
            self._weight=1.0
            for ind in range(self.cfg.nlayer):
                self._weight *= 0.5 * self.incdet_vec[ind].det()
        return self._weight

    @weight.setter
    def weight(self,val):
        self._weight = val


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

    def invalidate_gauge_update(self):
        self._energy = None
        self._mag_energy_op = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._el_energy_op_grad_vec = None

    def calculate_update_gamma_in(self,offset,update_mat):
        m_up, n_up = update_mat.shape
        gamma_in_old = self.gamma_in_sys[offset:offset + m_up,
                                         offset:offset + n_up]
        return -(update_mat - gamma_in_old)

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
        # Update the weight
        self.weight = 0.5 * np.sum(detval_vec)
        # Update the matrix inversion
        [ wi_gamma_in.update_index(update, ind_mat, ind_mat) for wi_gamma_in in self.wi_gamma_in_vec ]
        [ wi_gamma_out.update_index(update, ind_mat, ind_mat) for wi_gamma_out in self.wi_gamma_out_vec ]
        # Substitute in the array
        self.gamma_in_sys[ind_mat:ind_mat + 4,
                          ind_mat:ind_mat + 4] = gamma_in_subst
        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    def update_gauge_full_system(self,gaugeconfig):
        for ind,gauge in enumerate(gaugeconfig):
            self.update_gauge_ind(ind, gauge)

    def calculate_weight_attempt(self, link_ind, theta, all_factors=False):
        # There are two directions per vertex and two Majoranas per link
        ind_mat = 4 * link_ind
        rotmat = self.generate_rotmat(theta)
        gamma_in_subst = rotmat @ self.gamma_neutral_gauge @ np.transpose(rotmat)
        update = self.calculate_update_gamma_in(ind_mat, gamma_in_subst)
        return self.calculate_lognorm_inc(ind_mat, update, all_factors)


    def update_gauge_coord(self,coord,dir,theta):
        ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        self.update_gauge_ind(ind, theta)

    # Calculating the norm

    def calculate_lognorm(self,all_factors=False):
        return calculate_lognorm(self.gamma_in_sys, self.mat_d_vec, all_factors=all_factors)

    def calculate_lognorm_inc(self, offset, update, all_factors=False):
        cumval=0
        for ind in range(self.cfg.nlayer):
            detval = self.incdet_vec[ind].update_index(self.wi_gamma_in_vec[ind].inv(), update,
                                            offset, offset, store=False)
            if all_factors:
                detval-=self.gamma_in_sys.shape[0]*np.log(2)
                detval+=np.linalg.slogdet(self.mat_d_vec[0])[1]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            cumval += 0.5 * detval
        return cumval


    # Calculate gradients

    def compute_grad_norm_vec(self) -> np.ndarray:
        #The parameter order is [[dt1, dy1, dz1],[dt2,dy2,dz2]...]
        dest=[]
        for layerind in range(self.cfg.nlayer):
            dest.append(self.compute_grad_norm(layerind))
        return np.asarray(dest)

    def compute_grad_norm(self, layerind: int) -> np.ndarray:
        #The parameter order is [dt, dy, dz]
        dest=np.zeros(3)
        dest[0]=self.compute_grad_over_norm("t",layerind)
        dest[1]=self.compute_grad_over_norm("y",layerind)
        dest[2]=self.compute_grad_over_norm("z",layerind)
        return dest

    def gamma_maj_sys_deriv_vec(self, var: str) -> np.ndarray:
        if var == "t":
            return self.gamma_maj_sys_deriv_t_vec
        elif var == "y":
            return self.gamma_maj_sys_deriv_y_vec
        elif var == "z":
            return self.gamma_maj_sys_deriv_z_vec
        print("gamma_maj_sys_deriv: Invalid variable name", sys.stderr)
        return None

    def compute_grad_over_norm(self, var: str, layerind: int) -> float:
        """Compute the quotient of derivative of the norm over the norm itself.
        We can avoid a lot of factors by computing the quotient directly.

        Args:
            var (str): Name of the variable (t,y,z)
            layerind (int): Index of the layer

        Returns:
            float: Value of the gradient divided by the norm of the state
        """
        diff = self.wi_gamma_in_vec[layerind].inv()
        # 2 phys. Majorana modes per vertex
        offset = 2 * self.cfg.lattice.size
        # Extract only the part of the virtual-virtual correlations
        deriv_d = self.gamma_maj_sys_deriv_vec(var)[layerind][offset:, offset:]
        mat_d_inv=self.mat_d_inv_vec[layerind]
        return compute_grad_over_norm(self.gamma_in_sys, diff, deriv_d, mat_d_inv)


    # Observables

    @property
    def energy(self):
        if self._energy is None:
            self._energy = self.el_energy + self.mag_energy
        return self._energy

    @property
    def mag_energy_op(self):
        if self._mag_energy_op is None:
            nplaq = self.cfg.lattice.nplaquettes
            self._mag_energy_op = nplaq * self._compute_mag_energy_op()
        return self._mag_energy_op

    @property
    def el_energy_op(self):
        if self._el_energy_op is None:
            # The different layers can be separated into separate PEPS.
            nlinks = self.cfg.lattice.nlinks
            self._el_energy_op = nlinks * np.prod(self.el_energy_op_vec)
        return self._el_energy_op

    @property
    def el_energy_op_vec(self):
        if self._el_energy_op_vec is None:
            # This vector is the electric energy on a single link.
            # Otherwise, we get a power of nlinks in the product and the electric energy term (with prefactors) gets negative
            self._el_energy_op_vec = self._compute_el_energy_op_vec()
        return self._el_energy_op_vec

    @property
    def mag_energy(self):
        nplaq = self.cfg.lattice.nplaquettes
        mag_energy = self.cfg.g_mag * (2*nplaq - 2*self.mag_energy_op)
        return mag_energy

    @property
    def el_energy(self):
        nlinks=self.cfg.lattice.nlinks
        el_energy = self.cfg.g_el * (2*nlinks - 2*self.el_energy_op)
        return el_energy

    @property
    def el_energy_op_grad_vec(self):
        if self._el_energy_op_grad_vec is None:
            self._el_energy_op_grad_vec = self._compute_el_energy_op_grad_vec()
        return self._el_energy_op_grad_vec


    def _compute_el_energy_op_vec(self, use_trans_inv=True):
        if use_trans_inv:
            gamma_in_sys = self.gamma_in_sys
            norm_default=self.calculate_lognorm()
            # Number of fermions = # of sites
            single_site_offset = 4
            offset = 2 * self.cfg.lattice.size + single_site_offset
            # We have to cut one link from gamma_in_sys as well
            gamma_in_sys_tilde = gamma_in_sys[single_site_offset:,
                                            single_site_offset:]
            el_energy_bare=[]
            for ind in range(self.cfg.nlayer):
                #We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
                gamma_maj_sys = self.gamma_maj_sys_vec[ind]
                mat_a, mat_b, mat_d = extract_partial_covmats(gamma_maj_sys, offset)
                covmat_out = mat_a + \
                    mat_b @ np.linalg.inv(mat_d -
                                        gamma_in_sys_tilde) @ np.transpose(mat_b)
                covmat_out_virt = covmat_out[-single_site_offset:, -
                                            single_site_offset:]
                # The matrix elements yield only the real part of <P>
                norm_mod = calculate_lognorm(gamma_in_sys_tilde, [mat_d])
                el_energy_layer = 0.5 * (
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
        norm_default=self.calculate_lognorm()
        for layerind in range(self.cfg.nlayer):
            layer_derivative=[]
            # The matrices must be re-extracted here since we slice at different positions than usually
            # The offset is changed such that one virtual link is attributed to the physical part
            _, mat_b, mat_d = extract_partial_covmats(self.gamma_maj_sys_vec[layerind], offset)
            #TODO: We could also track this inverse
            diff_d_gamma_inv = np.linalg.inv(mat_d - gamma_in_sys_tilde)
            #TODO: This inverse can be calculated once and stored afterwards
            mat_d_inv=np.linalg.inv(mat_d)
            diff_d_inv_gamma_inv = np.linalg.inv(mat_d_inv - gamma_in_sys_tilde)

            norm_mod = calculate_lognorm(gamma_in_sys_tilde, [mat_d])
            for var in ["t", "y", "z"]:
                deriv_gamma_maj_sys = self.gamma_maj_sys_deriv_vec(var)[layerind]
                d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats(deriv_gamma_maj_sys, offset)
                gamma_out = d_mat_a + \
                        d_mat_b @ diff_d_gamma_inv @ np.transpose(mat_b) \
                        + mat_b @ diff_d_gamma_inv @ np.transpose(d_mat_b) \
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ np.transpose(mat_b)
                # The virtual mode is the last link on the bottom right of the covariance matrix
                covmat_out_virt = gamma_out[-single_site_offset:,
                                            -single_site_offset:]
                # Summand with derivative of the covariance matrix
                d_el_energy = 0.5 * (covmat_out_virt[0, 1] + covmat_out_virt[2, 3]) * np.exp(norm_mod - norm_default)
                # Summand with derivative of norms
                trace_def = self.compute_grad_over_norm(var, layerind)
                trace_mod = compute_grad_over_norm(gamma_in_sys_tilde, diff_d_inv_gamma_inv, d_mat_d, mat_d_inv)
                # In contrast to the formula, it seems like we have an additional factor of -2 in the equation.
                # This is due to the definition of compute_grad_over_norm with a factor of -1/2
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

    @property
    def gamma_maj_sys_deriv_t_vec(self):
        if self._gamma_maj_sys_deriv_t_vec is None:
            self._gamma_maj_sys_deriv_t_vec = [self._expand_gamma_maj_to_system(self.compute_gamma_maj_deriv(self.symbolvec[0], i)) for i in range(self.cfg.nlayer)
                                               ]
        return self._gamma_maj_sys_deriv_t_vec

    @property
    def gamma_maj_sys_deriv_y_vec(self):
        if self._gamma_maj_sys_deriv_y_vec is None:
            self._gamma_maj_sys_deriv_y_vec = [
                self._expand_gamma_maj_to_system(self.compute_gamma_maj_deriv(self.symbolvec[1], i)) for i in range(self.cfg.nlayer)
            ]
        return self._gamma_maj_sys_deriv_y_vec

    @property
    def gamma_maj_sys_deriv_z_vec(self):
        if self._gamma_maj_sys_deriv_z_vec is None:
            self._gamma_maj_sys_deriv_z_vec = [self._expand_gamma_maj_to_system(self.compute_gamma_maj_deriv(self.symbolvec[2], i)) for i in range(self.cfg.nlayer)
                                               ]
        return self._gamma_maj_sys_deriv_z_vec

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

    def compute_path(self,path):
        """Compute the observable corresponding the path given as an argument

        Args:
            path (list): List of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.
            This is the case if the link is traversed from right to left or from top to bottom.
        """
        theta_sum=0.
        for ind, conj in path:
            if conj:
                theta_sum+=self.gaugefieldvec[ind]
            else:
                theta_sum+=self.gaugefieldvec[ind]
        return np.exp(1.j*theta_sum)

    def compute_ferm_cov(self):
        """Compute the covariance matrix of the fermions in the system """
        return self.mat_a + self.mat_b@self.wi_gamma_out.inv()@np.transpose(self.mat_b)


def calculate_lognorm(gamma_in_sys: np.ndarray,
                      mat_d_vec: np.ndarray,
                      all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    cumval = 0
    nlayer=len(mat_d_vec)
    for ind in range(nlayer):
        mat_d = mat_d_vec[ind]
        if all_factors:
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d) / 2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))
        cumval += logval
    #The factor 1/2 is the square-root
    return cumval / 2


def compute_grad_over_norm(gamma_in_sys: np.ndarray, diff: np.ndarray,
                           deriv_d: np.ndarray,
                           mat_d_inv: np.ndarray) -> float:
    """Compute the gradient of the norm divided by the norm.
    The expression of deriv_d given to this function decides which derivative is computed

    Args:
        gamma_in_sys (np.ndarray): Gauged covariance matrix of the projectors
        diff (np.ndarray): (D^{-1}-gamma_in_sys)^{-1}
        deriv_d (np.ndarray): dD/d{alpha}: Derivative of the virtual-virtual covariance matrix
        mat_d_inv (np.ndarray): Inverse of D: D^{-1}

    Returns:
        float: Gradient of the norm divided by the norm.
    """
    # Extract only the part of the virtual-virtual correlations
    dest = -0.5 * np.trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
    return dest
