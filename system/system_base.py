from abc import ABC, abstractmethod
import numpy as np
import logging
import sys
import sympy
import utils
import gauge

class Z2System2DConfigBase(ABC):
    _nparams = 1

    def __init__(self, lattice, g2, g_gm, g2_mag, nlayer=1):
        #The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        self.nlayer = nlayer
        self.lattice = lattice

        self._parametervec = None

        #Parameters of the Hamiltonian
        self.g2 = g2
        self.g2_el = g2/2
        if g2_mag is None:
            self.g2_mag = 1./(2*g2)
        else:
            self.g2_mag = g2_mag
        self.g_gm = g_gm

    @property
    def paramvec(self):
        return self._parametervec

    @paramvec.setter
    def paramvec(self,val):
        if self.check_params(val):
            self._parametervec=val
            self.nlayer = len(val)
        else:
            logging.error("The set of parameters is not consistent.")
            sys.exit(1)

    def check_params(self,params):
        """Check the consistency of the input parameters.
        All arrays must have the same length.

        Args:
            params (list or np.ndarray): two dimensional array of input parameters
        """
        lenvec = np.asarray([len(x) for x in params])
        #We know that we need _nparams parameters for each layer
        return np.all(lenvec == self._nparams)

    def nvarparams(self):
        return self._nparams*self.nlayer

    def print_parametervec(self,symbolvec):
        for ind in range(self.nlayer):
            for symb,val in zip(symbolvec, self._parametervec[ind]):
                print(str(symb),val)
    
    @abstractmethod
    def make_pure_gauge(self):
        pass

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


def calculate_lognormvec(gamma_in_sys: np.ndarray,
                      mat_d_vec: np.ndarray,
                      all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    nlayer=len(mat_d_vec)
    dest=np.zeros(nlayer)
    for ind in range(nlayer):
        mat_d = mat_d_vec[ind]
        if all_factors:
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))-mat_d.shape[0]*np.log(2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))
        dest[ind]= logval
    #The factor 1/2 is the square-root
    return dest / 2

def calculate_lognorm(gamma_in_sys: np.ndarray,
                      mat_d_vec: np.ndarray,
                      all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    normvec=calculate_lognormvec(gamma_in_sys,mat_d_vec,all_factors=all_factors)
    return np.sum(normvec)


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


def calculate_lognormvec_inc(incdet_vec, det_mat_d_vec, n, all_factors=False):
    dest=[]
    for ind in range(len(incdet_vec)):
        detval = incdet_vec[ind].det()
        if all_factors:
            detval -= n * np.log(2)
            detval += det_mat_d_vec[ind]
        # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
        # The addition of the cumval is the multiplication of the indpendent PEPS
        dest.append(0.5 * detval)
    return dest


def calculate_lognorm_inc(incdet_vec, det_mat_d_vec, n, all_factors=False):
    lognormvec = calculate_lognormvec_inc(incdet_vec,
                                          det_mat_d_vec,
                                          n,
                                          all_factors=all_factors)
    return np.sum(lognormvec)


################## Z2System2DBase ######################

class Z2System2DBase(ABC):

    def __init__(self, cfg: Z2System2DConfigBase):
        self.cfg = cfg

        # Parameter based matrices
        self._symbolvec = None
        self._tmat_vec = None
        self._gamma_dirac_vec = None
        self._gamma_maj_vec = None
        self._gamma_maj_sys_vec = None

        #Partial covariance matrices 
        self._mat_a_vec = None
        self._mat_b_vec = None
        self._mat_d_vec = None
        self._det_mat_d_vec = None
        self._mat_d_inv_vec = None

        # Parameter dependent quantities for the electric energy
        self._mat_a_mod_vec = None
        self._mat_b_mod_vec = None
        self._mat_d_mod_vec = None
        self._det_mat_d_mod_vec = None
        self._mat_d_mod_inv_vec = None

        # Management of the gaugefields
        self.gamma_neutral_gauge = self.generate_gamma_gauge_neutral()
        self._gamma_in_sys = None
        self._gaugefieldvec = np.zeros(self.cfg.lattice.nlinks)
        self.gaugemgr = gauge.ZNGauge(2)

        # Weight
        self._weight = None
        
        # Gradients
        self._gamma_maj_sys_deriv_dict = None
        self._el_energy_op_grad_vec = None

        # Observables
        self._energy = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._mag_energy_op = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in_vec = None   #Tracks (D^-1 - gammain)^-1
        self._wi_gamma_out_vec = None  #Tracks (D - gammain)^-1
        self._incdet_vec = None        #Tracks det(D^-1 - gammain)

        self._wi_gamma_in_mod_vec = None #Tracks(Dmod^-1 - gammain)^-1
        self._wi_gamma_out_mod_vec = None  #Tracks (Dmod - gammain)^-1
        self._incdet_mod_vec = None #Tracks det(Dmod^-1 - gammain)


    def initialize(self):
        """Initialization function. 
        This is a good spot to copy essential data from the configuration.
        """
        return None
        

    def _exract_partial_covmatvec(self, offset):
        #We are assuming one physical mode per site
        mat_a_vec = []
        mat_b_vec = []
        mat_d_vec = []
        for ind in range(self.cfg.nlayer):
            mat_a, mat_b, mat_d = extract_partial_covmats(
                self.gamma_maj_sys_vec[ind], offset)
            mat_a_vec.append(mat_a)
            mat_b_vec.append(mat_b)
            mat_d_vec.append(mat_d)
        return mat_a_vec, mat_b_vec, mat_d_vec

    @abstractmethod
    def _create_symbolvec(self):
        pass

    @property
    def symbolvec(self):
        if self._symbolvec is None:
            self._symbolvec=self._create_symbolvec()
        return self._symbolvec

    @property
    @abstractmethod
    def tmat_symb(self):
        pass

    def compute_tmat_deriv(self,symb):
        tmat_symb = self.tmat_symb
        return np.asarray(sympy.diff(tmat_symb, symb)).astype(complex)

    def _compute_tmat(self, paramvec):
        tmat_eval = self.tmat_symb.evalf(subs={self.symbolvec[i]:paramvec[i] for i in range(len(paramvec))})
        return np.asarray(tmat_eval).astype(complex)

    @property
    def tmat_vec(self):
        """
        Generate the T-matrix vector(single virtual fermion on the link).
        Analytically, this mode order is not advantageous, 
        but is makes the reshuffling of the modes easier for gamma_in and M_D in the covariance matrix.

        Returns:
            [np.array]: parameter matrix T
        """
        if self._tmat_vec is None:
            self._tmat_vec = [
                self._compute_tmat(params) for params in self.cfg.paramvec
            ]
        return self._tmat_vec

    @property
    def gamma_dirac_vec(self):
        """Return the vector of covariance matrices in dirac modes.

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

    @abstractmethod
    def _expand_gamma_maj_to_system(self,covmat):
        pass

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
            offset = 2 * self.cfg.lattice.size
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec(offset)
        return self._mat_a_vec

    @property
    def mat_b_vec(self):
        """Extract the matrix for physical-virtual correlations.

        Returns:
            [np.array]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_vec is None:
            offset = 2 * self.cfg.lattice.size
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec(offset)
        return self._mat_b_vec

    @property
    def mat_d_vec(self):
        """Extract the matrix for virtual-virtual correlations.

        Returns:
            [np.array]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_vec is None:
            offset = 2 * self.cfg.lattice.size
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec(offset)
        return self._mat_d_vec

    @property
    def det_mat_d_vec (self):
        if self._det_mat_d_vec is None:
            self._det_mat_d_vec = [
                np.linalg.slogdet(mat_d)[1] for mat_d in self.mat_d_vec
            ]
        return self._det_mat_d_vec

    @property
    def mat_d_inv_vec(self):
        if self._mat_d_inv_vec is None:
            self._mat_d_inv_vec = [
                np.linalg.inv(mat_d) for mat_d in self.mat_d_vec
            ]
        return self._mat_d_inv_vec

    @property
    def mat_a_mod_vec(self):
        """Extract the matrix for physical-physical correlations and one virtual mode.
        This shifted matrix is used for the computation of the electric energy.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.

        Returns:
            [np.array]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_mod_vec is None:
            offset = 2 * self.cfg.lattice.size + 2 * self.cfg.nvirtmodes_link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = self._exract_partial_covmatvec(offset)
        return self._mat_a_mod_vec

    @property
    def mat_b_mod_vec(self):
        """Extract the matrix for physical-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)

        Returns:
            [np.array]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_mod_vec is None:
            offset = 2 * self.cfg.lattice.size + 2 * self.cfg.nvirtmodes_link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = self._exract_partial_covmatvec(offset)
        return self._mat_b_mod_vec

    @property
    def mat_d_mod_vec(self):
        """Extract the matrix for virtual-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)

        Returns:
            [np.array]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_mod_vec is None:
            offset = 2 * self.cfg.lattice.size + 2 * self.cfg.nvirtmodes_link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = self._exract_partial_covmatvec(offset)
        return self._mat_d_mod_vec

    @property
    def det_mat_d_mod_vec (self):
        if self._det_mat_d_mod_vec is None:
            self._det_mat_d_mod_vec = [
                np.linalg.slogdet(mat_d)[1] for mat_d in self.mat_d_mod_vec
            ]
        return self._det_mat_d_mod_vec

    @property
    def mat_d_mod_inv_vec(self):
        if self._mat_d_mod_inv_vec is None:
            self._mat_d_mod_inv_vec = [
                np.linalg.inv(mat_d) for mat_d in self.mat_d_mod_vec
            ]
        return self._mat_d_mod_inv_vec

    @abstractmethod
    def initialize_gamma_in_sys(self):
        pass

    @property
    def gamma_in_sys(self):
        if self._gamma_in_sys is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._gamma_in_sys

    @property
    def incdet_vec(self):
        if self._incdet_vec is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._incdet_vec

    @property
    def wi_gamma_in_vec(self):
        if self._wi_gamma_in_vec is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._wi_gamma_in_vec

    @property
    def wi_gamma_out_vec(self):
        if self._wi_gamma_out_vec is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._wi_gamma_out_vec

    @property
    def gamma_in_sys_mod(self):
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        return self.gamma_in_sys[single_link_offset:, single_link_offset:]

    @property
    def incdet_mod_vec(self):
        if self._incdet_mod_vec is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._incdet_mod_vec

    @property
    def wi_gamma_in_mod_vec(self):
        if self._wi_gamma_in_mod_vec is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._wi_gamma_in_mod_vec

    @property
    def wi_gamma_out_mod_vec(self):
        if self._wi_gamma_out_mod_vec is None:
            self._gamma_in_sys, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._wi_gamma_out_mod_vec

    ################## Computation of derivatives ######################
    
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

    def compute_gamma_maj_deriv(self,symb,layerind):
        gamma_dirac_deriv = self.compute_gamma_dirac_deriv(symb,layerind)
        m, _ = gamma_dirac_deriv.shape
        smat = utils.generate_smat(m)
        return np.real(smat@gamma_dirac_deriv@np.transpose(smat))

    def _generate_gamma_maj_sys_deriv_dict(self):
        dest={}
        for symb in self.symbolvec:
            dest[symb]=[self._expand_gamma_maj_to_system(self.compute_gamma_maj_deriv(symb, i)) for i in range(self.cfg.nlayer) ]
        return dest

    def gamma_maj_sys_deriv_vec(self, symb: sympy.Symbol) -> np.ndarray:
        if symb in self.symbolvec:
            if self._gamma_maj_sys_deriv_dict is None:
                self._gamma_maj_sys_deriv_dict = self._generate_gamma_maj_sys_deriv_dict()
            return self._gamma_maj_sys_deriv_dict[symb]
        else:
            print("gamma_maj_sys_deriv: Invalid variable name", sys.stderr)
        return None

    def compute_grad_norm_vec(self) -> np.ndarray:
        #The parameter order is [[dt1, dy1, dz1...],[dt2,dy2,dz2...]...]
        dest=[]
        for layerind in range(self.cfg.nlayer):
            dest.append(self.compute_grad_norm(layerind))
        return np.asarray(dest)

    def compute_grad_norm(self, layerind: int) -> np.ndarray:
        #The parameter order is the same as in the symbolvec [t1,y1,z1....]
        dest=np.zeros(len(self.symbolvec))
        for ind, symbol in enumerate(self.symbolvec):
            dest[ind]=self.compute_grad_over_norm(symbol,layerind)
        return dest

    ################## Weight management ######################

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

    def calculate_weight_attempt(self, link_ind, theta, all_factors=False):
        # There are two directions per vertex and two Majoranas per link
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        rotmat = self.generate_rotmat(theta)
        gamma_in_subst = rotmat @ self.gamma_neutral_gauge @ np.transpose(rotmat)
        update = self.calculate_update_gamma_in(ind_mat, gamma_in_subst)
        return self.update_lognorm_inc(ind_mat, update, all_factors)

    def calculate_lognorm(self,all_factors=False):
        return calculate_lognorm(self.gamma_in_sys, self.mat_d_vec, all_factors=all_factors)

    def calculate_lognormvec_inc(self, all_factors=False):
        return calculate_lognormvec_inc(self.incdet_vec,
                                        self.det_mat_d_vec,
                                        self.gamma_in_sys.shape[0],
                                        all_factors=all_factors)

    def calculate_lognorm_inc(self, all_factors=False):
        normvec = self.calculate_lognormvec_inc(all_factors=all_factors)
        return np.sum(normvec)

    def update_lognorm_inc(self, offset, update, all_factors=False):
        cumval=0
        for ind in range(self.cfg.nlayer):
            detval = self.incdet_vec[ind].update_index(self.wi_gamma_in_vec[ind].inv(), update,
                                            offset, offset, store=False)
            if all_factors:
                detval-=self.gamma_in_sys.shape[0]*np.log(2)
                detval+=np.linalg.slogdet(self.mat_d_vec[ind])[1]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            cumval += 0.5 * detval
        return cumval

    def compute_grad_over_norm(self, var: sympy.Symbol, layerind: int) -> float:
        """Compute the quotient of derivative of the norm over the norm itself.
        We can avoid a lot of factors by computing the quotient directly.

        Args:
            var (sympy.Symbol): Name of the variable
            layerind (int): Index of the layer

        Returns:
            float: Value of the gradient divided by the norm of the state
        """
        diff = self.wi_gamma_in_vec[layerind].inv()
        # 2 phys. Majorana modes per vertex, this is indepent of the number of copies or layers
        offset = 2 * self.cfg.lattice.size
        # Extract only the part of the virtual-virtual correlations
        deriv_d = self.gamma_maj_sys_deriv_vec(var)[layerind][offset:, offset:]
        mat_d_inv=self.mat_d_inv_vec[layerind]

        #TODO: We might save one matrix-matrix multiplication here
        return compute_grad_over_norm(self.gamma_in_sys, diff, deriv_d, mat_d_inv)



    ################## Local Gauge ######################

    @property
    def gaugefieldvec(self):
        return self._gaugefieldvec

    @gaugefieldvec.setter
    def gaugefieldvec(self,val):
        print(
            "Do not set the gaugefieldvec explicitly. Use 'update_gauge_ind'.", file=sys.stderr)

    @abstractmethod
    def generate_gamma_gauge_neutral(self):
        pass


    @abstractmethod
    def generate_rotmat(self,theta):
        pass


    def update_gauge_full_system(self,gaugeconfig):
        for ind,gauge in enumerate(gaugeconfig):
            self.update_gauge_ind(ind, gauge)


    def update_gauge_coord(self,coord,dir,theta):
        ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        self.update_gauge_ind(ind, theta)


    def calculate_update_gamma_in(self,offset,update_mat):
        m_up, n_up = update_mat.shape
        gamma_in_old = self.gamma_in_sys[offset:offset + m_up,
                                         offset:offset + n_up]
        return -(update_mat - gamma_in_old)

    def invalidate_gauge_update(self):
        self._energy = None
        self._mag_energy_op = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._el_energy_op_grad_vec = None

    ################## Observables ######################
    @abstractmethod
    def _compute_mag_energy_op(self):
        pass
    @abstractmethod
    def _compute_el_energy_op_vec_and_grad(self):
        pass

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
            self._el_energy_op_vec, self._el_energy_op_grad_vec = self._compute_el_energy_op_vec_and_grad()
        return self._el_energy_op_vec

    @property
    def mag_energy(self):
        nplaq = self.cfg.lattice.nplaquettes
        mag_energy = self.cfg.g2_mag * (2*nplaq - 2*self.mag_energy_op)
        return mag_energy

    @property
    def el_energy(self):
        nlinks=self.cfg.lattice.nlinks
        el_energy = self.cfg.g2_el * (2*nlinks - 2*self.el_energy_op)
        return el_energy

    @property
    def el_energy_op_grad_vec(self):
        if self._el_energy_op_grad_vec is None:
            self._el_energy_op_vec, self._el_energy_op_grad_vec = self._compute_el_energy_op_vec_and_grad()
        return self._el_energy_op_grad_vec

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