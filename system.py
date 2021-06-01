import numpy as np
import utils
import os
import sys
import lattice as lat
import gauge
import logging
from scipy.linalg import block_diag

################### U1MultilayerSystem2D ###################


class U1MultilayerSystem2DConfig:
    def __init__(self, paramdict, lattice):
        self.paramdict = paramdict
        if not self.check_paramdict(paramdict):
            logging.error("Different number of copies in parameters. Aborting")
            sys.exit(1)
        self.lattice = lattice

        self.yvec = paramdict["y"]
        self.zvec = paramdict["z"]
        self.tvec = paramdict["t"]
        self.ncopies = len(paramdict["y"])

    def check_paramdict(self, paramdict):
        lenarr = np.asarray([len(val) for _, val in paramdict.items()])
        return np.all(lenarr[-1] == lenarr)


class U1MultilayerSystem2D:
    def __init__(self, cfg):
        self.cfg = cfg

        self.tmat_ = None
        self.gamma_dirac_ = None
        self.gamma_maj_ = None
        self.gamma_maj_sys_ = None
        self.gamma_in_ = None
        self.gamma_in_sys_ = None

        self.gaugefieldvec_ = None
        self.gaugemgr_ = None

    def generate_tmat(self, t, y, z):
        dest = np.zeros((5, 4), dtype=complex)
        etap = np.exp(1.j * np.pi / 4.)
        y_block = np.array([[0, 1], [-1, 0]])
        z_block = np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)]])
        dest[1:3, 0:2] = y_block * y
        dest[3:5, 2:4] = y_block * y
        dest[1:3, 2:4] = z_block * z
        dest[1:3, 2:4] = -z_block.transpose() * z
        dest[0, :] = [t, etap ** 2 * t, etap*t, etap**3*t]
        return dest

    @property
    def tmat(self):
        if self.tmat_ is None:
            ncopies = self.cfg.ncopies
            self.tmat_ = np.zeros((ncopies, 5, 4), dtype=complex)
            yvec = self.cfg.yvec
            zvec = self.cfg.zvec
            tvec = self.cfg.tvec
            for ind, (t, y, z) in enumerate(zip(tvec, yvec, zvec)):
                self.tmat_[ind, :, :] = self.generate_tmat(t, y, z)
        return self.tmat_

    @property
    def gamma_dirac(self):
        #TODO: Implement
        pass

    @property
    def gamma_maj(self):
        #TODO: Implement
        pass

    @property
    def gamma_maj_sys(self):
        #TODO: Implement
        pass

    @property
    def gamma_in_sys(self):
        #TODO: Implement
        pass

################### U1MultilayerSystem3D ###################

###################### Z2System2D ##########################


class Z2System2DConfig:
    def __init__(self, paramdict, lattice, g2, g_gm, g_mag):
        self.paramdict = paramdict
        self.lattice = lattice

        #Parameters of the Hamiltonian
        self.g2 = g2
        self.g_el = g2
        if g_mag is None:
            self.g_mag = 1./g2
        else:
            self.g_mag = g_mag
        self.g_gm = g_gm


class Z2System2D:
    def __init__(self, cfg):
        self.cfg = cfg

        # Parameter dependent quantities
        self._tmat = None
        self._gamma_dirac = None
        self._gamma_maj = None
        self._gamma_maj_sys = None
        self._mat_a = None
        self._mat_b = None
        self._mat_d = None
        self._mat_d_inv = None

        # Management of the gaugefields
        self.gamma_neutral_gauge = self.generate_gamma_gauge_neutral()
        self._gamma_in_sys = None
        self._gaugefieldvec = np.zeros(self.cfg.lattice.nlinks)
        self.gaugemgr = gauge.ZNGauge(2)

        # Observables
        self._energy = None
        self._el_energy = None
        self._mag_energy = None

        # Weight
        self._weight = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in=None
        self._incdet=None

    def initialize(self):
        """Initialization function. 
        This is a good spot to copy essential data from the configuration.
        """
        return None

    @property
    def tmat(self):
        """
        Generate the T-matrix (single virtual fermion on the link).
        The mode ordering of this matrix is {p,l,r,d,u}.
        Analytically, this mode order is not advantageous, 
        but is makes the reshuffling of the modes easier for gamma_in and M_D in the covariance matrix.

        In the analytical part, we will keep the mode ordering {p,r,u,l,d} because the transformation matrices are easier to cope with.
        In the numerics, however, we will stick with {p,l,r,d,u}

        Returns:
            [np.array]: parameter matrix T
        """
        if self._tmat is None:
            paramdict = self.cfg.paramdict
            t = paramdict["t"]
            y = paramdict["y"]
            z = paramdict["z"]
            self._tmat = np.array([
                [0, -1.j*t, 1.j*t, t, -t],
                [1.j*t, 0, 1.j*y, z, 1.j*z],
                [-1.j*t, -1.j*y, 0, -1.j*z, -z],
                [-t, -z, 1.j*z, 0, -y],
                [t, -1.j*z, z, y, 0]],
                dtype=complex)
        return self._tmat

    @property
    def gamma_dirac(self):
        """Return the covariance matrix in dirac modes.
        The mode order of this matrix is {p,l,r,d,u,p_dag,l_dat,r_dag,u_dag,d_dag}.

        Returns:
            [np.array]: Covariance matrix in Dirac modes
        """
        if self._gamma_dirac is None:
            tmat = self.tmat
            self._gamma_dirac = utils.tmat_to_covariance_matrix(tmat)
        return self._gamma_dirac

    @property
    def gamma_maj(self):
        """Return the covariance matrix in Majorana modes.
        The mode order of this matrix is {p_1,p_2,l_1,l_2,r_1,r_2,d_1,d_2,u_1,u_2}.
        The definition of Majorana modes used is
            \gamma_1=c+c^\dagger
            \gamma_2=i(c-c^\dagger)

        Returns:
            [np.array]: Covariance matrix in Majorana modes
        """
        if self._gamma_maj is None:
            gamma_dirac = self.gamma_dirac
            m, _ = self.gamma_dirac.shape
            smat = utils.generate_smat(m)
            self._gamma_maj = np.real(smat@gamma_dirac@np.transpose(smat))
        return self._gamma_maj

    @property
    def gamma_maj_sys(self):
        """Return the covariance matrix of the full system in Majorana modes.
        The mode order is changed to fit the mode order of gamma_in.
        See documentation of gamma_in for details.

        Returns:
            [np.array]: Covariance matrix of the full system
        """
        if self._gamma_maj_sys is None:
            gamma_maj = self.gamma_maj
            amat = gamma_maj[:2, :2]
            bmat = gamma_maj[:2, 2:]
            dmat = gamma_maj[2:, 2:]
            nsites = self.cfg.lattice.size
            id = np.eye(nsites)
            # Extract the parts of the covariance matrix
            amat_sys = np.kron(id, amat)
            bmat_sys = np.kron(id, bmat)
            dmat_sys = np.kron(id, dmat)
            # Order the modes of the covariance matrix according to gamma_in_sys
            permbuilder = lat.PermutationBuilderGMS2D(self.cfg.lattice, nmodes_per_link=1)
            mat_perm = permbuilder.perm()
            self._gamma_maj_sys = mat_perm@np.block(
                [[amat_sys, bmat_sys], [-np.transpose(bmat_sys), dmat_sys]])@np.transpose(mat_perm)
        return self._gamma_maj_sys

    def initialize_gamma_in_sys(self):
        nlinks = self.cfg.lattice.nlinks
        id = np.eye(nlinks)
        neutral_gauge = self.gamma_neutral_gauge
        gamma_in_sys = np.kron(id, neutral_gauge)
        diff = self.mat_d_inv - gamma_in_sys
        wi_gamma_in=utils.WoodburyInverter(diff)
        incdet=utils.IncLogAbsDeterminant(diff)
        return gamma_in_sys, wi_gamma_in, incdet

    @property
    def gamma_in_sys(self):
        #TODO: Details about mode order
        if self._gamma_in_sys is None:
            self._gamma_in_sys, self._wi_gamma_in, self._incdet=self.initialize_gamma_in_sys()
        return self._gamma_in_sys

    @property
    def incdet(self):
        if self._incdet is None:
            self._gamma_in_sys, self._wi_gamma_in, self._incdet=self.initialize_gamma_in_sys()
        return self._incdet

    @property
    def wi_gamma_in(self):
        if self._wi_gamma_in is None:
            self._gamma_in_sys, self._wi_gamma_in, self._incdet=self.initialize_gamma_in_sys()
        return self._wi_gamma_in

    @property
    def mat_a(self):
        """Extract the matrix for physical-physical correlations.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        The mode ordering of the sites is identical to the site convention defined in the lattice class.

        Returns:
            [np.array]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a is None:
            self._mat_a, self._mat_b, self._mat_d = self.extract_partial_covmats()
        return self._mat_a

    @property
    def mat_b(self):
        """Extract the matrix for physical-virtual correlations.

        Returns:
            [np.array]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b is None:
            self._mat_a, self._mat_b, self._mat_d = self.extract_partial_covmats()
        return self._mat_b

    @property
    def mat_d(self):
        """Extract the matrix for virtual-virtual correlations.

        Returns:
            [np.array]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d is None:
            self._mat_a, self._mat_b, self._mat_d = self.extract_partial_covmats()
        return self._mat_d

    @property
    def mat_d_inv(self):
        if self._mat_d_inv is None:
            self._mat_d_inv = np.linalg.inv(self.mat_d)
        return self._mat_d_inv

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
            self._weight = 0.5 * self.incdet.det()
        return self._weight

    @weight.setter
    def weight(self,val):
        self._weight = val


    def extract_partial_covmats(self):
        gamma_maj_sys=self.gamma_maj_sys
        nsites=self.cfg.lattice.size
        #We are assuming one physical mode per site
        mat_a = gamma_maj_sys[:2*nsites, :2*nsites]
        mat_b = gamma_maj_sys[:2*nsites, 2*nsites:]
        mat_d = gamma_maj_sys[2*nsites:, 2*nsites:]
        return mat_a, mat_b, mat_d

    def generate_gamma_gauge_neutral(self):
        return np.real_if_close(1.j*np.kron(utils.pauliy, utils.paulix))

    #Gauging

    def generate_rotmat(self,theta):
        # TODO: Do we want to stagger here?
        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        rot_right=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        # We have only one left mode => 2 Majorana modes
        rot_left=np.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        dest=block_diag(rot_left,rot_right)
        return dest

    def invalidate_gauge_update(self):
        self._energy = None
        self._mag_energy = None
        self._el_energy = None

    def calculate_update_gamma_in(self,offset,update_mat):
        m_up,n_up=update_mat.shape
        gamma_in_old=self.gamma_in_sys[offset:offset+m_up,offset:offset+n_up]
        return -(update_mat-gamma_in_old)

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
        mat_inv = self.wi_gamma_in.inv()
        detval = self.incdet.update_index(mat_inv, update, ind_mat, ind_mat)
        # Update the weight
        self.weight = 0.5 * detval
        # Update the matrix inversion
        self.wi_gamma_in.update_index(update, ind_mat, ind_mat)
        # Substitute in the array
        self.gamma_in_sys[ind_mat:ind_mat + 4,
                          ind_mat:ind_mat + 4] = gamma_in_subst
        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    def calculate_weight_attempt(self, link_ind, theta, all_factors=False):
        # There are two directions per vertex and two Majoranas per link
        ind_mat=4*link_ind
        rotmat=self.generate_rotmat(theta)
        gamma_in_subst=rotmat@self.gamma_neutral_gauge@np.transpose(rotmat)
        update=self.calculate_update_gamma_in(ind_mat,gamma_in_subst)
        return self.calculate_lognorm_inc(ind_mat,update,all_factors)


    def update_gauge_coord(self,coord,dir,theta):
        ind=self.cfg.lattice.coord2ind_dir(coord,dir)
        self.update_gauge_ind(ind,theta)

    # Calculating the norm

    def calculate_lognorm(self,all_factors=False):
        # This is still the plain formula, without any update mechanism
        gamma_in_sys=self.gamma_in_sys
        mat_d=self.mat_d
        if all_factors:
            sign,logval=np.linalg.slogdet((np.eye(mat_d.shape[0])-gamma_in_sys@mat_d)/2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            sign,logval=np.linalg.slogdet((np.eye(mat_d.shape[0])-gamma_in_sys@mat_d))
        #The factor 1/2 is the square-root
        return logval/2

    def calculate_lognorm_inc(self, offset, update, all_factors=False):
        detval = self.incdet.update_index(self.wi_gamma_in.inv(), update,
                                          offset, offset, store=False)
        if all_factors:
            detval-=np.log(2**self.gamma_in_sys.shape[0])
            detval+=np.linalg.slogdet(self.mat_d)[1]
        # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
        return 0.5 * detval


    # Calculate gradients

    def gamma_maj_deriv_y(self):
        t=self.cfg.paramdict["t"]
        y=self.cfg.paramdict["y"]
        z=self.cfg.paramdict["z"]
        d = 1 + 4*t**2 + y**2 + 2*y*z + 2*z**2
        b = 1+y**2 - 2*y*z + 2*z**2
        alpha = y+z
        beta = y-z
        gamma = 1+z
        delta = 1+y
        dest=np.zeros((10,10))
        dest[0, 1] = (16*t**2*alpha)/d**2
        dest[0, 2] = (-4*t*(-1 + z)*alpha)/d**2
        dest[0, 3] = (-2*t*(d - 2*alpha**2))/d**2
        dest[0, 4] = (-4*t*alpha*gamma)/d**2
        dest[0, 5] = (-2*t*(d - 2*alpha**2))/d**2
        dest[0, 6] = (-4*t*z*alpha)/d**2
        dest[0, 7] = (-2*t*(d - 2*(-1 + alpha)*alpha))/d**2
        dest[0, 8] = (-4*t*z*alpha)/d**2
        dest[0, 9] = (-2*t*(d - 2*alpha*(1 + alpha)))/d**2

        dest[1, 2] = (2*t*(d - 2*alpha**2))/d**2
        dest[1, 3] = (-4*t*alpha*gamma)/d**2
        dest[1, 4] = (2*t*(d - 2*alpha**2))/d**2
        dest[1, 5] = (-4*t*(-1 + z)*alpha)/d**2
        dest[1, 6] = (2*t*(d - 2*alpha*(1 + alpha)))/d**2
        dest[1, 7] = (-4*t*z*alpha)/d**2
        dest[1, 8] = (2*t*(d - 2*(-1 + alpha)*alpha))/d**2
        dest[1, 9] = (-4*t*z*alpha)/d**2

        dest[2, 3] = (-2*(alpha + 2*t**2*alpha))/d**2 - (2*beta)/b**2
        dest[2, 4] = (d - 2*alpha**2)/d**2 + (b - 2*beta**2)/b**2
        dest[2, 5] = (-2*(2*t**2 + z)*alpha)/d**2 + (2*z*beta)/b**2
        dest[2, 6] = (-2*b*d*(t**2 + z + 2*y*z) + 4*(t**2 + y*z)
                      * (b*alpha + d*beta)*delta)/(b**2*d**2)
        dest[2, 7] = (-((b + 4*z*beta - 2*beta*delta)/b**2) +
                        (d - 2*alpha*(4*t**2 + 2*z + delta))/d**2)/2.
        dest[2, 8] = ((d - 2*(-1 + y + 2*z)*alpha)/d**2 -
                      (b + 2*beta - 2*y*beta + 4*z*beta)/b**2)/2.
        dest[2, 9] = (-((d + 2*alpha + 8*t**2*alpha - 2*y*alpha)/d**2) + (b + 2*beta - 2*y*beta)/b**2)/2.

        dest[3, 4] = (4*t**2*alpha - 2*z*alpha)/d**2 + (2*z*beta)/b**2
        dest[3, 5] = -((d - 2*alpha**2)/d**2) - (b - 2*beta**2)/b**2
        dest[3, 6] = (-(1/b) + (d - 2*(-1 - 4*t**2 + y + 2*z) *
                                alpha)/d**2 + (2*(-1 + y - 2*z)*beta)/b**2)/2.
        dest[3, 7] = (2*b*d*(t**2 + (-1 + 2*y)*z) - 4*b*(-1 + y) *
                        (t**2 + y*z)*alpha - 4*d*(-1 + y)*(t**2 + y*z)*beta)/(b**2*d**2)
        dest[3, 8] = ((b - 2*beta*delta)/b**2 +
                      (-d + 2*alpha*(4*t**2 + delta))/d**2)/2.
        dest[3, 9] = (-4*d*beta*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta)) + 2*b*(
            d*(t**2 + z) - 2*alpha*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta))))/(b**2*d**2)

        dest[4, 5] = (-2*(alpha + 2*t**2*alpha))/d**2 - (2*beta)/b**2
        dest[4, 6] = (-4*d*beta*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta)) + 2 * b*(
            d*(t**2 + z) - 2*alpha*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta))))/(b**2*d**2)
        dest[4,7]=(-((b - 2*beta*delta)/b**2) + (d - 2*alpha*(4*t**2 + delta))/d**2)/2.
        dest[4,8]=(2*b*d*(t**2 + (-1 + 2*y)*z) - 4*b*(-1 + y)*(t**2 + y*z)*alpha - 4*d*(-1 + y)*(t**2 + y*z)*beta)/(b**2*d**2)

        dest[4,9]=((-d + 2*(-1 - 4*t**2 + y + 2*z)*alpha)/d**2 + (b + 2*beta - 2*y*beta + 4*z*beta)/b**2)/2.

        dest[5,6]=-0.5*1/b + (d + 2*alpha + 8*t**2*alpha - 2*y*alpha)/(2.*d**2) + ((-1 + y)*beta)/b**2
        dest[5,7]=((d - 2*(-1 + y + 2*z)*alpha)/d**2 - (b + 2*beta - 2*y*beta + 4*z*beta)/b**2)/2.

        dest[5,8]=((b + 4*z*beta - 2*beta*delta)/b**2 + (-d + 2*alpha*(4*t**2 + 2*z + delta))/d**2)/2.
        dest[5,9]=(-2*b*d*(t**2 + z + 2*y*z) + 4*(t**2 + y*z)*(b*alpha + d*beta)*delta)/(b**2*d**2)

        dest[6,7]=(-2*(alpha + 2*t**2*alpha))/d**2 - (2*beta)/b**2
        dest[6,8]=z*((-2*alpha)/d**2 + (2*beta)/b**2)
        dest[6,9]=-((d + 4*t**2*alpha - 2*alpha**2)/d**2) - (b - 2*beta**2)/b**2

        dest[7,8]=(-d + 2*alpha*(2*t**2 + alpha))/d**2 - (b - 2*beta**2)/b**2
        dest[7,9]=2*z*((-y + z)/b**2 + alpha/d**2)

        dest[8,9]=(-2*(alpha + 2*t**2*alpha))/d**2 - (2*beta)/b**2

        return dest-np.transpose(dest)

    def gamma_maj_deriv_z(self):
        dest=np.zeros((10, 10))
        t=self.cfg.paramdict["t"]
        y=self.cfg.paramdict["y"]
        z=self.cfg.paramdict["z"]

        d = 1 + 4*t**2 + y**2 + 2*y*z + 2*z**2
        b = 1+y**2 - 2*y*z + 2*z**2
        alpha=y+z
        beta=y-z
        gamma=1+z
        delta=1+y
        eta=y+2*z

        dest[0, 1] = (16*t**2*eta)/d**2
        dest[0, 2] = (2*t*(d - 2*(-1 + z)*eta))/d**2
        dest[0, 3] = (-2*t*(d - 2*alpha*eta))/d**2
        dest[0, 4] = (2*t*(d - 2*gamma*eta))/d**2
        dest[0, 5] = (-2*t*(d - 2*alpha*eta))/d**2
        dest[0, 6] = (2*t*(d - 2*z*eta))/d**2
        dest[0, 7] = (-2*t*(d - 2*(-1 + alpha)*eta))/d**2
        dest[0, 8] = (2*t*(d - 2*z*eta))/d**2
        dest[0, 9] = (-2*t*(d - 2*(1 + alpha)*eta))/d**2

        dest[1, 2] = (2*t*(d - 2 * alpha * eta))/d**2
        dest[1, 3] = (2*t*(d - 2 * gamma*eta))/d**2
        dest[1, 4] = (2*t*(d - 2 * alpha * eta))/d**2
        dest[1, 5] = (2*t*(d - 2*(-1 + z) * eta))/d**2
        dest[1, 6] = (2*t*(d - 2*(1 + alpha) * eta))/d**2
        dest[1, 7] = (2*t*(d - 2*z * eta))/d**2
        dest[1, 8] = (2*t*(d - 2*(-1 + alpha) * eta))/d**2
        dest[1, 9] = (2*t*(d - 2*z * eta))/d**2

        dest[2, 3] = (2*(y - 2*z))/b**2 - (2*(eta + 2*t**2*eta))/d**2
        dest[2, 4] = -((b - 2*y*beta + 4*z*beta)/b**2) + (d - 2*alpha*eta)/d**2
        dest[2, 5] = -(1/b) - (2*(y - 2*z)*z)/b**2 + \
            (d - 2*(2*t**2 + z)*eta)/d**2
        dest[2, 6] = (-2*delta*(b*d*y + 2*d*(y - 2*z) *
                      (t**2 + y*z) - 2*b*(t**2 + y*z)*eta))/(b**2*d**2)
        dest[2, 7] = 1/b + ((y - 2*z)*(2*z - delta))/b**2 + \
            (d - (4*t**2 + 2*z + delta)*eta)/d**2
        dest[2, 8] = (b - (-1 + y - 2*z)*(y - 2*z)) / \
            b**2 + (d + eta - eta**2)/d**2
        dest[2, 9] = ((-1 + y)*(y - 2*z))/b**2 + ((-1 - 4*t**2 + y)*eta)/d**2

        dest[3, 4] = -(1/b) - (2*(y - 2*z)*z)/b**2 + \
            (d + 4*t**2*eta - 2*z*eta)/d**2
        dest[3, 5] = (b - 2*y*beta + 4*z*beta)/b**2 - (d - 2*alpha*eta)/d**2
        dest[3, 6] = (b - (-1 + y - 2*z)*(y - 2*z))/b**2 + \
            (d + eta + 4*t**2*eta - eta**2)/d**2
        dest[3, 7] = (2*(-1 + y)*(b*d*y + 2*d*(y - 2*z) *
                      (t**2 + y*z) - 2*b*(t**2 + y*z)*eta))/(b**2*d**2)
        dest[3, 8] = ((y - 2*z)*delta)/b**2 + ((4*t**2 + delta)*eta)/d**2
        dest[3, 9] = (4*d*(y - 2*z)*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta)) + 2*b*(d*(-1 -
                      2*t**2 + y - 6*z**2) - 2*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta))*eta))/(b**2*d**2)

        dest[4, 5] = (2*(y - 2*z))/b**2 - (2*(eta + 2*t**2*eta))/d**2
        dest[4, 6] = (4*d*(y - 2*z)*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta)) + 2*b*(d*(-1 -
                      2*t**2 + y - 6*z**2) - 2*(z*(-1 + y - 2*z**2) + t**2*(-2*z + delta))*eta))/(b**2*d**2)
        dest[4, 7] = -(((y - 2*z)*delta)/b**2) - ((4*t**2 + delta)*eta)/d**2
        dest[4, 8] = (2*(-1 + y)*(b*d*y + 2*d*(y - 2*z) *
                      (t**2 + y*z) - 2*b*(t**2 + y*z)*eta))/(b**2*d**2)
        dest[4, 9] = (-b + (-1 + y - 2*z)*(y - 2*z))/b**2 - \
            (d + eta + 4*t**2*eta - eta**2)/d**2

        dest[5, 6] = -(((-1 + y)*(y - 2*z))/b**2) + ((1 + 4*t**2 - y)*eta)/d**2
        dest[5, 7] = (b - (-1 + y - 2*z)*(y - 2*z)) / \
            b**2 + (d + eta - eta**2)/d**2
        dest[5, 8] = -(1/b) + ((y - 2*z)*(-2*z + delta))/b**2 + \
            (-d + (4*t**2 + 2*z + delta)*eta)/d**2
        dest[5, 9] = (-2*delta*(b*d*y + 2*d*(y - 2*z) *
                      (t**2 + y*z) - 2*b*(t**2 + y*z)*eta))/(b**2*d**2)

        dest[6, 7] = (2*(y - 2*z))/b**2 - (2*(eta + 2*t**2*eta))/d**2
        dest[6, 8] = 1/d - (b + 2*(y - 2*z)*z)/b**2 - (2*z*eta)/d**2
        dest[6, 9] = (b - 2*y*beta + 4*z*beta)/b**2 - \
            (d + 4*t**2*eta - 2*alpha*eta)/d**2

        dest[7, 8] = (b - 2*y*beta + 4*z*beta)/b**2 + \
            (-d + 2*(2*t**2 + alpha)*eta)/d**2
        dest[7, 9] = 1/b + (2*(y - 2*z)*z)/b**2 - (d - 2*z*eta)/d**2

        dest[8, 9] = (2*(y - 2*z))/b**2 - (2*(eta + 2*t**2*eta))/d**2

        return dest-np.transpose(dest)

    # Update the parameters

    # Observables

    @property
    def energy(self):
        if self._energy is None:
            self._energy = self.el_energy+self.mag_energy
        return self._energy

    @property
    def mag_energy(self):
        if self._mag_energy is None:
            self._mag_energy = self._compute_mag_energy()
        return self._mag_energy

    @property
    def el_energy(self):
        if self._el_energy is None:
            self._el_energy = self._compute_el_energy()
        return self._el_energy

    def _compute_el_energy(self, use_trans_inv=True):
        if use_trans_inv:
            nlinks=self.cfg.lattice.nlinks
            #We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
            gamma_maj_sys = self.gamma_maj_sys
            gamma_in_sys = self.gamma_in_sys
            # Number of fermions = # of sites
            single_site_offset = 4
            offset = 2*self.cfg.lattice.size+single_site_offset
            mat_a = gamma_maj_sys[:offset, :offset]
            mat_b = gamma_maj_sys[:offset, offset:]
            mat_d = gamma_maj_sys[offset:, offset:]
            # We have to cut one link from gamma_in_sys as well
            gamma_in_sys_tilde = gamma_in_sys[single_site_offset:,
                                              single_site_offset:]
            covmat_out = mat_a + \
                mat_b @ np.linalg.inv(mat_d -
                                      gamma_in_sys_tilde) @ np.transpose(mat_b)
            covmat_out_virt = covmat_out[-single_site_offset:, -
                                         single_site_offset:]
            el_energy_bare=0.5j*(covmat_out_virt[0,2]-covmat_out_virt[0,3]-1.j*covmat_out_virt[0,1]-1.j*covmat_out_virt[2,3])
            el_energy = nlinks*self.cfg.g2*(1-np.real(el_energy_bare))
        else:
            # Evaluate every link of the system
            logging.error("compute_el_energy: not implemented yet")
            el_energy = None
        return el_energy

    def _compute_mag_energy(self, use_trans_inv=True):
        if use_trans_inv:
            # Evaluate one plaquette and multiply by number of plaquettes
            nplaq = self.cfg.lattice.nplaquettes
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop(
                (0, 0), (1, 1))
            bare_energy = np.real(self.compute_path(wilson_plaquette))
            mag_energy = nplaq*self.cfg.g_mag*(1-bare_energy)
        else:
            # Evaluate every plaquette of the system
            logging.error("compute_mag_energy: not implemented yet")
            mag_energy = None
        return mag_energy

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
        # TODO: Track the inverse in the formula
        return self.mat_a + self.mat_b@np.linalg.inv(self.mat_d - self.gamma_in_sys)@np.transpose(self.mat_b)
