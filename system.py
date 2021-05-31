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

        # Management of the gaugefields
        self.gamma_neutral_gauge = self.generate_gamma_gauge_neutral()
        self._gamma_in_sys = None
        self._gaugefieldvec = np.zeros(self.cfg.lattice.nlinks)
        self.gaugemgr = gauge.ZNGauge(2)

        #Observables
        self._energy = None
        self._el_energy = None
        self._mag_energy = None

        #Weight
        self._weight = None

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
        if self._gamma_maj is None:
            gamma_dirac = self.gamma_dirac
            m, _ = self.gamma_dirac.shape
            smat = utils.generate_smat(m)
            self._gamma_maj = np.real(smat@gamma_dirac@np.transpose(smat))
        return self._gamma_maj

    @property
    def gamma_maj_sys(self):
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

    @property
    def gamma_in_sys(self):
        if self._gamma_in_sys is None:
            nlinks = self.cfg.lattice.nlinks
            id = np.eye(nlinks)
            neutral_gauge = self.gamma_neutral_gauge
            self._gamma_in_sys = np.kron(id, neutral_gauge)
        return self._gamma_in_sys
    
    @property
    def mat_a(self):
        if self._mat_a is None:
            self._mat_a, self._mat_b, self._mat_d = self.extract_partial_covmats()
        return self._mat_a

    @property
    def mat_b(self):
        if self._mat_b is None:
            self._mat_a, self._mat_b, self._mat_d = self.extract_partial_covmats()
        return self._mat_b

    @property
    def mat_d(self):
        if self._mat_d is None:
            self._mat_a, self._mat_b, self._mat_d = self.extract_partial_covmats()
        return self._mat_d
    
    @property
    def gaugefieldvec(self):
        return self._gaugefieldvec
    
    @gaugefieldvec.setter
    def gaugefieldvec(self,val):
        print(
            "Do not set the gaugefieldvec explicitly. Use 'update_gauge'.", file=sys.stderr)
    
    @property
    def weight(self):
        if self._weight is None:
            self._weight=self.calculate_lognorm()
        return self._weight
    
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
        self._weight = None

    def update_gauge_ind(self,ind,theta):
        #Update the gaugefield
        self.gaugefieldvec[ind]=theta
        #We have two directions per vertex and two Majoranas per link
        ind_mat=4*ind
        rotmat=self.generate_rotmat(theta)
        gamma_in_subst=rotmat@self.gamma_neutral_gauge@np.transpose(rotmat)
        # We get the matrix corresponding to the property
        gamma_in_sys=self.gamma_in_sys
        # and substitute in the array
        gamma_in_sys[ind_mat:ind_mat+4,ind_mat:ind_mat+4]=gamma_in_subst
        # Set the weight to None to recompute it
        self.invalidate_gauge_update()

    def update_gauge_coord(self,coord,dir,theta):
        ind=self.cfg.lattice.coord2ind_dir(coord,dir)
        self.update_gauge_ind(ind,theta)

    # Calculating the norm

    def calculate_lognorm(self):
        # This is still the plain formula, without any update mechanism
        gamma_in_sys=self.gamma_in_sys
        mat_d=self.mat_d
        sign,logval=np.linalg.slogdet((np.eye(mat_d.shape[0])-gamma_in_sys@mat_d)/2.)
        #The factor 1/2 is the square-root
        return logval/2

    # Calculate gradients

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
