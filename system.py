import numpy as np
import utils
import os
import sys

################### U1MultilayerSystem2D ###################


class U1MultilayerSystem2DConfig:
    def __init__(self, paramdict, lattice):
        self.paramdict = paramdict
        if not self.check_paramdict(paramdict):
            print("Different number of copies in parameters. Aborting",
                  file=sys.stderr)
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

        self.gaugefields_ = None
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
        self.g_el = g2/2
        if g_mag is None:
            self.g_mag = 1./g2
        else:
            self.g_mag = g_mag
        self.g_gm = g_gm


class Z2System2D:
    def __init__(self, cfg):
        self.cfg = cfg

        # Parameter dependent quantities
        self.tmat_ = None
        self.gamma_dirac_ = None
        self.gamma_maj_ = None
        self.gamma_maj_sys_ = None

        # Management of the gaugefields
        self.gamma_neutral_gauge = self.generate_gamma_gauge_neutral()
        self.gamma_in_sys_ = None
        self.gaugefields_ = None
        self.gaugemgr_ = None

        #Observables
        self.energy_ = None
        self.el_energy_ = None
        self.mag_energy_ = None

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
        if self.tmat_ is None:
            paramdict = self.cfg.paramdict
            t = paramdict["t"]
            y = paramdict["y"]
            z = paramdict["z"]
            self.tmat_ = np.array([
                [0, -1.j*t, 1.j*t, t, -t],
                [1.j*t, 0, 1.j*y, z, 1.j*z],
                [-1.j*t, -1.j*y, 0, -1.j*z, -z],
                [-t, -z, 1.j*z, 0, -y],
                [t, -1.j*z, z, y, 0]],
                dtype=complex)
        return self.tmat_

    @property
    def gamma_dirac(self):
        """Return the covariance matrix in dirac modes.
        The mode order of this matrix is {p,l,r,d,u,p_dag,l_dat,r_dag,u_dag,d_dag}.

        Returns:
            [np.array]: Covariance matrix in Dirac modes
        """
        if self.gamma_dirac_ is None:
            tmat = self.tmat
            self.gamma_dirac_ = utils.tmat_to_covariance_matrix(tmat)
        return self.gamma_dirac_

    @property
    def gamma_maj(self):
        if self.gamma_maj_ is None:
            gamma_dirac = self.gamma_dirac
            m, _ = self.gamma_dirac.shape
            smat = utils.generate_smat(m)
            self.gamma_maj_ = smat@gamma_dirac@np.transpose(smat)
        return self.gamma_maj_

    @property
    def gamma_maj_sys(self):
        if self.gamma_maj_sys_ is None:
            gamma_maj = self.gamma_maj
            amat = gamma_maj[:2, :2]
            bmat = gamma_maj[:2, 2:]
            dmat = gamma_maj[2:, 2:]
            nsites = self.cfg.lattice.get_size()
            id = np.eye(nsites)
            amat_sys = np.kron(id, amat)
            bmat_sys = np.kron(id, bmat)
            dmat_sys = np.kron(id, dmat)
            self.gamma_maj_sys_ = np.block(
                [[amat_sys, bmat_sys], [-np.transpose(bmat_sys), dmat_sys]])
        return self.gamma_maj_sys_

    @property
    def gamma_in_sys(self):
        if self.gamma_in_sys_ is None:
            nlinks = self.cfg.lattice.nlinks
            id = np.eye(nlinks)
            neutral_gauge = self.gamma_neutral_gauge
            self.gamma_in_sys_ = np.kron(id, neutral_gauge)
        return self.gamma_in_sys_

    def generate_gamma_gauge_neutral(self):
        return np.real_if_close(1.j*np.kron(utils.pauliy, utils.paulix))

    # Observables
    @property
    def energy(self):
        if self.energy_ is None:
            self.energy_ = self.el_energy+self.mag_energy
        return self.energy_

    @property
    def mag_energy(self):
        if self.mag_energy_ is None:
            self.mag_energy_ = self.compute_mag_energy()
        return self.mag_energy_

    @property
    def el_energy(self):
        if self.el_energy_ is None:
            self.el_energy_ = self.compute_el_energy()
        return self.el_energy_

    def compute_el_energy(self):
        #TODO: Implement electric energy
        return 0

    def compute_mag_energy(self):
        #TODO: Implement magnetic energy
        return 0
