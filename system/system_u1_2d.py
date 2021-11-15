import numpy as np
import sys
import logging

################### U1MultilayerSystem2D ###################


class U1MultilayerSystem2DConfig:
    def __init__(self, paramvec, lattice):
        self.paramvec = np.asarray(paramvec)
        if not self.check_paramvec(paramvec):
            logging.error("Different number of copies in parameters. Aborting")
            sys.exit(1)
        self.lattice = lattice

        self.tvec = paramvec[0]
        self.yvec = paramvec[1]
        self.zvec = paramvec[2]
        self.ncopies = len(paramvec[0])

    def check_paramvec(self, paramvec):
        lenarr = np.asarray([len(vec) for vec in paramvec])
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