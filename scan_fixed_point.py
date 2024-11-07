import numpy as np
import ggpeps
from ggpeps.caching import Cache
from ggpeps.system import Z2System2DConfig, Z2System2D_1c
from ggpeps.system import Z2System2D2CConfig, Z2System2D2C
from ggpeps.system import Z2System2D_G2C_F2C_Config, Z2System2D
from ggpeps.system import Z2System2D_G2C_F4C_Config, Z2System2D_G2C_F4C
from ggpeps.system import Z2System2D_8C_Config, Z2System2D_8C

from ggpeps import utils
from ggpeps import lattice as lat
from ggpeps.measurement import Measurement
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.exacteval import ExactEvaluator
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.minimizer import Minimizer, MinimizerConfig


el_ops = []
plaqs  = []
fms    = []

def scan_fixed_point_b():
    L = 2
    g_el = 1.
    g_mag = 1

    lattice = lat.Lattice2D(L, L)

    system_type = Z2System2D
    system_cfg = Z2System2D2CConfig(lattice, g_int=0, g_el=g_el, g_mag=g_mag, g_mass=0, g_chem=[0])


    # for b in np.arange(0,1.01,0.02):
    for phi in np.arange(0.,2*np.pi,0.1):
        b = np.exp(1j*phi)
        # b1=b
        # b2=np.sqrt(1-b1**2)+0.000001
        b1 = np.real(b)
        b2 = np.imag(b)+0.00001
        parvec_1 = np.array([0,0,0,0,0,0,0,0,0,b1,0,0,0,0,0,0,0,0,0,b2])
        parvec_2 = np.array([0,0,0,0,0,0,0,0,b1,0,0,0,0,0,0,0,0,0,b2,0])
        parvec_3 = np.array([0,0,0,0,0,0,0,b1,0,0,0,0,0,0,0,0,0,b2,0,0])
        parvec_4 = np.array([0,0,0,0,0,0,b1,0,0,0,0,0,0,0,0,0,b2,0,0,0])

        system_cfg.paramvec=[parvec_2]
        system = system_type(system_cfg)
        # u, s, vh = np.linalg.svd(system.tmat)
        # print("Singular values: ", s)
        eval_config = ExactEvaluatorConfig()
        ex_eval = ExactEvaluator(eval_config, system)
        res1 = ex_eval.evaluate()
        el_op = 1-res1[ "el_energy"]/16
        plaq  = 1-res1["mag_energy"]/8
        fm    =   res1["FM"]
        print("Re[b]: ", b1)
        print("Electric field: ", el_op)
        print("Plaquette: ",plaq) 
        print("FM: ", fm) 
        el_ops.append(el_op)
        plaqs.append(plaq)
        fms.append(fm)



if __name__ == "__main__":
    # compare_tmat()
    scan_fixed_point_b()
    print(el_ops)
    print(plaqs)

