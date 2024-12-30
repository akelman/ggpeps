import numpy as np
import pandas as pd
from ggpeps.lattice import Lattice2D
from ggpeps.system.system_z2_2d_G2c_F2c import Z2System2D, Z2System2D_G2C_F2C_Config
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.exacteval import ExactEvaluator
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.evaluator_manager import EvaluatorManager

L = 2

g_int0=1

for g0 in np.arange(0.1,2.,0.1):
    g    =  "{0:.1f}".format(g0)
    g_int = "{0:.3f}".format(g_int0)

    g_el  = "{0:.4f}".format(g0/2)
    g_mag = "{0:.4f}".format(1/(2*g0))

    name1="mc_data_to_compare/mass_0.0/g_{}_el_{}_mag_{}_int_1.0_mass_0.0".format(g,g_el,g_mag)
    name2="/summary_min_L_04-04_gel_{}_gmag_{}_gint_1.0000_gmass_0.0000_ncopy_02_nlayer_02.pkl".format(g_el,g_mag)
    name=name1+name2
    print(name)

    df = pd.read_pickle(name)
    print(df)

