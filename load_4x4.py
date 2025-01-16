import numpy as np
import sys
from ggpeps.lattice import Lattice2D
from ggpeps.system.system_z2_2d_G2c_F2c import Z2System2D, Z2System2D_G2C_F2C_Config
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.exacteval import ExactEvaluator
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.evaluator_manager import EvaluatorManager




L = 4

g_int0=1

FM_list=[]
ws_list=[]

g0 = float(sys.argv[1])
g    =  "{0:.1f}".format(g0)
g_el  = "{0:.4f}".format(g0/2)
g_mag = "{0:.4f}".format(1/(2*g0))
g_int = "{0:.1f}".format(g_int0)
name1="mc_data_to_compare/mass_0.0/g_{}_el_{}_mag_{}_int_1.0_mass_0.0".format(g,g_el,g_mag)
name2="mc_data_to_compare/mass_0.0/g_{}params/4x4/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(g,g_el,g_mag,g_int)
name=name1+name2
#name="params/2x2/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(g,g_el,g_mag,g_int)

lattice = Lattice2D(L, L)

system_type = Z2System2D
system_cfg = Z2System2D_G2C_F2C_Config(lattice, g0/2, 1/(2*g0), g_int0, 0, [0,0],num_pg_layer=1,num_fermionic_layer=1)

nlayer=2
param = np.load(name)
param = np.reshape(param, (nlayer, -1))

system_cfg.paramvec = param
system = system_type(system_cfg)

mc_config = MonteCarloEvaluatorConfig()
mc_config.warmup_steps = 50000
mc_config.meas_steps = 100000
mc_config.binsize = 1
mc_config.gauge_fixing = False

mc_mgr = EvaluatorManager(system_type, system_cfg, mc_config, 0)
mc_result = mc_mgr.simulate()
mc_result.print_stats()
mc_result.save()
mc_result.save()

