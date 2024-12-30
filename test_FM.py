import numpy as np
from ggpeps.lattice import Lattice2D 
from ggpeps.system.system_z2_2d_G2c_F2c import Z2System2D, Z2System2D_G2C_F2C_Config
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.exacteval import ExactEvaluator
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.evaluator_manager import EvaluatorManager




L = 2

g_int0=1

FM_1x1_list=[]
FM_2x2_list=[]
ws_list=[]

for g0 in np.arange(1.1,1.2,0.1):
    g    =  "{0:.1f}".format(g0)
    g_el  = "{0:.4f}".format(g0/2)
    g_mag = "{0:.4f}".format(1/(2*g0))
    g_int = "{0:.1f}".format(g_int0)
    name="params/2x2/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(g,g_el,g_mag,g_int)
    #name="params/4x4/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(g,g_el,g_mag,g_int)
    print("Starting ", name)

    lattice = Lattice2D(L, L)

    system_type = Z2System2D
    system_cfg = Z2System2D_G2C_F2C_Config(lattice, g0/2, 1/(2*g0), g_int0, 0, [0,0],num_pg_layer=1,num_fermionic_layer=1)

    nlayer=2
    param = np.load(name)
    param = np.reshape(param, (nlayer, -1))

    system_cfg.paramvec = param
    system = system_type(system_cfg)

    eval_config = ExactEvaluatorConfig()
    ex_eval = ExactEvaluator(eval_config, system)

    res = ex_eval.evaluate()

    FM_1x1=res["FM 1x1"]
    FM_2x2=res["FM 2x2"]
    w =res["wilson_loop_0-0_1x1"]
    FM_1x1_list.append(FM_1x1)
    FM_2x2_list.append(FM_2x2)
    ws_list.append(w)
    print(FM_1x1_list)
    print(FM_2x2_list)
    print(ws_list)


