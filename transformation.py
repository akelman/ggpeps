import numpy as np
from ggpeps.lattice import Lattice2D 
from ggpeps.system.system_z2_2d_G2c_F2c import Z2System2D, Z2System2D_G2C_F2C_Config
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.exacteval import ExactEvaluator
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.evaluator_manager import EvaluatorManager

L = 2

g_int0=1

g0=0.2
g    =  "{0:.1f}".format(g0)
g_el  = "{0:.4f}".format(g0/2)
g_mag = "{0:.4f}".format(1/(2*g0))
g_int = "{0:.1f}".format(g_int0)
name="params/2x2/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(g,g_el,g_mag,g_int)

def transf_gauge(par):
    t1 = par[0] + 1j * par[10]
    t2 = par[3] + 1j * par[13]
    y1, z1, y2, z2 = par[1] + 1j * par[11], par[2] + 1j * par[12], par[4] + 1j * par[14], par[5] + 1j * par[15]
    a, b, c, d = par[6] + 1j * par[16], par[7] + 1j * par[17], par[8] + 1j * par[18], par[9] + 1j * par[19]

    tau = np.array(
                       [
                [0, -1.0j * t1, 1.0j * t1, t1, -t1, -1.0j * t2, 1.0j * t2, t2, -t2],
                [ 1.0j * t1, 0, 1.0j * y1, z1, 1.0j * z1, -1.0j * a, -1.0j * c, -1.0j * b, -1.0j * d, ],
                [ -1.0j * t1, -1.0j * y1, 0, -1.0j * z1, -z1, 1.0j * c, 1.0j * a, 1.0j * d, 1.0j * b, ],
                [-t1, -z1, 1.0j * z1, 0, -y1, d, b, a, c],
                [t1, -1.0j * z1, z1, y1, 0, -b, -d, -c, -a],
                [1.0j * t2, 1.0j * a, -1.0j * c, -d, b, 0, 1.0j * y2, z2, 1.0j * z2],
                [ -1.0j * t2, 1.0j * c, -1.0j * a, -b, d, -1.0j * y2, 0, -1.0j * z2, -z2, ],
                [-t2, 1.0j * b, -1.0j * d, -a, c, -z2, 1.0j * z2, 0, -y2],
                [t2, 1.0j * d, -1.0j * b, -c, a, -1.0j * z2, z2, y2, 0],
            ]
    , dtype=complex)
    tau = tau[1:,1:]
    #print(tau@(tau.T.conj()))

    tau2 = tmat2(tau)
    mat_to_parvec = lambda tau: [
                # Real parts
    par[0],
    tau[0][1].imag,  # y1r
    tau[0][2].real,  # z1r
    par[3],
    tau[4][5].imag,  # y2r
    tau[4][6].real,  # z2r
    -tau[0][4].imag, # ar
    -tau[0][6].imag, # br
    -tau[0][5].imag, # cr
    -tau[0][7].imag, # dr

    # Imaginary parts
    par[10],
    -tau[0][1].real,  # y1i
    tau[0][2].imag,  # z1i
    par[13],
    -tau[4][5].real,  # y2i
    tau[4][6].imag,   # z2i
    tau[0][4].real,  # ai
    tau[0][6].real,  # bi
    tau[0][5].real,  # ci
    tau[0][7].real,  # di
    ]

    return list(mat_to_parvec(tau2))


def tmat2(tmat):
    u, s, vh = np.linalg.svd(tmat)
    return np.around(u @ np.linalg.inv(np.diag(s)) @ vh, decimals=14)

lattice = Lattice2D(L, L)

system_type = Z2System2D
system_cfg = Z2System2D_G2C_F2C_Config(lattice, g0/2, 1/(2*g0), g_int0, 0, [0,0],num_pg_layer=1,num_fermionic_layer=1)

nlayer=2
param = np.load(name)

param1 = np.reshape(param, (nlayer, -1))
print("param 1:", param1)

param2=np.zeros(40)
param2[0:20]  = transf_gauge(param[ 0:20])
# param2[20:40] = transf_gauge(param[20:40])
param2[20:40] = param[20:40]
param2 = np.reshape(param2, (nlayer, -1))
print("param 2:", param2)


def eval_en(par):
    system_cfg.paramvec = par
    system = system_type(system_cfg)
    eval_config = ExactEvaluatorConfig()
    ex_eval = ExactEvaluator(eval_config, system)
    res = ex_eval.evaluate()
    return res

res1 = eval_en(param1)
print("Energy 1", res1["energy"])
res2 = eval_en(param2)
print("Energy 2", res2["energy"])
