import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import utils

g_int0= 1.
g1 = 1.7
g2 = 1.8


def parvec(size, g0):
    g    =  "{0:.1f}".format(g0)
    g_el  = "{0:.4f}".format(g0/2)
    g_mag = "{0:.4f}".format(1/(2*g0))
    g_int = "{0:.1f}".format(g_int0)
    name = "{}_params/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(size,g,g_el,g_mag,g_int)
    print(name)
    vec = np.load(name)
    return vec

data_1 = parvec("2x2",g1)

data_2 = parvec("2x2",g2)
data_3 = parvec("2x2",g2)
data_3 = utils.vec2(data_2)

data_1 = data_1[data_1 != 0]
data_2 = data_2[data_2 != 0]
data_3 = data_3[data_3 != 0]

ticks = ["$y_1$","$z_1$","$y_2$","$z_2$","$a$","$b$","$c$","$d$","$y_1$","$z_1$","$y_2$","$z_2$","$a$","$b$","$c$","$d$","$t_1$","$a$","$b$","$c$","$d$","$t_1$","$a$","$b$","$c$","$d$"]

plt.plot(data_1, marker="o",linestyle="",label="2x2, g="+str(g1))
plt.plot(data_2, marker="s",linestyle="",label="2x2, g="+str(g2))
plt.plot(data_3, marker=">",linestyle="",label="2x2, transformed, g="+str(g2))
plt.xticks(range(26),ticks)
plt.grid(axis="x")
plt.legend()
plt.show()

