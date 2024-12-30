import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt

g_int0= 1.
g0 = 0.2


def parvec(size, g0):
    g    =  "{0:.1f}".format(g0)
    g_el  = "{0:.4f}".format(g0/2)
    g_mag = "{0:.4f}".format(1/(2*g0))
    g_int = "{0:.1f}".format(g_int0)
    name = "{}_params/g_{}_el_{}_mag_{}_int_{}_mass_0.0_extracted_paramvec.npy".format(size,g,g_el,g_mag,g_int)
    print(name)
    vec = np.load(name)
    vec = vec[vec != 0]
    return vec

def norm(parvec):
    return np.linalg.norm(parvec)

data_1 = parvec("2x2",g0)
data_2 = parvec("4x4",g0)
data_3 = parvec("6x6",g0)

ticks = ["$y_1$","$z_1$","$y_2$","$z_2$","$a$","$b$","$c$","$d$","$y_1$","$z_1$","$y_2$","$z_2$","$a$","$b$","$c$","$d$","$t_1$","$a$","$b$","$c$","$d$","$t_1$","$a$","$b$","$c$","$d$"]

plt.plot(data_1, marker="o",linestyle="",label="2x2")
plt.plot(data_2, marker="s",linestyle="",label="4x4")
plt.plot(data_3, marker=">",linestyle="",label="6x6")
plt.xticks(range(26),ticks)
plt.grid(axis="x")
plt.legend()
plt.title("g="+str(g0))
plt.show()

print(data_1)
print(data_2)
print(data_3)

print(np.linalg.norm(data_1))
print(np.linalg.norm(data_2))
print(np.linalg.norm(data_3))


plt.plot(np.abs(data_2-data_1), marker="o",linestyle="",label="diff 4-2")
plt.plot(np.abs(data_3-data_1), marker="s",linestyle="",label="diff 6-2")
plt.xticks(range(26),ticks)
plt.grid(axis="x")
plt.legend()
plt.title("g="+str(g0))
plt.show()

print(data_1)

# gvals = np.arange(0.1,2.9,0.1)
# plt.plot(gvals, [norm(parvec("2x2",g)) for g in gvals], marker="o",linestyle="",label="2x2")
# plt.plot(gvals, [norm(parvec("4x4",g)) for g in gvals], marker="s",linestyle="",label="4x4")
# plt.plot(gvals, [norm(parvec("6x6",g)) for g in gvals], marker=">",linestyle="",label="6x6")
# plt.ylim(4,6)
# plt.legend()

# plt.show()
