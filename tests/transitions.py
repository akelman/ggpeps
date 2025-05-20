import unittest
from unittest import skip

import numpy as np
import sympy as sp
from scipy.linalg import block_diag
import itertools


from ggpeps import lattice, utils
from ggpeps import system, exacteval
from ggpeps.modearray import generate_permutation_matrix


def forbidden_pairs(system, site_coord, dir):
    dest = []
    gauge_fields = system.cfg.gaugemgr.get_possible_gauge_values()
    pairs = list(itertools.combinations(gauge_fields, 2))
    for pair in pairs:
        if check_transition(system, pair[0], pair[1], site_coord, dir):
            dest.append(pair)
    return dest


def check_transition(system, g_old, g_new, site_coord, dir, layer=0):
    gamma_nerutral = system.cfg.generate_gamma_gauge_neutral_dict()[layer][dir]
    rot_mat = system.generate_rotmat(g_old, site_coord, dir)
    gamma_old = rot_mat @ gamma_nerutral @ np.transpose(rot_mat)
    rot_mat = system.generate_rotmat(g_new, site_coord, dir)
    gamma_new = rot_mat @ gamma_nerutral @ np.transpose(rot_mat)
    update_mat = gamma_old - gamma_new
    return np.linalg.det(update_mat) < 1e-10


if __name__ == "__main__":
    lat = lattice.Lattice2D(2, 2)
    num_pg_layer = 1
    num_fermionic_layer = 0
    nlayer = num_pg_layer + num_fermionic_layer
    unitcell_size = 1
    paramvec = np.random.rand(nlayer, unitcell_size, 20)
    cfg = system.D6System2D_Config(
        lat, 1, 1, 0, 0, None, num_pg_layer, num_fermionic_layer
    )
    cfg.paramvec = paramvec
    system_D6 = system.D2nSystem2D(cfg)
    system_D6.cfg.enforce_parameter_conditions(system_D6.cfg.paramvec)
    forbidden_pairs_x = forbidden_pairs(system_D6, (0, 0), 0)
    print(len(forbidden_pairs_x))
    doctionary = dict()
    for p in range(3):
        for q in range(2):
            doctionary[(p, q)] = system_D6.cfg.gaugemgr.get_representation(p, q)
    counter = 0
    for pair in forbidden_pairs_x:
        counter += 1
        print(counter)
        for key in doctionary:
            if np.allclose(doctionary[key], pair[0]):
                print("key", key, pair[0])
            if np.allclose(doctionary[key], pair[1]):
                print("key", key, pair[1])
    # forbidden_pairs_y = forbidden_pairs(system_D6, (0, 0), 1)
    # for pair in forbidden_pairs_y:
    #     print("The pairs y:", pair[0], pair[1])
    # print(system_D6.cfg.gaugemgr.get_possible_gauge_values())
