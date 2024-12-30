import numpy as np

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

def vec2(vec):
    vec2 = np.zeros(40)
    vec2[0:20]  = transf_gauge(vec[0:20])
    vec2[20:40] = vec[20:40]
    return vec2
