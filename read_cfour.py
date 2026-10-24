#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
from time import time
import jax.numpy as jnp
from jax import jit, vmap, random
from dataclasses import dataclass, field
# from memory_profiler import profile
from typing import Tuple
import jax.scipy.linalg #import eigh_tridiagonal
from jax.scipy.linalg import inv, sqrtm
import subprocess
def read_CFOUR():
    f = open('data_cfour.bin', 'rb')

    # number of basis function
    nbas = np.fromfile(f, dtype=np.int64, count=1)[0]
    # number of active orbitals
    n_act = np.fromfile(f, dtype=np.int64, count=1)[0]
    # number of inactive orbitals
    n_int = np.fromfile(f, dtype=np.int64, count=1)[0]
    # number of external orbitals
    n_ext = nbas - n_int - n_act
    # number of alpha and beta electrons
    m_a, m_b = np.fromfile(f, dtype=np.int64, count=2)
    # number of non-redundant orbital rotations and determinants
    n_rot, n_det = np.fromfile(f, dtype=np.int64, count=2)
    # number of alpha and beta ci strings s. t. n_a * n_b = n_det, for closed shell n_a == n_b
    n_a, n_b = np.fromfile(f, dtype=np.int64, count=2)

    # overlap integrals in the ao basis
    S = np.fromfile(f, dtype=np.float64, count=nbas ** 2)
    S = S.reshape((nbas, nbas), order='F')

    # molecular orbitals with C^T S C = 1
    C = np.fromfile(f, dtype=np.float64, count=nbas ** 2)
    C = C.reshape((nbas, nbas), order='F')

    # ci coefficients
    c = np.fromfile(f, dtype=np.float64, count=n_det)
    # c = c.reshape((n_a, n_b), order='F')

    # active 1-body reduced density matrix
    rdm1 = np.fromfile(f, dtype=np.float64, count=n_act ** 2)
    rdm1 = rdm1.reshape((n_act, n_act), order='F')

    # active 2-body reduced density matrix
    _ = np.fromfile(f, dtype=np.float64, count=n_act ** 4)
    # rdm2 = rdm2.reshape((n_act, n_act, n_act, n_act), order='F')

    # electronic energy
    ene = np.fromfile(f, dtype=np.float64, count=1)[0]

    # orbital gradient
    g_k = np.fromfile(f, dtype=np.float64, count=n_rot)
    # ci gradient
    g_ci = np.fromfile(f, dtype=np.float64, count=n_det)

    diag_k = np.fromfile(f, dtype=np.float64, count=n_rot)
    diag_k_AI = abs(diag_k[: n_int * n_act])
    diag_k_EI = abs(diag_k[n_int * n_act : n_int * (n_act + n_ext)])
    diag_k_EA = abs(diag_k[n_int * (n_act + n_ext):])
    diag_ci = abs(np.fromfile(f, dtype=np.float64, count=n_det))
    #diag_k = diag_k.reshape((n_act, n_act), order='F')

    # target index of saddle points
    k = np.fromfile(f, dtype=np.int64, count=1)[0]

    Hx_ci = []
    Hx_k = []
    for i in range(k):
        # ci hess-vec
        Hx_ci_tmp = np.fromfile(f, dtype=np.float64, count=n_det)
        # orbital hess-vec
        Hx_k_tmp = np.fromfile(f, dtype=np.float64, count=n_rot)

        Hx_ci.append(Hx_ci_tmp)
        Hx_k.append(Hx_k_tmp)
    # Spin <S^2> 
    spin = np.fromfile(f, dtype=np.float64, count=1)[0]
    # multiplicity 
    mult = np.fromfile(f, dtype=np.float64, count=1)[0]

    f.close()

    return [n_int, n_act, n_ext], [n_a, n_b], ene, c, C, \
            S, rdm1, [g_ci, g_k], [diag_ci, (diag_k_AI, diag_k_EI, diag_k_EA)], [Hx_ci, Hx_k], spin, mult
