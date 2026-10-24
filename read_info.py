#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from read_cfour import *
import jax.numpy as jnp
from jax import config, jit, random
from jax.scipy.linalg import sqrtm, inv


n, nab, ene, c, C, S, rdm1, _, precond, _, spin, mult = read_CFOUR()
print(n, nab, ene, c, C, S)
