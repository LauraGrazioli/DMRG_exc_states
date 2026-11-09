"""Fiedler-reorder an FCIDUMP by the exchange matrix K_ij = (ij|ji)."""
import sys, numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump
src, dst = sys.argv[1], sys.argv[2]
d = fcidump.read(src)
n = d['NORB']
eri = ao2mo.restore(1, d['H2'], n)
K = np.abs(np.einsum('ijji->ij', eri)); np.fill_diagonal(K, 0)
L = np.diag(K.sum(1)) - K
w, v = np.linalg.eigh(L)
perm = np.argsort(v[:, 1])
print("perm", perm.tolist())
h1 = d['H1'][np.ix_(perm, perm)]
eri = eri[np.ix_(perm, perm, perm, perm)]
orbsym = np.asarray(d['ORBSYM'])[perm]
fcidump.from_integrals(dst, h1, eri, n, d['NELEC'], nuc=d['ECORE'], ms=d['MS2'], orbsym=orbsym, tol=1e-15)
