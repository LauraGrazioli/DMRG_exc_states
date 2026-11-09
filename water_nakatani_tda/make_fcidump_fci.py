"""H2O / cc-pVDZ, 1s frozen (8e, 23o), C2v: FCIDUMP + per-irrep singlet FCI oracle.

Setup of Nakatani-Wouters-Van Neck-Chan, JCP 140, 024108 (2014), Table IV:
R(OH) = 1.8111 bohr, HOH = 104.45 deg, canonical RHF orbitals.
"""
import sys, time
import numpy as np
from pyscf import gto, scf, mcscf, fci, symm
from pyscf.tools import fcidump

out = sys.argv[1]
nroots = int(sys.argv[2]) if len(sys.argv) > 2 else 4
R, th = 1.8111, np.deg2rad(104.45)
y, z = R * np.sin(th / 2), R * np.cos(th / 2)
mol = gto.M(atom=f"O 0 0 0; H 0 {-y} {z}; H 0 {y} {z}", unit="Bohr",
            basis="cc-pvdz", symmetry="C2v", verbose=3)
mf = scf.RHF(mol).run(conv_tol=1e-12)
irr = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
print("E_RHF", mf.e_tot, "nao", mol.nao, "occ irreps", list(irr[:5]))

mc = mcscf.CASCI(mf, 23, 8)
fcidump.from_mcscf(mc, f"{out}/H2O.FCIDUMP", tol=1e-15, molpro_orbsym=True)
print("FCIDUMP written; act irreps", list(irr[1:]))

paper = {"A1": [-76.241697, -75.842487, -75.584080, -75.462977],
         "B1": [-75.939176, -75.401515, -75.381977],
         "A2": [-75.864445, -75.448180, -75.403286],
         "B2": [-75.765964, -75.696018, -75.322655]}
for ir, ref in paper.items():
    t = time.time()
    mc = mcscf.CASCI(mf, 23, 8)
    mc.fcisolver = fci.direct_spin0_symm.FCI(mol)
    mc.fcisolver.wfnsym = ir
    mc.fcisolver.nroots = nroots
    mc.fcisolver.conv_tol = 1e-10
    mc.fcisolver.max_cycle = 200
    mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, ss=0, shift=0.5)
    mc.verbose = 0
    e = np.atleast_1d(mc.kernel()[0])
    for k, ek in enumerate(e):
        rk = ref[k] if k < len(ref) else float("nan")
        print(f"{ir} root {k}: E_FCI = {ek:.6f}   paper E_conv = {rk:.6f}   diff = {1e3*(rk-ek):+.3f} mEh", flush=True)
    print(f"  ({time.time()-t:.0f} s)", flush=True)
