from pyscf import gto, scf, lib, dmrgscf, mcscf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg
import os

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

mol = gto.M(
    atom = '''
    O   -0.10237605   0.00000000  -0.07926801
    H   -0.10237605   0.00000000   1.81045851
    H    1.72715822   0.00000000  -0.55241774
    ''',
    basis = '6-31g',
    symmetry = False,
    unit = 'Bohr',
    charge = 0,
    spin = 0
)
mf = scf.RHF(mol)
mf.kernel()
print('done')

#####
#(AVAS = Automated Valence Active Space) procedure in PySCF. 
#It analyzes the molecular orbitals and automatically constructs
# a chemically motivated active space from the atomic orbitals you specify.
from pyscf.mcscf import avas
nactorb, nactelec, coeff = avas.avas(mf, ["O 2p", "O 2s", "H 1s"])
print('CAS = ', nactorb, nactelec)

lib.param.TMPDIR = os.path.abspath(lib.param.TMPDIR)

solvers = [dmrgscf.DMRGCI(mol, maxM=1000, tol=1E-10) for _ in range(3)]
weights = [1.0 / len(solvers)] * len(solvers)

solvers[0].spin = 0
solvers[1].spin = 0
solvers[2].spin = 0

for i, mcf in enumerate(solvers):
    mcf.runtimeDir = lib.param.TMPDIR + "/%d" % i
    mcf.scratchDirectory = lib.param.TMPDIR + "/%d" % i
    mcf.threads = 8
    mcf.memory = int(mol.max_memory / 1000) # mem in GB

mc = mcscf.CASSCF(mf, nactorb, nactelec)
mcscf.state_average_mix_(mc, solvers, weights)

mc.canonicalization = True
mc.natorb = True
mc.kernel(coeff)


ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, mc.ncore, mc.ncas, g2e_symm=1)
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
driver.write_fcidump(h1e, g2e, ecore=ecore, filename='./FCIDUMP', pg="c1", h1e_symm=True)

