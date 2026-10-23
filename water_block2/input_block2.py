from pyscf import gto, scf, mcscf, dmrgscf
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
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 6, 8)
mc.fcisolver = dmrgscf.DMRGCI(mol)
mc.canonicalization = False
#mc.canonicalization = True
dmrgscf.dryrun(mc)
