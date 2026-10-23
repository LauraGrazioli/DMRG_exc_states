from pyscf import gto, scf, mcscf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

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

# RHF case (for spin-adapted / non-spin-adapted DMRG)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 6, 8)
#ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, mc.ncore, mc.ncas, g2e_symm=8)
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, mc.ncore, mc.ncas, g2e_symm=1)
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
#driver.write_fcidump(h1e, g2e, ecore=ecore, filename='./FCIDUMP', pg="d2h", h1e_symm=True)
driver.write_fcidump(h1e, g2e, ecore=ecore, filename='./FCIDUMP', pg="c1", h1e_symm=True)

