# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..../adiabatic-tides")
sys.path.append("..../adiabatic-tides/")
import adiabatic_tides as at
import cusp_encounters.encounters_math as em
import cusp_encounters.cusp_distribution
from cusp_encounters.decorators import h5cache
from scipy.integrate import simpson as simps
import cusp_encounters.gammaray as gammaray
import Create_Orbits_Mpi
from MilkyWay_Orbits import MilkyWay

cachedir = "..../caches"
G = 43.0071057317063e-10 # Mpc (km/s)^2 / Msol 
mw = MilkyWay(adiabatic_contraction=True, cachedir=cachedir, mode="cautun_2020")

orbits = Create_Orbits_Mpi.orbits

cuspdis = cusp_encounters.cusp_distribution.CuspDistribution(cachedir=cachedir)
cusps = cuspdis.sample_cusps(100000)

shape = orbits["chi_star"].shape
nrepeats = int(np.ceil(np.prod(shape) / cusps["A"].shape[0]))
def deform(arr):
    return np.repeat(arr, nrepeats)[:np.prod(shape)].reshape(shape)

As, rcusps, rcores  = deform(cusps["A"]), deform(cusps["rcusp"]), deform(cusps["rcore"])

Bcusps, Bcores = em.Bresistance_of_r(rcusps, A=As), em.Bresistance_of_r(rcores, A=As)

mdm_per_samp = cusps["mdm_per_cusp"]

r = np.linalg.norm(orbits["pos"], axis=-1)
chistar = orbits["chi_star"][-1] * np.ones_like(orbits["chi_star"])
rperi = np.min(r, axis=0)

Bstar = 2.*np.pi*np.clip(chistar, 1e-10, None)*mw.G
Beff= em.sample_effective_B_hist(shape, Bminfac=1e-4, p=1.2, initial_sample=100000, cachefile="/Users/brandonkelpis/Desktop/Carnegie/Complete_Code/caches/Beff_hist.hdf5") * Bstar #deleted %cachedir part at end of function.
Beff_tide = np.sqrt(np.clip(42.2*mw.radial_mean_tide(rperi), 1e-10, None))*np.ones(r.shape)
Beff_tot= np.sqrt(Beff**2 + Beff_tide**2)

Js = em.Jcorecusp_in_4piA2(Beff, Bcores, Bcusps) * 4.*np.pi*As**2
Jstot = em.Jcorecusp_in_4piA2(Beff_tot, Bcores, Bcusps) * 4.*np.pi*As**2
Jstide = em.Jcorecusp_in_4piA2(Beff_tide, Bcores, Bcusps) * 4.*np.pi*As**2

J0s = em.Jcorecusp_in_4piA2(Bcusps*1e-7, Bcores, Bcusps) * 4.*np.pi*As**2

Bmax = em.sample_strongest_B_analytic(Bstar)
Jsmax = em.Jcorecusp_in_4piA2(Bmax, Bcores, Bcusps) * 4.*np.pi*As**2

rbins = np.logspace(-1,np.log10(1e3), 81)
mi = orbits["mass"]*np.ones(r.shape)
ri, perc = em.get_percentile_profile(r/1000., Js/J0s, xbins=rbins, weights=mi)
ri, perctot = em.get_percentile_profile(r/1000., Jstot/J0s, xbins=rbins, weights=mi)
# Each of these represents the total annihilation coming from inside each spherical shell
nxJ, _ = np.histogram(r/1000., weights=Js*mi/mdm_per_samp / len(r), bins=rbins)
nxJtot, _ = np.histogram(r/1000., weights=Jstot*mi/mdm_per_samp / len(r), bins=rbins)
nxJtide, _ = np.histogram(r/1000., weights=Jstide*mi/mdm_per_samp / len(r), bins=rbins)
nxJmax, _ = np.histogram(r/1000., weights=Jsmax*mi/mdm_per_samp / len(r), bins=rbins)
nxJ0s, _ = np.histogram(r/1000., weights=J0s*mi/mdm_per_samp / len(r), bins=rbins)
n, _ = np.histogram(r/1000., bins=rbins, weights=mi /len(r))
vbins = 4.*np.pi/3. * (rbins[1:]**3 - rbins[:-1]**3)

ri, percmax = em.get_percentile_profile(r/1000., Jsmax/J0s, xbins=rbins, weights=mi)

plt.loglog(ri, (mw.profile_contracted_nfw.self_density(ri/1e3)*1e-18)**2, label="NFW halo", color="black", lw=2)
plt.loglog(ri, nxJ0s/(vbins*1e9), label="cusps (unperturbed)", lw=3, color="C2", linestyle="dashed")
plt.loglog(ri, nxJtide/(vbins*1e9), label="cusps (tides)", lw=3, color="C0", linestyle="dashed")
plt.loglog(ri, nxJtot/(vbins*1e9), label="cusps (enc. + tides)", lw=3, color="red")

plt.legend(fontsize=12)
plt.xlim(.4,2e2)
plt.ylim(10**-7,0)
plt.grid()
plt.xlabel("r [kpc]", fontsize=14)
plt.ylabel(r"$\rm{d} J/\rm{d} V$ [$M_\odot^2 / \rm{pc}^3 / \rm{pc}^3$]", fontsize=14)


# This is for saving data to tables.

# halo = np.column_stack([ri,(mw.profile_contracted_nfw.self_density(ri/1e3)*1e-18)**2 ])
# unperturbed_cusp = np.column_stack([ri,nxJ0s/(vbins*1e9)])
# tidal_pull_cusp = np.column_stack([ri,nxJtide/(vbins*1e9)])
# enc_and_tide_cusp = np.column_stack([ri,nxJtot/(vbins*1e9)])
# addedup = ((mw.profile_contracted_nfw.self_density(ri/1e3)*1e-18)**2) +  nxJ0s/(vbins*1e9) +nxJtide/(vbins*1e9)+nxJtot/(vbins*1e9)
# total = np.column_stack([ri,addedup])

# np.savetxt('Halo_Total_Data.txt',halo, delimiter='\t', header='radius , DJ/DV', comments='', fmt = '%.17f' )
# np.savetxt('U_Cusp_Total_Data.txt',unperturbed_cusp, delimiter='\t', header='radius , DJ/DV', comments='', fmt = '%.17f' )
# np.savetxt('T_Cusp_Total_Data.txt',tidal_pull_cusp, delimiter='\t', header='radius , DJ/DV', comments='', fmt = '%.17f' )
# np.savetxt('E_C_Cusp_Total_Data.txt',enc_and_tide_cusp, delimiter='\t', header='radius , DJ/DV', comments='', fmt = '%.17f' )
# np.savetxt('Total_Data.txt',total, delimiter='\t', header='radius , DJ/DV', comments='', fmt = '%.17f' )

plt.show()





