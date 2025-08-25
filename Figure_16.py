import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(".../cusp-encounters/adiabatic-tides")
sys.path.append(".../adiabatic-tides/")
import adiabatic_tides as at
import cusp_encounters.encounters_math as em
import cusp_encounters.cusp_distribution
from cusp_encounters.decorators import h5cache
import cusp_encounters.gammaray as gammaray
import Create_Orbits_Mpi
from MilkyWay_Orbits import MilkyWay
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter

cachedir = ".../caches"
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

nxJ, _ = np.histogram(r/1000., weights=Js*mi/mdm_per_samp / len(r), bins=rbins)
nxJtot, _ = np.histogram(r/1000., weights=Jstot*mi/mdm_per_samp / len(r), bins=rbins)
nxJtide, _ = np.histogram(r/1000., weights=Jstide*mi/mdm_per_samp / len(r), bins=rbins)
nxJmax, _ = np.histogram(r/1000., weights=Jsmax*mi/mdm_per_samp / len(r), bins=rbins)
nxJ0s, _ = np.histogram(r/1000., weights=J0s*mi/mdm_per_samp / len(r), bins=rbins)
n, _ = np.histogram(r/1000., bins=rbins, weights=mi /len(r))
vbins = 4.*np.pi/3. * (rbins[1:]**3 - rbins[:-1]**3)

ri, percmax = em.get_percentile_profile(r/1000., Jsmax/J0s, xbins=rbins, weights=mi)

@h5cache(file="/Users/brandonkelpis/Desktop/Carnegie/Complete_Code/caches/angle_histogram.hdf5")
def angle_histograms(nsel=1000, d=765000, nalpha=100): # Changed to 765,000 due to Karwin et al.
    alpha, Jalpha0 = em.angle_histogram(r[0:nsel].flatten(), (J0s*mi/mdm_per_samp/nsel)[0:nsel].flatten(), nalpha=nalpha, d=d)
    alpha, Jalpha = em.angle_histogram(r[0:nsel].flatten(), (Js*mi/mdm_per_samp/nsel)[0:nsel].flatten(), nalpha=nalpha, d=d)
    alpha, Jalphatot = em.angle_histogram(r[0:nsel].flatten(), (Jstot*mi/mdm_per_samp/nsel)[0:nsel].flatten(), nalpha=nalpha, d=d)
    alpha, Jalphatide = em.angle_histogram(r[0:nsel].flatten(), (Jstide*mi/mdm_per_samp/nsel)[0:nsel].flatten(), nalpha=nalpha, d=d)
    alpha, Jalphamax = em.angle_histogram(r[0:nsel].flatten(), (Jsmax*mi/mdm_per_samp/nsel)[0:nsel].flatten(), nalpha=nalpha, d=d)

    rhalo = np.logspace(-2, 5, 100000) #rhalo = np.logspace(-2, 4, 100000)
    rcent = np.sqrt(rhalo[1:]*rhalo[:-1])
    Jshalo = (mw.profile_contracted_nfw.self_density(rcent*1e-3)*1e-18)**2 *4.*np.pi/3. * ((1e3*rhalo[1:])**3 - (1e3*rhalo[:-1])**3)
    alpha, Jtothalo = em.angle_histogram(rcent*1e3, Jshalo, nalpha=nalpha, d=d)
    
    return alpha, Jalpha0, Jalpha, Jalphatot, Jalphatide, Jalphamax, Jtothalo

alpha, Jalpha0, Jalpha, Jalphatot, Jalphatide, Jalphamax, Jtothalo = angle_histograms(nsel=1000)


from matplotlib.gridspec import GridSpec

fig, ax1= plt.subplots(figsize=(8,5))


nfac = gammaray.intragalactic_integral(1.)

ax1.loglog(alpha*180/np.pi, Jtothalo*nfac, label="NFW halo", color="grey", lw=3)

ax1.loglog(alpha*180/np.pi, (Jalpha0*nfac)*0.5, label="cusps (unperturbed)", lw=3, color="C2", linestyle="dashed")

ax1.loglog(alpha*180/np.pi, (Jalphatide*nfac)*0.5, label=r"cusps (tide only)", color="C0", linestyle="dashed", lw=3)

ax1.loglog(alpha*180/np.pi, (Jalphatot*nfac)*0.5, label="cusps (enc. + tide)", lw=3, color="red")

ax1.loglog(alpha*180/np.pi, ((Jalphatot*0.5)+Jtothalo)*nfac, label="total", lw=3, color="black", ls="dotted")

igrb = gammaray.igrb_integral()

ax1.legend(loc = "upper right", fontsize=13)
ax1.set_xlabel(r"$\theta$ (deg.)", fontsize=14)

ax1.set_xlim(10**-2, 14) 
ax1.set_xscale('log')
# x_ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]


for ax in ax1,:
    ax.set_yscale("log")
    ax.set_ylim(10**-10,10**0) 
    # ax.set_xticks(x_ticks)
ax1.grid("on")

# This is for the tangent radius in log scale. If you choose linear, just change the logs, uncomment "x_ticks" and the ax.set in the loop.

ticks = [ 0.4, 8.3, 14] # To place the tangent radius around the same points as Karwin et al.
ax2 = ax1.twiny()
ax2.set_xscale ('log')
ax2.set_xlim(ax1.get_xlim()) 
ax2.set_xticks(ticks)

angles = ax2.get_xticks()

distance = 765 * (angles*np.pi/180)
ax2.set_xticklabels(np.round(distance).astype(int))

ax2.set_xlabel ("Tangent Radius [kpc]")

ax1.set_ylabel(r"Intensity [ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=14)

plt.show()
