import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use (['science', 'notebook', 'grid'])
import sys
sys.path.append("...../cusp-encounters/adiabatic-tides")
sys.path.append(".../MilkyWay_Orbits")
from MilkyWay_Orbits import MilkyWay

# The purpose of this is to shows that the 3d mass distribution matches and not just the spherically averaged radial distribution.

# Correct parameters for disk called in instance to match Tamm et al.

mw= MilkyWay(mass_disk=  5.6e10,  scalelength_disk=5000, thickness_disk=1100) 

# Setting up contour parameters.

rr = np.linspace(0, 10,100)
zz = np.linspace(-2,2,100)
r,z = np.meshgrid(rr,zz)

#Stuckers code only takes a 3D Cartesian cooridnate.

xvec = np.stack([r,np.zeros_like(r),z], axis=-1)

# Plotting density contours.

def integrand_double_stellar_disk (r,z):
  pc = 0.01307 * (1000**3)
  dn = 3.273
  ac = 10.67
  q = .17
  N = 1.2
  a = np.sqrt(r**2 +((z**2)/q**2))
  b = (1/N)
  c = (a/ac)**b -1
 

  return pc*np.exp(-dn*(c))

func = integrand_double_stellar_disk(r,z)
func2 = mw.density(xvec*1000, components="s", dx=0.1)

# Plotting contours. In log scale.

fig, ax = plt.subplots()

tamm = ax.contour(r, z, np.log(func), levels = np.arange(0,20), linestyles='dashed')
stucker = ax.contour(r, z,np.log(func2*1e9),levels = np.arange(0,20) , linestyles='solid')

plt.clabel(tamm, inline=1, fontsize=10)
plt.clabel(stucker, inline=1, fontsize=10)

tamm_legend, labels = tamm.legend_elements()
stucker_legend, labels2 = stucker.legend_elements()
plt.legend([tamm_legend[0], stucker_legend[0]],['Tamm','Stucker'],loc="upper right")

plt.xlabel ('r [kpc]')
plt.ylabel ('z [kpc]')

plt.title("Density Contours")

plt.show()