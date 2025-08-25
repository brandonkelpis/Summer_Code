import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use (['science', 'notebook', 'grid'])
from numba import njit 
from scipy.integrate import dblquad,quad
from MilkyWay_Orbits import MilkyWay


# Calling instance for Stucker's Milky Way code.
mw1 = MilkyWay(mass_disk=5.6e10,  scalelength_disk=5000, thickness_disk=1100)

# X-axis
radii = np.geomspace(0.000001,1001,1001)

#Setting up observational integrands based off obersvational data from Tamm et al.
@njit
def integrand_NFW (r):
  pc = (1.10 * (10**-2)) *(1000**3)
  rc = 18
  a = r / rc
  b =(1 + (r/rc))**2
  c = a * b
  return(pc / c) * 4 * np.pi *r**2   # Area multiplied to return statement to obtain mass from single integral integration.


@njit
def integrand_double_bulge (y,x): # X and Y equivalent to rho and phi here (spherical coordinates).
  pc = 920100000
  dn = 7.769
  ac = 1.155
  q = .72
  N = 2.7
  a = x*np.sqrt((np.sin(y))**2 + (((np.cos(y))**2) / (q**2)))
  b = (1/N)
  c = (a/ac)**b -1
 
 

  return 2*np.pi*np.sin(y)*(x**2)*pc*np.exp(-dn*(c)) # Originally a triple integral (spherical coordinates) where theta was constant. Turned into double for faster computation.

@njit
def integrand_double_stellar_disk (y,x):
  pc = 0.01307 * (1000**3)
  dn = 3.273
  ac = 10.67
  q = .17
  N = 1.2
  a = x*np.sqrt((np.sin(y))**2 + (((np.cos(y))**2) / (q**2)))
  b = (1/N)
  c = (a/ac)**b -1
 

  return 2*np.pi*np.sin(y)*(x**2)*pc*np.exp(-dn*(c))

# Creating arrays to evaluate each radius (in kiloparsecs).

result_bulge = np.zeros(1001)
error_bulge = np.zeros(1001)

result_stellar_disk = np.zeros(1001)
error_stellar_disk = np.zeros(1001)

result_baryon = np.zeros(1001)
error_baryon = np.zeros(1001)

result_total = np.zeros(1001)
error_total = np.zeros(1001)

result_NFW = np.zeros(1001)
error_NFW = np.zeros(1001)


radii = np.geomspace(0.000001,1001,1001) # For logarithmic scale and faster computations.

# For loops to evaluate solar mass enclosed as a function of radius.

for i,r in enumerate(radii):
  inner_lower_limit = 0
  inner_upper_limit = r

  outer_lower_limit = 0
  outer_upper_limit = np.pi

  result_bulge[i], error_bulge[i] = dblquad(integrand_double_bulge, inner_lower_limit, inner_upper_limit, lambda x: outer_lower_limit, lambda x: outer_upper_limit)

for i,r in enumerate(radii):
  inner_lower_limit = 0
  inner_upper_limit = r

  outer_lower_limit = 0
  outer_upper_limit = np.pi

  result_stellar_disk[i], error_stellar_disk[i] = dblquad(integrand_double_stellar_disk, inner_lower_limit, inner_upper_limit, lambda x: outer_lower_limit, lambda x: outer_upper_limit)

for i,r in enumerate(radii):

  lower_limit = 0.000001
  upper_limit = r

  result_NFW[i], error_NFW[i] = quad(integrand_NFW, lower_limit, upper_limit)

# Setting up evaluated functions.

y1= result_bulge

y2= result_stellar_disk

baryon = np.array(result_stellar_disk) + np.array(result_bulge) 

y6 = baryon + np.array(result_NFW)

y7 = result_NFW

# Calling from Stucker's code.

stellar_mass = mw1.enclosed_mass(radii *10**3, components="s") 
bulge_mass = mw1.enclosed_mass(radii*10**3,components = 'b')
NFW_mass = mw1.enclosed_mass(radii*10**3, components= 'h')
baryon_mass = stellar_mass + bulge_mass
total_mass = NFW_mass + baryon_mass

# Stopping at virial radii.

i= np.argmax(radii > 213)


plt.xlabel ('r [kpc]')
plt.ylabel ('M ( < r) [$M_{\odot}$]')

plt.yscale('log')
plt.xscale('log')

plt.xlim(left=10**0) 
plt.xlim(right=10**3) 

plt.ylim (1e8, 1e13)

plt.plot(radii[:i],y2[:i], label = 'Stellar Disk', color = 'cornflowerblue')

plt.plot(radii[:i],y1[:i], label = 'Bulge', color = 'magenta')

plt.plot(radii[:i],baryon[:i], label = 'All Baryons', color = 'lime')

plt.plot(radii[:i],y6[:i], label = 'Total', color = 'darkgoldenrod')



plt.plot (radii[:i], y7[:i], label = 'NFW', color = 'dimgrey')

plt.plot(radii[:i], stellar_mass[:i], label = 'Stellar Disk Stücker', color = 'blue')
plt.plot(radii[:i], bulge_mass[:i], label = 'Bulge Stücker', color = 'purple')

plt.plot (radii[:i], NFW_mass[:i], label = 'NFW Stücker', color = 'black')
plt.plot (radii[:i],baryon_mass[:i], label = 'All Baryons Stücker', color = 'green')
plt.plot (radii[:i],total_mass[:i], label ='Total Stücker', color = 'orange')

plt.legend (loc = 'upper left',  fontsize = 8)

ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()