import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use (['science', 'notebook', 'grid'])
from MilkyWay_Orbits import MilkyWay

# Calling instances from Stuckers MilkyWay code.
mw3 = MilkyWay()

# X-axis. Logarithmic scale.

radii = np.geomspace(0.000001,1001,1001)

# Functions to be plotted from Stuckers code.

stellar_mass = mw3.enclosed_mass(radii *10**3, components="s") 
mass2 = mw3.enclosed_mass(radii *10**3, components="s") 
bulge_mass = mw3.enclosed_mass(radii*10**3,components = 'b')
NFW_mass = mw3.enclosed_mass(radii*10**3, components= 'h')

baryon_mass = stellar_mass + bulge_mass
total_mass = NFW_mass + baryon_mass

# Plot commands.

plt.xlabel ('r [kpc]')
plt.ylabel ('M ( < r) [$M_{\odot}$]')

plt.yscale('log')
plt.xscale('log')

plt.xlim(left=10**0) 
plt.xlim(right=10**3) 

plt.ylim (1e8, 1e13)

# Stopping curves at virial radii.
i = np.argmax(radii > 213)

plt.plot(radii[:i], stellar_mass[:i], label = 'Stellar Disk Stücker', color = 'blue')
plt.plot(radii[:i], bulge_mass[:i], label = 'Bulge Stücker', color = 'purple')
plt.plot (radii[:i], NFW_mass[:i], label = 'NFW Stücker', color = 'black')
plt.plot (radii[:i],baryon_mass[:i], label = 'All Baryons Stücker', color = 'green')
plt.plot (radii[:i],total_mass[:i], label ='Total Stücker', color = 'orange')

ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.legend (loc = 'upper left',  fontsize = 12)

plt.show()


