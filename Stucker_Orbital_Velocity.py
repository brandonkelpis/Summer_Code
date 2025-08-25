import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use (['science', 'notebook', 'grid'])
from MilkyWay_Orbits import MilkyWay

# Defining radius in kpc.

radii = np.linspace(0.1, 45, 45)

# Placing observer on x-y plane for easier calculations.

x = np.zeros_like(radii)
z = np.zeros_like(radii)

# Milky Way file only takes 3D Cartesian coodinates.

xaxi = np.stack([x, radii, z], axis=0).T

# Making instances.

mw = MilkyWay(conc_initial = 10 ,mass_halo=1e12)

# Calling accleration functions.

acc_h = mw.acceleration(xaxi *1000 , components='h')  
acc_b = mw.acceleration(xaxi *1000, components='b')
acc_s = mw.acceleration(xaxi *1000, components='s') 

# Converting to velocity.

v_h = np.sqrt(np.linalg.norm(acc_h,axis=1)*radii*1000)
v_b = np.sqrt(np.linalg.norm(acc_b,axis=1)*radii*1000)
v_s = np.sqrt(np.linalg.norm(acc_s,axis=1)*radii*1000)

acc_total = acc_h + acc_b + acc_s

acc_magn = np.linalg.norm(acc_total, axis=1)
acc_mag = np.abs(acc_total)

total_velocity = np.sqrt(acc_magn * radii *1000) 

# Plotting velocity as a function of radii.

plt.xlabel('r [kpc]')
plt.ylabel('$V_\mathrm{rot}\ (\mathrm{km}\ \mathrm{s}^{-1})$')
plt.plot(radii, total_velocity, color = 'purple',label = 'total')
plt.plot(radii, v_h,label = 'halo' )
plt.plot(radii, v_b,label = 'bulge' )
plt.plot(radii, v_s,label= 'stellar disk' )
plt.legend (loc = 'upper right',  fontsize = 8)

plt.ylim (0, 350)
plt.xlim (0, 40)

#Saving data from tables to help adjust Stucker's halo to match Tamms. Avoiding long computations.

data = np.column_stack([radii, total_velocity])
np.savetxt('Distance_Total_Data.txt',data, delimiter='\t', header='Distance, Total', comments='', fmt = '%.2f' )

plt.show()

