import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.integrate import nquad, tplquad
plt.style.use (['science', 'notebook', 'grid'])
from numba import njit 
from numpy import *
from MilkyWay_Orbits import MilkyWay

# Gravitational constant in (kpc*(km/s)**2) / solar mass

G = 4.30093e-6

# Virial radii.
distance = np.geomspace(0.1,213,213)

# @njit for faster calculations.
# Using Newtons gravitational acceleration integral.

@njit
def bulge_b(r,phi,theta,distance):
  pc = 920100000
  dn = 7.769
  ac = 1.155
  q = .72
  N = 2.7
  a = r*np.sqrt((np.sin(phi))**2 + (((np.cos(phi))**2) / (q**2)))
  b = (1/N)
  c = (a/ac)**b -1
  numer =r*np.sin(phi)*np.sin(theta)-distance # <a,b,c>. A and C will cancel out due to spherical symmetry.
  sinexphi = (np.sin(phi))**2
  cosextheta = (np.cos(theta))**2
  cosexphi = (np.cos(phi))**2
  rsqare = r**2
  middle_exp = r*np.sin(phi)*np.sin(theta) - distance
  denom = (np.sqrt(rsqare*sinexphi*cosextheta + (middle_exp)**2 + rsqare*cosexphi))**3

  return (numer/denom)*(G*np.sin(phi)*(r**2)*pc*np.exp(-dn*(c)))

#Create list to stores values for future integration. 

result_bulge_b = np.zeros_like(distance)
error_bulge_b = np.zeros_like(distance)

# Loop for inetgration.

for i,d in enumerate(distance):
  inner_lower_limit = 0
  inner_upper_limit = 213

  middle_lower_limit = 0
  middle_upper_limit = np.pi

  outer_lower_limit = 0
  outer_upper_limit = 2*np.pi

  result_bulge_b[i], error_bulge_b[i] = nquad(bulge_b, [(inner_lower_limit,inner_upper_limit),(middle_lower_limit,middle_upper_limit),(outer_lower_limit,outer_upper_limit)], args=(d,), opts = ({'points':[d]},{'points' : [np.pi/2]},{'points' : [np.pi/2]})) # the reason we are using d here for r, the infinity problem only happens at r=d, rho = pi/2, phi = pi/2 all at the same time

@njit
def stellar_disk_b(r,phi,theta,distance):
  pc = 0.01307 * (1000**3)
  dn = 3.273
  ac = 10.67
  q = .17
  N = 1.2
  a = r*np.sqrt((np.sin(phi))**2 + (((np.cos(phi))**2) / (q**2)))
  b = (1/N)
  c = (a/ac)**b -1
  numer =r*np.sin(phi)*np.sin(theta)-distance # <a,b,c>. A and C will cancel out due to spherical symmetry.
  sinexphi = (np.sin(phi))**2
  cosextheta = (np.cos(theta))**2
  cosexphi = (np.cos(phi))**2
  rsqare = r**2
  middle_exp = r*np.sin(phi)*np.sin(theta) - distance
  denom = (np.sqrt(rsqare*sinexphi*cosextheta + (middle_exp)**2 + rsqare*cosexphi))**3

  return (numer/denom)*(G*np.sin(phi)*(r**2)*pc*np.exp(-dn*(c)))


result_stellar_disk_b = np.zeros_like(distance)
error_stellar_disk_b = np.zeros_like(distance)

for i,d in enumerate(distance):
  inner_lower_limit = 0
  inner_upper_limit = 213

  middle_lower_limit = 0
  middle_upper_limit = np.pi

  outer_lower_limit = 0
  outer_upper_limit = 2*np.pi

  result_stellar_disk_b[i], error_stellar_disk_b[i] = nquad(stellar_disk_b, [(inner_lower_limit,inner_upper_limit),(middle_lower_limit,middle_upper_limit),(outer_lower_limit,outer_upper_limit)], args=(d,), opts = ({'points':[d]},{'points' : [np.pi/2]},{'points' : [np.pi/2]})) # the reason we are using d here for r, the infinity problem only happens at r=d, rho = pi/2, phi = pi/2 all at the same time

@njit
def NFW_b(r,phi,theta,distance):
  pc = (1.10 * (10**-2)) *(1000**3)
  rc = 18
  a = r / rc
  b =(1 + (r/rc))**2
  c = a * b
  numer =r*np.sin(phi)*np.sin(theta)-distance # <a,b,c>. A and C will cancel out due to spherical symmetry.
  sinexphi = (np.sin(phi))**2
  cosextheta = (np.cos(theta))**2
  cosexphi = (np.cos(phi))**2
  rsqare = r**2
  middle_exp = r*np.sin(phi)*np.sin(theta) - distance
  denom = (np.sqrt(rsqare*sinexphi*cosextheta + (middle_exp)**2 + rsqare*cosexphi))**3

  return (numer/denom)*(G*np.sin(phi)*(r**2))*(pc / c)


result_NFW_b = np.zeros_like(distance)
error_NFW_b = np.zeros_like(distance)

for i,d in enumerate(distance):
  inner_lower_limit = 0
  inner_upper_limit = 213

  middle_lower_limit = 0
  middle_upper_limit = np.pi

  outer_lower_limit = 0
  outer_upper_limit = 2*np.pi

  result_NFW_b[i], error_NFW_b[i] = nquad(NFW_b, [(inner_lower_limit,inner_upper_limit),(middle_lower_limit,middle_upper_limit),(outer_lower_limit,outer_upper_limit)], args=(d,), opts = ({'points':[d]},{'points' : [np.pi/2]},{'points' : [np.pi/2]}))

# Calulating velocities using centripedal acceleration formula.

velocity_bulge = np.sqrt(np.abs((result_bulge_b)*distance))

velocity_stellar_disk =np.sqrt(np.abs(result_stellar_disk_b*distance))

acceleration_total = np.abs((result_bulge_b+result_stellar_disk_b+result_NFW_b))

acceleration_baryons = np.abs(result_bulge_b+result_stellar_disk_b)

velocity_NFW = np.sqrt(np.abs(result_NFW_b*distance))

total = np.sqrt(acceleration_total*distance)

# Deifning radius in kpc.

radii = np.linspace(0.1, 45, 45)

# Placing observer on x-y plane for easier calculations.
x = np.zeros_like(radii)
z = np.zeros_like(radii)

# Instance only takes a 3D Cartesian cooridinate
xaxi = np.stack([x, radii, z], axis=0).T

# Purpose of overlay is to find model used in Tamms paper and have Stuckers match.
mw = MilkyWay(conc_initial = 7.4 ,mass_halo=2e12) 

# Calulating acceleration.
acc_h = mw.acceleration(xaxi *1000 , components='h')  
acc_b = mw.acceleration(xaxi *1000, components='b')
acc_s = mw.acceleration(xaxi *1000, components='s') 

# Calulating velocities.
v_h = np.sqrt(np.linalg.norm(acc_h,axis=1)*radii*1000)
v_b = np.sqrt(np.linalg.norm(acc_b,axis=1)*radii*1000)
v_s = np.sqrt(np.linalg.norm(acc_s,axis=1)*radii*1000)

acc_total = acc_h + acc_b + acc_s

acc_magn = np.linalg.norm(acc_total, axis=1)
acc_mag = np.abs(acc_total)

total_velocity = np.sqrt(acc_magn * radii *1000) 

# Plotting velocity as a function of radius.

plt.xlabel('r [kpc]')
plt.ylabel('$V_\mathrm{rot}\ (\mathrm{km}\ \mathrm{s}^{-1})$')
plt.plot(radii, total_velocity, color = 'purple',label = 'total stuck')

plt.ylim (0, 350)
plt.xlim (0, 40)

plt.plot(distance, total, label = 'total obsv', color = 'orange')

plt.legend (loc = 'upper right',  fontsize = 9)

ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()


