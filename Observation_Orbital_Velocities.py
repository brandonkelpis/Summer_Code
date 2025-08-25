import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.integrate import nquad
plt.style.use (['science', 'notebook', 'grid'])
from numba import njit 


# Defining constants and radii.
G = 4.30093e-6
distance = np.geomspace(0.1,213,213)

# @njit for faster computations.
# Using Newtons gravitational acceleration integral.

@njit
def bulge_b(r,phi,theta,distance):  # Need fourth variable data 'distance' for evaluating the observer at varying distances from the origin.
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

# Creating arrays to evaluate each acceleration.

result_bulge_b = np.zeros_like(distance)
error_bulge_b = np.zeros_like(distance)

# Looping over varying distances from origin.

for i,d in enumerate(distance):
  inner_lower_limit = 0
  inner_upper_limit = 213

  middle_lower_limit = 0
  middle_upper_limit = np.pi

  outer_lower_limit = 0
  outer_upper_limit = 2*np.pi

  result_bulge_b[i], error_bulge_b[i] = nquad(bulge_b, [(inner_lower_limit,inner_upper_limit),(middle_lower_limit,middle_upper_limit),(outer_lower_limit,outer_upper_limit)], args=(d,), opts = ({'points':[d]},{'points' : [np.pi/2]},{'points' : [np.pi/2]})) # Integral blows up at these points, must avoid.

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

  result_stellar_disk_b[i], error_stellar_disk_b[i] = nquad(stellar_disk_b, [(inner_lower_limit,inner_upper_limit),(middle_lower_limit,middle_upper_limit),(outer_lower_limit,outer_upper_limit)], args=(d,), opts = ({'points':[d]},{'points' : [np.pi/2]},{'points' : [np.pi/2]})) # Integral blows up at these points, must avoid.

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

  result_NFW_b[i], error_NFW_b[i] = nquad(NFW_b, [(inner_lower_limit,inner_upper_limit),(middle_lower_limit,middle_upper_limit),(outer_lower_limit,outer_upper_limit)], args=(d,), opts = ({'points':[d]},{'points' : [np.pi/2]},{'points' : [np.pi/2]})) # Integral blows up at these points, must avoid.

# Calculating velocities from acceleration integrals

velocity_bulge = np.sqrt(np.abs((result_bulge_b)*distance))

velocity_stellar_disk =np.sqrt(np.abs(result_stellar_disk_b*distance))

acceleration_total = np.abs((result_bulge_b+result_stellar_disk_b+result_NFW_b))

acceleration_baryons = np.abs(result_bulge_b+result_stellar_disk_b)

velocity_NFW = np.sqrt(np.abs(result_NFW_b*distance))

total = np.sqrt(acceleration_total*distance)



plt.xlabel ('r [kpc]')
plt.ylabel ('$V_\mathrm{rot}\ (\mathrm{km}\ \mathrm{s}^{-1})$')


plt.xlim(0,40) 
plt.ylim (0, 350)

plt.plot(distance,velocity_stellar_disk, label = 'stellar disk', color = 'blue')

plt.plot(distance,velocity_bulge, label = 'bulge', color = 'purple')

plt.plot(distance, np.sqrt(acceleration_baryons*distance), label = 'all baryons', color = 'green')

plt.plot(distance, total, label = 'total', color = 'orange')

plt.plot (distance, velocity_NFW,label = 'NFW', color = 'black')

plt.legend (loc = 'upper right',  fontsize = 9)

ax = plt.gca()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


#Saving data from tables to help adjust Stucker's halo to match Tamms. Avoiding long computations.

data = np.column_stack([distance, total])
np.savetxt('Distance_Total_Data.txt',data, delimiter='\t', header='Distance, Total', comments='', fmt = '%.2f' )




plt.show()


