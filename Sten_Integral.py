import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps,quad
from scipy.interpolate import interp1d

def N_GCE (E): 
  constant = (8.6*10**14) / 1000
  parentheses = E
  exponent = 0.27-(0.27*np.log(E))
  return (((1/(4*np.pi))*(constant*(parentheses**exponent)))* (1/(E**2)))* 1.05052e-19

lower_N_GCE_limit = 1
upper_N_GCE_limit = 100

result_N_GCE, error_N_GCE = quad(N_GCE, lower_N_GCE_limit,upper_N_GCE_limit)

djdvr = np.loadtxt('E_C_Cusp_Total_Data.txt', delimiter = '\t', skiprows=1)

annihilation_data = djdvr[:, 1]
radial_data = djdvr[:,0]

d = 765 #kpc
theta_range = np.linspace(0,0.244346095,80) #radians
theta_axis = np.linspace(0,14,80) #stairadians

def L_function(theta): 
  L_of_theta = np.sqrt((radial_data**2)-(d*np.sin(theta))**2)
  return L_of_theta

def djdvl_function(theta): # This is returning scalar values. I need the list of the scalar values for each theta input.
  L_of_djdvl= interp1d(L_function(theta), annihilation_data, kind = 'linear', fill_value = 0, bounds_error=False)
  return L_of_djdvl

result_function = np.zeros(80)
error_function = np.zeros(80)

for i,theta in enumerate(theta_range):

  lower_limit = 0
  upper_limit = 87.35

  result_function[i],error_function[i] = quad(djdvl_function(theta),lower_limit,upper_limit)

plt.yscale('log')
plt.xscale('log')
plt.plot(theta_range*180/np.pi, result_function*result_N_GCE*2)
plt.show()

