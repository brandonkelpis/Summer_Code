# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/brandonkelpis/cusp-encounters/adiabatic-tides")
sys.path.append("/Users/brandonkelpis/cusp-encounters")

import adiabatic_tides as at
import Paper_Code.cusp_encounters.milkyway_not_used
import cusp_encounters.encounters_math as em
import cusp_encounters.cusp_distribution
from cusp_encounters.decorators import h5cache
import scipy.integrate
import cusp_encounters.gammaray as gammaray

# %load_ext autoreload
cachedir = "../caches"

G = 43.0071057317063e-10 # Mpc (km/s)^2 / Msol 

mw = cusp_encounters.milkyway_not_used.MilkyWay(adiabatic_contraction=True, cachedir=cachedir, mode="cautun_2020")
import Create_Orbits_Mpi
orbits = Create_Orbits_Mpi