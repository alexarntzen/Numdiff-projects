import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from FiniteDifference import Shape
from DiseaseModel import ModifiedCrankNicolson
import Schemes as schemes
from cycler import cycler
default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))
plt.rc('axes', prop_cycle=default_cycler)

"""
This file makes the figure for the numerical experiments confirming convergence.
"""
# Parameters
T = 10 # Time
dim = 1  # Dimensions of simulation
a = 0.5
mu = 1
kListLarge = np.array([5,2,1])
kListSmall = np.array([0.1, 0.01, 0.001])
NList = np.array([25 ,50, 100])

def U_exact(x,t):
    return np.exp(a * t) * (x)


# Starting conditions for number of infected
def getU_0(x):
    return np.exp(0 * a) * (x)


def fDeriv(u):
    return a * u

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,4))


for k in kListSmall:
    diffList = []
    for N in NList:
        shapeObject = Shape(dim=dim, N=N)
        # Inserting into MCN object that solves the problem. Note neumann conditions are
        UList, times = ModifiedCrankNicolson.modifiedCrankNicolson(getU0=getU_0, fDeriv=fDeriv,
                                                                   scheme=schemes.makeLaplacian1D(1),
                                                                   shape=shapeObject, T=T, k=k, g=U_exact)
        t, x = np.ogrid[0:T+k:k,0:1+1/N-0.0001:1/N]
        Uex = U_exact(x,t)
        diffList.append(np.nanmax(abs(UList-Uex)))
    ax1.plot(NList,diffList,label=f"$k={k}$")

ax1.set_xlabel("$N$")
ax1.set_ylabel("Error")
ax1.legend()

for k in kListLarge:
    diffList = []
    for N in NList:
        shapeObject = Shape(dim=dim, N=N)
        # Inserting into MCN object that solves the problem. Note neumann conditions are
        UList, times = ModifiedCrankNicolson.modifiedCrankNicolson(getU0=getU_0, fDeriv=fDeriv,
                                                                   scheme=schemes.makeLaplacian1D(1),
                                                                   shape=shapeObject, T=T, k=k, g=U_exact)
        t, x = np.ogrid[0:T+k:k,0:1+1/N-0.0001:1/N]
        Uex = U_exact(x,t)
        diffList.append(np.nanmax(abs(UList-Uex)))
    ax2.plot(NList,diffList,label=f"$k={k}$")


ax2.set_xlabel("$N$")
ax2.set_ylabel("Error")
ax2.legend()

plt.savefig('testing_r.pdf')
