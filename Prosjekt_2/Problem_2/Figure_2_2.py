import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from DiseaseModel import DiseaseModel
import Schemes as schemes

"""
Using the 2D case. Solve the model 
The last two figures are then made. 
"""


def islands(x,y):
    if np.cos(10*x)**2 + np.cos(10*y)**2 > 1.5:
        return 1
    else:
        return 0


def moat(x,y):
    if 0.4 <= x <= 0.6 or y< 0.1 or y > 0.9:
        return 0
    else:
        return 1

# Parameters
N = 100
speed = 50
mu_S = speed*1e-3                                                        # rate of spread of susceptible [distance^2/time]
mu_I = speed*1e-3                                                        # rate of spread of infected [distance^2/time]
getBetaMoat= np.vectorize( lambda x, y: speed*2*moat(x,y))               # rate of susceptible getting sick [time^-1]
getBetaIslands = np.vectorize( lambda x, y: speed*2*islands(x,y))        # rate of susceptible getting sick [time^-1]

getGamma = lambda x, y: speed*1                                         # rate of people recovering/dying [time^-1]
T = 1                                                                   # Time
k = 0.001                                                               # Timestep
dim = 2                                                                 # Dimensions of simulation


# Starting conditions for number of infected
def getI_0_side(x, y):
    return np.where( (y-0.5)**2 + x**2 ==0 , 0.1, 0)

def getI_0_corner(x, y):
    return np.where(x + y == 0 , 0.1, 0)

# Starting conditions for number of suseptible
def getS_0(x, y):
    return 0*x+0*y+1


# Inserting into DiseaseModel object that solves the problem. Note neumann conditions are
moat = DiseaseModel(getS_0, getI_0_side, muS=mu_S, muI=mu_I,
                        schemeS=schemes.makeLaplacian2D(),
                        getBeta=getBetaMoat, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                        isNeumannFunc=schemes.isSqareBorderNeumann,
                        schemeNeumannFunc=schemes.makeSqareBorderNeumann2D())

islands = DiseaseModel(getS_0, getI_0_corner, muS=mu_S, muI=mu_I,
                        schemeS=schemes.makeLaplacian2D(),
                        getBeta=getBetaIslands, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                        isNeumannFunc=schemes.isSqareBorderNeumann,
                        schemeNeumannFunc=schemes.makeSqareBorderNeumann2D())





# Plotting functions
def plotImagesTimes(times,model, group= "I"):
    numAxis = len(times)
    fig, axis = plt.subplots(1, numAxis, figsize=(23, 4))
    imageArtist  = None
    for i in range(numAxis):
        timeIndex = times[i] // k
        imageArtist = model.plotImage(timeIndex=timeIndex, ax=axis[i], group=group,  cmap=cm.Greys, norm = colors.LogNorm(vmin=1e-4,vmax=1))
    fig.colorbar(imageArtist, ax=axis)
    return fig

def printEndResult(model):
    S, I, times = model.getSolution()
    total = np.sum(S[0] + I[0])

    print(f"Final suseptible: {np.sum(S[-1])/total}")
    print(f"Final infected: {np.sum(I[-1])/total}")
    print(f"Final removed: {1 - np.sum(S[-1]+ + I[-1])/total}")

plotImagesTimes([0.2,0.3,0.4,0.5],moat, group= "I")
plt.savefig('I_images_moat.pdf')

plotImagesTimes([0.2,0.5,0.7,0.9],islands, group= "I")
plt.savefig('I_images_islands.pdf')

print("moat:")
printEndResult(moat)

print("slands:")
printEndResult(islands)


