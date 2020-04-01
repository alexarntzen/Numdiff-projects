import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from DiseaseModel import DiseaseModel
import Schemes as schemes

"""
Using the 2D case. Solve the model 
The fist two figures are then made 
"""

# Parameters
N = 100
speed = 50
mu_S = speed*1e-3                                      # rate of spread of susceptible [distance^2/time]
mu_I = speed*1e-3                                      # rate of spread of infected [distance^2/time]

getBetaHighR = lambda x, y: speed*2         # rate of susceptible getting sick [time^-1]
getBetaLowR = lambda x, y: speed*0.5         # rate of susceptible getting sick [time^-1]

getGamma = lambda x, y: speed*1                       # rate of people recovering/dying [time^-1]
T = 1                                                   # Time
k = 0.001                                               # Timesteps
dim = 2                                                 # Dimensions of simulation


# Starting conditions for number of infected
def getI_0(x, y):
    return np.where(x + y == 0 , 0.01, 0)

# Starting conditions for number of suseptible
def getS_0(x, y):
    return 1 + 0*x + 0*y


# Inserting into DiseaseModel object that solves the problem. Note neumann conditions are
model_high_R = DiseaseModel(getS_0, getI_0, muS=mu_S, muI=mu_I,
                        schemeS=schemes.makeLaplacian2D(),
                        getBeta=getBetaHighR, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                        isNeumannFunc=schemes.isSqareBorderNeumann,
                        schemeNeumannFunc=schemes.makeSqareBorderNeumann2D())
model_low_R = DiseaseModel(getS_0, getI_0, muS=mu_S, muI=mu_I,
                        schemeS=schemes.makeLaplacian2D(),
                        getBeta=getBetaLowR, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                        isNeumannFunc=schemes.isSqareBorderNeumann,
                        schemeNeumannFunc=schemes.makeSqareBorderNeumann2D())


# Plotting functions
def plotImagesTimes(times,model, group="I"):
    numAxis = len(times)
    fig, axis = plt.subplots(1, numAxis, figsize=(23, 4))
    imageArtist  = None
    for i in range(numAxis):
        timeIndex = times[i] // k
        imageArtist = model.plotImage(timeIndex=timeIndex, ax=axis[i], group=group,  cmap=cm.Greys)
    fig.colorbar(imageArtist, ax=axis)
    return fig

def printEndResult(model):
    S, I, times = model.getSolution()
    total = np.sum(S[0] + I[0])

    print(f"Final suseptible: {np.sum(S[-1])/total}")
    print(f"Final infected: {np.sum(I[-1])/total}")
    print(f"Final removed: {1- np.sum(S[-1]+I[-1])/total}")

plotImagesTimes([0.2,0.3,0.4,0.5],model_high_R, group= "I")
plt.savefig('I_images_high_R.pdf')

plotImagesTimes([0.2,0.3,0.4,0.5],model_low_R, group= "I")
plt.savefig('I_images_low_R.pdf')


print("R=2:")
printEndResult(model_high_R)
print("R=2:")
printEndResult(model_low_R)
