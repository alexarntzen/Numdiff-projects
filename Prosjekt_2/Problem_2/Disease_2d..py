import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from DiseaseModel import DiseaseModel
import Schemes as schemes

"""
Using the 2D case. Solve the model 
1D is implemeted in Disease_1d.py 
"""

# Parameters
N = 50
mu_S = 3 # rate of spread of susieptible [distance^2/time]
mu_I = 3  # rate of spread of infected [distance^2/time]
getBeta = lambda x, y: 10*(np.abs(np.sin(15*x) + np.sin(15*y)))  # rate of suseptible getting sick [time^-1]
getGamma = lambda x, y: 1  # rate of people recovering/dying [time^-1]
T = 1
k = 0.001
dim = 2


# Starting conditions for number of infected
def getI_0(x, y):
    if x +y < 0.1:
        return 0.1
    else:
        return 0
# Starting conditions for number of suseptible
def getS_0(x, y):
    return 1 - getI_0(x, y)


# Inserting into DiseaseModel object that solves the problem. Note neumann conditions are
model_2D = DiseaseModel(np.vectorize(getS_0), np.vectorize(getI_0), muS=mu_S, muI=mu_I,
                        schemeS=schemes.makeLaplacian2D(),
                        getBeta=getBeta, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                        isNeumannFunc=schemes.isSqareBorderNeumann,
                        schemeNeumannFunc=schemes.makeSqareBorderNeumann2D())

#model_2D.plotTotalInTime()



# Plotting functions
def displayAnimation(animationLength=10):
    fig, (axS, axI) = plt.subplots(1, 2)

    anim = model_2D.applyDiseaseAnimation(axS, axI, animationLength=animationLength, norm=colors.LogNorm(vmin=10e-5,vmax=1), cmap=cm.Greys)
    plt.show()


def plot3D(time):
    fig = plt.figure(figsize=(4*2, 6), dpi=100)
    axS = fig.add_subplot(1, 2, 1, projection='3d')
    axI = fig.add_subplot(1, 2, 2, projection='3d')
    timeIndex = time // k
    model_2D.plot(timeIndex=timeIndex, ax=axS, group="S", label="Susceptible", ulabel="Density of people")
    model_2D.plot(timeIndex=timeIndex, ax=axI, group="I", label="Infected", ulabel="Density of people")

    plt.show()

def plotImages(time):
    fig, (axS, axI) = plt.subplots(1, 2)
    timeIndex = time // k
    model_2D.plotImage(timeIndex=timeIndex, ax=axS, group="S", cmap=cm.Greys)
    artist = model_2D.plotImage(timeIndex=timeIndex, ax=axI, group="I", cmap=cm.Greys)
    axI.figure.colorbar(artist, ax=[axS, axI])
    plt.show()

def plot3DTimes(times, group= "I"):
    numAxis = len(times)
    fig = plt.figure(figsize=(6*numAxis, 6), dpi=100)
    for i in range(numAxis):
        ax = fig.add_subplot(1, numAxis, i + 1 , projection='3d')
        timeIndex = times[i] // k
        model_2D.plot(timeIndex=timeIndex, ax=ax, group=group, ulabel="Density of people")
    plt.show()

def plotImagesTimes(times, group= "I"):
    numAxis = len(times)
    fig, axis = plt.subplots(1, numAxis)
    imageArtist  = None
    for i in range(numAxis):
        timeIndex = times[i] // k
        imageArtist = model_2D.plotImage(timeIndex=timeIndex, ax=axis[i], group=group,  cmap=cm.Greys)
    fig.colorbar(imageArtist, ax=axis)
    plt.show()

def printEndResult():
    S, I, times = model_2D.getSolution()
    total = np.sum(S[0] + I[0])

    print(f"Final suseptible: {np.sum(S[-1])/total}")
    print(f"Final infected: {np.sum(S[-1])/total}")
    print(f"Final removed: {1 - np.sum(S[-1]+ + I[-1])/total}")