import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from DiseaseModel import DiseaseModel
import Schemes as default

"""
Using the 2D case. Solve the model 
1D is not implemented yet 
"""

# Parameters
N = 40
mu_S = 0.5  # rate of spread of susieptible
mu_I = 2.5  # rate of spread of infected
getBeta = lambda x, y: 2  # rate of suseptible getting sick
getGamma = lambda x, y: 0.1  # rate of people recovering/dying
T = 1
k = 0.01
dim = 2


# v from problem description
def getI_0(x, y):
    if x + y < 0.2 :
        return 1
    else:
        return 0


def getS_0(x, y):
    return 1 - getI_0(x, y)


model_2D = DiseaseModel(np.vectorize(getS_0), np.vectorize(getI_0), muS=mu_S, muI=mu_I,
                        schemeS=default.makeLaplacian2D(),
                        getBeta=getBeta, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                        isNeumannFunc=default.isSqareBorderNeumann,
                        schemeNeumannFunc=default.makeSqareBorderNeumann2D())

#model_2D.plotTotalInTime()


def displayanimation(speed=10):
    fig, (axS, axI) = plt.subplots(1, 2)

    anim = model_2D.applyDiseaseAnimation(axS, axI, speed=speed, cmap=cm.Greys)
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
