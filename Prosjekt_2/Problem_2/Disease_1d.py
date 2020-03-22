import numpy as np
import matplotlib.pyplot as plt


from DiseaseModel import DiseaseModel
import Schemes as schemes

"""
Using the 1D case. Solve the model 
2D is implemeted in Disease_2d.py 
"""

# Parameters
N = 25
mu_S = 0.5                          # rate of spread of susieptible
mu_I = 2.5                          # rate of spread of infected
getBeta = lambda x:  2        # rate of suseptible getting sick
getGamma = lambda x:  2       # rate of people recovering/dying
T = 1
k = 0.01
dim = 1






# v from problem description
def getI_0(x):
    if x< 0.2 :
        return 1
    else:
        return 0


def getS_0(x):
    return 1 - getI_0(x)


model_1D = DiseaseModel(np.vectorize(getS_0), np.vectorize(getI_0), muS=mu_S, muI=mu_I, schemeS=schemes.makeLaplacian1D(1),
                    getBeta=getBeta, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                    isNeumannFunc=schemes.isSqareBorderNeumann, schemeNeumannFunc=schemes.makeSqareBorderNeumann1D(1), )


def plotTime2D(time):
    fig, (ax) = plt.subplots(1, 1)
    timeIndex = time//k
    model_1D.plot(timeIndex=timeIndex, ax=ax, label="Susceptible", group="S")
    model_1D.plot(timeIndex=timeIndex, ax=ax, label="Infected", group="I", ulabel="Densety of people")
    plt.legend()
    plt.show()

#model_1D.plotTotalInTime()

