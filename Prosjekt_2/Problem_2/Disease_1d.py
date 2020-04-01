import numpy as np
import matplotlib.pyplot as plt


from DiseaseModel import DiseaseModel
import Schemes as schemes

"""
Using the 1D case. Solve the model 
2D is implemeted in Disease_2d.py 
"""
def hole(x):
    if x > 0.3 and x<0.6:
        return 0
    else:
        return 1
# Parameters
freq = 1
N = 100
mu_S = freq * 0.5                          # rate of spread of susieptible
mu_I = freq * 0.5                         # rate of spread of infected
getBeta = lambda x: freq * 5  # rate of suseptible getting sick
getGamma = lambda x:  freq * 1       # rate of people recovering/dying
T = 10
k = 0.001
dim = 1


# getBeta = np.vectorize(getBeta)

# v from problem description
def getI_0(x):
    if x <= 0.1 :
        return (1-100*x**2)/100
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

model_1D.plotTotalInTime()

