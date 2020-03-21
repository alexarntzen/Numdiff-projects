import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib import cm


from DiseaseModel import DiseaseModel
import Schemes as default

"""
Using the 2D case. Solve the model 
1D is not implemented yet 
"""

# Parameters
N = 25
mu_S = 0.5                          # rate of spread of susieptible
mu_I = 2.5                          # rate of spread of infected
getBeta = lambda x, y:  2        # rate of suseptible getting sick
getGamma = lambda x, y:  2       # rate of people recovering/dying
T = 1
k = 0.01
dim = 2


# Newmann contitions
def g(x):
    return 0


# v from problem description
def getI_0(x, y):
    if x< 0.2 :
        return 1
    else:
        return 0


def getS_0(x, y):
    return 1 - getI_0(x,y)


test = DiseaseModel(g, np.vectorize(getS_0), np.vectorize(getI_0), muS=mu_S, muI=mu_I, schemeS=default.makeLaplacian(1),
                    getBeta=getBeta, getGamma=getGamma, T=T, k=k, N=N, dim=dim,
                    isNeumannFunc=default.isSqareBorderNeumann, schemeNeumannFunc=default.sqareBorderNeumann)

fig, (axS, axI) = plt.subplots(1, 2)

artistList = test.applyDiseaseAnimation(axS, axI, cmap=cm.Greys)

# fig.colorbar(artistList[0][0], ax=axI)
def I(frame):
    test.plotI(frame,show=True,title=f"t={frame}")