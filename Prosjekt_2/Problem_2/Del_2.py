import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from DiseaseModel import DiseaseModel
import Schemes as schemes

"""
Using the 2D case. Solve the model 
1D is not implemented yet 
"""



# Parameters
N = 25
mu_S = 0.1
mu_I = 0.1
beta = 0.1
gamma = 3
T = 20
k = 0.01
dim = 2
I0 = 0.2
# Newmann contitions
def g(x):
    return 0

# v from problem description
def getI_0(x,y):
    if x + 0.1*y <= 0.2:
        return 1
    else:
        return 0

def getS_0(x,y):
    if x <= 0.2:
        return 0
    else:
        return 1

#def getU_0(x,y):return x*y

test = DiseaseModel(g,np.vectorize(getS_0),np.vectorize(getI_0),muS = mu_S, muI = mu_I, schemeS = schemeMaker(1), beta=beta, gamma=gamma, T=T, k=k, N=N,dim=dim,
                        isNeumannFunc=isNeumann, schemeNeumannFunc=schemeNeumann)



def S(t):
    test.plotS(t,show=True,title=f"t={t}")

def I(frame):
    test.plotI(frame,show=True,title=f"t={frame}")

def R(frame):
    test.plotR(frame*k,show=True,title=f"t={frame*k}")

I(0)
# # Solving cases with BVP solve and plotting
# nList  = [10,30,50]
# fig = plt.figure(figsize=(18, 5))
# for i in range(len(nList)):
#     ax = fig.add_subplot(1, len(nList), i+1, projection='3d')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
#     ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
#
#     Oppg1dNormal= BVP.linear_elliptic(f, g, nList[i],scheme = scheme1)
#     Oppg1dNormal.plot(f"$N = {nList[i]}$",ax,view= (35,-20))
#     plt.savefig("Oppgave_d_Normal.pdf")
#
# fig = plt.figure(figsize=(18, 5))
# for i in range(len(nList)):
#     ax = fig.add_subplot(1, len(nList), i+1, projection='3d')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
#     ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
#
#     oppg1dNaumann = BVP.linear_elliptic(f, g, N, scheme=scheme1, isNeumannFunc=isNeumann, schemeNeumannFunc=schemeNeumann1)
#     oppg1dNaumann.plot(f"$N = {nList[i]}$", ax,view= (35,-20))
#     plt.savefig("Oppgave_d_Naumann.pdf",)
#
# fig = plt.figure(figsize=(18, 5))
# for i in range(len(nList)):
#     ax = fig.add_subplot(1, len(nList), i+1, projection='3d')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
#     ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
#
#     oppg1dLongStep = BVP.linear_elliptic(f, g, N, scheme=scheme1LongStep)
#     oppg1dLongStep.plot(f"$N = {nList[i]}$",ax,view= (35,-20))
#     plt.savefig("Oppgave_d_LongStep.pdf",)
