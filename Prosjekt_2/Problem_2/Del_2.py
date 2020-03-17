import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from DiseaseModel import DiseaseModel

"""
Using the 2D case. Solve the model 
1D is not implemented yet 
"""

# The BVP objects accepts functions that return arrays that determine the descritization at that point
def schemeMaker(constant):
    def scheme(position,BVPobject):
        h = BVPobject.h

        #   y -- >
        # x
        # |
        # V
        laplacian = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0, 1,  0]
        ])
        # scheme, schemecenter
        return laplacian/h**2, 1
    return scheme



def schemeNeumann(position,BVPobject):
    # 3 point derivative. O(h^2)
    h = BVPobject.h
    #   y -- >
    # x
    # |
    # V
    derivS = np.array([
        [0,   0,  0],
        [0,   1,  -1],
        [0,   0,  0]
    ])
    if position[1]== 0:
        deriv = derivS
    elif position[0] == 0:
        deriv = np.rot90(derivS,-1)
    elif position[1]== 1:
        deriv = np.rot90(derivS,-2)
    elif position[0]== 1:
        deriv = np.rot90(derivS,-3)
    else:
        print("Neumann conditions på en kant som ikke er en kant?")
        deriv=0
    # returning coefficients
    # left, rightG, rightF, schemecenter
    return deriv, 1, 0, 1

def isNeumann(position):
        return True

# Parameters
N = 10
mu_S = 0.1
mu_I = 0.1
beta = 0.1
gamma = 0.1
T = 2
k = 0.1
dim = 2
I0 = 0.2
# Newmann contitions
def g(x):
    return 0

# v from problem description
def getI_0(x,y):
    if x + y <= 0.3:
        return 0.5
    else:
        return 0

def getS_0(x,y):
    return 0.5

#def getU_0(x,y):return x*y

test = DiseaseModel(g,np.vectorize(getS_0),np.vectorize(getI_0),muS = mu_S, muI = mu_I, schemeS = schemeMaker(1), beta=beta, gamma=gamma, T=T, k=k, N=N,dim=dim,
                        isNeumannFunc=isNeumann, schemeNeumannFunc=schemeNeumann)



def S(frame):
    test.plotS(frame*k,show=True,title=f"t={frame*k}")

def I(frame):
    test.plotI(frame*k,show=True,title=f"t={frame*k}")

def R(frame):
    test.plotR(frame*k,show=True,title=f"t={frame*k}")

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
