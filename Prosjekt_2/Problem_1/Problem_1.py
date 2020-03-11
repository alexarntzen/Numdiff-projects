import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append('../Problem_2/')

import BVP_solver as BVP


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


#Only first order so far
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
T = 2
k = 0.1
dim = 2

def fDeriv(U):
    return U

# Newmann contitions
def g(x):
    return 0

# v from problem description
def getU_0(x,y):
    if x + y <= 0.3:
        return 0.5
    else:
        return 0

#def getU_0(x,y):return x*y

test = BVP.ModifiedCrankNicolson(g ,np.vectorize(getU_0),fDeriv = fDeriv,
                                 scheme = schemeMaker(1), T=T, k=k, N=N, dim=dim,
                                 isNeumannFunc=isNeumann, schemeNeumannFunc=schemeNeumann)

UList, times = test.solveBase()

#Do something with UList and times here

