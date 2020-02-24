import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import BVP_solver as BVP

"""
Running this file will store the relevant plots (ca 3) from problem 1d) on your computer
"""

# The BVP objects accepts functions that return arrays that determine the descritization at that point
def schemeMaker(position,BVPobject,constant,V):
    # The standard scheme for problem 1d. It takes inn position and outputs correct arrays for that point.
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

    grad = np.array([
        [              0, -V(position)[0],               0],
        [-V(position)[1],               0, V(position)[1] ],
        [              0,  V(position)[0],               0]
    ])
    return -laplacian*constant/h**2 + 1/(2*h)*grad

def schemeMakerLonStep(position,BVPobject,constant,V):
    # First order schmeme for longer stepsize on this problem.
    # Forward bacwards difference as gradient

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

    grad = np.array([
        [0,                 -V(position)[0] ,              0],
        [0, -V(position)[1] + V(position)[0], V(position)[1]],
        [0,                                0,              0]
    ])
    return -laplacian*constant/h**2 + 1/h*grad

def schemeMakerNeumann(position,BVPobject,constant,V):
    # 3 point derivative. O(h^2)

    h = BVPobject.h
    #   y -- >
    # x
    # |
    # V
    deriv = np.array([
        [0,   0,  0,  0],
        [0, 3/2, -2,1/2],
        [0,   0,  0,  0]
    ])
    # returning coefficients
    # left, rightG, rightF
    return deriv/h, 1, 0

def isNeumann(position):
    if position[1] == 0:
        return True

# Parameters
N = 10
mu = 1e-2

# Conditions on domain
def f(x):
    return 1

# Boundary contitions
def g(x):
    return 0

# v from problem description
def V(x):
    return [x[1],-x[0]]

# Make the functions for BVP solver
scheme1 = lambda postion,BVP: schemeMaker(postion,BVP,mu,V)
scheme1LongStep = lambda postion,BVP: schemeMakerLonStep(postion,BVP,mu,V)
schemeNeumann1 = lambda postion,BVP: schemeMakerNeumann(postion,BVP,mu,V)

# Solving cases with BVP solve and plotting
nList  = [10,30,50]
fig = plt.figure(figsize=(18, 5))
for i in range(len(nList)):
    ax = fig.add_subplot(1, len(nList), i+1, projection='3d')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))

    Oppg1dNormal= BVP.linear_elliptic(f, g, nList[i],scheme = scheme1)
    Oppg1dNormal.plot(f"$N = {nList[i]}$",ax,view= (35,-20))
    plt.savefig("Oppgave_d_Normal.pdf")

fig = plt.figure(figsize=(18, 5))
for i in range(len(nList)):
    ax = fig.add_subplot(1, len(nList), i+1, projection='3d')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))

    oppg1dNaumann = BVP.linear_elliptic(f, g, N, scheme=scheme1, isNeumannFunc=isNeumann, schemeNeumannFunc=schemeNeumann1)
    oppg1dNaumann.plot(f"$N = {nList[i]}$", ax,view= (35,-20))
    plt.savefig("Oppgave_d_Naumann.pdf",)

fig = plt.figure(figsize=(18, 5))
for i in range(len(nList)):
    ax = fig.add_subplot(1, len(nList), i+1, projection='3d')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))

    oppg1dLongStep = BVP.linear_elliptic(f, g, N, scheme=scheme1LongStep)
    oppg1dLongStep.plot(f"$N = {nList[i]}$",ax,view= (35,-20))
    plt.savefig("Oppgave_d_LongStep.pdf",)
