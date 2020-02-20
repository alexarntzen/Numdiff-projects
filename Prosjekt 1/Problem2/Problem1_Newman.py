import numpy as np
import scipy.linalg as lin
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.pyplot as plt
import scipy.linalg as lin
import BVP_solver2 as BVP

def isBoundary(position):
    if 0 in position or 1 in position:
        return True
    else:
        return False



def schemeMaker(position,h,constant,V):

    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                            [0, 1, 0]])

    grad  = np.array([[0, -V(position)[0], 0],
                    [-V(position)[1], 0,V(position)[1] ],
                    [0, V(position)[0], 0]])

    # x
    # |
    #  y - >
    return -laplacian*constant/h**2 + 1/(2*h)*grad

def schemeMakerLonStep(position,h,constant,V):

    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                            [0, 1, 0]])

    grad  = np.array([[0,-V(position)[0] , 0],
                    [0, -V(position)[1] + V(position)[0] ,V(position)[1] ],
                    [0,0 , 0]])

    # x
    # |
    #  y - >
    return -laplacian*constant/h**2 + 1/h*grad


def schemeMakerNeumann2(position,h,constant,V):

    laplacian = np.array([[0, 1, 0],
                          [0, -2, 0],
                           [0, 1, 0]])

    grad  = np.array([[0, -V(position)[0], 0],
                    [0, 0, 0 ],
                    [0, V(position)[0], 0]])

    deriv = np.array([[0, 3/2, 0],
                    [0, -2, 0 ],
                    [0, 1/2 , 0]])
    # x
    # |
    # v
    #  y - >
    print( 1/(2*h)*grad)
    right = (-2*constant/(h) + V(position)[1])
    return (-laplacian*constant/h**2 + 1/(2*h)*grad ), 1

#Case3 works better
def schemeMakerNeumann(position,h,constant,V):

    deriv = np.array([[0, 0, 0,0],
                    [0, 3/2, -2,1/2 ]])
    # x
    # |
    # v
    #  y - >
    #right, leftG, leftF
    return deriv/h, 1, 0


def isNeumann(position):
    if position[1] == 0:
        return True

dim = 2
N = 5
mu = 1e-3
def f(x):
    return 1

#Boundary contitions
def g(x):
    return 0

def V(x):
    return [x[1],-x[0]]

scheme1 = lambda postion,h: schemeMaker(postion,h,mu,V)
scheme1LongStep = lambda postion,h: schemeMakerLonStep(postion,h,mu,V)

schemeNeumann1 = lambda postion,h: schemeMakerNeumann(postion,h,mu,V)

test1 = BVP.linear_elliptic(f, g, N,scheme = scheme1)
#test1.plot("Oppgave d: Vanling")



test1 = BVP.linear_elliptic(f, g, N,scheme = scheme1LongStep)
test1.plot("Oppgave d: Long step ")

test2 = BVP.linear_elliptic(f, g, N,scheme = scheme1, isNeumannFunc = isNeumann, schemeNeumannFunc = schemeNeumann1 )
test2.plot("Oppgave d: Med neumann")

