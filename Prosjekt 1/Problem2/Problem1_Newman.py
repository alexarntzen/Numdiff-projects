import numpy as np
import scipy.linalg as lin
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.pyplot as plt
import scipy.linalg as lin
import BVP_solver as BVP

def isBoundary(position):
    if 0 in position or 1 in position:
        return True
    else:
        return False


def schemeMaker(postion,h,constant):

    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                            [0, 1, 0]])

    diff  = np.array([[0, postion[1], 0],
                    [-postion[0], 0,postion[0] ],
                    [0, -postion[1], 0]])
    # ^
    # |
    # y x - >
    return laplacian*constant/h**2 + 1/(2*h)*diff



dim = 2
N = 20
mu = 1
def f(x):
    return 1

#Boundary contitions
def g(x):
    return 0

scheme1 = lambda postion,h: schemeMaker(postion,h,mu)


test = BVP.linear_elliptic(f, g, N,scheme = scheme1)
test.plot("Uten neuman")
