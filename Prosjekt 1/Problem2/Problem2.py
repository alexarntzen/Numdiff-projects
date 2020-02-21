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


dim = 2
N = 30

tol = 10e-5
maxiter = 100
lam=2

#Not used
def f(x):
    return 0

#Boundary contitions
def g(x):
    return 1

def scheme(position,h):
    laplacian = np.array([[0,  1,  0],
                          [1, -4,  1],
                          [0,  1,  0]])

    # x
    # |
    #  y - >
    return laplacian/h**2


membrane = BVP.nonlinear_poisson(f, g, N, maxIterNewton = maxiter, lam = lam, dim = dim, length=1 )
membrane.plot("Microelectromechanical device")
membrane.summary()

