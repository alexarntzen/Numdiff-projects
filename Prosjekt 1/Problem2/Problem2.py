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
N = 20
tol = 10e-5
maxiter = 1000
lam=1.5
#Not used
def f(x):
    return 1

#Boundary contitions
def g(x):
    return 2

membrane = BVP.nonlinear_poisson(f, g, N, maxIterNewton = maxiter, lam = lam, dim = dim )
membrane.plot("Microelectromechanical device")
membrane.summary()