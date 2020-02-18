import numpy as np
import scipy.linalg as lin
import scipy.sparse as sparse             # Sparse matrices
import scipy.sparse.linalg as splin             # Sparse matrices

import matplotlib.pyplot as plt
newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plot
from matplotlib import cm
import BVP_solver as BVP

class nonlinear_poisson_sqare(BVP.Cube,BVP.Finite_difference):
    def __init__(self,f,g,N,dim=2,length=1, origin = 0, maxIterNewton = 1000,constantBoundary=1, lam=1.5,guess = None):
        BVP.Cube.__init__(self,N,dim,length, origin,constantBoundary )
        self.f = f
        self.g = g
        self.constantBoundary = constantBoundary
        self.lam = lam
        #dirichlet
        self.A, self.Fb =  self.getLinearizedBVP(self.scheme,
                                     self.g,
                                     self.numInternal,
                                     self.getIndexInternal,
                                     self.getCoordinateInternal,
                                     self.getPosition,
                                     self.isBoundarySqare,
                                     neumann=False)

        if guess == None:
            guess = np.ones(len(self.Fb))

        print("matrix_ferdig")
        self.U, self.error, self.iter = self.solveNonlinear1(A = self.A, u_n = guess, F_b = self.Fb,maxiter=maxIterNewton,constantBoundary= self.constantBoundary,lam=self.lam )
        self.X, self.Y = self.getMeshGrid()

    def getError(self):
        return self.error

    def getIter(self):
        return self.iter

    def summary(self):
        self.print_summary(self.error, self.iter)

    def plot(self,title="Microelectromechanical device"):
        self.plot2D(self.X, self.Y, self.U, title)

    # Cartesian coordinates. Important !
    @staticmethod
    def scheme(position):
        return np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]))

    #Ikke generell
    def solveNonlinear1(self,A,u_n,F_b,lam = 1.5,tol=10e-10,maxiter=100,constantBoundary=1):
        du = np.copy(u_n)
        error = 1
        iter = 0
        F = A @ u_n - self.h**2*lam/u_n**2 - F_b

        while iter < maxiter and error > tol:
            jacobian = A + np.diag(lam/u_n)*self.h**2

            du = splin.lgmres(jacobian,-F,du)[0]

            u_n += du

            F = A @ u_n - self.h**2*lam/u_n**2 - F_b

            error = lin.norm(F,2)
            iter += 1

        u_n = self.applyConstantBoundary(u_n,constantBoundary)
        return u_n,error,iter

    def getMeshGrid(self):
        return self.getPosition(np.ogrid[0:(self.N+1), 0:(self.N+1)])



dim = 2
N = 10
tol = 10e-10
maxiter = 20
lam=25
#Not used
def ftest(X,u,ud):
    return 1/u**2

#Boundary contitions
def g(v):
    return 1

test = nonlinear_poisson_sqare(ftest,g,N,dim,lam = lam, maxIterNewton=maxiter)
test.plot()
test.summary()