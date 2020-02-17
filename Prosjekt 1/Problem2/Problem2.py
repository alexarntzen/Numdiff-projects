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



class BVP_finite_difference:
    @staticmethod
    def getLinearizedBVP(scheme, g, numInternal, getInternalIndex, getCoordinate, getVecor, isBoundary , schemeCenter=1, neumann=False):
        A = sparse.lil_matrix((numInternal, numInternal), dtype=np.float64)
        boundaryF = np.zeros(numInternal)
        for internalIndex in range(numInternal):
            coordinate = getCoordinate(internalIndex)
            if not isBoundary(coordinate) or  neumann:
                #Scheme must return array  in same coordinate order as coordinates
                for arrayCoordinate, coeff in np.ndenumerate(scheme(getVecor(np.copy(coordinate)))):
                    shemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    boundary = isBoundary(shemeCoordinate)
                    if boundary and not neumann:
                        boundaryF[internalIndex] += - coeff *g(getVecor(np.copy(coordinate)))
                    elif not boundary or neumann:
                        A[internalIndex, getInternalIndex(shemeCoordinate)] = coeff
        if(neumann):
            return sparse.csr_matrix(A)

        else:
            return sparse.csr_matrix(A), boundaryF

    @staticmethod
    def print_summary(error, iter):
        print("Local error: ", error)
        print("The method used", iter, "iterations" )


# Denne kan også brukes i oppgave 1a
class Lattice():
    def __init__(self, shape = (4,4), size=(1,1), origin=0):
        self.shape = np.array(shape)
        self.size = np.array(size)
        self.box = self.size / self.shape
        self.N = np.max(self.shape)
        self.Ni = self.N - 1
        self.length = np.max(size)
        self.h = self.length/self.N
        self.origin = origin
        self.basis = self.getBasis(shape)
        self.basisInternal = self.getBasis(shape - 1)
        self.numInternal = np.prod(self.shape - 1)
        self.dim = len(self.shape)

    @staticmethod
    def getBasis(shape):
        return np.cumprod(np.append(1,shape[:-1]))

    @staticmethod
    def getIndexLattice(coordinates, basis):
        return np.dot(coordinates, basis)

    def getIndexInternal(self, coordinates):
        return self.getIndexLattice(coordinates - 1, self.basisInternal )

    @staticmethod
    def getCoordinateLattice(index,basis):
        return - np.diff(index - (index % basis), append=0)//basis

    def getCoordinateInternal(self,internalIndex):
        return self.getCoordinateLattice(internalIndex, self.basisInternal ) + 1

    @staticmethod
    def getPositionLattice(coordinate, box, origin):
        return origin + coordinate * box

    def getPosition(self, coordinate):
        return self.getPositionLattice(coordinate, self.box,self.origin)

    @staticmethod
    def getLattice(vector, shape):
        return np.reshape(vector,shape + 1, order="F")

    def getLatticeInternal(self, internalVector):
        return self.getLattice(internalVector,self.shape-2)

    def plot2D(self, X, Y, U, title=""):
        Ugird = self.getLattice(U,self.shape)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Ugird, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
        # Set initial view angle

        # Set labels and show figure
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$u(x,y)$')
        ax.set_title(title)
        plt.show()


#Denne kan også brukes i oppgave 1a, cartesian
class BVP_n_qube(Lattice):
    def __init__(self,N,dim=2,length=1, origin = 0,constantBoundary=0):
        Lattice.__init__(self,np.full(dim,N),np.full(dim,length),origin)
        self.constantBoundary = constantBoundary

    def isBoundaryGrid(self, coordinate):
        if (self.N in coordinate) or (0 in coordinate):
            return True
        else:
            return False

    def applyConstantBoundary(self,internalVector, boundary=1):
        return np.pad(self.getLatticeInternal(internalVector), (1, 1), 'constant', constant_values=boundary).ravel()


class nonlinear_poisson_sqare(BVP_n_qube,BVP_finite_difference):
    def __init__(self,f,g,N,dim=2,length=1, origin = 0, maxIterNewton = 1000,constantBoundary=1,guess = None):
        BVP_n_qube.__init__(self,N,dim,length, origin,constantBoundary )
        self.f = f
        self.g = g
        self.constantBoundary = constantBoundary
        #dirichlet
        self.A, self.Fb =  self.getLinearizedBVP(self.scheme,
                                     self.g,
                                     self.numInternal,
                                     self.getIndexInternal,
                                     self.getCoordinateInternal,
                                     self.getPosition,
                                     self.isBoundaryGrid,
                                     neumann=False)

        if guess == None:
            guess = np.ones(len(self.Fb))

        print("matrix_ferdig")
        self.U, self.error, self.iter = self.solveNonlinear1(A = self.A, u_n = guess, F_b = self.Fb,maxiter=maxIterNewton,constantBoundary= self.constantBoundary )
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
maxiter = 10

def ftest(X,u,ud):
    return 1/u**2
def gtest(v):
    return 1

test = nonlinear_poisson_sqare(ftest,gtest,N,dim,maxIterNewton=10)
test.plot()
test.summary()