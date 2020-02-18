import numpy as np
import scipy.linalg as lin
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.pyplot as plt

newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm


class finite_difference:
    @staticmethod
    def getLinearizedDirichlet(scheme,f, g, maxIndex, getIndex, getCoordinate, getVecor, isBoundary, schemeCenter=1):
        A = sparse.lil_matrix((maxIndex, maxIndex), dtype=np.float64)
        boundaryF = np.zeros(maxIndex)
        interiorF = np.zeros(maxIndex)
        geometry = np.full((3,maxIndex),False,dtype=bool)
        for index in range(maxIndex):
            coordinate = getCoordinate(index)
            if not isBoundary(coordinate):
                geometry[0][index] = True
                # Scheme must return array  in same coordinate order as coordinates
                interiorF[index] = f(getVecor(coordinate))
                for arrayCoordinate, coeff in np.ndenumerate(scheme(getVecor(np.copy(coordinate)))):
                    #Could have used isomorphism property here
                    schemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    schemeIndex = getIndex(schemeCoordinate)
                    if isBoundary(schemeCoordinate):
                        boundaryF[index] += - coeff * g(getVecor(np.copy(coordinate)))
                        geometry[1][schemeIndex] = True
                    elif coeff != 0:
                        A[index, schemeIndex] = coeff
        np.logical_and(np.logical_not(geometry[0]),np.logical_not(geometry[1]),geometry[2])
        return sparse.csr_matrix(A[geometry[0]])[:,geometry[0]],interiorF[geometry[0]], boundaryF[geometry[0]], geometry

    @staticmethod
    def applyBoundary(U_interior, g, geometry,getCoordinate, getVecor):
        maxIndex = geometry.shape[1]
        U = np.zeros(maxIndex, dtype=np.float64)
        interiorIndexs = np.flatnonzero(geometry[0])
        bounderyIndexs = np.flatnonzero(geometry[1])
        exteriorIndexs = np.flatnonzero(geometry[2])
        U[interiorIndexs] = U_interior
        U[bounderyIndexs] = [g(getVecor(getCoordinate(index))) for index in bounderyIndexs]
        U[exteriorIndexs] = np.nan
        return U

# Denne kan også brukes i oppgave 1a
class lattice():
    def __init__(self, shape=(4, 4), size=(1, 1), origin=0):
        self.shape = np.array(shape)
        self.size = np.array(size)
        self.box = self.size / self.shape
        self.N = np.max(self.shape)
        self.length = np.max(size)
        self.h = self.length / self.N
        self.origin = origin
        self.basis = self.getBasis(shape +1 )
        self.maxIndex = np.prod(self.shape + 1)
        self.dim = self.shape.ndim

    @staticmethod
    def getBasis(shape):
        return np.cumprod(np.append(1, shape[:-1]))

    #isomorphism Z_N^dim --> Z_{N*dim}
    @staticmethod
    def getIndexLattice(coordinates, basis):
        return np.dot(coordinates, basis)

    def getIndex(self, coordinates):
        return self.getIndexLattice(coordinates , self.basis)

    #Inverse isomorphism
    @staticmethod
    def getCoordinateLattice(index, basis):
        return - np.diff(index - (index % basis), append=0) // basis

    def getCoordinate(self, internalIndex):
        return self.getCoordinateLattice(internalIndex, self.basis)

    @staticmethod
    def getPositionLattice(coordinate, box, origin):
        return origin + coordinate * box

    def getPosition(self, coordinate):
        return self.getPositionLattice(coordinate, self.box, self.origin)

    @staticmethod
    def getLattice(vector, shape):
        return np.reshape(vector, shape + 1, order="F")

    #Only for 2D
    def plot2D(self, X, Y, U, title=""):
        Ugrid = self.getLattice(U, self.shape)
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Ugrid, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
        # Set initial view angle

        # Set labels and show figure
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$u(x,y)$')
        ax.set_title(title)
        plt.show()

    #Only for 2D
    def getMeshGrid(self):
        return self.getPosition(np.ogrid[0:(self.N + 1), 0:(self.N + 1)])


class sqare(lattice):
    def __init__(self, N, dim=2, length=1, origin=0, constantBoundary=0):
        lattice.__init__(self, np.full(dim, N), np.full(dim, length), origin)
        self.constantBoundary = constantBoundary

    def getLatticeInternalSqare(self, internalVector):
        return self.getLattice(internalVector, self.shape - 2)

    def isBoundarySqare(self, coordinate):
        if (self.N in coordinate) or (0 in coordinate):
            return True
        else:
            return False

    def applyConstantBoundary(self, internalVector, boundary=1):
        return np.pad(self.getLatticeInternalSqare(internalVector), (1, 1), 'constant', constant_values=boundary).ravel()

# Denne kan også brukes i oppgave 1a, cartesian
class shape(lattice):
    def __init__(self, N, isBoundaryFunction, dim=2, length=1, origin=0):
        lattice.__init__(self, np.full(dim, N), np.full(dim, length), origin)
        if isBoundaryFunction is None:
            self.isBoundaryFunction = self.isBoundarySqare
        else:
            self.isBoundaryFunction = lambda coordinate : isBoundaryFunction(self.getPosition(coordinate))

    def isBoundarySqare(self, coordinate):
        if (self.N in coordinate) or (0 in coordinate):
            return True
        else:
            return False

# Cartesian coordinates. Important !
def defaultScheme(position):
    return np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]))

class solve_interface(shape,finite_difference):
    def __init__(self,f,g,N,isBoundaryFunction,scheme=None,dim=2,length=1, origin = 0):
        shape.__init__(self,N,isBoundaryFunction,dim,length, origin )
        self.f = f
        self.g = g
        if scheme is None:
            self.scheme = defaultScheme
        else:
            self.scheme  = lambda position : scheme(position,self.h)
        #dirichlet
        self.U = None
        self.A, self.Fb, self.Fi, self.geometry = self.getLinearizedDirichlet(self.scheme,
                                                                             self.f,
                                                                             self.g,
                                                                             self.maxIndex,
                                                                             self.getIndex,
                                                                             self.getCoordinate,
                                                                             self.getPosition,
                                                                             self.isBoundaryFunction)
    


    def plot(self,title=""):
        self.X, self.Y = self.getMeshGrid()
        self.plot2D(self.X, self.Y, self.U, title)


class linear_elliptic(solve_interface):
    def __init__(self,f,g,N,isBoundaryFunction=None,scheme=defaultScheme,dim=2,length=1, origin = 0):
        solve_interface.__init__(self,f,g,N,isBoundaryFunction,scheme,dim,length,origin)
        U_internal = splin.spsolve(self.A, self.Fi + self.Fb)
        self.U = finite_difference.applyBoundary(U_internal, self.g, self.geometry,self.getCoordinate, self.getPosition)


#F is not used because
class nonlinear_poisson(solve_interface):
    def __init__(self,f,g,N,isBoundaryFunction=None,scheme=defaultScheme,dim=2,length=1, origin = 0, maxIterNewton = 1000, lam=1.5,guess = None):
        solve_interface.__init__(self,f,g,N,isBoundaryFunction,scheme,dim,length,origin)
        self.lam = lam
        if guess == None:
            guess = np.ones(len(self.Fb))

        U_internal, self.error, self.iter = self.solveNonlinear2(A=self.A, u_n=guess, F_b=self.Fb, maxiter=maxIterNewton, lam=self.lam)
        self.U = finite_difference.applyBoundary(U_internal, self.g, self.geometry,self.getCoordinate, self.getPosition)

    # Ikke generell
    def solveNonlinear2(self, A, u_n, F_b, lam=1.5, tol=10e-10, maxiter=100):
        du = np.copy(u_n)
        error = 1
        iter = 0
        F = A @ u_n - self.h ** 2 * lam / u_n ** 2 - F_b

        while iter < maxiter and error > tol:
            jacobian = A + np.diag(lam / u_n) * self.h ** 2

            du = splin.lgmres(jacobian, -F, du)[0]

            u_n += du

            F = A @ u_n - self.h ** 2 * lam / u_n ** 2 - F_b

            error = np.nanmax(np.abs(F))
            iter += 1

        return u_n, error, iter

    def getError(self):
        return self.error

    def getIter(self):
        return self.iter

    def summary(self):
        print("Error estimate: ", self.error)
        print("The method used", self.iter, "iterations")