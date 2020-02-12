import numpy as np
from scipy.linalg import solve
import scipy.sparse as sparse             # Sparse matrices
from scipy.sparse.linalg import spsolve   # Linear solver for sparse matrices
import matplotlib.pyplot as plt
newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plot
from matplotlib import cm



#Denne kan brukes når med Dirichlet conditions
class finite_difference_BVP:
    @staticmethod
    def getLinearizedBVPDirichlet(scheme, g, numInternal, getInternalIndex, getCoordinate, getVecor,isBoundary, schemeCenter=1):
        A = sparse.dok_matrix((numInternal, numInternal), dtype=np.float32)
        Right = np.zeros(numInternal)
        for internalIndex in range(numInternal):
            coordinate = getCoordinate(internalIndex)
            for arrayCoordinate, coeff in np.ndenumerate(scheme(getVecor(coordinate))):
                shemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                if isBoundary(shemeCoordinate):
                    Right[internalIndex] = -coeff * g(getVecor(shemeCoordinate))
                else:
                    A[internalIndex, getInternalIndex(shemeCoordinate)] = coeff

        return A, Right


#Denne kan også brukes i oppgave 1a
class grid_BVP(finite_difference_BVP):
    def __init__(self,N,f,g,dim=2):
        self.N = N
        self.h = 1/N
        self.f = f
        self.g = g
        self.dim = dim

    def getInternalIndexGrid(self, coordinates):
        return np.dot(np.array(coordinates) - 1, np.array([self.N ** n for n in range(self.dim - 1, -1, -1)]))

    def getCoordinate(self, internalIndex):
        divider = np.array([self.N ** n for n in range(self.dim - 1, -1, -1)])
        return -np.diff(internalIndex - (internalIndex % divider))  + 1

    def getVecor(self):
        pass

    def isBoundary(coordinate, ):
        return False


class nonlinearPisson(grid_BVP):

    def __init__(self,N,f,g,dim):
        grid_BVP.__init__(self,N,f,g,dim)

    def constanScheme(self, vector):
        return np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0])) * self.h ** 2
