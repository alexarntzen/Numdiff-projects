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



def get_Axy_square(bvp, N):
    # Gridsize
    h = 1.0 / n

    # Total number of unknowns is N = (n+1)*(n+1)
    N = (n + 1) * (n + 1)

    # Define zero matrix A of right size and insert 0
    A = sp.dok_matrix((N, N))

    # Define FD entries of A
    hh = h * h
    for i in range(1, n):
        for j in range(1, n):
            A[I(i, j, n), I(i, j, n)] = 4 / hh  # U_ij
            A[I(i, j, n), I(i - 1, j, n)] = -1 / hh  # U_{i-1,j}
            A[I(i, j, n), I(i + 1, j, n)] = -1 / hh  # U_{i+1,j}
            A[I(i, j, n), I(i, j - 1, n)] = -1 / hh  # U_{i,j-1}
            A[I(i, j, n), I(i, j + 1, n)] = -1 / hh  # U_{i,j+1}



    # N, Number of intervals
    # Gridsize
    h = 1 / N
    # Total number of unknowns
    Ni = N + 1
    Ni2 = Ni * Ni
    # Make the grid
    x, y = np.ogrid[bvp.a:bvp.b:Ni * 1j, bvp.a:bvp.b:Ni * 1j]

    # Define zero matrix A of right size and insert 0
    A = sparse.dok_matrix((Ni2, Ni2))
    # Define FD entries of A
    muhh = - bvp.mu / (h * h)
    h2 = 1 / (2 * h)

    # get V
    v1, v2 = bvp.V(x, y)
    v1 = v1.ravel()
    v2 = v2.ravel()
    for i in range(1, N):
        for j in range(1, N):
            index = I(i, j, N)
            A[index, index] = -4 * muhh  # U_ij, U_p
            A[index, index - N - 1] = muhh - v2[i] * h2  # U_{i,j-1}, U_s
            A[index, index + N + 1] = muhh + v2[i] * h2  # U_{i,j+1}, U_n
            A[index, index - 1] = muhh - v1[j] * h2  # U_{i-1,j}, U_w
            A[index, index + 1] = muhh + v1[j] * h2  # U_{i+1,j}, U_e

        # Incorporate boundary conditions
        # Add boundary values related to unknowns from the first and last grid ROW
        for j in [0, N]:
            for i in range(0, Ni):
                index = I(i, j, N)
                A[index, index] = 1

        # Add boundary values related to unknowns from the first and last grid COLUMN
        for i in [0, N]:
            for j in range(0, Ni):
                index = I(i, j, N)
                A[index, index] = 1

    return A, x, y




class nonlinearPoisson():
    def __init__(self, f):
        self.f = f






class finite_difference_BVP:

    @staticmethod
    def getLinearizedBVP(scheme, g, numInternal, getInternalIndex, getCoordinate, getVecor,isBoundary, schemeCenter=1):
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

    def __init__(self):
        grid_BVP.__init__(self)
    def constanScheme(self, vector):
        return np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0])) * self.h ** 2








    def solve():

        N = 4
        Ni = N-1

        Ni2 = Ni**2
        # Construct a sparse A-matrix
        B = sparse.diags(Test.flatten(), [n for n in range(Test.size)], shape=(N + 1, N + 1), format="lil")

        A = sparse.kron(sparse.eye(Ni), B)
        C = sparse.diags([1, 1], [-Ni, Ni], shape=(Ni2, Ni2), format="lil")
        A = (A + C).tocsr()  # Konverter til csr-format (nødvendig for spsolve)

        C.toarray()

        A = sparse.csr_matrix(np.array([[1, 0], [0,1]]))
        sparse.kron(A, B).toarray()
        B
        A

        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        h = 1/N