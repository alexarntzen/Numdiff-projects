import numpy as np
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)

class finite_difference:
    """
    Class so that important functions have a namespace
    """
    @staticmethod
    def getLinearizedInterior(scheme,f, g, maxIndex, getIndex, getCoordinate, getVecor, isBoundary, schemeCenter=1):
        """
        Function return the appropriate linearized BVP problem for parameters given. Only on the internal points.
        :param scheme: function returning array of coefficients.
        :param f: function returning conditions.
        :param g: function  returning boundry conditions.
        :param maxIndex: number of points in computation .
        :param getIndex: function returning index from coordinates.
        :param getCoordinate: function returning coordinates from index.
        :param getVecor: function returning position from coordinates.
        :param isBoundary: return true if point is on boundary
        :param schemeCenter: for a given scheme return the origion coordinate for the array
        :return A: matrix discretization from scheme
        :return Fi: Right side of linearized BVP problem for interior points
        :return Fb: Right side of linearized BVP problem for boundary points
        :return geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        """
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
                        boundaryF[index] += - coeff * g(getVecor(np.copy(schemeCoordinate)))
                        geometry[1][schemeIndex] = True
                    elif coeff != 0:
                        A[index, schemeIndex] = coeff
        np.logical_and(np.logical_not(geometry[0]),np.logical_not(geometry[1]),geometry[2])
        return sparse.csr_matrix(A[geometry[0]])[:,geometry[0]],interiorF[geometry[0]], boundaryF[geometry[0]], geometry

    @staticmethod
    def applyBoundaryInterior(U_interior, g, geometry,getCoordinate, getVecor):
        """
        :param U_interior: list form of u for interior points
        :param g: function  returning boundry conditions.
        :param geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        :param getCoordinate: function returning coordinates from index.
        :param getVecor: function returning position from coordinates.
        :return U: list of u values for all points
        """
        maxIndex = geometry.shape[1]
        U = np.zeros(maxIndex, dtype=np.float64)
        interiorIndexs = np.flatnonzero(geometry[0])
        bounderyIndexs = np.flatnonzero(geometry[1])
        exteriorIndexs = np.flatnonzero(geometry[2])
        U[interiorIndexs] = U_interior
        U[bounderyIndexs] = [g(getVecor(getCoordinate(index))) for index in bounderyIndexs]
        U[exteriorIndexs] = np.nan
        return U

    @staticmethod
    def getLinearized(scheme,f, g, maxIndex, getIndex, getCoordinate, getVecor, isBoundary,isNeumann,schemeNeumann, schemeCenter=1):
        """
        Function return the appropriate linearized BVP problem for parameters given. Only on the internal points.
        :param scheme: function returning array of coefficients.
        :param f: function returning conditions.
        :param g: function  returning boundry conditions.
        :param maxIndex: number of points in computation .
        :param getIndex: function returning index from coordinates.
        :param getCoordinate: function returning coordinates from index.
        :param getVecor: function returning position from coordinates.
        :param isBoundary: return true if point is on boundary
        :param isNeumann: Function Returning true if point has Neumann conditions
        :param schemeNeumann: Scheme for Neumann conditions on that point.
        :param schemeCenter: for a given scheme return the origion coordinate for the array
        :return A: matrix discretization from scheme
        :return Fi: Right side of linearized BVP problem for interior points
        :return Fb: Right side of linearized BVP problem for boundary points
        :return geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        """
        A = sparse.lil_matrix((maxIndex, maxIndex), dtype=np.float64)
        boundaryF = np.zeros(maxIndex)
        interiorF = np.zeros(maxIndex)
        geometry = np.full((3,maxIndex),False,dtype=bool)
        for index in range(maxIndex):
            coordinate = getCoordinate(index)
            if not isBoundary(coordinate):
                # Scheme must return array  in same coordinate order as coordinates
                interiorF[index] = f(getVecor(coordinate))
                geometry[0][index] = True
                for arrayCoordinate, coeff in np.ndenumerate(scheme(getVecor(np.copy(coordinate)))):
                    #Could have used isomorphism property here
                    schemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    schemeIndex = getIndex(schemeCoordinate)
                    if isBoundary(schemeCoordinate) and not isNeumann(getVecor(schemeCoordinate)) :
                        A[schemeIndex,schemeIndex] = 1
                        boundaryF[schemeCoordinate]  = g(getVecor(schemeCoordinate))
                        geometry[1][schemeIndex] = True

                    if coeff != 0:
                        A[index, schemeIndex] = coeff

            elif isNeumann(getVecor(coordinate)):
                left, rightG, rightF = schemeNeumann(getVecor(np.copy(coordinate)))
                for arrayCoordinate, coeff in np.ndenumerate(left):
                    schemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    schemeIndex = getIndex(schemeCoordinate)
                    if coeff != 0:
                        geometry[1][schemeIndex] = True
                        A[index, schemeIndex] = coeff
                boundaryF[index] = rightG*g(getVecor(np.copy(coordinate))) + rightF* f(getVecor(np.copy(coordinate)))
                geometry[1][index] = True

        np.logical_and(np.logical_not(geometry[0]),np.logical_not(geometry[1]),geometry[2])
        indexes = np.logical_or(geometry[0], geometry[1])
        return sparse.csr_matrix(A[indexes])[:,indexes],interiorF[indexes], boundaryF[indexes], geometry

    @staticmethod
    def applyBoundary(U_interior, geometry,getCoordinate, getVecor):
        maxIndex = geometry.shape[1]
        U = np.zeros(maxIndex, dtype=np.float64)
        interiorIndexs = np.flatnonzero(np.logical_or(geometry[0],geometry[1]))
        exteriorIndexs = np.flatnonzero(geometry[2])
        U[interiorIndexs] = U_interior
        U[exteriorIndexs] = np.nan
        return U


class lattice():
    """
    Class handling isomorphism from  {Z_{N+1}}^dim to {Z_{{N+1}^dim}}
    """
    def __init__(self, shape=(4, 4), size=(1, 1), origin=0):
        self.shape = np.array(shape)
        self.size = np.array(size)
        self.box = self.size / (self.shape-1)
        self.N = np.max(self.shape - 1)
        self.length = np.max(size)
        self.h = self.length / self.N
        self.origin = origin
        self.basis = self.getBasis(self.shape )
        self.maxIndex = np.prod(self.shape )
        self.dim = self.shape.ndim

    @staticmethod
    def getBasis(shape):
        # Get the coordinates in {Z_{{N+1}^dim}} from isomorphism of basis in {Z_{N+1}}^dim
        return np.cumprod(np.append(1, shape[:-1]))

    @staticmethod
    def getIndexLattice(coordinates, basis):

        # isomorphism {Z_{N+1}}^dim --> Z_{{N+1}^{cim}}
        # There shoud be a % maxIndex to get proper isomorphism, but it is only needed on error cases.
        return np.dot(coordinates, basis)

    def getIndex(self, coordinates):
        return self.getIndexLattice(coordinates , self.basis)

    @staticmethod
    def getCoordinateLattice(index, basis):
        # Inverse isomorphism  Z_{N^{dim}} --> {Z_N}^dim
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
        return np.reshape(vector, shape , order="F")

    def getMeshGrid(self):
        # Only for 2D, Grdid for plotting
        return self.getPosition(np.ogrid[0:(self.N + 1), 0:(self.N + 1)])


# Denne kan også brukes i oppgave 1a, cartesian
class shape(lattice):
    """
    Class to handle the boundary and create sqares or other shapes
    """
    def __init__(self, N, isBoundaryFunc, dim=2, length=1, origin=0):
        lattice.__init__(self, np.full(dim, N+1), np.full(dim, length), origin)
        if isBoundaryFunc is None:
            self.isBoundaryFunc = self.isBoundarySqare
        else:
            self.isBoundaryFunc = lambda coordinate : isBoundaryFunc(self.getPosition(coordinate))

    def isBoundarySqare(self, coordinate):
        if (self.N in coordinate) or (0 in coordinate):
            return True
        else:
            return False


#Default functions for not Neumann conditions
def isNeumannFalse(position):
    return False

def schemeNeumannDefault(position):
    return 0, 0, 0

class solve_interface(shape,finite_difference):
    """
    Base class for inheritance. Do not use directly. Takes inn all necessary parameters for solving the base case.
    """
    def __init__(self,f,g,N,isBoundaryFunc=None,scheme=None,dim=2,length=1, origin = 0,isNeumannFunc = None, schemeNeumannFunc =None, interior=False):
        shape.__init__(self,N,isBoundaryFunc,dim,length, origin )

        self.f = f
        self.g = g
        self.interior = interior


        # Makes the passed arguments to a form finite_difference will accept.
        if isNeumannFunc is None:
            self.isNeumannFunc = isNeumannFalse
        else:
            self.isNeumannFunc = isNeumannFunc

        if scheme is None:
            self.scheme = self.defaultScheme
        else:
            self.scheme  = lambda position : scheme(position,self)

        if schemeNeumannFunc is None:
            self.schemeNeumannFunc = schemeNeumannDefault
        else:
            self.schemeNeumannFunc  = lambda position : schemeNeumannFunc(position,self)

        self.U = None

        #For calculating only the interor points. Nesseseary for nonlinear problems
        if interior:
            self.A, self.Fi, self.Fb, self.geometry = finite_difference.getLinearizedInterior(self.scheme,
                                                                                             self.f,
                                                                                             self.g,
                                                                                             self.maxIndex,
                                                                                             self.getIndex,
                                                                                             self.getCoordinate,
                                                                                             self.getPosition,
                                                                                             self.isBoundaryFunc)

        #For calculating  the interor points and the boundary points. Nesseseary for Neumann conditions
        else:
            self.A, self.Fi, self.Fb, self.geometry = finite_difference.getLinearized(self.scheme,
                                                                                      self.f,
                                                                                      self.g,
                                                                                      self.maxIndex,
                                                                                      self.getIndex,
                                                                                      self.getCoordinate,
                                                                                      self.getPosition,
                                                                                      self.isBoundaryFunc,
                                                                                      self.isNeumannFunc,
                                                                                      self.schemeNeumannFunc
                                                                                      )

    def defaultScheme(self, position):
        # Cartesian coordinates. Important!. 5-point stencil for laplacian.
        return np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0])) / self.h ** 2

    def plot(self,title=None,ax=None,zlabel='$u(x,y)$',show=False,view=None):
        X, Y = self.getMeshGrid()
        Ugrid = self.getLattice(self.U, self.shape)
        if ax is None:
            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = fig.gca(projection='3d')
            show = True
        ax.plot_surface(X, Y, Ugrid, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
        if view is not None:
            ax.view_init(view[0], view[1])
        # Set initial view angle
        # Set labels and show figure
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        if zlabel is not None:
            ax.set_zlabel('$z$')
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()

class linear_elliptic(solve_interface):
    """
    Class for solving linear elliptic PDEs.
    """
    def __init__(self,f,g,N,isBoundaryFunction=None,scheme=None,dim=2,length=1, origin = 0,
                 isNeumannFunc = None, schemeNeumannFunc =None):
        """
        :param scheme: function returning array of coefficients.
        :param f: function returning conditions.
        :param g: function  returning boundry conditions.
        :param isBoundaryFunction: return true if point is on boundary
        :param isBoundaryFunction:
        :param length: length of sides
        :param origin: for plotting
        :param isNeumannFunc: Function Returning true if point has Neumann conditions
        :param schemeNeumannFunc: Scheme for Neumann conditions on that point.
        """
        #Make discrimination
        solve_interface.__init__(self,f,g,N,isBoundaryFunction,scheme,dim,length,origin,isNeumannFunc , schemeNeumannFunc )

        #Solve system
        U_internal = splin.spsolve(self.A, self.Fi + self.Fb)

        #Apply np.nan on boundary
        self.U = finite_difference.applyBoundary(U_internal, self.geometry,self.getCoordinate, self.getPosition)


def fDefault(x):
    return 0

class nonlinear_poisson(solve_interface):
    #Class for solving the nonlinear problem \Delta u =1/u^2

    def __init__(self,g,N,f = fDefault, isBoundaryFunction=None,scheme=None,dim=2,length=1, origin = 0, tol = 1e-10,
                 maxIterNewton = 1000, lam=1.5,guess = None,isNeumannFunc = None, schemeNeumannFunc =None):
        # F is not used as it is 1/u^2
        #Make discrimination
        solve_interface.__init__(self,f,g,N,isBoundaryFunction,scheme,dim,length,origin,isNeumannFunc, schemeNeumannFunc, interior = True)
        self.tol = tol
        self.lam = lam
        if guess == None:
            guess = np.ones(len(self.Fb))
        #Solve the nonlinear problem
        U_internal, self.error, self.iter = self.solveNonlinear1(A=self.A, u_n=guess, F_b=self.Fb, maxiter=maxIterNewton, lam=self.lam, tol= self.tol)
        #Apply boundary
        self.U = finite_difference.applyBoundaryInterior(U_internal,self.g,self.geometry,self.getCoordinate, self.getPosition)

    @staticmethod
    def solveNonlinear1(A, u_n, F_b, lam=1.5, tol=10e-10, maxiter=100):
        """
        Solve the nonlinear eqation Au = \lambda/u^2 + F_b using Newton's method.
        :param A: descretization of the problem
        :param u_n: The inial guess for the solution u
        :param F_b: Right hand side, using boundary points
        :param lam: \lambda as described
        :param tol: tolerance of max(abs( Au -\lambda/u^2 - F_b))
        :param maxiter: maximum number of iterations of Newton's method.
        :return U: solution
        :return error : max(abs( Au -\lambda/u^2 - F_b))
        :return iter: iterations
        """
        du = np.copy(u_n)
        error = 100
        iter = 0
        F = A @ u_n - lam / u_n ** 2 + F_b

        while iter < maxiter and error > tol:
            #Standard newthons method

            jacobian = A + np.diag(lam / u_n)

            du = splin.lgmres(jacobian, -F, du)[0]

            u_n += du

            F = A @ u_n - lam / u_n **2 - F_b


            error = np.nanmax(np.abs(F))
            iter += 1

        return u_n, error, iter

    def summary(self):
        print("Error estimate: ", self.error)
        print("The method used", self.iter, "iterations")