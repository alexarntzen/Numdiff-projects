import numpy as np
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm


class FiniteDifference:
    """
    Class so that important functions have a namespace
    """
    @staticmethod
    def getLinearizedInterior(scheme, f, g, maxIndex, getIndex, getCoordinate, getVecor, isBoundary):
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
        :return A: matrix discretization from scheme
        :return Fi: Right side of linearized BVP problem for interior points
        :return Fb: Right side of linearized BVP problem for boundary points
        :return geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        """
        A = sparse.lil_matrix((maxIndex, maxIndex), dtype=np.float64)
        boundaryF = np.zeros(maxIndex)
        interiorF = np.zeros(maxIndex)
        geometry = np.full((3, maxIndex), False, dtype=bool)
        for index in range(maxIndex):
            coordinate = getCoordinate(index)
            if not isBoundary(coordinate):
                geometry[0][index] = True
                # Scheme must return array  in same coordinate order as coordinates
                interiorF[index] = f(getVecor(coordinate))
                S, schemeCenter = scheme(getVecor(np.copy(coordinate)))
                for arrayCoordinate, coeff in np.ndenumerate(S):
                    # Could have used isomorphism property here
                    schemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    schemeIndex = getIndex(schemeCoordinate)
                    if isBoundary(schemeCoordinate):
                        boundaryF[index] += - coeff * g(getVecor(np.copy(schemeCoordinate)))
                        geometry[1][schemeIndex] = True
                    elif coeff != 0:
                        A[index, schemeIndex] = coeff
        np.logical_and(np.logical_not(geometry[0]), np.logical_not(geometry[1]), geometry[2])
        return sparse.csr_matrix(A[geometry[0]])[:, geometry[0]], interiorF[geometry[0]], boundaryF[
            geometry[0]], geometry

    @staticmethod
    def applyBoundaryInterior(U_interior, g, geometry, getCoordinate, getVecor):
        """
        :param U_interior: list form of u for interior points
        :param g: function  returning boundry conditions.
        :param geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        :param getCoordinate: function returning coordinates from index.
        :param getVecor: function returning position from coordinates.
        :return U: list of u values for all points in domain and exterior
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
    def getLinearized(scheme, f, g, maxIndex, getIndex, getCoordinate, getVecor, isBoundary, isNeumann, schemeNeumann,
                      ):
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
        :return A: matrix discretization from scheme
        :return Fi: Right side of linearized BVP problem for interior points
        :return Fb: Right side of linearized BVP problem for boundary points
        :return geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        """
        A = sparse.lil_matrix((maxIndex, maxIndex), dtype=np.float64)
        boundaryF = np.zeros(maxIndex)
        interiorF = np.zeros(maxIndex)
        geometry = np.full((3, maxIndex), False, dtype=bool)
        for index in range(maxIndex):
            coordinate = getCoordinate(index)
            if not isBoundary(coordinate):
                # Scheme must return array  in same coordinate order as coordinates
                interiorF[index] = f(getVecor(coordinate))
                geometry[0][index] = True
                S, schemeCenter = scheme(getVecor(np.copy(coordinate)))
                for arrayCoordinate, coeff in np.ndenumerate(S):
                    # Could have used isomorphism property here
                    schemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    schemeIndex = getIndex(schemeCoordinate)
                    if isBoundary(schemeCoordinate) and not isNeumann(getVecor(schemeCoordinate)):
                        A[schemeIndex, schemeIndex] = 1
                        boundaryF[schemeCoordinate] = g(getVecor(schemeCoordinate))
                        geometry[1][schemeIndex] = True

                    if coeff != 0:
                        A[index, schemeIndex] = coeff

            elif isNeumann(getVecor(coordinate)):
                left, rightG, rightF, schemeCenter = schemeNeumann(getVecor(np.copy(coordinate)))
                for arrayCoordinate, coeff in np.ndenumerate(left):
                    schemeCoordinate = coordinate + arrayCoordinate - schemeCenter
                    schemeIndex = getIndex(schemeCoordinate)
                    if coeff != 0:
                        A[index, schemeIndex] = coeff
                boundaryF[index] = rightG * g(getVecor(np.copy(coordinate))) + rightF * f(getVecor(np.copy(coordinate)))
                geometry[1][index] = True

        np.logical_and(np.logical_not(geometry[0]), np.logical_not(geometry[1]), geometry[2])
        indexes = np.logical_or(geometry[0], geometry[1])
        return sparse.csr_matrix(A[indexes])[:, indexes], interiorF[indexes], boundaryF[indexes], geometry

    @staticmethod
    def applyBoundary(U_domian, geometry, g, getCoordinate, getVecor):
        """
        :param U_domain: list form of u for all points in the domain 
        :param g: function  returning boundry conditions.
        :param geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        :param getCoordinate: function returning coordinates from index.
        :param getVecor: function returning position from coordinates.
        :return U: list of u values for all points in domain now with boundary points having vales from g
        """
        domainIndexs = np.flatnonzero(np.logical_or(geometry[0], geometry[1]))
        bounderyIndexs = np.flatnonzero(geometry[1])
        maxIndex = len(geometry[0])

        if maxIndex == len(U_domian):
            U = U_domian
        else:
            U = np.zeros(maxIndex, dtype=np.float64)
            U[domainIndexs] = U_domian

        U[bounderyIndexs] = [g(getVecor(getCoordinate(index))) for index in bounderyIndexs]

        return U[domainIndexs]

    @staticmethod
    def applyExterior(U_domian, geometry):
        """
        :param U_interior: list form of u for all points in the domain. 
        :param g: function  returning boundry conditions.
        :param geometry: list of list specifying which values are on 0: interior, 1: boundary, 2: exterior
        :param getCoordinate: function returning coordinates from index.
        :param getVecor: function returning position from coordinates.
        :return U: list of u values for all points in domain and exterior
        """

        domainIndexs = np.flatnonzero(np.logical_or(geometry[0], geometry[1]))
        exteriorIndexs = np.flatnonzero(geometry[2])
        maxIndex = len(geometry[0])

        if len(U_domian) == maxIndex:
            U = U_domian
        else:
            U = np.zeros(maxIndex, dtype=np.float64)
            U[domainIndexs] = U_domian

        U[exteriorIndexs] = np.nan

        return U

    @staticmethod
    def getDomainGeometry(geometry):
        domainIndexs = np.flatnonzero(np.logical_or(geometry[0], geometry[1]))
        domainGeometry = geometry[:, domainIndexs]


class Lattice():
    """
    Class handling isomorphism from  {Z_{N+1}}^dim to {Z_{{N+1}^dim}}. In addition serves as namespace
    """

    def __init__(self, shape=(4, 4), size=(1, 1), origin=0):
        self.shape = np.array(shape)
        self.size = np.array(size)
        self.box = self.size / (self.shape - 1)
        self.N = np.max(self.shape - 1)
        self.length = np.max(size)
        self.h = self.length / self.N
        self.origin = origin
        self.basis = Lattice.getBasis(self.shape)
        self.maxIndex = np.prod(self.shape)
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
        return self.getIndexLattice(coordinates, self.basis)

    @staticmethod
    def makeGetIndex(lattice):
        return lambda coordinates : Lattice.getIndexLattice(coordinates,lattice.basis)

    @staticmethod
    def getCoordinateLattice(index, basis):
        # Inverse isomorphism  Z_{N^{dim}} --> {Z_N}^dim
        return - np.diff(index - (index % basis), append=0) // basis

    def getCoordinate(self, internalIndex):
        return self.getCoordinateLattice(internalIndex, self.basis)

    @staticmethod
    def makeGetCoordinate(lattice):
        return lambda internalIndex : Lattice.getCoordinateLattice(internalIndex, lattice.basis)

    @staticmethod
    def getPositionLattice(coordinate, box, origin):
        return origin + coordinate * box

    def getPosition(self, coordinate):
        return self.getPositionLattice(coordinate, self.box, self.origin)

    @staticmethod
    def makeGetPosition(lattice ):
        return lambda coordinate : Lattice.getPositionLattice(coordinate, lattice.box, lattice.origion)


    @staticmethod
    def getLatticeLatticeL(vector, shape):
        return np.reshape(vector, shape, order="F")

    def getLattice(self,vector):
        return np.reshape(vector, self.shape, order="F")

    @staticmethod
    def makeGetLattice(lattice):
        return lambda vector: Lattice.getLatticeLatticeL(vector, lattice.shape)

    @staticmethod
    def getVectorLattice(lattice, maxIndex):
        return np.reshape(lattice, (maxIndex), order="F")

    def getVector(self,lattice):
        return np.reshape(lattice, (self.maxIndex), order="F")

    @staticmethod
    def makeGetVector(lattice):
        return lambda Array: Lattice.getVectorLattice(Array, lattice.maxIndex)

    @staticmethod
    def getMeshGridLattice(lattice):
        # Only for 2D, Grid for plotting
        dim = len(lattice.basis)
        if dim == 1:
            return lattice.getPosition(np.ogrid[0:(lattice.N + 1)])
        elif dim == 2:
            return lattice.getPosition(np.ogrid[0:(lattice.N + 1), 0:(lattice.N + 1)])



    def getMeshGrid(self):
        # Only for 2D, Grid for plotting
        dim = len(self.basis)
        if dim == 1:
            return self.getPosition(np.ogrid[0:(self.N + 1)])
        elif dim == 2:
            return self.getPosition(np.ogrid[0:(self.N + 1), 0:(self.N + 1)])

# Denne kan også brukes i oppgave 1a, cartesian
class Shape(Lattice):
    """
    Class to handle the boundary and create sqares or other shapes
    """

    def __init__(self, N, isBoundaryFunc, dim=2, length=1, origin=0):
        if dim == 0:
            dim = 1
            N = 0
        Lattice.__init__(self, np.full(dim, N + 1), np.full(dim, length), origin)
        if isBoundaryFunc is None:
            self.isBoundaryFunc = Shape.makeIsBoundarySqare(self)
        else:
            self.isBoundaryFunc = lambda coordinate: isBoundaryFunc(Lattice.makeGetPosition(self)(coordinate))

    @staticmethod
    def makeIsBoundarySqare(lattice):
        def isBoundarySqare(coordinate):
            if (lattice.N  in coordinate) or (0 in coordinate):
                return True
            else:
                return False
        return isBoundarySqare


# Default functions for not Neumann conditions
def isNeumannFalse(position):
    return False


def schemeNeumannDefault(position):
    return 0, 0, 0


def getSystem(shape, f, g, scheme, isNeumannFunc=None, schemeNeumannFunc=None, interior=False):

    # Makes the passed arguments to a form FiniteDifference will accept.
    if isNeumannFunc is None:
        isNeumannFunc = isNeumannFalse
    else:
        isNeumannFunc = isNeumannFunc

    scheme = lambda position: scheme(position, shape)

    if schemeNeumannFunc is None:
        schemeNeumannFunc = schemeNeumannDefault
    else:
        schemeNeumannFunc = lambda position: schemeNeumannFunc(position, shape)


    # For calculating only the interor points. Nesseseary for nonlinear problems
    if interior:
        Amatrix, Finternal, Fboundary, geometry  = FiniteDifference.getLinearizedInterior(scheme,
                                                                                         f,
                                                                                         g,
                                                                                         shape.maxIndex,
                                                                                         Shape.makeGetIndex(shape),
                                                                                         Shape.makeGetCoordinate(shape),
                                                                                         Shape.makeGetPosition(shape),
                                                                                         shape.isBoundaryFunc )

    # For calculating  the interor points and the boundary points. Nesseseary for Neumann conditions and some nonlinear probles
    else:
        Amatrix, Finternal, Fboundary, geometry = FiniteDifference.getLinearized(scheme,
                                                                                 f,
                                                                                 g,
                                                                                 shape.maxIndex,
                                                                                 Shape.makeGetIndex(shape),
                                                                                 Shape.makeGetCoordinate(shape),
                                                                                 Shape.makeGetPosition(shape),
                                                                                 shape.isBoundaryFunc ,
                                                                                 isNeumannFunc,
                                                                                 schemeNeumannFunc)

    return Amatrix, Finternal, Fboundary, geometry




class SolveInterface(FiniteDifference):
    """
    Base class for inheritance. Do not use directly. Takes inn all necessary parameters for solving the base case.
    """

    def __init__(self,shape, f, g, scheme=None, isNeumannFunc=None, schemeNeumannFunc=None, interior=False):
        self.shapeObject = shape
        self.f = f
        self.g = g
        self.interior = interior

        # Makes the passed arguments to a form FiniteDifference will accept.
        if isNeumannFunc is None:
            self.isNeumannFunc = isNeumannFalse
        else:
            self.isNeumannFunc = isNeumannFunc

        if scheme is None:
            self.scheme = self.defaultScheme
        else:
            self.scheme = lambda position: scheme(position, self.shapeObject)

        if schemeNeumannFunc is None:
            self.schemeNeumannFunc = schemeNeumannDefault
        else:
            self.schemeNeumannFunc = lambda position: schemeNeumannFunc(position, self.shapeObject)

        self.U = None

        # For calculating only the interor points. Nesseseary for nonlinear problems
        if interior:
            self.A, self.Fi, self.Fb, self.geometry = FiniteDifference.getLinearizedInterior(self.scheme,
                                                                                             self.f,
                                                                                             self.g,
                                                                                             self.shapeObject.maxIndex,
                                                                                             self.shapeObject.getIndex,
                                                                                             self.shapeObject.getCoordinate,
                                                                                             self.shapeObject.getPosition,
                                                                                             self.shapeObject.isBoundaryFunc)

        # For calculating  the interor points and the boundary points. Nesseseary for Neumann conditions
        else:
            self.A, self.Fi, self.Fb, self.geometry = FiniteDifference.getLinearized(self.scheme,
                                                                                     self.f,
                                                                                     self.g,
                                                                                     self.shapeObject.maxIndex,
                                                                                     self.shapeObject.getIndex,
                                                                                     self.shapeObject.getCoordinate,
                                                                                     self.shapeObject.getPosition,
                                                                                     self.shapeObject.isBoundaryFunc,
                                                                                     self.isNeumannFunc,
                                                                                     self.schemeNeumannFunc)


    def plot(self, title=None, ax=None, zlabel='$u(x,y)$', show=False, view=None, ulabel="U"):
        coordinates = self.shapeObject.getMeshGrid()
        Ufull = FiniteDifference.applyExterior(self.U,self.geometry)
        Ugrid = self.shapeObject.getLattice(Ufull)
        if Ugrid.ndim == 1:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                show = True
            ax.plot(*coordinates, Ugrid,ulabel=ulabel)  # Surface-plot
            # Set initial view angle
            # Set labels and show figure
            ax.set_xlabel('$x$')
            if zlabel is not None:
                ax.set_ylabel('$z$')
        elif Ugrid.ndim == 2:
            if ax is None:
                fig = plt.figure(figsize=(8, 6), dpi=100)
                ax = fig.gca(projection='3d')
                show = True
            ax.plot_surface(*coordinates, Ugrid, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
            if view is not None:
                ax.view_init(view[0], view[1])
            # Set initial view angle
            # Set labels and show figure
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            if zlabel is not None:
                ax.set_zlabel('$z$')
        else:
            print("Shape only supports plotting in 1 and 2 dimentions")
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()

def fDefault(x):
    return np.array([0])

class ModifiedCrankNicolson(SolveInterface):
    def __init__(self, g, getU0=None, fDeriv=fDefault, scheme=None, T=10, k=1, N=4, isBoundaryFunction=None, dim=2, length=1, origin=0,
                 isNeumannFunc=None, schemeNeumannFunc=None):
        shapeObject = Shape(N, isBoundaryFunction, dim, length, origin)
        SolveInterface.__init__(self, shapeObject, fDefault, g, scheme, isNeumannFunc, schemeNeumannFunc)
        self.getU0 = getU0
        self.fDeriv = fDeriv
        self.T = T
        self.k = k
        self.UList = None
        self.times = None

    def solveBase(self):
        U_0 = self.shapeObject.getVector(self.getU0(*self.shapeObject.getMeshGrid()))
        domainGeometry = self.getDomainGeometry(self.geometry)
        return self.modifiedCrankNicolsonSolver(U_0, self.fDeriv, T = self.T, k = self.k, diffOperator = self.A, Fb = self.Fb,isBoundaryList=domainGeometry[1])

    @staticmethod
    def modifiedCrankNicolsonSolver(U_0 = np.array([0]), f = fDefault , T = 10, k = 1, diffOperator = None,Fb=np.array(0), isBoundaryList=None,mu=1 ):
        maxIndex = len(U_0)
        if diffOperator is not None:
            Left = sparse.lil_matrix(sparse.identity(maxIndex ,format="csr")- k/2*diffOperator)
            Left[isBoundaryList] = diffOperator[isBoundaryList]
        else:
            Left = sparse.identity(maxIndex)
            diffOperator=np.zeros((maxIndex, maxIndex))
        times = np.arange(0,T+k,k)
        U = np.zeros((len(times), maxIndex))
        U[0] = U_0
        for t in range(1,len(times)):
            right = U[t - 1] + k / 2 * diffOperator @ U[t - 1] + k * f(U[t - 1])
            if isBoundaryList is not None:
                right[isBoundaryList] = Fb[isBoundaryList]
            U_temp = splin.lgmres(Left, right, U[t - 1])[0]
            U[t] = U_temp + k / 2 * (f(U_temp) - f(U[t - 1]))
        return U, times

def DiseaseModelF(beta, gamma):
    # currying
    def F(U):
        S = U[:len(U)//2]
        I = U[len(U)//2:]
        return np.concatenate((-beta * S * I, S * I - gamma * I))
    return F


class DiseaseModel(ModifiedCrankNicolson):
    """
    Class for solving linear elliptic PDEs.
    """
    def __init__(self, g, getU0_I,muS,muI,schemeS=None, beta=1, gamma=1,T=10, k=1, N=4, isBoundaryFunction=None, dim=2, length=1, origin=0,
                 isNeumannFunc=None, schemeNeumannFunc=None):
        """
        :param scheme: function returning array of coefficients.
        :param f: function returning time derivative.
        :param g: function  returning boundry conditions.
        :param isBoundaryFunction: return true if point is on boundary
        :param isBoundaryFunction:
        :param length: length of sides
        :param origin: for plotting
        :param isNeumannFunc: Function Returning true if point has Neumann conditions
        :param schemeNeumannFunc: Scheme for Neumann conditions on that point.
        """
        # Make discrimination

        ModifiedCrankNicolson.__init__(self,g,getU0_I, DiseaseModelF(beta,gamma), schemeS, T, k, N, isBoundaryFunction, dim = dim, length = length, origin = 0, isNeumannFunc = isNeumannFunc, schemeNeumannFunc = schemeNeumannFunc)

        self.muS, self.myI, self.beta, self.gamma = muS, muI, T, k

        self.diffOperator = sparse.bmat([[self.A*muS, None],[None, self.A*muI]],format = "lil")

        self.F_b = np.concatenate((self.Fb,self.Fb))
        geometrySI = np.concatenate((self.geometry,self.geometry),axis=1)
        domainIndexs = np.flatnonzero(np.logical_or(geometrySI[0], geometrySI[1]))
        domainGeometry = geometrySI[:, domainIndexs]

        #Assuming R = 0 at t = 0
        I_0 = self.shapeObject.getVector(getU0_I(*self.shapeObject.getMeshGrid()))
        S_0 =  np.ones(self.shapeObject.maxIndex) - I_0
        U_0 = np.concatenate((S_0,I_0))

        self.UList, self.times = self.modifiedCrankNicolsonSolver(U_0, self.fDeriv, self.T, self.k, self.diffOperator, self.F_b, domainGeometry[1])

    def plotS(self, time=0, title=None, ax=None, zlabel='$u(x,y)$', show=False, view=None, ulabel="U"):
        timeIndex = int(time//self.k)
        self.U = self.UList[timeIndex,:self.shapeObject.maxIndex]
        self.plot(title=title, ax=ax,  show=show, view=view, zlabel="Relative number of people", ulabel="S")

    def plotI(self,time=0,title=None, ax=None, show=False, view=None):
        timeIndex = int(time//self.k)
        self.U = self.UList[timeIndex,self.shapeObject.maxIndex:]
        self.plot(title=title, ax=ax,  show=show, view=view, zlabel="Relative number of people", ulabel="I")

    def plotR(self,time=0,title=None, ax=None, show=False, view=None):
        timeIndex = int(time//self.k)
        S, I =  self.UList[timeIndex,:self.shapeObject.maxIndex], self.UList[timeIndex,self.shapeObject.maxIndex:]
        # Er denne koden helt uleselig? Synd, for den er skikkelig kul.
        R = 1 - S - I
        self.U = R
        self.plot(title=title, ax=ax,  show=show, view=view, zlabel="Relative number of people", ulabel="R")

    def getSolution(self):
        SList, IList =  self.UList[:,:self.shapeObject.maxIndex], self.UList[:,self.shapeObject.maxIndex:]
        RList = 1 -SList -IList
        return SList, IList, RList, self.times
