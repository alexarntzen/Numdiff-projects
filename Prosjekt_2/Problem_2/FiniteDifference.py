import numpy as np
import scipy.sparse as sparse  # Sparse matrices
import matplotlib.pyplot as plt
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
        return domainGeometry

    @staticmethod
    def getSystem(shape, f, g, scheme, isNeumannFunc=None, schemeNeumannFunc=None, interior=False):

        # Makes the passed arguments to a form FiniteDifference will accept.
        if isNeumannFunc is None:
            isNeumannFunc = isNeumannFalse

        schemePassed = lambda position: scheme(position, shape)

        if schemeNeumannFunc is None:
            schemeNeumannFuncPassed = schemeNeumannDefault
        else:
            schemeNeumannFuncPassed = lambda position: schemeNeumannFunc(position, shape)

        # For calculating only the interor points. Nesseseary for nonlinear problems
        if interior:
            Amatrix, Finternal, Fboundary, geometry = FiniteDifference.getLinearizedInterior(schemePassed,
                                                                                             f,
                                                                                             g,
                                                                                             shape.maxIndex,
                                                                                             Shape.makeGetIndex(shape),
                                                                                             Shape.makeGetCoordinate(
                                                                                                 shape),
                                                                                             Shape.makeGetPosition(
                                                                                                 shape),
                                                                                             shape.isBoundaryFunc)

        # For calculating  the interor points and the boundary points. Nesseseary for Neumann conditions and some nonlinear probles
        else:
            Amatrix, Finternal, Fboundary, geometry = FiniteDifference.getLinearized(schemePassed,
                                                                                     f,
                                                                                     g,
                                                                                     shape.maxIndex,
                                                                                     Shape.makeGetIndex(shape),
                                                                                     Shape.makeGetCoordinate(shape),
                                                                                     Shape.makeGetPosition(shape),
                                                                                     shape.isBoundaryFunc,
                                                                                     isNeumannFunc,
                                                                                     schemeNeumannFuncPassed)

        return Amatrix, Finternal, Fboundary, geometry


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

    @staticmethod
    def makeGetIndex(lattice):
        return lambda coordinates : Lattice.getIndexLattice(coordinates,lattice.basis)

    @staticmethod
    def getCoordinateLattice(index, basis):
        # Inverse isomorphism  Z_{N^{dim}} --> {Z_N}^dim
        return - np.diff(index - (index % basis), append=0) // basis

    @staticmethod
    def makeGetCoordinate(lattice):
        return lambda internalIndex : Lattice.getCoordinateLattice(internalIndex, lattice.basis)

    @staticmethod
    def getPositionLattice(coordinate, box, origin):
        return origin + coordinate * box

    @staticmethod
    def makeGetPosition(lattice ):
        return lambda coordinate : Lattice.getPositionLattice(coordinate, lattice.box, lattice.origin)

    @staticmethod
    def getLatticeVector(vector, lattice):
        return np.reshape(vector, lattice.shape, order="F")

    @staticmethod
    def getVectorLattice(lattice, maxIndex):
        return np.reshape(lattice, (maxIndex), order="F")

    @staticmethod
    def makeGetVector(lattice):
        return lambda Array: Lattice.getVectorLattice(Array, lattice.maxIndex)

    @staticmethod
    def getMeshGrid(lattice):
        # Only for 2D, Grid for plotting
        dim = len(lattice.basis)
        if dim == 1:
            return Lattice.makeGetPosition(lattice)(np.ogrid[0:(lattice.N + 1)])
        elif dim == 2:
            return Lattice.makeGetPosition(lattice)(np.ogrid[0:(lattice.N + 1), 0:(lattice.N + 1)])



# Denne kan også brukes i oppgave 1a, cartesian
class Shape(Lattice):
    """
    Class to handle the boundary and create sqares or other shapes
    """
    def __init__(self, N, isBoundaryFunc=None, dim=2, length=1, origin=0):
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

    @staticmethod
    def plotOnShape(U, shape, title=None, ax=None, zlabel='$u(x,y)$', show=False, view=None, ulabel="U", geometry=None):
        coordinates = Shape.getMeshGrid(shape)
        if geometry is not None:
            Ufull = FiniteDifference.applyExterior(U, geometry)
        else:
            Ufull = U
        Ugrid = Shape.getLatticeVector(Ufull, shape)
        if Ugrid.ndim == 1:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                show = True
            ax.plot(*coordinates, Ugrid, ulabel=ulabel)  # Surface-plot
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
