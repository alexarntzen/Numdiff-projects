import numpy as np
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.animation as animation

from FiniteDifference import Shape, FiniteDifference


# Default functions for not Neumann conditions


def fDefault(*args):
    return np.array([0])


class ModifiedCrankNicolson():
    @staticmethod
    def modifiedCrankNicolson(g, getU0=None, fDeriv=fDefault, scheme=None, T=10, k=1, shape=None,
                              isNeumannFunc=None, schemeNeumannFunc=None, mu=1):

        Amatrix, _, Fboundary, geometry = FiniteDifference.getSystem(shape=shape,
                                                                     f=fDefault,
                                                                     g=g,
                                                                     scheme=scheme,
                                                                     isNeumannFunc=isNeumannFunc,
                                                                     schemeNeumannFunc=schemeNeumannFunc)

        domainIndexs = np.flatnonzero(np.logical_or(geometry[0], geometry[1]))
        isBoundaryList = FiniteDifference.getDomainGeometry(geometry)[1]
        U_0 = Shape.getVectorLattice(getU0(*Shape.getMeshGrid(shape)), shape)[domainIndexs]

        return ModifiedCrankNicolson.modifiedCrankNicolsonSolver(U_0, fDeriv, T=T, k=k, diffOperator=Amatrix,
                                                                 Fb=Fboundary,
                                                                 isBoundaryList=isBoundaryList)

    # TODO: Add get bate, gamma.
    @staticmethod
    def modifiedCrankNicolsonSolver(U_0=0, f=fDefault, T=10, k=1, diffOperator=None, Fb=np.array(0),
                                    isBoundaryList=None, mu=1):
        maxIndex = np.size(U_0)
        if diffOperator is not None:
            Left = sparse.csr_matrix(sparse.identity(maxIndex, format="csr") - k / 2 * diffOperator)
            Left[isBoundaryList] = diffOperator[isBoundaryList]
        else:
            Left = sparse.identity(maxIndex)
            diffOperator = np.zeros((maxIndex, maxIndex))
        times = np.arange(0, T + k, k)
        U = np.zeros((len(times), maxIndex))
        U[0] = U_0
        for t in range(1, len(times)):
            right = U[t - 1] + k / 2 * diffOperator @ U[t - 1] + k * f(U[t - 1])
            if isBoundaryList is not None:
                right[isBoundaryList] = Fb[isBoundaryList]
            U_temp = splin.lgmres(Left, right, U[t - 1])[0]
            U[t] = U_temp + k / 2 * (f(U_temp) - f(U[t - 1]))
        return U, times


class DiseaseModel():
    """
    Class for solving linear elliptic PDEs.
    """

    def __init__(self, g, getU0_S, getU0_I, muS, muI, schemeS, getBeta, getGamma, T=10, k=1, N=4,
                 isBoundaryFunction=None, dim=2, length=1, origin=0,
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
        self.T, self.k = T, k
        self.shapeObject = Shape(N=N, isBoundaryFunc=isBoundaryFunction, dim=dim, length=length, origin=origin)

        AmatrixS, _S, FboundaryS, self.geometryS = FiniteDifference.getSystem(shape=self.shapeObject,
                                                                              f=fDefault,
                                                                              g=g,
                                                                              scheme=schemeS,
                                                                              isNeumannFunc=isNeumannFunc,
                                                                              schemeNeumannFunc=schemeNeumannFunc)

        self.geometryI = self.geometryS

        self.Fboundary = np.concatenate((FboundaryS, FboundaryS))

        self.diffOperator = sparse.bmat([[AmatrixS * muS, None], [None, AmatrixS * muI]], format="lil")

        geometrySI = np.concatenate((self.geometryS, self.geometryI), axis=1)
        domainGeometrySI = FiniteDifference.getDomainGeometry(geometrySI)
        self.domaiSize = len(domainGeometrySI[0])

        # Assuming R = 0 at t = 0
        I_0 = Shape.getVectorLattice(getU0_I(*Shape.getMeshGrid(self.shapeObject)), self.shapeObject)
        S_0 = Shape.getVectorLattice(getU0_S(*Shape.getMeshGrid(self.shapeObject)), self.shapeObject)
        self.U_0 = np.concatenate((S_0, I_0))[np.logical_or(geometrySI[0], geometrySI[1])]

        diseaseModelF = DiseaseModel.getDiseaseModelF(getBeta,getGamma,self.shapeObject)

        self.UList, self.times = ModifiedCrankNicolson.modifiedCrankNicolsonSolver(self.U_0,
                                                                                   f=diseaseModelF,
                                                                                   T=T, k=k,
                                                                                   diffOperator=self.diffOperator,
                                                                                   Fb=self.Fboundary,
                                                                                   isBoundaryList=domainGeometrySI[1])

    def plotS(self, timeIndex=0, title=None, ax=None, show=False, view=None):
        U = self.UList[timeIndex, :self.domaiSize // 2]
        return Shape.plotOnShape(U, self.shapeObject, title=title, ax=ax, show=show, view=view,
                                 ulabel="S", geometry=self.geometryS)

    def plotI(self, timeIndex=0, title=None, ax=None, show=False, view=None):
        U = self.UList[timeIndex, self.domaiSize // 2:]
        return Shape.plotOnShape(U, self.shapeObject, title=title, ax=ax, show=show, view=view,
                                 ulabel="I", geometry=self.geometryI)

    def plotSImage(self, timeIndex=0, title=None, ax=None, show=False, view=None, animated=False, colorbar=False):
        U = self.UList[timeIndex, :self.domaiSize // 2]
        return Shape.plotImage2d(U, self.shapeObject, title="Susceptible", ax=ax, show=show, geometry=self.geometryS,
                                 animated=animated, colorbar=colorbar, vmin=0, vmax=1)

    def plotIImage(self, timeIndex=0, ax=None, show=False, view=None, animated=False, colorbar=False):
        U = self.UList[timeIndex, self.domaiSize // 2:]
        return Shape.plotImage2d(U, self.shapeObject, title="Infected", ax=ax, show=show, geometry=self.geometryI,
                                 animated=True, colorbar=False, vmin=0, vmax=1)


    def applyDiseaseAnimation(self, axS, axI, **kvargs):
        artistlist = []

        for timeIndex in range(len(self.times)):
            S, I = self.UList[timeIndex, :self.domaiSize // 2], self.UList[timeIndex, self.domaiSize // 2:]

            artistS = Shape.plotImage2d(S, self.shapeObject, ax=axS, geometry=self.geometryS, title="Susceptible",
                                        animated=True, colorbar=False, vmin=0, vmax=1, **kvargs)

            artistI = Shape.plotImage2d(I, self.shapeObject, ax=axI, geometry=self.geometryI, title="Infected",
                                        animated=True, colorbar=False, vmin=0, vmax=1, **kvargs)
            artistlist.append([artistS, artistI])
            if timeIndex == 0 :
                axI.figure.colorbar(artistI, ax=[axS,axI])

        return animation.ArtistAnimation(axI.figure, artistlist, blit=True, )

    def getSolution(self):
        SList, IList = self.UList[:, :self.domaiSize // 2], self.UList[:, self.domaiSize // 2:]
        return SList, IList, self.times

    @staticmethod
    def DiseaseModelF(beta, gamma):
        # currying
        def F(U):
            S = U[:len(U) // 2]
            I = U[len(U) // 2:]
            return np.concatenate((-beta * S * I, S * I - gamma * I))

        return F

    @staticmethod
    def getDiseaseModelF(getBeta, getGamma,shape):
        # currying
        betaLattice = getBeta(*Shape.getMeshGrid(shape))
        gammaLattice = getGamma(*Shape.getMeshGrid(shape))
        if  np.size(betaLattice) == 1 and np.size(gammaLattice) == 1:
            beta = betaLattice
            gamma = gammaLattice
        else:
            beta = Shape.getVectorLattice(betaLattice, shape)
            gamma = Shape.getVectorLattice(gammaLattice, shape)
        def F(U):
            S, I = np.split(U,2)
            return np.concatenate((-beta * S * I, S * I - gamma * I))
        return F

    # TODO: plot time evolution with R



