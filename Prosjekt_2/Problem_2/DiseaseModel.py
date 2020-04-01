import numpy as np
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FiniteDifference import Shape, FiniteDifference

"""
This file contains the modified numerical solver and an implementation for solving a disease model- 
"""

# Default function used since the problem is not in the form Ax = F is not used
def fZero(*args):
    return np.array([0])

# Newmann contitions
def gOne(*args):
    return 1

# Newmann contitions
def gDefault(*args):
    return 0


class ModifiedCrankNicolson():
    @staticmethod
    def modifiedCrankNicolson(getU0=None, fDeriv=fZero, scheme=None, T=10, k=1, shape=None,
                              isNeumannFunc=None, schemeNeumannFunc=None, g=gDefault):
        Amatrix, Finterior, Fboundary, geometry = FiniteDifference.getSystem(shape=shape,
                                                                             f=fZero,
                                                                             g=gOne,
                                                                             scheme=scheme,
                                                                             isNeumannFunc=isNeumannFunc,
                                                                             schemeNeumannFunc=schemeNeumannFunc,
                                                                             interior=True)
        if shape.dim == 1:
            def getBoundaryVals(t):
                return Shape.getVectorLattice(g(Shape.getMeshGrid(shape), t), shape)
            U_0 = Shape.getVectorLattice(getU0(Shape.getMeshGrid(shape)), shape)

        else:
            def getBoundaryVals(t):
                return Shape.getVectorLattice(g(*Shape.getMeshGrid(shape), t), shape)
            U_0 = Shape.getVectorLattice(getU0(*Shape.getMeshGrid(shape)), shape)

        return ModifiedCrankNicolson.modifiedCrankNicolsonSolver(U_0, fDeriv, T=T, k=k, diffOperator=Amatrix,
                                                                 boundaryCoeffs=-Fboundary, g=getBoundaryVals,
                                                                 geometry=geometry)

    @staticmethod

    def modifiedCrankNicolsonSolver(U_0=0, f=fZero, T=10, k=1, diffOperator=np.array(0), boundaryCoeffs=np.array(0), g=gOne, geometry=[None, None, None]):
        """
        This is the most impotant funciton that actually does the temportal iteration
        """
        U_0 = np.array(U_0)
        internalIndex = np.size(U_0[geometry[0]])
        maxIndex = np.size(U_0)
        Left = sparse.csc_matrix(sparse.identity(internalIndex, format="csc") - k / 2 * diffOperator)
        solveLeft = splin.factorized(Left)
        times = np.arange(0, T + k, k)
        U = np.zeros((len(times), maxIndex))
        U[0] = U_0
        for t in range(1, len(times)):
            if geometry[0] is None:
                boundaryValsPrev =g(times[t-1])
                boundaryPrev = boundaryCoeffs * boundaryValsPrev
                boundaryVals = g(times[t])
                boundary = boundaryCoeffs * boundaryVals
                right = U[t - 1] + k / 2 * sparse.csc_matrix.dot(diffOperator, U[t - 1]) + k * f(U[t - 1]) + k / 2 * (boundaryPrev + boundary)
                U_temp = solveLeft(right)
                U[t] = U_temp + k / 2 * (f(U_temp) - f(U[t - 1]))
            else:
                boundaryValsPrev = (g(times[t-1])*np.ones(maxIndex))[geometry[0]]
                boundaryPrev = boundaryCoeffs*boundaryValsPrev
                boundaryValsFull = np.array(g(times[t]))*np.ones(maxIndex)
                boundaryVals = boundaryValsFull[geometry[0]]
                boundary = boundaryCoeffs * boundaryVals
                right = U[t - 1, geometry[0]] + k / 2 * sparse.csc_matrix.dot(diffOperator,  U[t - 1, geometry[0]]) + k * f(U[t - 1, geometry[0]]) + k/2*(boundaryPrev + boundary)
                U_temp = solveLeft(right)
                Uinter = U_temp + k / 2 * (f(U_temp) - f(U[t - 1,geometry[0]]))
                U[t] = FiniteDifference.applyBoundaryInterior(Uinter, geometry=geometry, boundaryVector=boundaryValsFull)
        return U, times

# Function used to sum the final result.
def sumList(UList):
    tempU = np.sum(UList,axis=-1)
    if np.ndim(tempU) == 1:
        return tempU
    else:
        return sumList(tempU)

class DiseaseModel():
    """
    Class for solving linear elliptic PDEs.
    """

    def __init__(self, getU0_S, getU0_I, muS, muI, schemeS, getBeta, getGamma, T=10, k=1, N=4,
                 isBoundaryFunction=None, dim=2, length=1, origin=0,
                 isNeumannFunc=None, schemeNeumannFunc=None, g=gDefault):
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

        AmatrixS, FinternalS, FboundaryS, self.geometryS = FiniteDifference.getSystem(shape=self.shapeObject,
                                                                                      f=fZero,
                                                                                      g=gOne,
                                                                                      scheme=schemeS,
                                                                                      isNeumannFunc=isNeumannFunc,
                                                                                      schemeNeumannFunc=schemeNeumannFunc,
                                                                                      interior=True)


        self.geometryI = self.geometryS
        self.Fboundary = np.concatenate((muS * FboundaryS, muI * FboundaryS))

        self.diffOperator = sparse.bmat([[AmatrixS * muS, None], [None, AmatrixS * muI]], format="csc")

        geometrySI = np.concatenate((self.geometryS, self.geometryI), axis=1)
        domainGeometrySI = FiniteDifference.getDomainGeometry(geometrySI)
        self.domainSize = len(domainGeometrySI[0])

        # Assuming R = 0 at t = 0
        if self.shapeObject.dim == 1:
            I_0 = Shape.getVectorLattice(getU0_I(Shape.getMeshGrid(self.shapeObject)), self.shapeObject)
            S_0 = Shape.getVectorLattice(getU0_S(Shape.getMeshGrid(self.shapeObject)), self.shapeObject)
        else:
            I_0 = Shape.getVectorLattice(getU0_I(*Shape.getMeshGrid(self.shapeObject)), self.shapeObject)
            S_0 = Shape.getVectorLattice(getU0_S(*Shape.getMeshGrid(self.shapeObject)), self.shapeObject)
        self.U_0 = np.concatenate((S_0, I_0))[np.logical_or(geometrySI[0], geometrySI[1])]

        diseaseModelF = DiseaseModel.getDiseaseModelF(getBeta, getGamma, self.shapeObject)

        self.UList, self.times = ModifiedCrankNicolson.modifiedCrankNicolsonSolver(self.U_0,
                                                                                   f=diseaseModelF,
                                                                                   T=T, k=k,
                                                                                   diffOperator=self.diffOperator,
                                                                                   boundaryCoeffs=-self.Fboundary,
                                                                                   geometry=geometrySI,
                                                                                   g = gDefault)

    def plot(self, timeIndex=0, ax=None, group="I", show=False, view=None, **kwargs):
        if group == "S":
            U = self.UList[int(timeIndex), :self.domainSize // 2]
            title = f"Susceptible at t = {timeIndex * self.k}"
        elif group == "I":
            U = self.UList[int(timeIndex), self.domainSize // 2:]
            title = f"Infected at t = {timeIndex * self.k}"
        else:
            print(f"Group: {group} not found")
            return None
        return Shape.plotOnShape(U, self.shapeObject, ax=ax, show=show,
                                 view=view, geometry=self.geometryS, title=title, **kwargs)

    def plotImage(self, timeIndex=0, ax=None, group="I", show=False, animated=False, **kwargs):
        if group == "S":
            U = self.UList[int(timeIndex), :self.domainSize // 2]

        elif group == "I":
            U = self.UList[int(timeIndex), self.domainSize // 2:]
        else:
            print(f"Group: {group} not found")
            return None
        max = np.nanmax(U)
        return Shape.plotImage2d(U, self.shapeObject, ax=ax, show=show, geometry=self.geometryS, animated=animated,
                                 **kwargs)

    def applyDiseaseAnimation(self, axS, axI, animationLength=10, **kvargs):
        artistlist = []
        fps = 24
        numFrames = animationLength*fps
        numTimes = self.UList.shape[0]
        step = numTimes//numFrames
        max = np.nanmax(self.UList)

        for timeIndex in range(0, len(self.times), step):
            S, I = self.UList[timeIndex, :self.domainSize // 2], self.UList[timeIndex, self.domainSize // 2:]

            artistS = Shape.plotImage2d(S, self.shapeObject, ax=axS, geometry=self.geometryS,
                                        title=f"Susceptible",
                                        animated=True, colorbar=False, **kvargs)

            artistI = Shape.plotImage2d(I, self.shapeObject, ax=axI, geometry=self.geometryI,
                                        title=f"Infected",
                                        animated=True, colorbar=False,  **kvargs)

            artistlist.append([artistS, artistI])
            if timeIndex == 0:
                axI.figure.colorbar(artistI, ax=[axS, axI])

        return animation.ArtistAnimation(axI.figure, artistlist, blit=True, interval=1000/fps)

    def getSolution(self):
        SList, IList = self.UList[:, :self.domainSize // 2], self.UList[:, self.domainSize // 2:]
        return SList, IList, self.times

    @staticmethod
    def DiseaseModelF(beta, gamma):
        # currying
        def F(U):
            S, I = np.split(U,2)
            return np.concatenate((-beta * S * I, beta * S * I - gamma * I))

        return F

    @staticmethod
    def getDiseaseModelF(getBeta, getGamma, shape):
        # currying
        if shape.dim == 1:
            betaLattice = getBeta(Shape.getMeshGrid(shape))
            gammaLattice = getGamma(Shape.getMeshGrid(shape))
        else:
            betaLattice = getBeta(*Shape.getMeshGrid(shape))
            gammaLattice = getGamma(*Shape.getMeshGrid(shape))

        if np.size(betaLattice) == 1:
            beta = betaLattice
        else:
            beta = Shape.getVectorLattice(betaLattice, shape)

        if np.size(gammaLattice) == 1:
            gamma = gammaLattice
        else:
            gamma = Shape.getVectorLattice(gammaLattice, shape)

        def F(U):
            S, I = np.split(U, 2)
            return np.concatenate((-beta * S * I, beta*S * I - gamma * I))

        return F



    def plotTotalInTime(self):
        SList, IList, times = self.getSolution()
        totPeople = np.sum(SList[0] + IList[0])
        #Assuming no people leaves the area
        totSList, totIList = sumList(SList), sumList(IList)
        totRList = totPeople - (totSList + totIList)

        fig, ax = plt.subplots(1)
        ax.plot(times, totSList/totPeople, label="Susceptible")
        ax.plot(times, totIList/totPeople, label="Infected")
        ax.plot(times, totRList/totPeople, label="Removed")

        ax.set_xlabel("Time")
        ax.set_ylabel("Fraction of people")
        ax.set_title("Disease spread over time")
        plt.legend()
        plt.show()



