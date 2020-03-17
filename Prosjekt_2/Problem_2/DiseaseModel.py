import numpy as np
import scipy.sparse as sparse  # Sparse matrices
import scipy.sparse.linalg as splin  # Sparse matrices


from FiniteDifference import Shape, FiniteDifference

# Default functions for not Neumann conditions
def isNeumannFalse(position):
    return False

def schemeNeumannDefault(position):
    return 0, 0, 0


def fDefault(x):
    return 0

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
        isBoundaryList = FiniteDifference.getDomainGeometry(geometry)[1]
        U_0 = Shape.makeGetVector(shape)(getU0(*Shape.getMeshGrid(shape)))

        return ModifiedCrankNicolson.modifiedCrankNicolsonSolver(U_0, fDeriv, T=T, k=k, diffOperator=Amatrix, Fb=Fboundary,
                                           isBoundaryList=isBoundaryList)

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
    def __init__(self, g,getU0_S, getU0_I,muS,muI,schemeS=None, beta=1, gamma=1,T=10, k=1, N=4, isBoundaryFunction=None, dim=2, length=1, origin=0,
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

        AmatrixI, _I, FboundaryI, self.geometryI = FiniteDifference.getSystem(shape=self.shapeObject,
                                                                     f=fDefault,
                                                                     g=g,
                                                                     scheme=schemeS,
                                                                     isNeumannFunc=isNeumannFunc,
                                                                     schemeNeumannFunc=schemeNeumannFunc)

        self.Fboundary = np.concatenate((FboundaryS,FboundaryI))


        self.diffOperator = sparse.bmat([[AmatrixS*muS, None],[None, AmatrixI*muI]],format = "lil")

        geometrySI = np.concatenate((self.geometryS,self.geometryI),axis=1)
        domainIndexs = np.flatnonzero(np.logical_or(geometrySI[0], geometrySI[1]))
        domainGeometry = geometrySI[:, domainIndexs]

        #Assuming R = 0 at t = 0
        I_0 = Shape.makeGetVector(self.shapeObject)(getU0_I(*Shape.getMeshGrid(self.shapeObject)))
        S_0 =  Shape.makeGetVector(self.shapeObject)(getU0_S(*Shape.getMeshGrid(self.shapeObject)))
        self.U_0 = np.concatenate((S_0,I_0))

        self.UList, self.times = ModifiedCrankNicolson.modifiedCrankNicolsonSolver(self.U_0, f = DiseaseModel.DiseaseModelF(beta, gamma), T = T, k = k, diffOperator=self.diffOperator, Fb=self.Fboundary, isBoundaryList=domainGeometry[1])

    def plotS(self, time=0, title=None, ax=None, zlabel='$u(x,y)$', show=False, view=None, ulabel="U"):
        timeIndex = int(time//self.k)
        U = self.UList[timeIndex,:self.shapeObject.maxIndex]
        Shape.plotOnShape(U,self.shapeObject,title=title, ax=ax,  show=show, view=view, zlabel="Relative number of people", ulabel="S",geometry=self.geometryS)

    def plotI(self,time=0,title=None, ax=None, show=False, view=None):
        timeIndex = int(time//self.k)
        U = self.UList[timeIndex,self.shapeObject.maxIndex:]
        Shape.plotOnShape(U,self.shapeObject,title=title, ax=ax,  show=show, view=view, zlabel="Relative number of people", ulabel="I",geometry=self.geometryI)

    def getSolution(self):
        SList, IList =  self.UList[:,:self.shapeObject.maxIndex], self.UList[:,self.shapeObject.maxIndex:]
        return SList, IList,  self.times

    @staticmethod
    def DiseaseModelF(beta, gamma):
        # currying
        def F(U):
            S = U[:len(U) // 2]
            I = U[len(U) // 2:]
            return np.concatenate((-beta * S * I, S * I - gamma * I))

        return F