
import BVP_solver as BVP


dim = 2
N = 10
tol = 10e-10
maxiter = 10
lam=1.5
#Not used
def ftest(X,u,ud):
    return 1/u**2


#Boundary contitions
def g(v):
    return 1

test = BVP.nonlinear_poisson(ftest,g,N)
test.plot()
test.summary()