import matplotlib.pyplot as plt
import BVP_solver as BVP

"""
Running this file will store  relevant plots (ca from problem 2 on your computer
"""

#Boundary contitions
def g(x):
    return 1


N = 40
tol = 10e-5
maxiter = 100
lam=1.5

lamList = [1.5,2,3]

#Solve and plot using solver from BVP_solver.py
fig = plt.figure(figsize=(18, 5))
for i in range(len(lamList)):
    print(i)
    ax = fig.add_subplot(1, len(lamList), i+1, projection='3d')
    membrane = BVP.nonlinear_poisson(g, N, maxIterNewton=maxiter, lam=lamList[i], length=1, tol= tol)
    membrane.plot(f"$N = {lamList[i]}$",ax,zlabel="u")
    plt.savefig("Oppgave_2.pdf")





