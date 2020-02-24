import numpy as np
import matplotlib.pyplot as plt
import BVP_solver as BVP
import matplotlib.ticker as ticker


"""
Running this file will store  relevant plots (ca from problem 2 on your computer
"""

#Boundary contitions
def g(x):
    return 1

N = 50
tol = 1e-10
maxiter = 100
lam=1.5

maxList = []

lamList = [1.5,3,3.5]
#Solve and plot using solver from BVP_solver.py
fig = plt.figure(figsize=(18, 5))
for i in range(len(lamList)):
    ax = fig.add_subplot(1, len(lamList), i+1, projection='3d')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))

    membrane = BVP.nonlinear_poisson(g, N, maxIterNewton=maxiter, lam=lamList[i], length=1, tol= tol)
    membrane.plot(f"$\lambda = {lamList[i]}$", ax, zlabel="u")
    maxList.append([lamList[i],np.nanmax(np.abs(membrane.U-1))])
    print(maxList)
    plt.savefig("Oppgave_2_1.pdf")
    membrane.summary()





