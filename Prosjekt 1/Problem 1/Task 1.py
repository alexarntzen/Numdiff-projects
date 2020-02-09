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


# A class to hold the parameters of the BVP
# -mu Delta u + V * Nabla u = f, on (a,b)x(a,b), u = g on Boundary, with Dirichlet boundary conditions
class BVP(object):
    def __init__(self, f, V, Vb, gs, gn, gw, ge, a=0, b=1, mu=1, uexact=None):
        self.f = f       # Source function
        self.V = V       # function representing the vector on grid
        self.Vb = Vb     # function representing the vector on boundary
        self.gs = gs     # S boundary condition
        self.gn = gn     # N boundary condition
        self.gw = gw     # W boundary condition
        self.ge = ge     # E boundary condition
        self.a = a       # Interval
        self.b = b
        self.mu = mu    # constant mu
        self.uexact = uexact # The exact solution, if known.



def solve_BVP_and_plot(bvp, N, test, plot=True):
    #N, Number of intervals

    # Lag gitteret
    x = np.linspace(bvp.a, bvp.b, N + 1)
    y = x.copy()
    h = 1 / N
    muhh = -bvp.mu / (h * h)

    # Inner points
    xi = x[1:-1]
    yi = y[1:-1]

    Xi, Yi = np.meshgrid(xi, yi)
    Ni = N - 1       # Number of inner points in each direction
    Ni2 = Ni * Ni    # Number of inner points in total


    # Construct a sparse A-matrix

    # -mu * Delta u
    B = sparse.diags([1, -4, 1], [-1, 0, 1], shape=(Ni, Ni), format="lil")
    A = sparse.kron(sparse.eye(Ni), B)
    A += sparse.diags([1, 1], [-Ni, Ni], shape=(Ni2, Ni2), format="lil")
    A *= muhh

    #v * grad u, v represents the vector in the grid
    v = bvp.V(Xi, Yi, Ni, Ni2)
    C = sparse.diags([-v[0, 1:], v[0, :-1]], [-1, 1], shape=(Ni2, Ni2), format="lil")  # V_1 * (Ue - Uw)
    D = sparse.diags([-v[1, 1:], v[1, :-1]], [-Ni, Ni], shape=(Ni2, Ni2), format="lil")  # V_1 * (Un - Us)
    A += (C + D) / (2 * h)
    # finished A
    A = A.tocsr()  # Konverter til csr-format (nødvendig for spsolve)

    # Construct the right hand side for -mu * Delta u
    Db = np.zeros(Ni2)
    # Include the boundary conditions for -mu * Delta u
    Db[0:Ni] -= bvp.gs(xi)                         # y=0
    Db[Ni2-Ni:Ni2] -= bvp.gn(xi)                   # y=1
    Db[0:Ni2:Ni] -= bvp.gw(yi)                     # x=0
    Db[Ni-1:Ni2:Ni] -= bvp.ge(yi)                  # x=1
    Db *= muhh
    # Construct the right hand side for V * Nabla u
    Nb = np.zeros(Ni2)
    # Get vector representation for boundary
    vb = bvp.Vb(xi, yi, Ni, Ni2)
    # Include the boundary conditions for V * Nabla u
    Nb[0:Ni] += vb[1] * bvp.gs(xi)                         # y=0
    Nb[Ni2 - Ni:Ni2] -= vb[1] * bvp.gn(xi)                 # y=1
    Nb[0:Ni2:Ni] += vb[0] * bvp.gw(yi)                     # x=0
    Nb[Ni - 1:Ni2:Ni] -= vb[0] * bvp.ge(yi)                # x=1
    Nb *= 2 * h
    # Include the source funtions

    b = Db + Nb + bvp.f(Xi, Yi).flatten()

    # Solve the system.
    Ui = spsolve(A, b)

    # Make an array to store the solution
    U = np.zeros((N+1, N+1))

    # Reshape the Ui-vector to an array, and insert into Ui.
    U[1:-1, 1:-1] = np.reshape(Ui, (Ni, Ni))

    # Include the boundary conditions
    U[0, :] = bvp.gs(x)
    U[N, :] = bvp.gn(x)
    U[:, 0] = bvp.gw(y)
    U[:, N] = bvp.ge(y)

    if N == 4:
        # For inspection: Print A and b
        print('A =\n ', A.toarray())           # A konverteres til en full matrise først
        print('\nb=\n', b)

    X, Y = np.meshgrid(x, y)
    if plot:
        # Plot the solution
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, U, cmap=cm.coolwarm)  # Surface-plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Løsning av ligningen gitt i ' + test)
    try:
        U_exact = bvp.uexact(X, Y)
        # Print out the numerical solution and the error
        if N == 4:
            print('\nU=\n', U)
            print("N = ", N)
        print('The error is {:.2e}'.format(np.max(np.max(abs(U - U_exact)))))
        if plot:
            # Plot the exact-solution
            fig = plt.figure(2)
            X, Y = np.meshgrid(x, y)
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, U_exact, cmap=cm.coolwarm)  # Surface-plot
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Løsning av ligningen gitt i ' + test + ' exact')
            plt.show()

            # Plot the error
            fig = plt.figure(2)
            X, Y = np.meshgrid(x, y)
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, np.abs(U - U_exact), cmap=cm.coolwarm)  # Surface-plot
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('error  i ' + test)
            plt.show()
        return U, U_exact
    except:
        print("No excat solution given")
        return U, None


def convergence(bvp, P):
    # Measure the error for different stepsizes.
    # Require bvp.uexact to be set.
    # Number of different stepsizes
    Nconv = np.zeros(P)  #log(h) = log(1/N) = -log(N)
    Econv = np.zeros(P)
    N = 10  # The least number of intervals
    for p in range(P):
        U, U_exact = solve_BVP_and_plot(bvp, N, "", plot=False)
        Eh = U_exact-U
        Econv[p] = np.max(np.max(np.abs(Eh)))
        Nconv[p] = N
        N *= 2  # Double the number of intervals (or )
    order = np.polyfit(np.log(Nconv), np.log(Econv), 1)[0]   # Measure the order
    return Nconv, Econv, order

def plot_convergence(N, E, p):
    plt.loglog(N, E, 'o-', label='p={:.2f}'.format(p))
    """vet ikke denne helt"""
    #plt.loglog(N, (1) ** 2 * 7 / 16 * np.exp(1), '--', label='upper bound')
    plt.grid('on')
    plt.xlabel('N')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def TEST_0(N, P=4):
    print("------------------------------------------")
    print("Test 0, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: x**3
    gn = lambda x: 2+x**3
    gw = lambda y: 2*y**2
    ge = lambda y: 1+2*y**2
    f = lambda x, y: 6*x+4
    uexact = lambda x, y: x ** 3 + 2 * y ** 2

    def V(x, y, Ni, Ni2):
        return np.zeros(shape=(2, Ni, Ni)).reshape((2, Ni2))
    def Vb(x, y, Ni, Ni2):
        return np.zeros(shape=(2, Ni))

    test = BVP(f, V, Vb, gs, gn, gw, ge, mu=-1, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_0")
    print("Test convergence for TEST_0")
    Nconv, Econv, order = convergence(test, P=P)
    plot_convergence(Nconv, Econv, order)
    print("Convergence order: ", order)
    print("------------------------------------------")


def TEST_1(N, P=8):
    print("------------------------------------------")
    print("Test 1, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.sin(x) * np.sin(1)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.sin(1) * np.sin(y)
    f = lambda x, y: 2 * np.sin(x) * np.sin(y) + np.sin(x)*np.cos(y) + np.cos(x) * np.sin(y)
    uexact = lambda x, y: np.sin(x) * np.sin(y)


    def V(x, y, Ni, Ni2):
        v = np.ones(Ni2)
        return np.array([v, v])
    def Vb(x, y, Ni, Ni2):
        v = np.ones(Ni)
        return np.array([v, v])

    test = BVP(f, V, Vb, gs, gn, gw, ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_1")
    print("------------------------------------------")
    print("Test convergence for TEST_1")
    Nconv, Econv, order = convergence(test, P=P)
    plot_convergence(Nconv, Econv, order)
    print("Convergence order: ", order)
    print("------------------------------------------")









def V_d(x, y, Ni, Ni2):
    return np.array([y, -x]).reshape((2, Ni2))



TEST_0(10)
TEST_1(10, P=4)

