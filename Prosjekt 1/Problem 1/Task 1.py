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


# Define index mapping
def I(i, j, N):
    return j * (N + 1) + i


def get_Axy_square(bvp, N):
    # N, Number of intervals
    # Gridsize
    h = 1 / N
    # Total number of unknowns
    Ni = N + 1
    Ni2 = Ni * Ni
    # Make the grid
    x, y = np.ogrid[bvp.a:bvp.b:Ni * 1j, bvp.a:bvp.b:Ni * 1j]

    # Define zero matrix A of right size and insert 0
    A = sparse.dok_matrix((Ni2, Ni2))
    # Define FD entries of A
    muhh = - bvp.mu / (h * h)
    h2 = 1 / (2 * h)

    # get V
    v1, v2 = bvp.V(x, y)
    v1 = v1.ravel()
    v2 = v2.ravel()
    for i in range(1, N):
        for j in range(1, N):
            index = I(i, j, N)
            A[index, index] = -4 * muhh  # U_ij, U_p
            A[index, index - N - 1] = muhh - v2[i] * h2  # U_{i,j-1}, U_s
            A[index, index + N + 1] = muhh + v2[i] * h2  # U_{i,j+1}, U_n
            A[index, index - 1] = muhh - v1[j] * h2  # U_{i-1,j}, U_w
            A[index, index + 1] = muhh + v1[j] * h2  # U_{i+1,j}, U_e

        # Incorporate boundary conditions
        # Add boundary values related to unknowns from the first and last grid ROW
        for j in [0, N]:
            for i in range(0, Ni):
                index = I(i, j, N)
                A[index, index] = 1

        # Add boundary values related to unknowns from the first and last grid COLUMN
        for i in [0, N]:
            for j in range(0, Ni):
                index = I(i, j, N)
                A[index, index] = 1

    return A, x, y

# A class to hold the parameters of the BVP
# -mu div grad u + V * grad u = f, on (a,b)x(a,b), u = g on Boundary, with Dirichlet boundary conditions
class BVP(object):
    def __init__(self, f, V, gs, gn, gw, ge, a=0, b=1, mu=1, uexact=None, I=I, get_Axy=get_Axy_square):
        self.f = f       # Source function
        self.V = V       # function representing the vector
        self.gs = gs     # S boundary condition
        self.gn = gn     # N boundary condition
        self.gw = gw     # W boundary condition
        self.ge = ge     # E boundary condition
        self.a = a       # Interval
        self.b = b
        self.mu = mu    # constant mu
        self.uexact = uexact  # The exact solution, if known.
        self.I = I      # index mapping
        self.get_Axy = get_Axy # function to get matrix A and grid x,y



def apply_bcs(F, G, N, I):
    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = np.array([I(i, j, N) for j in [0, N] for i in range(0, N + 1)])
    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = np.array([I(i, j, N) for i in [0, N] for j in range(0, N + 1)])
    F[bc_indices] = G[bc_indices]

    return F


def g(x, y, bvp, N):
    G = np.zeros((N+1, N+1))
    G[:, 0] = bvp.gs(x).ravel()
    G[:, -1] = bvp.gn(x).ravel()
    G[0, :] = bvp.gw(y).ravel()
    G[-1, :] = bvp.ge(y).ravel()
    return G



def solve_BVP_and_plot(bvp, N, test, plot=True, neumann=True):
    # Make grid and matrix
    A, x, y = bvp.get_Axy(bvp, N)
    F = bvp.f(x, y).ravel()

    if neumann:
        G = g(x, y, bvp, N).ravel()
    else:
        G = np.zeros_like(F)  # placeholder

    # Apply bcs
    F = apply_bcs(F, G, N, I)

    # Solve
    A_csr = A.tocsr()
    U = spsolve(A_csr, F).reshape((N+1, N+1))

    if plot:
        plot2D(x, y, U, "Numerical solution of " + test)

    try:
        U_exact = bvp.uexact(x, y)
        err = np.abs(U - U_exact)
        print('The error is {:.2e}'.format(np.max(np.max(err))) + ", N = " + str(N))

        if plot:
            plot2D(x, y, U_exact, "Exact solution of " + test)
            plot2D(x, y, err, "Error of " + test)

        return err

    except:
        print("No excat solution given")
        return None



def plot2D(X, Y, Z, title=""):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
    # Set initial view angle
    ax.view_init(30, 225)

    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()


def convergence(bvp, P):
    # Measure the error for different stepsizes.
    # Require bvp.uexact to be set.
    # Number of different stepsizes
    Hconv = np.zeros(P)  #log(h) = log(1/N) = -log(N)
    Econv = np.zeros(P)
    N = 10  # The least number of intervals
    for p in range(P):
        Eh = solve_BVP_and_plot(bvp, N, "", plot=False)
        Econv[p] = np.max(np.max(Eh))
        Hconv[p] = 1 / N
        N *= 2  # Double the number of intervals (or )
    order = np.polyfit(np.log(Hconv), np.log(Econv), 1)[0]   # Measure the order
    return Hconv, Econv, order

def plot_convergence(H, E, p):
    plt.loglog(H, E, 'o-', label='p={:.2f}'.format(p))
    """vet ikke denne helt"""
    #plt.loglog(N, (1) ** 2 * 7 / 16 * np.exp(1), '--', label='upper bound')
    plt.grid('on')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def TEST_0(N, P=4): #As given in exercise
    print("------------------------------------------")
    print("Test 0, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: x**3
    gn = lambda x: 2+x**3
    gw = lambda y: 2*y**2
    ge = lambda y: 1+2*y**2
    f = lambda x, y: 6*x+4 + 0*y
    uexact = lambda x, y: x ** 3 + 2 * y ** 2

    def V(x, y):
        n = len(x)
        return np.zeros(n), np.zeros(n)

    test = BVP(f, V, gs, gn, gw, ge, mu=-1, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_0")
    print("------------------------------------------")
    #print("Test convergence for TEST_0")
    #Hconv, Econv, order = convergence(test, P=P)
    #plot_convergence(Hconv, Econv, order)
    #print("Convergence order: ", order)
    #print("------------------------------------------")


def TEST_1(N, P=4):
    print("------------------------------------------")
    print("Test 1, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.sin(x) * np.sin(1)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.sin(1) * np.sin(y)
    f = lambda x, y: 2 * np.sin(x) * np.sin(y)
    uexact = lambda x, y: np.sin(x) * np.sin(y)

    def V(x, y):
        n = len(x)
        return np.zeros(n), np.zeros(n)

    test = BVP(f, V, gs, gn, gw, ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_1")
    print("------------------------------------------")
    print("Test convergence for TEST_1")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_2(N, P=4):
    print("------------------------------------------")
    print("Test 2, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.sin(x) * np.sin(1)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.sin(1) * np.sin(y)
    f = lambda x, y: 2 * np.sin(x) * np.sin(y) + np.sin(x) * np.cos(y) + np.cos(x) * np.sin(y)
    uexact = lambda x, y: np.sin(x) * np.sin(y)


    def V(x, y):
        n = len(x)
        return np.ones(n), np.ones(n)

    test = BVP(f, V, gs, gn, gw, ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_2")
    print("------------------------------------------")
    print("Test convergence for TEST_2")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_3(N, P=4):
    print("------------------------------------------")
    print("Test 3, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: 5 * np.pi * np.pi * np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     + np.pi * np.cos(1 * np.pi * x) * np.sin(2 * np.pi * y) + 2 * np.pi * np.sin(1 * np.pi * x) * np.cos(2 * np.pi * y)
    uexact = lambda x, y: np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y)


    def V(x, y):
        n = len(x)
        return np.ones(n), np.ones(n)

    test = BVP(f, V, gs, gn, gw, ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_3")
    print("------------------------------------------")
    print("Test convergence for TEST_3")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def Task_1d(N, P=4):
    print("------------------------------------------")
    print("Task 1d, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: np.ones((len(x), len(x)))

    def V(x, y):
        return y, -x

    test = BVP(f, V, gs, gn, gw, ge, mu=1e-2)
    solve_BVP_and_plot(test, N, "Task 1d")
    print("------------------------------------------")





TEST_0(4)
TEST_1(10)
TEST_2(10)
TEST_3(10)
Task_1d(100)
