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


def get_Axy(bvp, N):
    # N, Number of intervals
    # Gridsize
    h = (bvp.b - bvp.a) / N
    # Total number of unknowns
    N2 = (N + 1) * (N + 1)
    # Make the grid
    y, x = np.ogrid[bvp.a:bvp.b:(N + 1) * 1j, bvp.a:bvp.b:(N + 1) * 1j]

    # Define zero matrix A of right size and insert 0
    A = sparse.dok_matrix((N2, N2))
    # Define FD entries of A
    muhh = - bvp.mu / (h * h)
    h2 = 1 / (2 * h)

    for i in range(1, N):
        for j in range(1, N):
            index = bvp.I(i, j, N)
            v1, v2 = bvp.V(x[0, i], y[j, 0])
            A[index, index] = - 4 * muhh  # U_ij, U_p
            A[index, index - N - 1] = muhh - v2 * h2  # U_{i,j-1}, U_s
            A[index, index + N + 1] = muhh + v2 * h2  # U_{i,j+1}, U_n
            A[index, index - 1] = muhh - v1 * h2  # U_{i-1,j}, U_w
            A[index, index + 1] = muhh + v1 * h2  # U_{i+1,j}, U_e

    # Incorporate boundary conditions
    A = bvp.BC(A, N, bvp)
    return A, x, y


def get_Axy_square_longer_step_1d(bvp, N):
    # N, Number of intervals
    # Gridsize
    h = (bvp.b - bvp.a) / N
    # Total number of unknowns
    N2 = (N + 1) * (N + 1)
    # Make the grid
    y, x = np.ogrid[bvp.a:bvp.b:(N + 1) * 1j, bvp.a:bvp.b:(N + 1) * 1j]

    # Define zero matrix A of right size and insert 0
    A = sparse.dok_matrix((N2, N2))
    # Define FD entries of A
    muhh = - bvp.mu / (h * h)

    for i in range(1, N):
        for j in range(1, N):
            index = bvp.I(i, j, N)
            v1, v2 = bvp.V(x[0, i], y[j, 0])
            A[index, index] = - 4 * muhh - v2 / h + v1 / h  # U_ij, U_p
            A[index, index - N - 1] = muhh   # U_{i,j-1}, U_s
            A[index, index + N + 1] = muhh + v2 / h  # U_{i,j+1}, U_n
            A[index, index - 1] = muhh - v1 / h  # U_{i-1,j}, U_w
            A[index, index + 1] = muhh  # U_{i+1,j}, U_e

    # Incorporate boundary conditions
    A = bvp.BC(A, N, bvp)
    return A, x, y




def BC_square(A, N, bvp):
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0, N]:
        for i in range(0, N + 1):
            index = bvp.I(i, j, N)
            A[index, index] = 1

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0, N]:
        for j in range(0, N + 1):
            index = bvp.I(i, j, N)
            A[index, index] = 1

    return A


def BC_circle_quadrant(A, N, bvp):
    # Add boundary values for the square first
    A = BC_square(A, N, bvp)
    # N, Number of intervals
    # Gridsize
    h = (bvp.b - bvp.a) / N
    # Total number of unknowns
    N2 = (N + 1) * (N + 1)
    # Make the grid
    y, x = np.ogrid[bvp.a:bvp.b:(N + 1) * 1j, bvp.a:bvp.b:(N + 1) * 1j]

    # Define FD entries of A
    muhh = - bvp.mu / (h * h)
    h1 = 1 / h
    h2 = 1 / (2 *h)

    for i in range(1, N):
        for j in range(1, N):
            # clean up, set everything to zero
            if x[0, i] ** 2 + y[j, 0] ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                v1, v2 = bvp.V(x[0, i], y[j, 0])
                A[index, index] = 0  # U_ij
                # clean up
                A[index, index - N - 1] = 0  # U_{i,j-1}, U_s
                A[index, index + N + 1] = 0  # U_{i,j+1}, U_n
                A[index, index - 1] = 0  # U_{i-1,j}, U_w
                A[index, index + 1] = 0  # U_{i+1,j}, U_e

    # for x, by p2(x)
    for i in range(1, N):
        for j in range(1, N):
            xp = x[0, i]
            yp = y[j, 0]
            if (xp + h) ** 2 + yp ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                v1, v2 = bvp.V(xp, yp)
                rho = (np.sqrt(1 - yp ** 2) - xp) / h
                A[index, index] += - muhh / rho + v2 * h1  # U_ij, U_p
                A[index, index - N - 1] += 0   # U_{i,j-1}, U_s
                A[index, index + N + 1] += 0 # U_{i,j+1}, U_n
                A[index, index - 1] += - muhh / rho + muhh / (rho * (1 + rho))  # U_{i-1,j}, U_w
                A[index, index + 1] += muhh / (rho * (1 + rho)) - v2 * h1   # U_{i+1,j}, U_e

    # for y, by p2(y)
    for i in range(1, N):
        for j in range(1, N):
            xp = x[0, i]
            yp = y[j, 0]
            if (yp + h) ** 2 + xp ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                v1, v2 = bvp.V(xp, yp)
                rho = (np.sqrt(1 - xp ** 2) - yp) / h
                A[index, index] += - muhh / rho + v1 * h1  # U_ij, U_p
                A[index, index - N - 1] += - muhh / rho + muhh / (rho * (1 + rho))  # U_{i,j-1}, U_s
                A[index, index + N + 1] += muhh / (rho * (1 + rho)) - v1 * h1    # U_{i,j+1}, U_n
                A[index, index - 1] += 0  # U_{i-1,j}, U_w
                A[index, index + 1] += 0     # U_{i+1,j}, U_e

    # for x, by p2(x)
    for i in range(1, N):
        for j in range(1, N):
            xp = x[0, i]
            yp = y[j, 0]
            if (xp + h) ** 2 + yp ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                v1, v2 = bvp.V(xp, yp)
                rho = (np.sqrt(1 - yp ** 2) - xp) / h
                if A[index, index] == - muhh / rho + v2 * h1:
                    A[index, index] += - 2 * muhh
                if A[index, index - 1] == 0:
                    A[index, index - 1] += muhh - v1 * h2  # U_{i-1,j}, U_w
                if A[index, index + 1] == 0:
                    A[index, index + 1] += muhh + v1 * h2  # U_{i+1,j}, U_e

    # for y, by p2(y), note y is "x"
    for i in range(1, N):
        for j in range(1, N):
            xp = x[0, i]
            yp = y[j, 0]
            if (yp + h) ** 2 + xp ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                v1, v2 = bvp.V(xp, yp)
                rho = (np.sqrt(1 - xp ** 2) - yp) / h
                if A[index, index] == - muhh / rho + v1 * h1:
                    A[index, index] += - 2 * muhh
                if A[index, index - N - 1] == 0:
                    A[index, index - N - 1] += muhh - v2 * h2   # U_{i,j-1}, U_s
                if A[index, index + N + 1] == 0:
                    A[index, index + N + 1] += muhh + v2 * h2  # U_{i,j+1}, U_n


    for i in range(1, N):
        for j in range(1, N):
            if x[0, i] ** 2 + y[j, 0] ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                A[index, index] = 1  # U_ij
                # clean up
                A[index, index - N - 1] = 0  # U_{i,j-1}, U_s
                A[index, index + N + 1] = 0  # U_{i,j+1}, U_n
                A[index, index - 1] = 0  # U_{i-1,j}, U_w
                A[index, index + 1] = 0  # U_{i+1,j}, U_e

    return A


def BC_square_neumann(A, N, bvp):

    h = 1 / N
    muhh = - bvp.mu / (h * h)
    h2 = 1 / (2 * h)
    N2 = (N + 1) * (N + 1)
    y, x = np.ogrid[bvp.a:bvp.b:(N + 1) * 1j, bvp.a:bvp.b:(N + 1) * 1j]

    # y = 1, North, Neumann
    for i in range(0, N + 1):
        index = bvp.I(i, N, N)
        v1, v2 = bvp.V(x[0, i], y[N, 0])
        A[index, index] = - 4 * muhh - v1 * h2  # U_ij, U_p
        # test for out of bounds
        if index - 1 >= bvp.I(0, N, N):
            A[index, index - 1] = 2 * muhh - v1 * h2  # U_{i-1,j}, U_w
        if index - N - 1 >= 0:
            A[index, index - N - 1] = 2 * muhh  # U_{i,j-1}, U_s
        if index + N + 1 <= N2 - 1:
            A[index, index + 1] = muhh - v1 * h2  # U_{i,j+1}, U_e

    # x = 0, west Dirichlet
    for j in range(0, N + 1):
        index = bvp.I(0, j, N)
        A[index, index] = 1
        # clean up
        if index + 1 <= bvp.I(N, j, N):
            A[index, index + 1] = 0
        if index - N - 1 >= 0:
            A[index, index - N - 1] = 0
        if index + N + 1 <= N2 - 1:
            A[index, index + N + 1] = 0

    # y = 0, south
    for i in range(0, N + 1):
        index = bvp.I(i, 0, N)
        v1, v2 = bvp.V(x[0, i], y[0, 0])
        A[index, index] = 1  # U_ij, U_p
        # clean up
        if index - 1 >= 0:
            A[index, index - 1] = 0  # U_{i-1,j}, U_w
        if index + N + 1 <= N2 - 1:
            A[index, index + N + 1] = 0  # U_{i,j+1}, U_n
        if index + 1 <= N + 1:
            A[index, index + 1] = 0  # U_{i+1,j}, U_e

    # x = 1, east Dirichlet
    for j in range(0, N + 1):
        index = bvp.I(N, j, N)
        A[index, index] = 1
        # clean up
        if index + N + 1 <= N2 - 1:
            A[index, index + 1] = 0
        if index - N - 1 >= 0:
            A[index, index - N - 1] = 0
        if index - 1 >= 0:
            A[index, index - 1] = 0
    return A


def apply_bcs_square(F, G, N, bvp):
    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = np.array([bvp.I(i, j, N) for j in [0, N] for i in range(0, N + 1)])
    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = np.array([bvp.I(i, j, N) for i in [0, N] for j in range(0, N + 1)])
    F[bc_indices] = G[bc_indices]
    return F


def apply_bsc_circle_quadrant(F, G, N, bvp):
    # Add boundary values for the square first
    F = apply_bcs_square(F, G, N, bvp)
    # Make the grid
    y, x = np.ogrid[bvp.a:bvp.b:(N + 1) * 1j, bvp.a:bvp.b:(N + 1) * 1j]
    for i in range(1, N):
        for j in range(1, N):
            if x[0, i] ** 2 + y[j, 0] ** 2 >= bvp.c:
                index = bvp.I(i, j, N)
                F[index] = G[index]

    return F



def G_square(x, y, bvp, N):
    G = np.zeros((N+1, N+1))
    G[0, :] = bvp.gs(x).ravel()
    G[-1, :] = bvp.gn(x).ravel()
    G[:, 0] = bvp.gw(y).ravel()
    G[:, -1] = bvp.ge(y).ravel()
    return G


def G_circle_quadrant(x, y, bvp, N):
    # ge and gn may not exist
    # gw and gs exist, so do them, first
    G = sparse.dok_matrix((N + 1, N + 1))
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            if x[0, i] ** 2 + y[j, 0] ** 2 >= bvp.c:
                G[i, j] = bvp.gc(x[0, i], y[j, 0])
    G = G.toarray()
    G[0, :] = bvp.gs(x).ravel()
    G[:, 0] = bvp.gw(y).ravel()

    return G


def G_square_neumann(x, y, bvp, N):
    h = 1 / N
    G = np.zeros((N + 1, N + 1))
    F = bvp.f(x, y)
    # east  neumann
    v1, v2 = bvp.V(x[0, :], y[N, 0])
    G[-1, :] = F[N, :].ravel() + (2 * bvp.mu / h - v2) * bvp.gn(x).ravel()
    # west, north dirichlet
    G[:, -1] = bvp.ge(y).ravel()
    G[:, 0] = bvp.gw(y).ravel()
    G[0, :] = bvp.gs(x).ravel()
    return G


# A class to hold the parameters of the BVP
# -mu div grad u + V * grad u = f, on (a,b)x(a,b), u = g on Boundary, with Dirichlet boundary conditions
class BVP(object):
    def __init__(self, f, V, gs=None, gn=None, gw=None, ge=None, a=0, b=1, mu=1, uexact=None, I=I, get_Axy=get_Axy,
                 BC=BC_square, apply_bcs=apply_bcs_square, G=G_square, c=1, gc=None):
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
        self.get_Axy = get_Axy  # Get matrix A and x, y
        self.BC = BC  # apply boundary condition in matrix
        self.apply_bcs = apply_bcs  # apply boundary in F
        self.G = G  # Boundary condition for F.
        self.c = c # constant for circle quadrant boundary condition, i.e. x^2 + y^2 = c
        self.gc = gc # Boundary condition for circle quadrant boundary condition, i.e. x^2 + y^2 = c


def solve_BVP_and_plot(bvp, N, test, plot=True, view=225):
    # Make grid and matrix
    A, x, y = get_Axy(bvp, N)
    F = bvp.f(x, y).ravel()

    G = bvp.G(x, y, bvp, N).ravel()
    # Apply bcs
    F = bvp.apply_bcs(F, G, N, bvp)
    # Solve
    A_csr = A.tocsr()
    U = spsolve(A_csr, F).reshape((N+1, N+1))

    if plot:
        plot2D(x, y, U, "Numerical solution of " + test, view=view)
    try:
        U_exact = bvp.uexact(x, y)
        err = np.abs(U - U_exact)
        print('The error is {:.2e}'.format(np.max(np.max(err))) + ", N = " + str(N))

        if plot:
            plot2D(x, y, U_exact, "Exact solution of " + test, view=view)
            plot2D(x, y, err, "Error of " + test, view=view)
        return err

    except:
        print("No excat solution given")
        return None



def plot2D(X, Y, Z, title="", view=225):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
    # Set initial view angle
    ax.view_init(30, view)
    """In plots and code, x and y are iterchanged, meaningig x_plot=y and y_plot=x"""
    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u(x,y)$')
    ax.set_title(title)


def convergence(bvp, P, N=10):
    # Measure the error for different stepsizes.
    # Require bvp.uexact to be set.
    # Number of different stepsizes
    Hconv = np.zeros(P)  #log(h) = log(1/N) = -log(N)
    Econv = np.zeros(P)
    for p in range(P):
        Eh = solve_BVP_and_plot(bvp, N, "", plot=False)
        Econv[p] = np.max(np.max(Eh))
        Hconv[p] = 1 / N
        N *= 2  # Double the number of intervals
    order = np.polyfit(np.log(Hconv), np.log(Econv), 1)[0]   # Measure the order
    return Hconv, Econv, order

def plot_convergence(H, E, p):
    plt.figure()
    plt.loglog(H, E, 'o-', label='p={:.2f}'.format(p))
    #plt.loglog(H, ..., '--', label='upper bound')
    plt.grid('on')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.legend()


def TEST_0(N, P=4): #As given in exercise, to see that boundary cond. is implemented rigth
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
        return 0, 0

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, mu=-1, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_0")
    print("------------------------------------------")


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
        return 0, 0

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
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
        return 1, 1

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_2")
    print("------------------------------------------")
    print("Test convergence for TEST_2")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_3(N, P=4, c1=1, c2=1):
    print("------------------------------------------")
    print("Test 3, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: 5 * np.pi * np.pi * np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     + c1 * np.pi * np.cos(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     + c2 * 2 * np.pi * np.sin(1 * np.pi * x) * np.cos(2 * np.pi * y)
    uexact = lambda x, y: np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y)
    def V(x, y):
        return c1, c2

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_3")
    print("------------------------------------------")
    print("Test convergence for TEST_3")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")

def TEST_3_1c(N, P=4, c1=1, c2=1):
    print("------------------------------------------")
    print("Test 3 1c, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: 1 - x * x
    gw = lambda y: 1 - y * y
    gc = lambda x, y: 0
    f = lambda x, y: 4 - c1 * 2 * x - c2 * 2 * y
    def uexact(x, y):
        n = np.shape(x)[1]
        uexact = sparse.dok_matrix((n, n))
        for i in range(0, n):
            for j in range(0, n):
                if x[0, i] ** 2 + y[j, 0] ** 2 >= 1:
                    uexact[i, j] = 0
                else:
                    uexact[i, j] = 1 - x[0, i] ** 2 - y[j, 0] ** 2

        return uexact.toarray()

    def V(x, y):
        return c1, c2

    test = BVP(f, V, gs=gs, gw=gw, gc=gc, uexact=uexact, BC=BC_circle_quadrant, apply_bcs=apply_bsc_circle_quadrant, G=G_circle_quadrant)
    solve_BVP_and_plot(test, N, "TEST_3 1c", view=45)
    print("------------------------------------------")
    print("Test convergence for TEST_3 1c")
    Hconv, Econv, order = convergence(test, P=P, N=10)
    plot_convergence(Hconv, Econv, order)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_4_n(N, P=4, c1=1, c2=1):
    print("------------------------------------------")
    print("Test 3 n, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: x ** 2  # Dirichlet, y = 0
    gn = lambda x: 2 * np.ones_like(x)  # Neumann, y = 1
    gw = lambda y: y ** 2  # Dirichlet, x = 0
    ge = lambda y: y ** 2 + 1  # Dirichlet, x = 1
    f = lambda x, y: - 4 + c1 * 2 * x + c2 * 2 * y
    uexact = lambda x, y: x ** 2 + y ** 2
    def V(x, y):
        return c1, c2

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact, BC=BC_square_neumann, G=G_square_neumann)
    solve_BVP_and_plot(test, N, "TEST_4 n")
    print("------------------------------------------")
    print("Test convergence for TEST_4 n")
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
    f = lambda x, y: np.ones((np.shape(x)[1], np.shape(y)[0]))

    def V(x, y):
        return y, -x

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, mu=1e-2)
    solve_BVP_and_plot(test, N, "Task 1d", view=255)
    print("------------------------------------------")


def Task_1d_neumann(N, P=4):
    print("------------------------------------------")
    print("Task 1d n, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: np.ones((np.shape(x)[1], np.shape(y)[0]))

    def V(x, y):
        return y, -x

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, mu=1e-2, BC=BC_square_neumann, G=G_square_neumann)
    solve_BVP_and_plot(test, N, "Task 1d n", view=255)
    print("------------------------------------------")


def Task_1d_long_step(N, P=4):
    print("------------------------------------------")
    print("Task 1d long step, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: np.ones((np.shape(x)[1], np.shape(y)[0]))

    def V(x, y):
        return y, -x

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, mu=1e-2, get_Axy=get_Axy_square_longer_step_1d)
    solve_BVP_and_plot(test, N, "Task 1d long step", view=255)
    print("------------------------------------------")





#TEST_0(4)
#TEST_1(10)
#TEST_2(10)
#TEST_3(100, c1=1, c2=-1)
#TEST_3_1c(10)
TEST_4_n(20)
#Task_1d(100)
#Task_1d_neumann(100)
#Task_1d(30)
#Task_1d_long_step(30)



plt.show()