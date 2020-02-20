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
from matplotlib import ticker



"""Functions used in class"""
# Define index mapping
def I(i, j, N):
    return j * (N + 1) + i

# Normal function to get the matrix A, for convergence of order 2, also returns x, y for the grid
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
    # some constans
    muhh = - bvp.mu / (h * h)
    h2 = 1 / (2 * h)

    # set in the inner values
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
    return A, x, y

# Function to get the matrix A for the circle quadrant, order 1 for small h/ big N, also returns x, y
def get_Axy_circle_quadrant(bvp, N):
    # N, Number of intervals
    # Gridsize
    h = (bvp.b - bvp.a) / N
    # Total number of unknowns
    N2 = (N + 1) * (N + 1)
    # Make the grid
    y, x = np.ogrid[bvp.a:bvp.b:(N + 1) * 1j, bvp.a:bvp.b:(N + 1) * 1j]
    # Define zero matrix A of right size and insert 0
    A = sparse.dok_matrix((N2, N2))
    # some constants
    muhh = - bvp.mu / (h * h)
    h1 = 1 / h
    h2 = 1 / (2 * h)

    # Matrix to store what kind of point we are working on.
    P = sparse.dok_matrix((N+1, N+1))
    # 0: normal inner point, or point outside of the circle quadrant
    # 1: The point to the east is on the circle quadrant, but not the north point
    # 2: The point to the north is on the circle quadrant, but not the east point
    # 3: Both the point to the east and the north are on the circle quadrant
    # classify the point like given over
    for i in range(1, N):
        for j in range(1, N):
            xp = x[0, i]
            yp = y[j, 0]
            if (xp + h) ** 2 + yp ** 2 >= bvp.c and xp ** 2 + (yp + h) ** 2 < bvp.c:
                P[i, j] = 1
            if xp ** 2 + (yp + h) ** 2 >= bvp.c and (xp + h) ** 2 + yp ** 2 < bvp.c:
                P[i, j] = 2
            if (xp + h) ** 2 + (yp + h) ** 2 >= bvp.c and \
                    (xp + h) ** 2 + yp ** 2 < bvp.c and xp ** 2 + (yp + h) ** 2 < bvp.c:
                P[i, j] = 3

    # Give the points "there values" according to P matrix
    for i in range(1, N):
        for j in range(1, N):
            index = bvp.I(i, j, N)
            xp = x[0, i]
            yp = y[j, 0]
            v1, v2 = bvp.V(xp, yp)
            if P[i, j] == 0:
                A[index, index] = - 4 * muhh  # U_ij, U_p
                A[index, index - N - 1] = muhh - v2 * h2  # U_{i,j-1}, U_s
                A[index, index + N + 1] = muhh + v2 * h2  # U_{i,j+1}, U_n
                A[index, index - 1] = muhh - v1 * h2  # U_{i-1,j}, U_w
                A[index, index + 1] = muhh + v1 * h2  # U_{i+1,j}, U_e
            if P[i, j] == 1:
                rho = (np.sqrt(bvp.c - yp ** 2) - xp) / h
                A[index, index] = - 2 * muhh - 2 * muhh / rho \
                                  - (2 * xp / (rho * h * h) + h1 / rho - h1) * v1  # U_ij, U_p
                A[index, index - 1] = 2 * muhh / (1 + rho) \
                                       + (2 * xp / (h * h * (rho + 1)) - h1 * rho / (rho + 1)) * v1  # U_{i-1,j}, U_w
                A[index, index + 1] = 2 * muhh / (rho * (rho + 1)) \
                                       + (2 * xp / (h * h * rho * (rho + 1)) + h1 / (
                            rho * (rho + 1))) * v1  # U_{i+1,j}, U_e
                A[index, index - N - 1] = muhh - v2 * h2  # U_{i,j-1}, U_s
                A[index, index + N + 1] = muhh + v2 * h2  # U_{i,j+1}, U_n
            if P[i, j] == 2:
                eta = (np.sqrt(bvp.c - xp ** 2) - yp) / h
                A[index, index] += - 2 * muhh - 2 * muhh / eta \
                                   - (2 * yp / (eta * h * h) + h1 / eta - h1) * v2  # U_ij, U_p
                A[index, index - N - 1] += 2 * muhh / (1 + eta) \
                                           + (2 * yp / (h * h * (eta + 1)) - h1 * eta / (
                            eta + 1)) * v2  # U_{i,j-1}, U_s
                A[index, index + N + 1] += 2 * muhh / (eta * (eta + 1)) \
                                           + (2 * yp / (h * h * eta * (eta + 1)) + h1 / (
                        eta * (eta + 1))) * v2  # U_{i,j+1}, U_n
                A[index, index - 1] += muhh - v1 * h2  # U_{i-1,j}, U_w
                A[index, index + 1] += muhh + v1 * h2  # U_{i+1,j}, U_e
            if P[i, j] == 3:
                rho = (np.sqrt(bvp.c - yp ** 2) - xp) / h
                A[index, index] = - 2 * muhh - 2 * muhh / rho \
                                  - (2 * xp / (rho * h * h) + h1 / rho - h1) * v1  # U_ij, U_p
                A[index, index - 1] = 2 * muhh / (1 + rho) \
                                      + (2 * xp / (h * h * (rho + 1)) - h1 * rho / (rho + 1)) * v1  # U_{i-1,j}, U_w
                A[index, index + 1] = 2 * muhh / (rho * (rho + 1)) \
                                      + (2 * xp / (h * h * rho * (rho + 1)) + h1 / (
                        rho * (rho + 1))) * v1  # U_{i+1,j}, U_e
                eta = (np.sqrt(bvp.c - xp ** 2) - yp) / h
                A[index, index] += - 2 * muhh - 2 * muhh / eta \
                                   - (2 * yp / (eta * h * h) + h1 / eta - h1) * v2  # U_ij, U_p
                A[index, index - N - 1] += 2 * muhh / (1 + eta) \
                                           + (2 * yp / (h * h * (eta + 1)) - h1 * eta / (
                        eta + 1)) * v2  # U_{i,j-1}, U_s
                A[index, index + N + 1] += 2 * muhh / (eta * (eta + 1)) \
                                           + (2 * yp / (h * h * eta * (eta + 1)) + h1 / (
                        eta * (eta + 1))) * v2  # U_{i,j+1}, U_n

    # Boundary condition of circle quadrant
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

    # Add boundary values for the square
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
    return A, x, y


# Function to get A matrix for longer stepsizes in 1d, order 1 (for small h/ big N), also returns x, y
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
    # one constant
    muhh = - bvp.mu / (h * h)

    # set in the inner values
    for i in range(1, N):
        for j in range(1, N):
            index = bvp.I(i, j, N)
            v1, v2 = bvp.V(x[0, i], y[j, 0])
            A[index, index] = - 4 * muhh + v1 / h - v2 / h  # U_ij, U_p
            A[index, index - N - 1] = muhh   # U_{i,j-1}, U_s
            A[index, index + N + 1] = muhh + v2 / h  # U_{i,j+1}, U_n
            A[index, index - 1] = muhh - v1 / h  # U_{i-1,j}, U_w
            A[index, index + 1] = muhh  # U_{i+1,j}, U_e

    # Incorporate boundary conditions
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

    return A, x, y


# function to apply the boundary condition G to F in the right places on a square
def apply_bcs_square(F, G, N, bvp):
    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = np.array([bvp.I(i, j, N) for j in [0, N] for i in range(0, N + 1)])
    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = np.array([bvp.I(i, j, N) for i in [0, N] for j in range(0, N + 1)])
    F[bc_indices] = G[bc_indices]
    return F


# function to apply the boundary condition G to F in the right places on a circle quadrant
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


# Function for Boundary condition on square
def G_square(x, y, bvp, N):
    G = np.zeros((N+1, N+1))
    G[0, :] = bvp.gs(x).ravel()
    G[-1, :] = bvp.gn(x).ravel()
    G[:, 0] = bvp.gw(y).ravel()
    G[:, -1] = bvp.ge(y).ravel()
    return G


# Function for Boundary condition on circle quadrant
def G_circle_quadrant(x, y, bvp, N):
    # ge and gn may do not exist, do gc
    G = sparse.dok_matrix((N + 1, N + 1))
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            if x[0, i] ** 2 + y[j, 0] ** 2 >= bvp.c:
                G[i, j] = bvp.gc(x[0, i], y[j, 0])
    G = G.toarray()
    # gw and gs exist, so do them last
    G[0, :] = bvp.gs(x).ravel()
    G[:, 0] = bvp.gw(y).ravel()

    return G

"""Class"""
# A class to hold the parameters of the BVP
# -mu div grad u + V * grad u = f, on (a,b)x(a,b), u = g on Boundary, with Dirichlet boundary conditions
class BVP(object):
    def __init__(self, f, V, gs=None, gn=None, gw=None, ge=None, a=0, b=1, mu=1, uexact=None, I=I, get_Axy=get_Axy,
                 apply_bcs=apply_bcs_square, G=G_square, c=1, gc=None):
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
        self.apply_bcs = apply_bcs  # apply boundary in F
        self.G = G  # Boundary condition for F.
        self.c = c  # constant for circle quadrant boundary condition, i.e. x^2 + y^2 = c
        self.gc = gc  # Boundary condition for circle quadrant boundary condition, i.e. x^2 + y^2 = c

"""Functions not used in class"""
# solve the BVP and plot it if value plot=True
def solve_BVP_and_plot(bvp, N, test, plot=True, view=225, save=False):
    # Make grid and matrix
    A, x, y = bvp.get_Axy(bvp, N)
    F = bvp.f(x, y).ravel()
    G = bvp.G(x, y, bvp, N).ravel()
    # Apply bcs
    F = bvp.apply_bcs(F, G, N, bvp)
    # Solve
    A_csr = A.tocsr()
    U = spsolve(A_csr, F).reshape((N+1, N+1))

    try:
        U_exact = bvp.uexact(x, y)
        err = np.abs(U - U_exact)
        print('The error is {:.2e}'.format(np.max(np.max(err))) + ", N = " + str(N))
        if plot:
            fig = plt.figure(num=test, figsize=(18, 6), dpi=100)
            fig.suptitle("Nummerical solution, Exact solution and Error of" +test, fontsize=20)
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')

            ax1 = plot2D(ax1, x, y, U, view=view)
            ax2 = plot2D(ax2, x, y, U_exact, view=view,)
            ax3 = plot2D(ax3, x, y, err, view=view, zlabel="err")

            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            ax3.w_zaxis.set_major_formatter(formatter)

            # Set to "save" to True to save the plot
            plt.subplots_adjust(hspace=0.3, wspace=0.45)
            if save:
                plt.savefig(get_name_of_plot(test) + ".pdf", bbox_inches='tight')

        return err
    except:
        print("No excat solution given")
        fig = plt.figure(num=test, figsize=(6, 6), dpi=100)
        fig.suptitle("Nummerical solution, Exact solution and Error of" + test, fontsize=20)
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1 = plot2D(ax1, x, y, U, view=view)
        # Set to "save" to True to save the plot
        plt.subplots_adjust(hspace=0.3, wspace=0.45)
        # Set to "save" to True to save the plot
        if save:
            plt.savefig(get_name_of_plot(test) + ".pdf", bbox_inches='tight')
        return None


# Function to find the order of convergence
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


"""Functions related to plotting"""
# fix name of plot, for saving
def get_name_of_plot(title):
    for i in range(len(title)):
        if title[i] == " ":
            title[i] = "_"
    return title


# Plot x, y, and u(x, y)
def plot2D(ax, X, Y, Z, view=270, zlabel='$u(x,y)$'):
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)  # Surface-plot
    # Set initial view angle
    ax.view_init(30, view)

    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel(zlabel)
    return ax

# Function to make a convergence plot
def plot_convergence(H, E, p, title, save=False):
    plt.figure()
    plt.loglog(H, E, 'o-', label='p={:.2f}'.format(p))
    plt.grid('on')
    plt.xlabel('h')
    plt.ylabel('error')
    plt.title(title)
    plt.legend()

    # Set to "save" to True to save the plot
    if save:
        plt.savefig(get_name_of_plot(title) + ".pdf", bbox_inches='tight')


"""Code for Tests and tasks"""

def TEST_1(N, P=4, save=False):
    # Test just the possion problem
    print("------------------------------------------")
    print("Test 1, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.sin(x) * np.sin(1)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.sin(1) * np.sin(y)
    f = lambda x, y: 2 * np.sin(x) * np.sin(y)

    # excat solution
    uexact = lambda x, y: np.sin(x) * np.sin(y)

    # function acting as vector
    def V(x, y):
        return 0, 0

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_1", save=save)
    print("------------------------------------------")
    print("Test convergence for TEST_1")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order, "Convergence for TEST_1", save=save)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_2(N, P=4, save=False):
    print("------------------------------------------")
    print("Test 2, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.sin(x) * np.sin(1)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.sin(1) * np.sin(y)
    f = lambda x, y: 2 * np.sin(x) * np.sin(y) + np.sin(x) * np.cos(y) + np.cos(x) * np.sin(y)

    # exact solution
    uexact = lambda x, y: np.sin(x) * np.sin(y)

    # function acting as vector
    def V(x, y):
        return 1, 1

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_2", save=save)
    print("------------------------------------------")
    print("Test convergence for TEST_2")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order, "Convergence for TEST_2", save=save)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_3(N, P=4, c1=1, c2=1, save=False):
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

    # exact solution
    uexact = lambda x, y: np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y)

    # function acting as vector
    def V(x, y):
        return c1, c2

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "TEST_3", save=save)
    print("------------------------------------------")
    print("Test convergence for TEST_3")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order, "Convergence for TEST_3", save=save)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")

def TEST_4(N, P=4, save=False):
    print("------------------------------------------")
    print("Test 3, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: 5 * np.pi * np.pi * np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     + x * np.pi * np.cos(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     + y * 2 * np.pi * np.sin(1 * np.pi * x) * np.cos(2 * np.pi * y)
    # exact solution
    uexact = lambda x, y: np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y)

    # function acting as vector
    def V(x, y):
        return x, y

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact)
    solve_BVP_and_plot(test, N, "", save=save)
    print("------------------------------------------")
    print("Test convergence for TEST_4")
    Hconv, Econv, order = convergence(test, P=P)
    plot_convergence(Hconv, Econv, order, "Convergence for TEST_3", save=save)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def TEST_4_long_step(N, P=4, save=False):
    print("------------------------------------------")
    print("Test 3, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: 5 * np.pi * np.pi * np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     + y * np.pi * np.cos(1 * np.pi * x) * np.sin(2 * np.pi * y) \
                     - x * 2 * np.pi * np.sin(1 * np.pi * x) * np.cos(2 * np.pi * y)

    # exact solution
    uexact = lambda x, y: np.sin(1 * np.pi * x) * np.sin(2 * np.pi * y)

    # function acting as vector
    def V(x, y):
        return y, -x

    test = BVP(f, V, gs=gs, gn=gn, gw=gw, ge=ge, uexact=uexact, get_Axy=get_Axy_square_longer_step_1d)
    solve_BVP_and_plot(test, N, "TEST_4 long step", save=save)
    print("------------------------------------------")
    print("Test convergence for TEST_4 long step")
    Hconv, Econv, order = convergence(test, P=P, N=10)
    plot_convergence(Hconv, Econv, order, "Convergence for TEST_4 long step", save=save)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")

def TEST_1c(N, P=4, c1=1, c2=1, save=False):
    print("------------------------------------------")
    print("Test 3 1c, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: 1 - x * x
    gw = lambda y: 1 - y * y
    gc = lambda x, y: 0
    f = lambda x, y: 4 - c1 * 2 * x - c2 * 2 * y

    # exact solution
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

    # function acting as vector
    def V(x, y):
        return c1, c2

    test = BVP(f, V, gs=gs, gw=gw, gc=gc, uexact=uexact, get_Axy=get_Axy_circle_quadrant, apply_bcs=apply_bsc_circle_quadrant, G=G_circle_quadrant)
    solve_BVP_and_plot(test, N, "TEST_1c", view=45, save=save)
    print("------------------------------------------")
    print("Test convergence for TEST_1c")
    Hconv, Econv, order = convergence(test, P=P, N=65)
    plot_convergence(Hconv, Econv, order, "Convergence for TEST_1c", save=save)
    print("Convergence order: " + "{:.2f}".format(order))
    print("------------------------------------------")


def Task_1d(N, P=4, save=False):
    print("------------------------------------------")
    print("Task 1d, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: np.ones((np.shape(x)[1], np.shape(y)[0]))

    # function acting as vector
    def V(x, y):
        return y, -x

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, mu=1e-2)
    solve_BVP_and_plot(test, N, "Task 1d", view=255, save=save)
    print("------------------------------------------")


def Task_1d_long_step(N, P=4, save=False):
    print("------------------------------------------")
    print("Task 1d long step, N = ", N)
    # Boundary conditions and source functions.
    gs = lambda x: np.zeros_like(x)
    gn = lambda x: np.zeros_like(x)
    gw = lambda y: np.zeros_like(y)
    ge = lambda y: np.zeros_like(y)
    f = lambda x, y: np.ones((np.shape(x)[1], np.shape(y)[0]))

    # function acting as vector
    def V(x, y):
        return y, -x

    test = BVP(f, V,  gs=gs, gn=gn, gw=gw, ge=ge, mu=1e-2, get_Axy=get_Axy_square_longer_step_1d)
    solve_BVP_and_plot(test, N, "Task 1d long step", view=255, save=save)
    print("------------------------------------------")



"""Run tests"""

#TEST_1(10, save=False)
#TEST_2(10, save=False)
#TEST_3(100, c1=1, c2=-1, save=False)
TEST_4(100, save=False)
#TEST_4_long_step(10, save=False)

"""Run tasks"""
#TEST_1c(50, save=False)
#Task_1d(100)
#Task_1d(10, save=False)
#Task_1d_long_step(10, save=False)

# save=True to save the plots


plt.show()