import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import scipy.integrate as integrate
#import BVP_solver as BVP

def getDiseaseModelF(beta, gamma):
    # currying
    def F(t,N):
        return np.array([-beta * N[0]*N[1], beta * N[0] * N[1] - gamma * N[1], gamma * N[1]])
    return F

mu_S = 0.1
mu_I = 0.1
beta = 1
gamma = 1
T = 20
I0=0.1


sol = integrate.solve_ivp(getDiseaseModelF(beta,gamma), t_span = [0,T], y0 =np.array([1-I0,I0,0]), method="RK23")
[S,I,R] = sol.y
plt.plot(sol.t, S, label="S")
plt.plot(sol.t, I, label="I")
plt.plot(sol.t, R, label="R")
plt.title(f"beta = ${beta}, gamma = ${gamma}, scipy")

plt.legend()
plt.show()

