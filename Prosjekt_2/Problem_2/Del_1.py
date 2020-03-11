import numpy as np
import matplotlib.pyplot as plt
import BVP_solver as BVP

"""
 Check that algoritm gives correct result. This is compared to the reuslt from Del_1_test.py
"""

# Parameters
beta = 1
gamma = 1
T = 10
k = 1
I0 = 0.2

# Make sure the solver works for 0D case.
U, times = BVP.modified_CrankNicolson([1 - I0, I0],BVP.DiseaseModelF(1,1), T=20, k=k)
[S,I] = U.T
R = 1 - S - I
plt.plot(times, S, label="S")
plt.plot(times, I, label="I")
plt.plot(times, R, label="R")
plt.legend()
plt.title(f"$ \\beta = {beta}, \\gamma = {gamma}$")
plt.show()
