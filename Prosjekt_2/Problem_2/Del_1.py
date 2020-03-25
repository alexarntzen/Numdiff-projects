import numpy as np
import matplotlib.pyplot as plt
from DiseaseModel import ModifiedCrankNicolson, DiseaseModel


"""
 Check that algoritm gives correct result. This is compared to the reuslt from Del_1_test.py
"""

# Parameters
beta = 10
gamma = 10
T = 1
k = 0.001
I0 = 0.1

# Make sure the solver works for 0D case.
U, times = ModifiedCrankNicolson.modifiedCrankNicolsonSolver([1 - I0, I0],DiseaseModel.DiseaseModelF(beta,gamma), T=T, k=k)
[S,I] = U.T
R = 1 - S - I
plt.plot(times, S, label="S")
plt.plot(times, I, label="I")
plt.plot(times, R, label="R")
plt.legend()
plt.title(f"$ \\beta = {beta}, \\gamma = {gamma}$")
plt.show()
