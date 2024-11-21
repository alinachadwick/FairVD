import matplotlib.pyplot as plt
# from utility_sim import *
import numpy as np
from itertools import permutations

# this plots in 3 dimensions only!
alphas = np.array([[0.5, 0.4, 0.5],
				   [0.3, 0.6, 0.2],
				   [0.6, 0.2, 0.3]]) # rows is metrics, column is functions

alphas = np.random.rand(3,3)
betas = np.array([[0.6, 0.5, 0.4],
				  [0.4, 0.5, 0.6]]) # row is voter, column is beta value per function
gamma = 1
sample = 5000

k = betas.shape[1]
planes = permutations(range(alphas.shape[0]), 2) # all pairs of alphas

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for first, second in planes:
	a1, a2 = alphas[first], alphas[second]
	diff = a1 - a2
	y = np.linspace(-1, 1, 10)
	z = np.linspace(-1, 1, 10)
	Y, Z = np.meshgrid(y, z)
	X = (-diff[0]*Y -diff[1]*Z)/diff[2]
	ax.plot_surface(X, Y, Z, alpha = 0.5)
ax.set_ylim([-1, 1])
ax.set_xlim([-1, 1])


plt.show()