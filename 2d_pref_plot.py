import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import matplotlib.patches as patches

# Fixed alphas and betas values for 2D case
alphas = np.array([[0.5, 0.4],
                   [0.3, 0.6],
                   [0.6, 0.2]])  # rows are metrics, columns are functions

betas = np.array([[0.6, 0.5],
                  [0.4, 0.7],
                  [0.7, 0.3]])  # rows are voter preference points

# alphas = np.array([[0.5,0.4],[0.4,0.3]])
# betas = np.array([[0.6, 0.5],
#                   [0.7, 0.3]])
gamma = 0.1
sample = 5000

planes = combinations(range(alphas.shape[0]), 2)  # all pairs of alphas

fig, ax = plt.subplots()

# Add legend handles for the lines
handles = []

index = 0
labels = [r"$H_{M_2,M_1}$", r"$H_{M_1,M_3}$", r"$H_{M_2,M_3}$"]
for first, second in planes:
    a1, a2 = alphas[first], alphas[second]
    diff = a2 - a1
    x = np.linspace(-1, 1, 100)
    y = (-diff[0] * x) / diff[1]  # Line equation in 2D: y = -(a1/a2) * x
    # label = r"$H_{M_{%d},M_{%d}}$"%(first + 1, second + 1)  # Line label
    label = labels[index]
    line, = ax.plot(x, y, alpha=0.5, label=label)
    handles.append(line)
    index += 1

# Add the two points (\beta_x and \beta_y) and their hollow squares
points = betas
point_labels = [r"$\beta_%d$"%(i + 1) for i in range(len(betas))]
square_size = gamma # Size of the square

for i, (point, label) in enumerate(zip(points, point_labels)):
    # Add the point
    dot = ax.plot(point[0], point[1], 'o', label=label, markersize=8)
    handles.append(dot[0])
    
    # Add a hollow square around the point
    square = patches.Rectangle((point[0] - square_size / 2, point[1] - square_size / 2), 
                                square_size, square_size, 
                                edgecolor=dot[0].get_color(), facecolor='none', lw=1.5)
    ax.add_patch(square)

# Set axis limits and labels
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("Fairness Notion 1 Weight")
ax.set_ylabel("Fairness Notion 2 Weight")

# Add legend
ax.legend(handles=handles, loc="upper left")

plt.gca().set_aspect('equal')

# Show plot
plt.savefig("2dPrefPlot.pdf", format = "pdf", bbox_inches = "tight")