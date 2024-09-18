# Alina Chadwick
# VD Fairness
# Sept 2024

import math 
import matplotlib.pyplot as plt
import numpy as np

def circle_approximate_flip(b1, b2, d1_over_d2, a):
    r = a
    x = (b2 + d1_over_d2 * b1) * (1/(2 * d1_over_d2))
    y = (b2 + d1_over_d2 * b1) * 0.5
    c = math.sqrt((x - b1)**2 + (y - b2)**2)
    if c > r:
        return 0
    else:
        little_theta = math.acos(c/r)
        big_theta = 2 * little_theta
        sector_area = 0.5 * r**2 * (big_theta - math.sin(big_theta))
        circle_area = math.pi * (r**2)
        flip_prop = sector_area/circle_area
        return flip_prop

print(circle_approximate_flip(2, 1, 100, 0.3))


def visualize(b1, b2, alpha, step):
    x_vals = []
    y_vals = []
    d1_over_d2 = 0.01
    while d1_over_d2 < 1000:
        y = circle_approximate_flip(b1, b2, d1_over_d2, alpha)
        x = d1_over_d2
        if x is not None and y is not None and y != 0:
            x_vals.append(x)
            y_vals.append(y)
        d1_over_d2 +=step

    xpoints = np.array(x_vals)
    ypoints = np.array(y_vals)
    # return (xpoints, ypoints)
    plt.plot(x_vals, y_vals, '.')
    title = "Circle: b1 = " + str(b1) + ", b2 = " + str(b2) + ", a = " + str(alpha) + ", step = " + str(step)
    plt.title(title)
    plt.xlabel("d1/d2 value")
    plt.ylabel("approximate flip percentage")

    plt.savefig(title + '.png')
    plt.show()

# visualize(1, 2, .1, .01)
    


