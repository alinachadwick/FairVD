# Alina Chadwick
# Fairness Notions and VD

import matplotlib.pyplot as plt
import numpy as np

def approximate_flip(b1, b2, d1_over_d2, a):
    d2_over_d1 = 1/(d1_over_d2)
    upper_triangle_approx = (1/(4*a*a)) * (0.5) * (b2 + a - (d1_over_d2) * (b1 - a)) * ((b2 + a)*(d2_over_d1) - (b1 - a))
    skinny_trap_approx = (1/(2*a)) * (d2_over_d1* b2 - b1 + a)
    fat_trap_approx = (1/(2*a)) * (b2 + a - ((d1_over_d2)* b1))
    
    if b2 < b1:
        if d1_over_d2 >= ((b2 + a)/(b1 + a)) and d1_over_d2 <= ((b2 + a)/(b1 - a)): # upper triangle
            print("reverse" + str(d1_over_d2))
            return upper_triangle_approx
        elif d1_over_d2 < ((b2 + a)/(b1 + a)) and d1_over_d2 >= ((b2)/(b1)): # fat trapezoid
            print(d1_over_d2)
            return fat_trap_approx
        else:
            return 0
    else:
        if d1_over_d2 >= ((b2 - a)/(b1 - a)) and d1_over_d2 <= ((b2 + a)/(b1 - a)): # upper triangle:
            print("reverse" + str(d1_over_d2))
            return upper_triangle_approx
        elif d1_over_d2 < ((b2 - a)/(b1 - a)) and d1_over_d2 >= ((b2)/ (b1)): # skinny trapezoid
            print(d1_over_d2)
            return skinny_trap_approx
        else:
            return 0

def visualize(b1, b2, alpha, step):
    x_vals = []
    y_vals = []
    d1_over_d2 = 0.01
    while d1_over_d2 < 1000:
        y = approximate_flip(b1, b2, d1_over_d2, alpha)
        x = d1_over_d2
        if x is not None and y is not None and y != 0:
            x_vals.append(x)
            y_vals.append(y)
        d1_over_d2 +=step

    xpoints = np.array(x_vals)
    ypoints = np.array(y_vals)
    # return (xpoints, ypoints)
    plt.plot(x_vals, y_vals, 'o')
    title = "b1 = " + str(b1) + ", b2 = " + str(b2) + ", a = " + str(alpha) + ", step = " + str(step)
    plt.title(title)
    plt.xlabel("d1/d2 value")
    plt.ylabel("approximate flip percentage")

    plt.savefig(title + '.png')
    plt.show()


def colorgradient(d1_over_d2, alpha):
    point_dict = {}
    for b1 in np.arange(0, 1, alpha/10):
        for b2 in np.arange(0, 1, alpha/10):
            point = (b1, b2)
            if b1 != 0 and b1 + alpha != 0 and b1 - alpha != 0:
                prob = approximate_flip(b1, b2, d1_over_d2, alpha)
                point_dict[point] = prob

    red_x = [] # 
    red_y = []
    orange_x = []
    orange_y = []
    yellow_x = []
    yellow_y = []
    green_x = []
    green_y = []
    blue_x = []
    blue_y = []
    indigo_x = []
    indigo_y = []
    purple_x = []
    purple_y = []
    xs = np.arange(0, 1, alpha/10)
    ys = []
    for x in xs:
        y = d1_over_d2 * x
        ys.append(y)

    for pointpair in point_dict:
        x = pointpair[0]
        y = pointpair[1]
        if point_dict[pointpair] == 0.5:
            red_x.append(x)
            red_y.append(y)
        elif point_dict[pointpair] >= 0.4:
            orange_x.append(x)
            orange_y.append(y)
        elif point_dict[pointpair] >= 0.3:
            yellow_x.append(x)
            yellow_y.append(y)
        elif point_dict[pointpair] >= 0.2:
            green_x.append(x)
            green_y.append(y)
        elif point_dict[pointpair] >= 0.1:
            blue_x.append(x)
            blue_y.append(y)
        elif point_dict[pointpair] > 0.0:
            indigo_x.append(x)
            indigo_y.append(y)
        else:
            purple_x.append(x)
            purple_y.append(y)
    
    plt.plot(red_x, red_y, 'o', color="red", label="flip probability = 0.5")
    plt.plot(orange_x, orange_y, 'o', color="orange", label="flip probability >= 0.4")
    plt.plot(yellow_x, yellow_y, 'o', color="yellow", label="flip probability >= 0.3")
    plt.plot(green_x, green_y, 'o', color="green", label="flip probability >= 0.2")
    plt.plot(blue_x, blue_y, 'o', color="blue", label="flip probability >= 0.1")
    plt.plot(indigo_x, indigo_y, 'o', color="indigo", label="flip probability > 0.0")
    # plt.plot(purple_x, purple_y, 'x', color="#B19CD9", label="flip probability = 0.0")
    plt.plot(xs, ys, color="black")
    plt.xlim(alpha, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    title = "d1_over_d2 = " + str(d1_over_d2) + ", a = " + str(alpha) 
    plt.title(title)
    plt.xlabel("b1 value")
    plt.ylabel("b2 value")

    plt.savefig(title + '.png')
    plt.show()

colorgradient(2, 0.1)
# visualize(3, 7, 2, 0.1)
