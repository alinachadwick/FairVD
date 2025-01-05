import numpy as np
from itertools import permutations
from math import floor, ceil


def normalize_to_bounds(x, bounds):
	return x
	return np.minimum(bounds[1], np.maximum(bounds[0], bounds[0] + (x - bounds[0])/(bounds[1]- bounds[0])))

def generate_alphas(m: int, k: int, bounds: tuple = (0.0, 1.0), gaussian: bool = False, true_distribution = [], sd: float = 1.0):
	if not gaussian:
		unnorm_alpha = np.random.rand(m, k)
	else:
		if len(true_distribution) == 0:
			mean = (bounds[0] + bounds[1])/2
			unnorm_alpha = (np.random.normal(mean, sd, size = (m, k))) #max is to ensure no negative values
		else:
			unnorm_alpha = np.random.normal(true_distribution, sd)
	return normalize_to_bounds(unnorm_alpha, bounds)

def generate_betas(n: int, k: int, bounds: tuple = (0.0, 1.0), case:int = 0, sd: float = 1.0, space_param: float = 0.5):
	'''
	case:
	0 -> uniformly drawn from across bounds
	1 -> antagonistically normally drawn
	2 -> normally drawn from same mean
	3 -> uniformly drawn from two bounds

	space_param should be a float between 0 and 1. In each case refers to:
	case 0: does nothing
	case 1: locations of means = (0.5 +/- 0.5*space_param)*(bounds[1]-bounds[0])+bounds[0]
	case 2: mean for distribution = space_param*(bounds[1]-bounds[0])+bounds[0]
	case 3: uniform distribution bounds = (bounds[0], bounds[0]+space_param*(bounds[1]- bounds[0])), (bounds[1]- space_param*(bounds[1]-bounds[0], bounds[1]))

	'''
	if case == 1: # antagonistic & normal
		smaller_mean = (0.5 - 0.5 * space_param)*(bounds[1] - bounds[0]) + bounds[0]
		larger_mean = (0.5 + 0.5 * space_param)*(bounds[1] - bounds[0]) + bounds[0]
		smaller_betas = normalize_to_bounds(np.random.normal(smaller_mean, sd/2, size =(floor(n/2), k)), bounds)
		larger_betas = normalize_to_bounds(np.random.normal(larger_mean, sd/2, size =(ceil(n/2), k)), bounds)
		unnorm_betas = np.vstack([smaller_betas, larger_betas])
	elif case == 2: # normally distributed betas around center point
		mean = space_param*(bounds[1] -bounds[0]) + bounds[0]
		unnorm_betas = np.random.normal(mean, sd, size = (n, k))
	elif case == 3: # antagonistic & uniform
		smaller_bounds = (bounds[0], bounds[0]+(0.5*space_param)*(bounds[1]- bounds[0]))
		larger_bounds = (bounds[1] - (0.5*space_param)*(bounds[1]-bounds[0]), bounds[1])
		smaller_betas = generate_betas(floor(n/2), k, smaller_bounds, case = 0, sd = sd)
		larger_betas = generate_betas(ceil(n/2), k, larger_bounds, case = 0, sd = sd)
		return np.vstack([smaller_betas, larger_betas]) # return because already normalized
	else:
		unnorm_betas = np.random.rand(n, k)
	return normalize_to_bounds(unnorm_betas, bounds)

def generate_epsilon(n: int, k: int, bounds: tuple = (-0.1, 0.1), normal = False, sd = 0.1):
	mean_error = (bounds[0] + bounds[1])/2
	if normal:
		return np.random.normal(mean_error, sd, size = (n, k))
	else:
		return normalize_to_bounds(np.random.rand(n, k), bounds)