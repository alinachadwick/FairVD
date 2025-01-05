import numpy as np
from itertools import permutations
from math import log2, floor, pow, factorial, ceil
import matplotlib.pyplot as plt

def get_column_widths(matrix):
	rows, cols = matrix.shape
	col_widths = []
	
	for j in range(cols):
		max_width = 0
		for i in range(rows):
			# Calculate the width needed for each entry, accounting for decimals and sign
			entry = f"{matrix[i, j]:.6g}"  # Limit to 6 significant figures to avoid overly long entries
			max_width = max(max_width, len(entry))
		col_widths.append(max_width)
	
	return col_widths

def print_matrix(matrix):
	rows, cols = matrix.shape
	col_widths = get_column_widths(matrix)

	# Print column headers with adjusted spacing
	print(" " * 4, end="")
	for j in range(cols):
		print(f" f{j} ".ljust(col_widths[j] + 1), end="")
	print()
	
	# Print top border
	print("   +" + "-".join("-" * (w + 1) for w in col_widths))
	
	# Print rows with row labels and values
	for i in range(rows):
		print(f"m{i} |", end="")  # Row label
		for j in range(cols):
			# Print each element, right-aligned with the calculated column width
			print(f"{matrix[i, j]:>{col_widths[j]}.6g} ", end=" ")
		print()  # New line at end of row

def get_ranked_scores(alphas, betas):
	scores = []
	for i in range(len(betas)):
		scores.append(alphas*betas[i])
	votersScores = np.dot(alphas, betas.T).T
	return votersScores, np.array([indexes_sortByVal(vs) for vs in votersScores])

def indexes_sortByVal(l, reverse = True):
	return sorted(range(len(l)), key = lambda x: l[x], reverse = reverse)

if __name__ == "__main__":
	# Parameters:
	alphas = np.array([[0.5, 0.4],
					[0.3, 0.6],
					[0.6, 0.2]])  # rows are metrics, columns are functions

	betas = np.array([[0.6, 0.5],
					[0.4, 0.7],
					[0.7, 0.3]])  # rows are voter preference points

	gamma = 0.1
	sample = 5000

	print_matrix(alphas)


	print("True scores & preferences:\n----------")
	votersScores, votersRankings = get_ranked_scores(alphas, betas)

	for voter in range(len(betas)):
		print("Voter %d's scores:"%voter + str(dict(enumerate(votersScores[voter]))))
		print("         rankings:" + str(votersRankings[voter]))

	counter = [{} for i in betas]
	for s in range(sample):
		errorMat = np.random.rand(betas.shape[0], betas.shape[1])*gamma # assumes uniform probability
		simBeta = betas.copy()
		simBeta += errorMat
		scores, ranking = get_ranked_scores(alphas, simBeta)
		for i, r in enumerate(ranking):
			if str(r) in counter[i]:
				counter[i][str(r)] += 1
			else:
				counter[i][str(r)] = 1

	# normalizing to probability
	counter = [{i:c[i]/sample for i in c.keys()} for c in counter]

	print(counter)