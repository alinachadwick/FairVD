import numpy as np

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

def indexes_sortByVal(l, reverse = True): # current
	return sorted(range(len(l)), key = lambda x: l[x], reverse = reverse)

def indexSortWithTies(l, reverse = True):
	sortedIndexes = indexes_sortByVal(l, reverse = reverse)
	prevScore = float("inf")
	indexesWithTies = []
	currTieSet = set()
	for i in sortedIndexes:
		val = l[i]
		if val == prevScore:
			currTieSet.add(i)
		else:
			if len(currTieSet) == 0:
				indexesWithTies.append({i})
			else:
				indexesWithTies.append(currTieSet)
				currTieSet = set()
	if len(currTieSet) == 0:
		indexesWithTies.add(currTieSet)
	return indexesWithTies