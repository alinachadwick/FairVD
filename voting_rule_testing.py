from utility_sim import *
from value_generators import *
from scipy.stats import kendalltau
import csv
import matplotlib.pyplot as plt
import numpy as np

def kendalltau_distance(*args, **kwargs):
	tau_coef = kendalltau(*args, **kwargs)
	return (1 - tau_coef.statistic) / 2

def kendall_tau_distance_with_ties(x, y, tieListX, tieListY):
	"""
	Calculate the Kendall Tau distance (Tau-c based) between two lists x and y,
	incorporating the handling of ties based on provided tieLists for x and y.

	Parameters:
		x (list or np.ndarray): First ranking.
		y (list or np.ndarray): Second ranking.
		tieListX (list of tuples): List of tuples, where each tuple contains indices that are considered tied in x.
		tieListY (list of tuples): List of tuples, where each tuple contains indices that are considered tied in y.

	Returns:
		float: Kendall Tau distance between x and y.
	"""
	if len(x) != len(y):
		raise ValueError("Both lists must have the same length.")

	# Convert inputs to numpy arrays for optimized computations
	x = np.array(x)
	y = np.array(y)
	n = len(x)

	# Function to check if two indices are tied based on the tieList
	def are_tied(i, j, tieList):
		for tie_group in tieList:
			if i in tie_group and j in tie_group:
				return True
		return False

	# Initialize concordant, discordant, and tie counts
	kt_distance = 0
	for i in range(n):
		for j in range(i + 1, n):
			if are_tied(i, j, tieListX) or are_tied(i, j, tieListY):  # Tie in x or y
				kt_distance += 0.5
			else:  # Compare normally
				if (x[i] - x[j]) * (y[i] - y[j]) > 0:
					kt_distance += 0  # Concordant pair
				else:
					kt_distance += 1  # Discordant pair

	# Normalize the distance to be between 0 and 1
	max_pairs = n * (n - 1) / 2  # Total number of possible pairs
	normalized_distance = kt_distance / max_pairs

	return normalized_distance

def threshold_vector(vector, t):
	return (vector > t).astype(int)

def approvalRaw(alphas, betas, t):
	voterScores, _ = get_ranked_scores(alphas, betas)
	return approval(voterScores, t)

def approval(voterScores, t):
	voterThresholds = threshold_vector(voterScores, t)
	mechScores = np.sum(voterThresholds, axis=0)

	# Map scores to their indices to identify ties
	score_to_indices = {}
	for idx, score in enumerate(mechScores):
		if score not in score_to_indices:
			score_to_indices[score] = []
		score_to_indices[score].append(idx)

	# Create tieList based on groups of tied indices
	tieList = [set(indices) for indices in score_to_indices.values() if len(indices) > 1]
	tieList.sort(key = lambda x: min(x))  # Sort by the smallest index in each group

	# Create orderedIndexes
	orderedIndexes = indexes_sortByVal(mechScores)

	return orderedIndexes, tieList

def bordaRaw(alphas, betas):
	_, voterRankings = get_ranked_scores(alphas, betas)
	return borda(voterRankings)

def borda(voterRankings):
	scores = [0.0] * voterRankings.shape[1]
	# can be optimized using np functions
	for i in voterRankings:
		for s, j in enumerate(i):
			scores[j] += (voterRankings.shape[1] - s)
	return indexes_sortByVal(scores)

def copelandRaw(alphas, betas):
	_, voterRankings = get_ranked_scores(alphas, betas)
	return copeland(voterRankings)

def copeland(voterRankings):
	m = voterRankings.shape[1]
	n = voterRankings.shape[0]

	# Precompute rankings as positional lookups for each candidate
	position_lookup = np.argsort(voterRankings, axis=1)  # Shape: (n, m)
	
	# Initialize pairwise win counts
	pairwise_wins = np.zeros((m, m), dtype=int)
	
	# Calculate pairwise wins
	for i in range(m):
		for j in range(i + 1, m):  # Only need to compare i < j
			# Count wins for i vs j
			wins_for_i = np.sum(position_lookup[:, i] < position_lookup[:, j])
			wins_for_j = n - wins_for_i

			if wins_for_i > wins_for_j:
				pairwise_wins[i, j] += 1
				pairwise_wins[j, i] -= 1
			elif wins_for_i < wins_for_j:
				pairwise_wins[j, i] += 1
				pairwise_wins[i, j] -= 1

	# Compute Copeland scores
	copeland_scores = np.sum(pairwise_wins > 0, axis=1)

	# Return indices sorted by Copeland scores
	return np.argsort(-copeland_scores)  # Descending order of scores

def scoresum(voterScores):
	return indexes_sortByVal(sum(voterScores))

def scoresumRaw(alphas, betas):
	voterScores, _ = get_ranked_scores(alphas, betas)
	return scoresum(voterScores)

# Function to save data to CSV
def save_to_csv(filename, x_values, borda_values, copeland_values, sum_values):
	with open(filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["X Values", "Borda KT", "Copeland KT", "Sum KT"])
		for x, borda, copeland, sum_kt in zip(x_values, borda_values, copeland_values, sum_values):
			writer.writerow([x, borda, copeland, sum_kt])

# Function to run simulations and compute average Kendall Tau distance

if __name__ == "__main__":
	
	# Define parameter configurations
	alpha_methods = [False, True]  # gaussian = False, True
	beta_cases = [0, 1, 2, 3]      # case = 0, 1, 2, 3
	epsilon_methods = [False, True]  # normal = False, True

	# Number of alternatives (m) and voters (n)
	m = 8
	n = 10
	k = 5
	simulations = 10000  # Number of simulations per configuration

	# Storage for results
	results = []
	
	for alpha_gaussian in alpha_methods:
		for beta_case in beta_cases:
			for epsilon_normal in epsilon_methods:
				# Initialize KT scores
				b_netKT = 0.0
				c_netKT = 0.0
				s_netKT = 0.0

				# Simulation loop
				for _ in range(simulations):
					# Generate parameters for the simulation
					alphas = generate_alphas(m, k, gaussian=alpha_gaussian)
					true_beta = generate_betas(n, k, case=beta_case)
					true_scores, true_ranking = get_ranked_scores(true_beta, alphas)

					true_sum_ranking = scoresum(true_scores)
					true_borda_ranking = borda(true_ranking)
					true_copeland_ranking = copeland(true_ranking)

					error = generate_epsilon(n, k, bounds=(-0.2, 0.2), sd=0.1, normal = epsilon_normal)
					obs_beta = true_beta + error

					# Get observed scores and rankings
					obs_scores, obs_ranking = get_ranked_scores(obs_beta, alphas)

					# Calculate rankings using different methods
					borda_ranking = borda(obs_ranking)
					copeland_ranking = copeland(obs_ranking)
					sum_ranking = scoresum(obs_scores)

					# Accumulate KT distances
					b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
					c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
					s_netKT += kendalltau_distance(true_sum_ranking, sum_ranking)

				# Average KT distances for this configuration
				avg_borda_kt = b_netKT / simulations
				avg_copeland_kt = c_netKT / simulations
				avg_sum_kt = s_netKT / simulations

				# Store results
				results.append({
					"alpha_gaussian": alpha_gaussian,
					"beta_case": beta_case,
					"epsilon_normal": epsilon_normal,
					"avg_borda_kt": avg_borda_kt,
					"avg_copeland_kt": avg_copeland_kt,
					"avg_sum_kt": avg_sum_kt
				})

	# Print results
	for result in results:
		print(
			f"Alpha Gaussian: {result['alpha_gaussian']}, "
			f"Beta Case: {result['beta_case']}, "
			f"Epsilon Normal: {result['epsilon_normal']} -> "
			f"Borda KT: {result['avg_borda_kt']:.3f}, "
			f"Copeland KT: {result['avg_copeland_kt']:.3f}, "
			f"Sum KT: {result['avg_sum_kt']:.3f}"
		)
