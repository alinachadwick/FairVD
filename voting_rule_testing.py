from utility_sim import *
from value_generators import *
from scipy.stats import kendalltau
import csv
import matplotlib.pyplot as plt
import numpy as np

def bordaRaw(alphas, betas):
	_, voterRankings = get_ranked_scores(alphas, betas)
	return borda(voterRankings)

def borda(voterRankings):
	scores = [0.0] * voterRankings.shape[1]
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
	m = 5
	n = 5
	k = 2
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
					true_ranking = scoresumRaw(true_beta, alphas)
					error = generate_epsilon(n, k, bounds=(-0.2, 0.2), sd=0.1, normal = epsilon_normal)
					obs_beta = true_beta + error

					# Get observed scores and rankings
					obs_scores, obs_ranking = get_ranked_scores(obs_beta, alphas)

					# Calculate rankings using different methods
					borda_ranking = borda(obs_ranking)
					copeland_ranking = copeland(obs_ranking)
					sum_ranking = scoresum(obs_scores)

					# Accumulate KT distances
					b_netKT += kendalltau(true_ranking, borda_ranking).statistic
					c_netKT += kendalltau(true_ranking, copeland_ranking).statistic
					s_netKT += kendalltau(true_ranking, sum_ranking).statistic

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
