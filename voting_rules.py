from value_generators import *
from modelling import *
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

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

def kendalltau_distance(*args, **kwargs):
	tau_coef = kendalltau(*args, **kwargs)
	return (1 - tau_coef.statistic) / 2

def kendall_tau_distance_with_ties(x, y, tieListX, tieListY, tiePenalty=True, arbitraryBreak=False, simulateBoth=False):
    if len(x) != len(y):
        raise ValueError("Both lists must have the same length.")

    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    max_pairs = n * (n - 1) / 2

    tie_lookup_x = [set(group) for group in tieListX]
    tie_lookup_y = [set(group) for group in tieListY]

    def are_tied(i, j, tie_groups):
        return any(i in group and j in group for group in tie_groups)

    if simulateBoth:
        disc_with, partial_with = 0, 0
        disc_without = 0

        for i, j in combinations(range(n), 2):
            tied_x = are_tied(i, j, tie_lookup_x)
            tied_y = are_tied(i, j, tie_lookup_y)

            # With ties
            if arbitraryBreak or not (tied_x or tied_y):
                concordant = (x[i] - x[j]) * (y[i] - y[j]) > 0
                if not concordant:
                    disc_with += 1
            else:
                if tiePenalty:
                    partial_with += 0.5

            # Without ties (i.e., simulate as if arbitrary break resolves all ties)
            concordant_no_ties = (x[i] - x[j]) * (y[i] - y[j]) > 0
            if not concordant_no_ties:
                disc_without += 1

        kt_with = (disc_with + partial_with) / max_pairs
        kt_without = disc_without / max_pairs
        return kt_with, kt_without

    else:
        disc, partial = 0, 0
        for i, j in combinations(range(n), 2):
            tied_x = are_tied(i, j, tie_lookup_x)
            tied_y = are_tied(i, j, tie_lookup_y)

            if arbitraryBreak or not (tied_x or tied_y):
                concordant = (x[i] - x[j]) * (y[i] - y[j]) > 0
                if not concordant:
                    disc += 1
            else:
                if tiePenalty:
                    partial += 0.5

        return (disc + partial) / max_pairs