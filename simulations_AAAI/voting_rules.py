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

def ranked_pairs(voterRankings):
    m = voterRankings.shape[1]  # number of candidates
    n = voterRankings.shape[0]  # number of voters

    # Precompute rankings as positional lookups for each candidate
    position_lookup = np.argsort(voterRankings, axis=1)  # Shape: (n, m)
    
    # Calculate pairwise margins
    pairwise_margins = np.zeros((m, m), dtype=int)
    
    for i in range(m):
        for j in range(i + 1, m):  # Only need to compare i < j
            # Count wins for i vs j
            wins_for_i = np.sum(position_lookup[:, i] < position_lookup[:, j])
            wins_for_j = n - wins_for_i
            
            margin = wins_for_i - wins_for_j
            if margin > 0:
                pairwise_margins[i, j] = margin
                pairwise_margins[j, i] = -margin
            elif margin < 0:
                pairwise_margins[j, i] = -margin
                pairwise_margins[i, j] = margin
    
    # Create list of all pairwise comparisons with their margins
    pairs = []
    for i in range(m):
        for j in range(m):
            if i != j and pairwise_margins[i, j] > 0:
                pairs.append((i, j, pairwise_margins[i, j]))
    
    # Sort pairs by margin in descending order (strongest defeats first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Build directed graph using ranked pairs algorithm
    # Start with empty graph and add edges that don't create cycles
    graph = np.zeros((m, m), dtype=bool)
    
    def creates_cycle(graph, start, end):
        """Check if adding edge start->end would create a cycle"""
        if start == end:
            return True
        
        # Use DFS to check if there's already a path from end to start
        visited = np.zeros(m, dtype=bool)
        stack = [end]
        
        while stack:
            current = stack.pop()
            if current == start:
                return True
            if visited[current]:
                continue
            visited[current] = True
            
            # Add all nodes that current points to
            for next_node in range(m):
                if graph[current, next_node] and not visited[next_node]:
                    stack.append(next_node)
        
        return False
    
    # Add edges in order of decreasing margin, skipping those that create cycles
    for winner, loser, margin in pairs:
        if not creates_cycle(graph, winner, loser):
            graph[winner, loser] = True
    
    # Calculate final ranking based on the tournament graph
    # Count the number of candidates each candidate beats
    beats_count = np.sum(graph, axis=1)
    
    # Return indices sorted by number of victories (descending order)
    return np.argsort(-beats_count)

def minimax_condorcet(voterRankings):
    m = voterRankings.shape[1]  # number of candidates
    n = voterRankings.shape[0]  # number of voters

    # Precompute rankings as positional lookups for each candidate
    position_lookup = np.argsort(voterRankings, axis=1)  # Shape: (n, m)
    
    # Calculate pairwise defeat margins
    pairwise_margins = np.zeros((m, m), dtype=int)
    
    for i in range(m):
        for j in range(i + 1, m):  # Only need to compare i < j
            # Count wins for i vs j
            wins_for_i = np.sum(position_lookup[:, i] < position_lookup[:, j])
            wins_for_j = n - wins_for_i
            
            margin_i_over_j = wins_for_i - wins_for_j
            margin_j_over_i = wins_for_j - wins_for_i
            
            # Store the margin by which each candidate beats the other
            # Positive values mean the row candidate beats the column candidate
            if margin_i_over_j > 0:
                pairwise_margins[i, j] = margin_i_over_j
                pairwise_margins[j, i] = 0  # j doesn't beat i
            elif margin_j_over_i > 0:
                pairwise_margins[j, i] = margin_j_over_i
                pairwise_margins[i, j] = 0  # i doesn't beat j
            # If tied, both remain 0
    
    # Calculate minimax scores
    # For each candidate, find their worst defeat (largest margin they lose by)
    minimax_scores = np.zeros(m)
    
    for i in range(m):
        # Find the largest margin by which candidate i is defeated
        worst_defeat = 0
        for j in range(m):
            if i != j and pairwise_margins[j, i] > worst_defeat:
                worst_defeat = pairwise_margins[j, i]
        minimax_scores[i] = worst_defeat
    
    # Return indices sorted by minimax scores (ascending order - lower is better)
    # We negate to get descending order since lower minimax scores are better
    return np.argsort(minimax_scores)

def schulze(voterRankings):
    m = voterRankings.shape[1]  # number of candidates
    n = voterRankings.shape[0]  # number of voters

    # Precompute rankings as positional lookups for each candidate
    position_lookup = np.argsort(voterRankings, axis=1)  # Shape: (n, m)
    
    # Calculate pairwise preferences matrix
    pairwise_preferences = np.zeros((m, m), dtype=int)
    
    for i in range(m):
        for j in range(m):
            if i != j:
                # Count how many voters prefer candidate i over candidate j
                preferences_i_over_j = np.sum(position_lookup[:, i] < position_lookup[:, j])
                pairwise_preferences[i, j] = preferences_i_over_j
    
    # Initialize the strongest path matrix
    # strongest_paths[i][j] = strength of strongest path from i to j
    strongest_paths = np.zeros((m, m), dtype=int)
    
    # Initialize direct strengths between candidates
    for i in range(m):
        for j in range(m):
            if i != j:
                if pairwise_preferences[i, j] > pairwise_preferences[j, i]:
                    # i beats j directly
                    strongest_paths[i, j] = pairwise_preferences[i, j]
                else:
                    # i doesn't beat j directly
                    strongest_paths[i, j] = 0
    
    # Floyd-Warshall algorithm to find strongest paths
    # For each intermediate candidate k
    for k in range(m):
        # For each pair of candidates i, j
        for i in range(m):
            for j in range(m):
                if i != j and i != k and j != k:
                    # Check if path i->k->j is stronger than direct path i->j
                    # Strength of path is the minimum of its weakest link
                    path_strength = min(strongest_paths[i, k], strongest_paths[k, j])
                    if path_strength > strongest_paths[i, j]:
                        strongest_paths[i, j] = path_strength
    
    # Calculate Schulze scores
    # A candidate's score is the number of other candidates they beat
    # (have a stronger path to than the reverse path)
    schulze_scores = np.zeros(m, dtype=int)
    
    for i in range(m):
        for j in range(m):
            if i != j:
                if strongest_paths[i, j] > strongest_paths[j, i]:
                    schulze_scores[i] += 1
    
    # Return indices sorted by Schulze scores (descending order)
    return np.argsort(-schulze_scores)