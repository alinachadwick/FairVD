from voting_rule_testing import *
from utility_sim import *
from value_generators import *
from scipy.stats import kendalltau
import csv
import matplotlib.pyplot as plt
import numpy as np


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
    tieList.sort(key=lambda x: min(x))  # Sort by the smallest index in each group

    # Create orderedIndexes
    orderedIndexes = indexes_sortByVal(mechScores)

    return orderedIndexes, tieList