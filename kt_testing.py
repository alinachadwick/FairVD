import random
from scipy.stats import kendalltau


import numpy as np

def kendall_tau_distance_with_ties(x, y):
    """
    Calculate the Kendall Tau distance (Tau-c based) between two lists x and y,
    incorporating the handling of ties as outlined in the pseudo-code.
    
    Parameters:
        x (list or np.ndarray): First ranking.
        y (list or np.ndarray): Second ranking.
        
    Returns:
        float: Kendall Tau distance between x and y.
    """
    if len(x) != len(y):
        raise ValueError("Both lists must have the same length.")
    
    # Convert inputs to numpy arrays for optimized computations
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    
    # Initialize concordant, discordant, and tie counts
    kt_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] and y[i] != y[j]:  # Tie in x but not in y
                kt_distance += 0.5
            elif y[i] == y[j] and x[i] != x[j]:  # Tie in y but not in x
                kt_distance += 0.5
            elif x[i] == x[j] and y[i] == y[j]:  # Tie in both x and y
                kt_distance += 0
            else:  # Compare normally
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    kt_distance += 0  # Concordant pair
                else:
                    kt_distance += 1  # Discordant pair
    
    # Normalize the distance to be between 0 and 1
    max_pairs = n * (n - 1) / 2  # Total number of possible pairs
    normalized_distance = kt_distance / max_pairs
    
    return normalized_distance

# Example usage
x = [1, 2, 2, 3, 4]
y = [4, 5, 5, 3, 1]
print("Kendall Tau Distance with Ties:", kendall_tau_distance_with_ties(x, y))


for i in range(10):
	l = 4
	x = list(range(l))
	rank_a = x.copy()
	rank_b = x.copy()
	random.shuffle(rank_a)
	random.shuffle(rank_b)

	index_a = [rank_a.index(i) for i in range(l)]
	index_b = [rank_b.index(i) for i in range(l)]
	
	rank_kt = kendalltau(rank_a, rank_b).statistic
	index_kt = kendalltau(index_a, index_b).statistic
	same = 1
	if rank_kt != index_kt:
		same = 0
	print("Same: %d; Ranks: %.3f ; Index: %.3f"%(same, rank_kt, index_kt))