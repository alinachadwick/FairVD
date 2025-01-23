from utility_sim import *
from value_generators import *
from scipy.stats import kendalltau
import csv
import matplotlib.pyplot as plt
from voting_rule_testing import *
from math import comb



def pairwise_tie_index(m, ties):
	total_pairwise_ties = 0
	for s in ties:
		total_pairwise_ties += comb(len(s), 2)
	return total_pairwise_ties / comb(m, 2)

def run_simulations_for_variable(variable_name, values, fixed_params, simulations=10000, beta_mode = 0, thresholds = [0.1,0.3,0.5,0.7,0.9]):
	avg_borda_kt = []
	avg_copeland_kt = []
	avg_sum_kt = []
	avg_approval_kt = {t:[] for t in thresholds}
	avg_approval_true_ties = {t:[] for t in thresholds}
	avg_approval_obs_ties = {t:[] for t in thresholds}
	avg_approval_true_PW_ties = {t:[] for t in thresholds}
	avg_approval_obs__PW_ties = {t:[] for t in thresholds}

	for value in values:
		if value % 10 == 0:
			print("Finished %d simulations for var %s"%(value, variable_name))
		# b_netKT, c_netKT, s_netKT = 0.0, 0.0, 0.0, 
		a_netKT =  {t:0.0 for t in thresholds}
		a_netTrueTies = {t:0 for t in thresholds}
		a_pairwiseTrueTies = {t:0 for t in thresholds}
		a_netObsTies = {t:0 for t in thresholds}
		a_pairwiseObsTies = {t:0 for t in thresholds}

		for _ in range(simulations):
			# Update the variable of interest
			if variable_name == "m":
				m, k, n = value, fixed_params["k"], fixed_params["n"]
			elif variable_name == "k":
				m, k, n = fixed_params["m"], value, fixed_params["n"]
			elif variable_name == "n":
				m, k, n = fixed_params["m"], fixed_params["k"], value

			alphas = generate_alphas(m, k, gaussian=False)
			true_beta = generate_betas(n, k, case=beta_mode)
			
			true_scores, true_ranking = get_ranked_scores(alphas, true_beta)
			# true_borda_ranking = borda(true_ranking)
			# true_copeland_ranking = copeland(true_ranking)
			# true_score_ranking = scoresum(true_scores)
			true_threshold_rankings = {}
			true_threshold_ties = {}
			for t in thresholds:
				true_approval_ranking, true_approval_ties = approval(true_scores/k, t)
				true_threshold_rankings[t] = true_approval_ranking
				true_threshold_ties[t] = true_approval_ties
				a_netTrueTies[t] += sum(map(lambda x: len(x), true_approval_ties))
				a_pairwiseTrueTies[t] += pairwise_tie_index(m, true_approval_ties)
			
			error = generate_epsilon(n, k, bounds=(-0.2, 0.2), normal=False)
			obs_beta = true_beta + error

			obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)
			# borda_ranking = borda(obs_ranking)
			# copeland_ranking = copeland(obs_ranking)
			# sum_ranking = scoresum(obs_scores)
			threshold_rankings = {}
			threshold_ties = {}
			for t in thresholds:
				approval_ranking, approval_ties = approval(obs_scores/k, t)
				threshold_rankings[t] = approval_ranking
				threshold_ties[t] = approval_ties
				a_netObsTies[t] += sum(map(lambda x: len(x), approval_ties))
				a_pairwiseObsTies[t] += pairwise_tie_index(m, approval_ties)

			# b_netKT += kendalltau_distance(true_score_ranking, borda_ranking)
			# c_netKT += kendalltau_distance(true_score_ranking, copeland_ranking)
			# s_netKT += kendalltau_distance(true_score_ranking, sum_ranking)

			# b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
			# c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
			# s_netKT += kendalltau_distance(true_score_ranking, sum_ranking)
			for t in thresholds:
				a_netKT[t] += kendall_tau_distance_with_ties(true_threshold_rankings[t], threshold_rankings[t], true_threshold_ties[t], threshold_ties[t])
		# avg_borda_kt.append(b_netKT / simulations)
		# avg_copeland_kt.append(c_netKT / simulations)
		# avg_sum_kt.append(s_netKT / simulations)
		for t in thresholds:
			avg_approval_kt[t].append(a_netKT[t] / simulations)
			avg_approval_true_ties[t].append(a_netTrueTies[t] / (simulations * m))
			avg_approval_obs_ties[t].append(a_netObsTies[t] / (simulations * m))
			avg_approval_true_PW_ties[t].append(a_pairwiseTrueTies[t] / simulations)
			avg_approval_obs__PW_ties[t].append(a_pairwiseObsTies[t] / simulations)

	# return avg_approval_kt, avg_approval_true_ties, avg_approval_obs_ties
	return avg_approval_kt, avg_approval_true_PW_ties, avg_approval_obs__PW_ties

if __name__ == "__main__":
	# Parameters
	simulations = 10000  # Number of simulations
	fixed_params = {"m": 8, "k": 5, "n": 10}  # Fixed parameters for other variables
	thresholds = [0.1,0.3,0.5,0.7,0.9]

	mvalues = range(2,51)
	kvalues = range(2,31)
	nvalues = range(2,101)

	color_list = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

	beta_modes = ("HU", "AN", "HN", "AU")
	for mode, beta_text in enumerate(beta_modes):
		print("Testing", beta_text)
		# Run simulations for each variable
		avg_approval_kt_m, avg_approval_true_ties_m, avg_approval_obs_ties_m = run_simulations_for_variable("m", mvalues, fixed_params, simulations, mode, thresholds=thresholds)
		# Save data for number of alternatives (m)
		save_to_csv("%s_APPROVAL_kendall_tau_vs_m.csv"%(beta_text), mvalues, avg_approval_kt_m, avg_approval_true_ties_m, avg_approval_obs_ties_m)
		print("Finished testing for m")

		avg_approval_kt_k, avg_approval_true_ties_k, avg_approval_obs_ties_k = run_simulations_for_variable("k", kvalues, fixed_params, simulations, mode, thresholds=thresholds)
		# Save data for number of fairness notions (k)
		save_to_csv("%s_APPROVAL_kendall_tau_vs_k.csv"%(beta_text), kvalues, avg_approval_kt_k, avg_approval_true_ties_k, avg_approval_obs_ties_k)
		print("Finished testing for k")

		avg_approval_kt_n, avg_approval_true_ties_n, avg_approval_obs_ties_n = run_simulations_for_variable("n", nvalues, fixed_params, simulations, mode, thresholds=thresholds)
		# Save data for number of players (n)
		save_to_csv("%s_APPROVAL_kendall_tau_vs_n.csv"%(beta_text), nvalues, avg_approval_kt_n, avg_approval_true_ties_n, avg_approval_obs_ties_n)
		print("Finished testing for n")

		# Save the plots as images
		paramss = [["m", avg_approval_kt_m, avg_approval_true_ties_m, avg_approval_obs_ties_m, mvalues],["k", avg_approval_kt_k, avg_approval_true_ties_k, avg_approval_obs_ties_k, kvalues],["n", avg_approval_kt_n, avg_approval_true_ties_n, avg_approval_obs_ties_n, nvalues]]
		
		counter = 1
		for p in paramss:
			changeVar, avg_approval_kt, avg_approval_true_ties, avg_approval_obs_ties, varValues = p

			# First figure
			plt.figure(counter)
			ax1 = plt.subplot(111)
			for t in thresholds:
				ax1.plot(varValues, avg_approval_kt[t], label="t = %.1f"%(t))
			ax1.set_xlabel(changeVar)
			ax1.set_ylabel("Average Kendall Tau Distance")
			ax1.set_title(f"[{beta_text}] Kendall Tau Distance vs. {changeVar}")

			# Adjust plot and position legend outside
			box1 = ax1.get_position()
			ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
			ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

			plt.savefig(f"{beta_text}_APPROVAL_kendall_tau_vs_{changeVar}.png")
			plt.savefig(f"{beta_text}_APPROVAL_kendall_tau_vs_{changeVar}.pdf", format="pdf", bbox_inches="tight")

			counter += 1

			# Second figure
			plt.figure(counter)
			ax2 = plt.subplot(111)
			for ci, t in enumerate(thresholds):
				# ax2.plot(varValues, avg_approval_true_ties[t], label=f"True; t = {t:.1f}")
				# ax2.plot(varValues, avg_approval_obs_ties[t], label=f"Obs; t = {t:.1f}")
				ax2.plot(varValues, avg_approval_true_ties[t], label=f"True; t = {t:.1f}", color = color_list[ci])
				ax2.plot(varValues, avg_approval_obs_ties[t], label=f"Obs; t = {t:.1f}", color = color_list[ci],  linestyle = ":")
			ax2.set_xlabel(changeVar)
			ax2.set_ylabel("% of Tied Pairs")
			ax2.set_title(f"[{beta_text}] % Tied Pairs vs. {changeVar}")

			# Adjust plot and position legend outside
			box2 = ax2.get_position()
			ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
			ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

			plt.savefig(f"{beta_text}_APPROVAL_ties_vs_{changeVar}.png")
			plt.savefig(f"{beta_text}_APPROVAL_ties_vs_{changeVar}.pdf", format="pdf", bbox_inches="tight")

			counter += 1
			plt.close('all')
