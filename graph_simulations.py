from utility_sim import *
from value_generators import *
from scipy.stats import kendalltau
import csv
import matplotlib.pyplot as plt
from voting_rule_testing import *

def run_simulations_for_variable(variable_name, values, fixed_params, simulations=10000, beta_mode = 0):
	avg_borda_kt = []
	avg_copeland_kt = []
	avg_sum_kt = []

	for value in values:
		if value % 10 == 0:
			print("Finished %d simulations for var %s"%(value, variable_name))
		b_netKT, c_netKT, s_netKT = 0.0, 0.0, 0.0

		for _ in range(simulations):
			# Update the variable of interest
			if variable_name == "m":
				m, k, n = value, fixed_params["k"], fixed_params["n"]
			elif variable_name == "k":
				m, k, n = fixed_params["m"], value, fixed_params["n"]
			elif variable_name == "n":
				m, k, n = fixed_params["m"], fixed_params["k"], value

			alphas = generate_alphas(m, k, gaussian=False)
			true_beta = generate_betas(n, k, case=2)
			
			true_scores, true_ranking = get_ranked_scores(alphas, true_beta)
			true_borda_ranking = borda(true_ranking)
			true_copeland_ranking = copeland(true_ranking)
			true_score_ranking = scoresum(true_scores)
			
			error = generate_epsilon(n, k, bounds=(-0.2, 0.2), normal=False)
			obs_beta = true_beta + error

			obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)
			borda_ranking = borda(obs_ranking)
			copeland_ranking = copeland(obs_ranking)
			sum_ranking = scoresum(obs_scores)

			b_netKT += kendalltau(true_score_ranking, borda_ranking).statistic
			c_netKT += kendalltau(true_score_ranking, copeland_ranking).statistic
			s_netKT += kendalltau(true_score_ranking, sum_ranking).statistic

			# b_netKT += kendalltau(true_borda_ranking, borda_ranking).statistic
			# c_netKT += kendalltau(true_copeland_ranking, copeland_ranking).statistic
			# s_netKT += kendalltau(true_score_ranking, sum_ranking).statistic

		avg_borda_kt.append(b_netKT / simulations)
		avg_copeland_kt.append(c_netKT / simulations)
		avg_sum_kt.append(s_netKT / simulations)

	return avg_borda_kt, avg_copeland_kt, avg_sum_kt

# Parameters
simulations = 10000  # Number of simulations
fixed_params = {"m": 8, "k": 5, "n": 10}  # Fixed parameters for other variables

mvalues = range(2,51)
kvalues = range(2,31)
nvalues = range(2,101)

beta_modes = ("HU", "AN", "HN", "AU")
for mode, beta_text in enumerate(beta_modes):
	print("Testing", beta_text)
	# Run simulations for each variable
	avg_borda_m, avg_copeland_m, avg_sum_m = run_simulations_for_variable("m", mvalues, fixed_params, simulations, mode)
	# Save data for number of alternatives (m)
	save_to_csv("%s_kendall_tau_vs_m.csv"%(beta_text), mvalues, avg_borda_m, avg_copeland_m, avg_sum_m)
	print("Finished testing for m")

	avg_borda_k, avg_copeland_k, avg_sum_k = run_simulations_for_variable("k", kvalues, fixed_params, simulations, mode)
	# Save data for number of fairness notions (k)
	save_to_csv("%s_kendall_tau_vs_k.csv"%(beta_text), kvalues, avg_borda_k, avg_copeland_k, avg_sum_k)
	print("Finished testing for k")

	avg_borda_n, avg_copeland_n, avg_sum_n = run_simulations_for_variable("n", nvalues, fixed_params, simulations, mode)
	# Save data for number of players (n)
	save_to_csv("%s_kendall_tau_vs_n.csv"%(beta_text), nvalues, avg_borda_n, avg_copeland_n, avg_sum_n)
	print("Finished testing for n")

	# Save the plots as images
	plt.figure(1)
	plt.plot(mvalues, avg_borda_m, label="Borda", color="blue")
	plt.plot(mvalues, avg_copeland_m, label="Copeland", color="red")
	plt.plot(mvalues, avg_sum_m, label="Sum", color="green")
	plt.xlabel("Number of Alternatives (m)")
	plt.ylabel("Average Kendall Tau Distance")
	plt.legend()
	plt.title("[%s] Kendall Tau Distance vs. Number of Alternatives (m)"%(beta_text))
	plt.savefig("%s_kendall_tau_vs_m.png"%(beta_text))

	plt.figure(2)
	plt.plot(kvalues, avg_borda_k, label="Borda", color="blue")
	plt.plot(kvalues, avg_copeland_k, label="Copeland", color="red")
	plt.plot(kvalues, avg_sum_k, label="Sum", color="green")
	plt.xlabel("Number of Fairness Notions (k)")
	plt.ylabel("Average Kendall Tau Distance")
	plt.legend()
	plt.title("[%s] Kendall Tau Distance vs. Number of Fairness Notions (k)"%(beta_text))
	plt.savefig("%s_kendall_tau_vs_k.png"%(beta_text))

	plt.figure(3)
	plt.plot(nvalues, avg_borda_n, label="Borda", color="blue")
	plt.plot(nvalues, avg_copeland_n, label="Copeland", color="red")
	plt.plot(nvalues, avg_sum_n, label="Sum", color="green")
	plt.xlabel("Number of Players (n)")
	plt.ylabel("Average Kendall Tau Distance")
	plt.legend()
	plt.title("[%s] Kendall Tau Distance vs. Number of Players (n)"%(beta_text))
	plt.savefig("%s_kendall_tau_vs_n.png"%(beta_text))