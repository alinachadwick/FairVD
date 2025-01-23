from utility_sim import *
from value_generators import *
from scipy.stats import kendalltau
import csv
import matplotlib.pyplot as plt
from voting_rule_testing import *

def kendalltau_distance(*args, **kwargs):
	tau_coef = kendalltau(*args, **kwargs)
	return (1 - tau_coef.statistic) / 2

def run_simulations_for_variable(variable_name, values, fixed_params, simulations=10000, beta_mode=0, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
	avg_borda_kt = []
	avg_copeland_kt = []
	avg_sum_kt = []
	avg_threshold_kt = {t: [] for t in thresholds}

	for value in values:
		if value % 10 == 0:
			print("Finished %d simulations for var %s" % (value, variable_name))
		b_netKT, c_netKT, s_netKT = 0.0, 0.0, 0.0
		a_netKT = {t: 0.0 for t in thresholds}

		for _ in range(simulations):
			if variable_name == "m":
				m, k, n = value, fixed_params["k"], fixed_params["n"]
			elif variable_name == "k":
				m, k, n = fixed_params["m"], value, fixed_params["n"]
			elif variable_name == "n":
				m, k, n = fixed_params["m"], fixed_params["k"], value

			alphas = generate_alphas(m, k, gaussian=False)
			true_beta = generate_betas(n, k, case=beta_mode)

			true_scores, true_ranking = get_ranked_scores(alphas, true_beta)
			true_borda_ranking = borda(true_ranking)
			true_copeland_ranking = copeland(true_ranking)
			true_score_ranking = scoresum(true_scores)

			true_approval_rankings = {t: approval(true_scores / k, t)[0] for t in thresholds}

			error = generate_epsilon(n, k, bounds=(-0.2, 0.2), normal=False)
			obs_beta = true_beta + error

			obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)
			borda_ranking = borda(obs_ranking)
			copeland_ranking = copeland(obs_ranking)
			sum_ranking = scoresum(obs_scores)

			b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
			c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
			s_netKT += kendalltau_distance(true_score_ranking, sum_ranking)

			for t in thresholds:
				approval_ranking, _ = approval(obs_scores / k, t)
				a_netKT[t] += kendalltau_distance(true_approval_rankings[t], approval_ranking)

		avg_borda_kt.append(b_netKT / simulations)
		avg_copeland_kt.append(c_netKT / simulations)
		avg_sum_kt.append(s_netKT / simulations)
		for t in thresholds:
			avg_threshold_kt[t].append(a_netKT[t] / simulations)

	return avg_borda_kt, avg_copeland_kt, avg_sum_kt, avg_threshold_kt

def get_best_approval_line(avg_threshold_kt, thresholds):
	best_threshold = min(thresholds, key=lambda t: sum(avg_threshold_kt[t]) / len(avg_threshold_kt[t]))
	return best_threshold, avg_threshold_kt[best_threshold]

if __name__ == "__main__":
	simulations = 10000
	fixed_params = {"m": 8, "k": 5, "n": 10}

	mvalues = range(2, 51)
	kvalues = range(2, 31)
	nvalues = range(2, 101)

	beta_modes = ("HU", "AN", "HN", "AU")
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

	for mode, beta_text in enumerate(beta_modes):
		print("Testing", beta_text)

		avg_borda_m, avg_copeland_m, avg_sum_m, avg_threshold_m = run_simulations_for_variable("m", mvalues, fixed_params, simulations, mode, thresholds)
		best_threshold_m, best_line_m = get_best_approval_line(avg_threshold_m, thresholds)

		plt.figure(1)
		plt.plot(mvalues, avg_borda_m, label="Borda", color="blue")
		plt.plot(mvalues, avg_copeland_m, label="Copeland", color="red")
		plt.plot(mvalues, avg_sum_m, label="Sum", color="green")
		plt.plot(mvalues, best_line_m, label=f"Approval; t = {best_threshold_m:.1f}", color="purple", linestyle="--")
		plt.xlabel("Number of Alternatives (m)")
		plt.ylabel("Average Kendall Tau Distance")
		plt.legend()
		plt.title(f"[{beta_text}] Kendall Tau Distance vs. Number of Alternatives (m)")
		plt.savefig(f"{beta_text}_kendall_tau_vs_m.png")
		plt.savefig(f"{beta_text}_kendall_tau_vs_m.pdf", format="pdf", bbox_inches="tight")

		avg_borda_k, avg_copeland_k, avg_sum_k, avg_threshold_k = run_simulations_for_variable("k", kvalues, fixed_params, simulations, mode, thresholds)
		best_threshold_k, best_line_k = get_best_approval_line(avg_threshold_k, thresholds)

		plt.figure(2)
		plt.plot(kvalues, avg_borda_k, label="Borda", color="blue")
		plt.plot(kvalues, avg_copeland_k, label="Copeland", color="red")
		plt.plot(kvalues, avg_sum_k, label="Sum", color="green")
		plt.plot(kvalues, best_line_k, label=f"Approval; t = {best_threshold_k:.1f}", color="purple", linestyle="--")
		plt.xlabel("Number of Fairness Notions (k)")
		plt.ylabel("Average Kendall Tau Distance")
		plt.legend()
		plt.title(f"[{beta_text}] Kendall Tau Distance vs. Number of Fairness Notions (k)")
		plt.savefig(f"{beta_text}_kendall_tau_vs_k.png")
		plt.savefig(f"{beta_text}_kendall_tau_vs_k.pdf", format="pdf", bbox_inches="tight")

		avg_borda_n, avg_copeland_n, avg_sum_n, avg_threshold_n = run_simulations_for_variable("n", nvalues, fixed_params, simulations, mode, thresholds)
		best_threshold_n, best_line_n = get_best_approval_line(avg_threshold_n, thresholds)

		plt.figure(3)
		plt.plot(nvalues, avg_borda_n, label="Borda", color="blue")
		plt.plot(nvalues, avg_copeland_n, label="Copeland", color="red")
		plt.plot(nvalues, avg_sum_n, label="Sum", color="green")
		plt.plot(nvalues, best_line_n, label=f"Approval; t = {best_threshold_n:.1f}", color="purple", linestyle="--")
		plt.xlabel("Number of Voters (n)")
		plt.ylabel("Average Kendall Tau Distance")
		plt.legend()
		plt.title(f"[{beta_text}] Kendall Tau Distance vs. Number of Voters (n)")
		plt.savefig(f"{beta_text}_kendall_tau_vs_n.png")
		plt.savefig(f"{beta_text}_kendall_tau_vs_n.pdf", format="pdf", bbox_inches="tight")

		plt.close('all')