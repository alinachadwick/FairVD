from utilities import *
from modelling import *
import numpy as np
from voting_rules import *
from value_generators import *
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor

def is_multiple_of(value, base=0.05, tol=1e-8):
	return math.isclose(value % base, 0.0, abs_tol=tol)

def run_intra_inter_simulations(simulations, m, n, k, dist_name, intra_values, inter_values, alpha_generator, plot_interval=1):
	lower_bound = 0.0
	results = []

	for intra in intra_values:
		for inter in inter_values:
			b_netKT = 0.0
			c_netKT = 0.0
			s_netKT = 0.0

			for _ in range(simulations):
				alphas = alpha_generator(m, k, intra, inter, lower_bound)
				true_beta = generate_betas(n, k, case=0)
				true_scores, true_ranking = get_ranked_scores(alphas, true_beta)
				true_sum_ranking = scoresum(true_scores)
				true_borda_ranking = borda(true_ranking)
				true_copeland_ranking = copeland(true_ranking)

				error = generate_epsilon(n, k, bounds=(-0.1, 0.1), sd=0.1, normal=False)
				obs_beta = true_beta + error
				obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)

				borda_ranking = borda(obs_ranking)
				copeland_ranking = copeland(obs_ranking)
				sum_ranking = scoresum(obs_scores)

				b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
				c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
				s_netKT += kendalltau_distance(true_sum_ranking, sum_ranking)

			avg_borda_kt = b_netKT / simulations
			avg_copeland_kt = c_netKT / simulations
			avg_sum_kt = s_netKT / simulations

			results.append({
				"m": m,
				"n": n,
				"k": k,
				"distribution": dist_name,
				"Intra": intra,
				"Inter": inter,
				"Borda": avg_borda_kt,
				"Copeland": avg_copeland_kt,
				"Sum": avg_sum_kt
			})

	# Export CSV
	filename = f"results_{dist_name}_m{m}_n{n}_k{k}.csv"
	export_results_to_csv(results, filename)

	df_results = pd.DataFrame(results)

	# 1. KT vs Intra (for each inter)
	unique_inters = sorted(set(df_results["Inter"]))
	for idx, inter in enumerate(unique_inters):
		if idx % plot_interval != 0:
			continue
		subset = df_results[df_results["Inter"] == inter]
		plot_graphs(
			df=subset,
			x_col="Intra",
			y_cols=["Borda", "Copeland", "Sum"],
			title=f"{dist_name.capitalize()} | Inter={inter:.2f} | KT vs Intra (m={m}, n={n}, k={k})",
			filename=f"{dist_name}_kt_vs_intra_inter{inter:.2f}_m{m}_n{n}_k{k}"
		)

	# 2. KT vs Inter (for each intra)
	unique_intras = sorted(set(df_results["Intra"]))
	for idx, intra in enumerate(unique_intras):
		if idx % plot_interval != 0:
			continue
		subset = df_results[df_results["Intra"] == intra]
		plot_graphs(
			df=subset,
			x_col="Inter",
			y_cols=["Borda", "Copeland", "Sum"],
			title=f"{dist_name.capitalize()} | Intra={intra:.2f} | KT vs Inter (m={m}, n={n}, k={k})",
			filename=f"{dist_name}_kt_vs_inter_intra{intra:.2f}_m{m}_n{n}_k{k}"
		)


def run_intra_inter_simulations_with_approval(simulations, m, n, k, dist_name, intra_values, inter_values, alpha_generator, plot_interval=1, thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]):
	results = []
	lower_bound = 0.0

	def compute_approval_kt(t, true_scores, obs_scores, k, simulate_both=True):
		true_ranking, true_ties = approval(true_scores / k, t)
		obs_ranking, obs_ties = approval(obs_scores / k, t)
		ktTies, ktNoTies = kendall_tau_distance_with_ties(
			true_ranking, obs_ranking, true_ties, obs_ties,
			tiePenalty=True, arbitraryBreak=False, simulateBoth=simulate_both
		)
		return t, ktTies, ktNoTies

	for intra in intra_values:
		for inter in inter_values:
			b_netKT = 0.0
			c_netKT = 0.0
			s_netKT = 0.0
			a_netKT_with_penalty = {t: 0.0 for t in thresholds}
			a_netKT_no_penalty = {t: 0.0 for t in thresholds}

			for _ in range(simulations):
				alphas = alpha_generator(m, k, intra, inter, lower_bound)
				true_beta = generate_betas(n, k, case=0)
				true_scores, true_ranking = get_ranked_scores(alphas, true_beta)
				true_sum_ranking = scoresum(true_scores)
				true_borda_ranking = borda(true_ranking)
				true_copeland_ranking = copeland(true_ranking)

				error = generate_epsilon(n, k, bounds=(-0.1, 0.1), sd=0.1, normal=False)
				obs_beta = true_beta + error
				obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)

				borda_ranking = borda(obs_ranking)
				copeland_ranking = copeland(obs_ranking)
				sum_ranking = scoresum(obs_scores)

				b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
				c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
				s_netKT += kendalltau_distance(true_sum_ranking, sum_ranking)

				with ThreadPoolExecutor() as executor:
					futures = [executor.submit(compute_approval_kt, t, true_scores, obs_scores, k) for t in thresholds]
					for future in futures:
						t, ktTies, ktNoTies = future.result()
						a_netKT_with_penalty[t] += ktTies
						a_netKT_no_penalty[t] += ktNoTies

			avg_borda_kt = b_netKT / simulations
			avg_copeland_kt = c_netKT / simulations
			avg_sum_kt = s_netKT / simulations

			result_entry = {
				"m": m,
				"n": n,
				"k": k,
				"distribution": dist_name,
				"Intra": intra,
				"Inter": inter,
				"Borda": avg_borda_kt,
				"Copeland": avg_copeland_kt,
				"Sum": avg_sum_kt
			}
			for t in thresholds:
				result_entry[f"Approval (t={t}, penalty=1)"] = a_netKT_with_penalty[t] / simulations
				result_entry[f"Approval (t={t}, penalty=0)"] = a_netKT_no_penalty[t] / simulations

			results.append(result_entry)

	filename = f"results_{dist_name}_m{m}_n{n}_k{k}.csv"
	export_results_to_csv(results, filename)
	df_results = pd.DataFrame(results)

	unique_inters = sorted(set(df_results["Inter"]))
	for idx, inter in enumerate(unique_inters):
		if idx % plot_interval != 0:
			continue
		subset = df_results[df_results["Inter"] == inter]
		plot_graphs(
			df=subset,
			x_col="Intra",
			y_cols=["Borda", "Copeland", "Sum"],
			title=f"{dist_name.capitalize()} | Inter Dist.={inter:.2f} | KT vs Intra (m={m}, n={n}, k={k})",
			filename=f"{dist_name}_kt_vs_intra_inter{inter:.2f}_m{m}_n{n}_k{k}"
		)
		plot_approval_only_graph(
			df=subset,
			x_col="Intra",
			thresholds=thresholds,
			title=f"{dist_name.capitalize()} | Inter Dist.={inter:.2f} | Approval KT vs Intra (m={m}, n={n}, k={k})",
			filename=f"{dist_name}_approval_kt_vs_intra_inter{inter:.2f}_m{m}_n{n}_k{k}"
		)

	unique_intras = sorted(set(df_results["Intra"]))
	for idx, intra in enumerate(unique_intras):
		if idx % plot_interval != 0:
			continue
		subset = df_results[df_results["Intra"] == intra]
		plot_graphs(
			df=subset,
			x_col="Inter",
			y_cols=["Borda", "Copeland", "Sum"],
			title=f"{dist_name.capitalize()} | Intra Dist.={intra:.2f} | KT vs Inter (m={m}, n={n}, k={k})",
			filename=f"{dist_name}_kt_vs_inter_intra{intra:.2f}_m{m}_n{n}_k{k}"
		)
		plot_approval_only_graph(
			df=subset,
			x_col="Inter",
			thresholds=thresholds,
			title=f"{dist_name.capitalize()} | Intra Dist.={intra:.2f} | Approval KT vs Inter (m={m}, n={n}, k={k})",
			filename=f"{dist_name}_approval_kt_vs_inter_intra{intra:.2f}_m{m}_n{n}_k{k}"
		)
		
def run_parameter_simulations_with_approval(simulations, m, n, k):
	# Define parameter configurations
	alpha_methods = [False, True]  # gaussian = False, True
	beta_cases = [0]      # case = 0, 1, 2, 3
	epsilon_methods = [False, True]  # normal = False, True

	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

	# Storage for results
	results = []

	for alpha_gaussian in alpha_methods:
		for beta_case in beta_cases:
			for epsilon_normal in epsilon_methods:
				b_netKT = 0.0
				c_netKT = 0.0
				s_netKT = 0.0
				a_netKT_with_penalty = {t: 0.0 for t in thresholds}
				a_netKT_no_penalty = {t: 0.0 for t in thresholds}

				for _ in range(simulations):
					alphas = generate_alphas(m, k, gaussian=alpha_gaussian)
					true_beta = generate_betas(n, k, case=beta_case)
					true_scores, true_ranking = get_ranked_scores(alphas, true_beta)

					true_sum_ranking = scoresum(true_scores)
					true_borda_ranking = borda(true_ranking)
					true_copeland_ranking = copeland(true_ranking)

					# Precompute approval rankings from true scores
					true_threshold_rankings = {}
					true_threshold_ties = {}
					for t in thresholds:
						ranking, ties = approval(true_scores / k, t)
						true_threshold_rankings[t] = ranking
						true_threshold_ties[t] = ties

					error = generate_epsilon(n, k, bounds=(-0.1, 0.1), sd=0.1, normal=epsilon_normal)
					obs_beta = true_beta + error
					obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)

					borda_ranking = borda(obs_ranking)
					copeland_ranking = copeland(obs_ranking)
					sum_ranking = scoresum(obs_scores)

					b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
					c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
					s_netKT += kendalltau_distance(true_sum_ranking, sum_ranking)

					# Approval voting evaluation
					for t in thresholds:
						obs_ranking, obs_ties = approval(obs_scores / k, t)
						kt_with, kt_without = kendall_tau_distance_with_ties(
							true_threshold_rankings[t], obs_ranking,
							true_threshold_ties[t], obs_ties,
							tiePenalty=True, arbitraryBreak=False, simulateBoth=True
						)
						a_netKT_with_penalty[t] += kt_with
						a_netKT_no_penalty[t] += kt_without

				# Average KT distances
				avg_borda_kt = b_netKT / simulations
				avg_copeland_kt = c_netKT / simulations
				avg_sum_kt = s_netKT / simulations

				result_entry = {
					"alpha_gaussian": alpha_gaussian,
					"beta_case": beta_case,
					"epsilon_normal": epsilon_normal,
					"avg_borda_kt": avg_borda_kt,
					"avg_copeland_kt": avg_copeland_kt,
					"avg_sum_kt": avg_sum_kt
				}
				for t in thresholds:
					result_entry[f"approval_t{t}_penalty1"] = a_netKT_with_penalty[t] / simulations
					result_entry[f"approval_t{t}_penalty0"] = a_netKT_no_penalty[t] / simulations

				results.append(result_entry)

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
		for t in thresholds:
			print(
				f"  Approval KT (t={t}, penalty=1): {result[f'approval_t{t}_penalty1']:.3f}, "
				f"penalty=0: {result[f'approval_t{t}_penalty0']:.3f}"
			)

	filename = f"paramresults__m{m}_n{n}_k{k}.csv"
	export_results_to_csv(results, filename)

def run_parameter_simulations(simulations, m, n, k):
	# Define parameter configurations
	alpha_methods = [False, True]  # gaussian = False, True
	beta_cases = [0]               # case = 0, 1, 2, 3
	epsilon_methods = [False, True]  # normal = False, True

	# Storage for results
	results = []

	for alpha_gaussian in alpha_methods:
		for beta_case in beta_cases:
			for epsilon_normal in epsilon_methods:
				b_netKT = 0.0
				c_netKT = 0.0
				s_netKT = 0.0
				rp_netKT = 0.0
				mmc_netKT = 0.0
				schulze_netKT = 0.0

				for _ in range(simulations):
					alphas = generate_alphas(m, k, gaussian=alpha_gaussian)
					true_beta = generate_betas(n, k, case=beta_case)
					true_scores, true_ranking = get_ranked_scores(alphas, true_beta)

					true_sum_ranking = scoresum(true_scores)
					true_borda_ranking = borda(true_ranking)
					true_copeland_ranking = copeland(true_ranking)

					# new additions: additional PMC voting rules
					true_rp_ranking = ranked_pairs(true_ranking)
					true_mm_condorcet_ranking = minimax_condorcet(true_ranking)
					true_schulze_ranking = schulze(true_ranking)

					error = generate_epsilon(n, k, bounds=(-0.1, 0.1), sd=0.1, normal=epsilon_normal)
					obs_beta = true_beta + error
					obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)

					borda_ranking = borda(obs_ranking)
					copeland_ranking = copeland(obs_ranking)
					sum_ranking = scoresum(obs_scores)

					rp_ranking = ranked_pairs(obs_ranking)
					mm_condorcet_ranking = minimax_condorcet(obs_ranking)
					schulze_ranking = schulze(obs_ranking)

					b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
					c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
					s_netKT += kendalltau_distance(true_sum_ranking, sum_ranking)
					rp_netKT += kendalltau_distance(true_rp_ranking, rp_ranking)
					mmc_netKT += kendalltau_distance(true_mm_condorcet_ranking, mm_condorcet_ranking)
					schulze_netKT += kendalltau_distance(true_schulze_ranking, schulze_ranking)

				# Average KT distances
				avg_borda_kt = b_netKT / simulations
				avg_copeland_kt = c_netKT / simulations
				avg_sum_kt = s_netKT / simulations
				avg_rp_kt = rp_netKT / simulations
				avg_mmc_kt = mmc_netKT / simulations
				avg_schulze_kt = schulze_netKT / simulations
				
				result_entry = {
					"alpha_gaussian": alpha_gaussian,
					"beta_case": beta_case,
					"epsilon_normal": epsilon_normal,
					"avg_borda_kt": avg_borda_kt,
					"avg_copeland_kt": avg_copeland_kt,
					"avg_sum_kt": avg_sum_kt,
					"avg_rp_kt": avg_rp_kt,
					"avg_mmc_kt": avg_mmc_kt,
					"avg_schulze_kt": avg_schulze_kt
				}

				results.append(result_entry)

	# Print results
	for result in results:
		print(
			f"Alpha Gaussian: {result['alpha_gaussian']}, "
			f"Beta Case: {result['beta_case']}, "
			f"Epsilon Normal: {result['epsilon_normal']} -> "
			f"Borda KT: {result['avg_borda_kt']:.3f}, "
			f"Copeland KT: {result['avg_copeland_kt']:.3f}, "
			f"Sum KT: {result['avg_sum_kt']:.3f}"
			f"Ranked Pairs KT: {result['avg_rp_kt']:.3f}, "
			f"Minimax Condorcet KT: {result['avg_mmc_kt']:.3f}, "
			f"Schulze KT: {result['avg_schulze_kt']:.3f}"
		)

	filename = f"paramresults__m{m}_n{n}_k{k}.csv"
	export_results_to_csv(results, filename)

def run_variable_simulations_with_plots(simulations, fixed_params={"m": 8, "n": 10, "k": 5}):
	mvalues = range(2, 51)
	kvalues = range(2, 31)
	nvalues = range(2, 101)

	output_dir = os.path.join(script_dir, "output")
	os.makedirs(output_dir, exist_ok=True)  # Create output folder if not exists

	def run_single(case, variable_name, values):
		avg_borda_kt = []
		avg_copeland_kt = []
		avg_sum_kt = []
		records = []

		for value in values:
			if value % 10 == 0:
				print("Finished %d simulations for var %s in case %d" % (value, variable_name, case))
			b_netKT, c_netKT, s_netKT = 0.0, 0.0, 0.0

			for _ in range(simulations):
				if variable_name == "m":
					m, k, n = value, fixed_params["k"], fixed_params["n"]
				elif variable_name == "k":
					m, k, n = fixed_params["m"], value, fixed_params["n"]
				elif variable_name == "n":
					m, k, n = fixed_params["m"], fixed_params["k"], value

				alphas = generate_alphas(m, k, gaussian=False)
				true_beta = generate_betas(n, k, case=case)

				true_scores, true_ranking = get_ranked_scores(alphas, true_beta)
				true_borda_ranking = borda(true_ranking)
				true_copeland_ranking = copeland(true_ranking)
				true_score_ranking = scoresum(true_scores)

				error = generate_epsilon(n, k, bounds=(-0.1, 0.1), normal=False)
				obs_beta = true_beta + error
				obs_scores, obs_ranking = get_ranked_scores(alphas, obs_beta)

				borda_ranking = borda(obs_ranking)
				copeland_ranking = copeland(obs_ranking)
				sum_ranking = scoresum(obs_scores)

				b_netKT += kendalltau_distance(true_borda_ranking, borda_ranking)
				c_netKT += kendalltau_distance(true_copeland_ranking, copeland_ranking)
				s_netKT += kendalltau_distance(true_score_ranking, sum_ranking)

			avg_borda = b_netKT / simulations
			avg_copeland = c_netKT / simulations
			avg_sum = s_netKT / simulations

			avg_borda_kt.append(avg_borda)
			avg_copeland_kt.append(avg_copeland)
			avg_sum_kt.append(avg_sum)

			records.append({
				variable_name: value,
				"Borda": avg_borda,
				"Copeland": avg_copeland,
				"Sum": avg_sum
			})

		df = pd.DataFrame(records)
		case_name = ["HU", "AU", "HN", "AN"][case]
		csv_filename = os.path.join(output_dir, f"kt_vs_{variable_name}_{case_name}.csv")
		df.to_csv(csv_filename, index=False)
		return avg_borda_kt, avg_copeland_kt, avg_sum_kt

	for case in range(4):
		case_name = ["HU", "AU", "HN", "AN"][case]
		print(f"Testing case {case_name}")
		for variable_name, values, xlabel in [
				("m", mvalues, "Number of Alternatives (m)"),
				("k", kvalues, "Number of Fairness Notions (k)"),
				("n", nvalues, "Number of Voters (n)")
			]:
			avg_borda, avg_copeland, avg_sum = run_single(case, variable_name, values)

			plt.figure()
			plt.plot(values, avg_borda, label="Borda", color="blue")
			plt.plot(values, avg_copeland, label="Copeland", color="red")
			plt.plot(values, avg_sum, label="Sum", color="green")
			plt.xlabel(xlabel)
			plt.ylabel("Average Kendall Tau Distance")
			plt.legend()
			plt.title(f"{case_name}: Kendall Tau Distance vs. {xlabel}")

			png_filename = os.path.join(output_dir, f"kt_vs_{variable_name}_{case_name}.png")
			pdf_filename = os.path.join(output_dir, f"kt_vs_{variable_name}_{case_name}.pdf")
			plt.savefig(png_filename)
			plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
			plt.close()