from simulators import *

intra_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
inter_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

simulations = 100000

# run_intra_inter_simulations_with_approval(simulations, 8, 10, 5, "uniform", intra_values, inter_values, generate_uniform_corr_alphas, 1)
# run_intra_inter_simulations(simulations, 20, 50, 5, "uniform", intra_values, inter_values, generate_uniform_corr_alphas, 1)
# run_intra_inter_simulations(simulations, 20, 10, 5, "uniform", intra_values, inter_values, generate_uniform_corr_alphas, 1)
# run_intra_inter_simulations(simulations, 8, 50, 5, "uniform", intra_values, inter_values, generate_uniform_corr_alphas, 1)

# run_intra_inter_simulations_with_approval(simulations, 8, 10, 5, "normal", intra_values, inter_values, generate_normal_corr_alphas, 1)
# run_intra_inter_simulations(simulations, 20, 50, 5, "normal", intra_values, inter_values, generate_normal_corr_alphas, 1)
# run_intra_inter_simulations(simulations, 20, 10, 5, "normal", intra_values, inter_values, generate_normal_corr_alphas, 1)
# run_intra_inter_simulations(simulations, 8, 50, 5, "normal", intra_values, inter_values, generate_normal_corr_alphas, 1)

run_parameter_simulations(simulations, 8, 10, 5)

# run_variable_simulations_with_plots(simulations)