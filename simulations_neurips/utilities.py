import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# Create an 'output' folder in that same directory
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output folder: {output_dir}")

def export_results_to_csv(results, filename="corr_results.csv"):
	if not results:
		print("No results to export.")
		return

	filepath = os.path.join(output_dir, filename)
	with open(filepath, mode="w", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
		writer.writeheader()
		writer.writerows(results)

	print(f"Results exported to: {filepath}")

def plot_graphs(df, x_col, y_cols, title, filename, dist = True):
	if isinstance(y_cols, str):
		y_cols = [y_cols]

	df_sorted = df.sort_values(by=x_col)

	max_y_val = max(df_sorted[y].max() for y in y_cols)
	y_max_scaled = max_y_val * 1.1 if max_y_val > 0.005 else 0.005

	plt.figure(figsize=(8, 5))
	for y in y_cols:
		plt.plot(df_sorted[x_col], df_sorted[y], marker='o', markersize=4, label=y)

	plt.xlabel(x_col.capitalize() + " Dist." if dist else x_col.capitalize())
	plt.ylabel("Kendall Tau Distance")
	plt.title(title)
	plt.ylim(0, y_max_scaled)
	plt.xlim(df_sorted[x_col].min(), df_sorted[x_col].max())
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend()
	plt.tight_layout()

	plot_path = os.path.join(output_dir, f"{filename}.pdf")
	plt.savefig(plot_path)
	plt.close()
	print(f"Plot saved to: {plot_path}")
	return plot_path

def plot_approval_only_graph(df, x_col, thresholds, title, filename, dist = True):
    import matplotlib.pyplot as plt
    import os

    for penalty_flag in [1, 0]:
        plt.figure(figsize=(8, 5))
        y_max = 0
        suffix = "tiePenal" if penalty_flag else "noPenal"

        # Find max y-value across all thresholds for this penalty mode
        for t in thresholds:
            col = f"Approval (t={t}, penalty={penalty_flag})"
            if col in df.columns:
                y_vals = df[col].values
                y_max = max(y_max, max(y_vals))

        y_max = y_max * 1.1 if y_max > 0.005 else 0.005

        # Plot each threshold line
        for t in thresholds:
            col = f"Approval (t={t}, penalty={penalty_flag})"
            if col in df.columns:
                plt.plot(df[x_col], df[col], marker='o', label=f"Threshold={t}")

        plt.xlabel(x_col.capitalize() + " Dist." if dist else x_col.capitalize())
        plt.ylabel("Kendall Tau Distance")
        plt.title(title + (" (Penalty Applied)" if penalty_flag else " (No Penalty)"))
        plt.ylim(0, y_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"approval_{suffix}_{filename}.pdf")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to: {plot_path}")
