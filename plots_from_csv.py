import os
import pandas as pd
import matplotlib.pyplot as plt

input_dir = "./input"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

def plot_graph(df, x_col, y_cols, title, filename):
    df_sorted = df.sort_values(by=x_col)
    y_max = max(df_sorted[y].max() for y in y_cols)
    y_max_scaled = y_max * 1.1 if y_max > 0.005 else 0.005
    y_min_scaled = -0.02 * y_max_scaled  # Add small bottom margin

    plt.figure(figsize=(8, 5))
    markers = {'Borda': 'o', 'Copeland': '+', 'Sum': 'x'}
    any_plotted = False
    for y in y_cols:
        if y in df_sorted.columns:
            plt.plot(df_sorted[x_col], df_sorted[y], linestyle='None', marker=markers.get(y, 'o'), markersize=5, label=y)
            any_plotted = True

    plt.xlabel(x_col + " Dist.")
    plt.ylabel("Kendall Tau Distance")
    plt.title(title)
    plt.ylim(y_min_scaled, y_max_scaled)
    x_min = df_sorted[x_col].min()
    x_max = df_sorted[x_col].max()
    x_range = x_max - x_min
    plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    if any_plotted:
        plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def plot_approval_graphs(df, x_col, thresholds, dist_name, fixed_val, fixed_type, m, n, k):
    marker_cycle = ['o', '+', 'x', '^', 's']
    for penalty in [1, 0]:
        suffix = "tiePenal" if penalty == 1 else "noPenal"
        y_max = 0
        plotted_cols = []
        for t in thresholds:
            col = f"Approval (t={t}, penalty={penalty})"
            if col in df.columns:
                y_max = max(y_max, df[col].max())
                plotted_cols.append((t, col))
        if not plotted_cols:
            continue  # Skip plotting if no data is found

        y_max = y_max * 1.1 if y_max > 0.005 else 0.005
        y_min = -0.02 * y_max  # Small bottom margin
        plt.figure(figsize=(8, 5))
        for idx, (t, col) in enumerate(plotted_cols):
            plt.plot(df[x_col], df[col], linestyle='None', marker=marker_cycle[idx % len(marker_cycle)], markersize=5, label=f"Threshold={t}")

        plt.xlabel(x_col + " Dist.")
        plt.ylabel("Kendall Tau Distance")

        if x_col.lower() == "inter":
            title = f"{dist_name.capitalize()} | Intra Dist.={fixed_val:.2f} | Approval KT vs Inter (m={m}, n={n}, k={k})"
        else:
            title = f"{dist_name.capitalize()} | Inter Dist.={fixed_val:.2f} | Approval KT vs Intra (m={m}, n={n}, k={k})"

        plt.title(title + (" (Penalty Applied)" if penalty else " (No Penalty)"))
        plt.ylim(y_min, y_max)
        x_min = df[x_col].min()
        x_max = df[x_col].max()
        x_range = x_max - x_min
        plt.xlim(x_min - 0.02 * x_range, x_max + 0.02 * x_range)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        filename = f"approval_{suffix}_{dist_name}_approval_kt_vs_{x_col.lower()}_{fixed_type.lower()}{fixed_val:.2f}_m{m}_n{n}_k{k}"
        path = os.path.join(output_dir, f"{filename}.pdf")
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")

for fname in os.listdir(input_dir):
    if not fname.endswith(".csv"):
        continue

    filepath = os.path.join(input_dir, fname)
    df = pd.read_csv(filepath)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    filename_base = fname.replace(".csv", "")
    dist_parts = filename_base.split("_")
    dist = dist_parts[1] if len(dist_parts) > 1 else "unknown"

    # Attempt to extract m, n, k from filename
    m, n, k = 0, 0, 0
    for part in dist_parts:
        if part.startswith("m"):
            try:
                m = int(part[1:])
            except:
                pass
        if part.startswith("n"):
            try:
                n = int(part[1:])
            except:
                pass
        if part.startswith("k"):
            try:
                k = int(part[1:])
            except:
                pass

    if "Intra" in df.columns and "Inter" in df.columns:
        # Group by Inter for each Intra
        for intra_val in sorted(df["Intra"].unique()):
            subset = df[df["Intra"] == intra_val]
            meta = f"{dist.capitalize()} | Intra={intra_val:.2f}"
            file_tag = f"{dist}_kt_vs_inter_intra{intra_val:.2f}_{filename_base}"
            plot_graph(subset, "Inter", ["Borda", "Copeland", "Sum"], f"{meta} | KT vs Inter", file_tag)
            plot_approval_graphs(subset, "Inter", thresholds, dist, intra_val, "intra", m, n, k)

        # Group by Intra for each Inter
        for inter_val in sorted(df["Inter"].unique()):
            subset = df[df["Inter"] == inter_val]
            meta = f"{dist.capitalize()} | Inter={inter_val:.2f}"
            file_tag = f"{dist}_kt_vs_intra_inter{inter_val:.2f}_{filename_base}"
            plot_graph(subset, "Intra", ["Borda", "Copeland", "Sum"], f"{meta} | KT vs Intra", file_tag)
            plot_approval_graphs(subset, "Intra", thresholds, dist, inter_val, "inter", m, n, k)
