import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

# Parameters
n = 4
inter = 0.1
intra = 0.05
intra_expand = 0.04  # 4% for intra
inter_expand = 0.02  # 2% extend for inter

# Expanded endpoints for intra
intra_start, intra_end = -intra * intra_expand, intra * (1 + intra_expand)
# Slightly extended endpoints for inter
inter_start, inter_end = -inter * inter_expand, inter * (1 + inter_expand)

# Generate segment ranges
x_ranges = [(i * inter, i * inter + intra) for i in range(n)]

# Plot setup
fig, ax = plt.subplots(figsize=(8, 5))

for i, (x_start, x_end) in enumerate(x_ranges):
    y = n - i
    ax.plot([x_start, x_end], [y, y], marker='o', linestyle='-', color='black')

# Intra distance bracket (above top line)
intra_y = n + 0.3
ax.annotate('', xy=(intra_start, intra_y), xytext=(intra_end, intra_y),
            arrowprops=dict(arrowstyle='<->', lw=1.5))
ax.text((intra_start + intra_end) / 2, intra_y + 0.07, 'Intra Distance', ha='center', fontsize=10)

# Inter distance bracket (below second line)
inter_y = n - 1 - 0.3
ax.annotate('', xy=(inter_start, inter_y), xytext=(inter_end, inter_y),
            arrowprops=dict(arrowstyle='<->', lw=1.5))
ax.text((inter_start + inter_end) / 2, inter_y + 0.07, 'Inter Distance', ha='center', fontsize=10)

# Blue annotations for group at n = 2 (y = 2)
mu_x = 0.225
y_mu = 2

# Dot at mu
ax.plot(mu_x, y_mu, 'o', color='blue')
ax.text(mu_x, y_mu + 0.09, r'$\mu$', color='blue', fontsize=12, ha='center')

# Horizontal arrow from left bound to mu
sigma_start = x_ranges[2][0]  # left bound of group 2 (i=2)
ax.annotate('', xy=(sigma_start-0.0025, y_mu - 0.12), xytext=(mu_x+0.0025, y_mu - 0.12),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text((sigma_start + mu_x) / 2, y_mu - 0.29, r'$\sigma$', color='blue', fontsize=12, ha='center')

# Formatting
ax.set_xlabel("Feature Score", fontsize=11)
ax.set_ylabel("Group Index (n)", fontsize=11)
ax.set_yticks(range(1, n + 1))
ax.set_xlim(-0.02, 0.4)
ax.set_ylim(0.3, n + 0.8)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
