# Plot spatio-temporal quality surface for different velocities of motion.
# Run `ex_custom_video_source.py` first to generate the data file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

df = pd.read_csv('spatiotemp_results.csv', sep=',', skipinitialspace=True)

velocities = sorted(df['velocity'].unique())
n = len(velocities)

colors = [cm.Accent(i / max(n - 1, 1)) for i in range(n)]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

res_map = df[['res_scale', 'res_label']].drop_duplicates().sort_values('res_scale')
res_scales = res_map['res_scale'].tolist()
res_labels = res_map['res_label'].tolist()

fr_map = df[['fr_scale', 'fr_label']].drop_duplicates().sort_values('fr_scale')
fr_scales = fr_map['fr_scale'].tolist()
fr_labels = fr_map['fr_label'].tolist()

for color, vel in zip(colors, velocities):
    subset = df[df['velocity'] == vel]
    pivot = subset.pivot_table(index='fr_scale', columns='res_scale', values='Q_JOD')

    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values

    ax.plot_surface(X, Y, Z, color=color, edgecolor='k', linewidth=0.3, alpha=0.6)

ax.set_xticks(res_scales)
ax.set_xticklabels(res_labels)
ax.set_yticks(fr_scales)
ax.set_yticklabels(fr_labels)
ax.set_xlabel('Resolution')
ax.set_ylabel('Frame rate [fps]')
ax.set_zlabel('Quality')
# ax.set_title('Spatio-temporal quality')

legend_handles = [Patch(facecolor=color, label=f'{vel} deg/s') for color, vel in zip(colors, velocities)]
ax.legend(handles=legend_handles, title='velocity')

plt.tight_layout()
plt.show()
