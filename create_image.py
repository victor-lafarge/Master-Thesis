"""
Test FULL LLM script to generate visualisations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Helper class for beautiful 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

# Display configuration
plt.style.use('seaborn-v0_8-whitegrid')

np.random.seed(422)

# ==========================================
# 0. Data Preparation (3D)
# ==========================================
n_users = 5
claims_per_user = 3
n_claims = n_users * claims_per_user

# High spread for points
user_centers = np.random.uniform(low=0, high=25, size=(n_users, 3))

points = []
labels = []
colors = []
user_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']

for i in range(n_users):
    user_claims = np.random.normal(loc=user_centers[i], scale=2.8, size=(claims_per_user, 3))
    points.append(user_claims)
    for j in range(claims_per_user):
        labels.append(f"U{i+1}-C{j+1}")
        colors.append(user_colors[i])

points = np.vstack(points)

# Common legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'User {i+1}',
                          markerfacecolor=user_colors[i], markersize=10) for i in range(n_users)]

# Depth-based size
z_min, z_max = points[:, 2].min(), points[:, 2].max()
sizes_3d = 60 + 200 * (z_max - points[:, 2]) / (z_max - z_min)

# Grid bounds
padding = 4
x_lim = [points[:, 0].min() - padding, points[:, 0].max() + padding]
y_lim = [points[:, 1].min() - padding, points[:, 1].max() + padding]
z_lim = [points[:, 2].min() - padding, points[:, 2].max() + padding]

def setup_axes_and_grid(ax):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    
    # Hide tick labels (numbers) but keep the ticks so grid stays
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    
    # Remove the small tick marks (graduations)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.zaxis.set_tick_params(size=0)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    origin = [x_lim[0], y_lim[1], z_lim[0]]
    
    # Draw arrows with low zorder
    arrow_props = dict(mutation_scale=20, arrowstyle='-|>', color='black', linewidth=2, zorder=0)
    
    # X Axis
    a_x = Arrow3D([origin[0], x_lim[1]], [origin[1], origin[1]], [origin[2], origin[2]], **arrow_props)
    ax.add_artist(a_x)
    
    # Y Axis
    a_y = Arrow3D([origin[0], origin[0]], [origin[1], y_lim[0]], [origin[2], origin[2]], **arrow_props)
    ax.add_artist(a_y)
    
    # Z Axis
    a_z = Arrow3D([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], z_lim[1]], **arrow_props)
    ax.add_artist(a_z)
    
    # Labels with low zorder
    ax.text(x_lim[1] + 1.5, origin[1], origin[2], "X", fontsize=15, fontweight='bold', zorder=0)
    ax.text(origin[0], y_lim[0] - 2.5, origin[2], "Y", fontsize=15, fontweight='bold', zorder=0)
    ax.text(origin[0], origin[1], z_lim[1] + 1.5, "Z", fontsize=15, fontweight='bold', zorder=0)

# ==========================================
# IMAGE 1 : 3D Embedding Space
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False  # Use manual zorder instead of depth sorting
ax.view_init(elev=20, azim=-35)

# 1. Setup axes FIRST so they are behind
setup_axes_and_grid(ax)

# 2. Plot points SECOND with high zorder
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes_3d, edgecolors='black', alpha=1, depthshade=False, zorder=10)

ax.set_title("3D Embedding Space (15 Claims, 5 Users)", fontsize=16, pad=25)
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Users", fontsize=11)

plt.tight_layout()
plt.savefig("1_embedding_space_3d.png", dpi=300, bbox_inches='tight')
print("Image 1 generated: 1_embedding_space_3d.png")

# ==========================================
# IMAGE 2 : KNN Graph Construction (K=3)
# ==========================================
K = 3
nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(points)
distances, indices = nbrs.kneighbors(points)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False  # Use manual zorder instead of depth sorting
ax.view_init(elev=20, azim=-35)

# 1. Setup axes FIRST
setup_axes_and_grid(ax)

# 2. Plot graph and points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes_3d, edgecolors='black', alpha=1, depthshade=False, zorder=10)

for i in range(len(points)):
    for j in range(1, K+1):
        neighbor_idx = indices[i][j]
        ax.plot([points[i, 0], points[neighbor_idx, 0]], 
                [points[i, 1], points[neighbor_idx, 1]], 
                [points[i, 2], points[neighbor_idx, 2]], 
                color='#7f8c8d', linestyle='-', linewidth=1.5, alpha=0.6, zorder=5)

ax.set_title(f"KNN Graph Construction (K={K}) in 3D", fontsize=16, pad=25)
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Users", fontsize=11)

plt.tight_layout()
plt.savefig("2_knn_graph_3d.png", dpi=300, bbox_inches='tight')
print("Image 2 generated: 2_knn_graph_3d.png")

# ==========================================
# IMAGE 3 : Centroid Graph Construction
# ==========================================

# Compute centroids for each user
centroids = np.array([points[i*claims_per_user:(i+1)*claims_per_user].mean(axis=0) for i in range(n_users)])

# Compute pairwise distances between centroids
from scipy.spatial.distance import pdist, squareform
centroid_dists = squareform(pdist(centroids))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False  # Use manual zorder instead of depth sorting
ax.view_init(elev=20, azim=-35)

# 1. Setup axes FIRST
setup_axes_and_grid(ax)

# 2. Plot claims as small faded dots
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=40, 
           edgecolors='gray', alpha=0.25, depthshade=False, zorder=5)

# 3. Draw dashed lines from each claim to its centroid
for i in range(n_users):
    for j in range(claims_per_user):
        idx = i * claims_per_user + j
        ax.plot([points[idx, 0], centroids[i, 0]], 
                [points[idx, 1], centroids[i, 1]], 
                [points[idx, 2], centroids[i, 2]], 
                color=user_colors[i], linestyle='--', linewidth=0.8, alpha=0.4, zorder=4)

# 4. Draw edges between centroids (weighted by 1/distance)
# Normalize weights for line thickness
max_dist = centroid_dists[centroid_dists > 0].max()
for i in range(n_users):
    for j in range(i+1, n_users):
        d = centroid_dists[i, j]
        weight = 1.0 / d
        # Normalize thickness: closer centroids = thicker line
        thickness = 0.5 + 4.0 * (1.0 - d / max_dist)
        alpha_edge = 0.3 + 0.6 * (1.0 - d / max_dist)
        ax.plot([centroids[i, 0], centroids[j, 0]], 
                [centroids[i, 1], centroids[j, 1]], 
                [centroids[i, 2], centroids[j, 2]], 
                color='#2c3e50', linestyle='-', linewidth=thickness, alpha=alpha_edge, zorder=8)

# 5. Plot centroids as large diamonds
centroid_sizes = 250
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           c=user_colors, s=centroid_sizes, edgecolors='black', linewidths=2,
           marker='D', alpha=1, depthshade=False, zorder=15)

# Legend with both claims and centroids
legend_claims = [Line2D([0], [0], marker='o', color='w', label=f'User {i+1} claims',
                        markerfacecolor=user_colors[i], markersize=7, alpha=0.4) for i in range(n_users)]
legend_centroids = [Line2D([0], [0], marker='D', color='w', label=f'User {i+1} centroid',
                           markerfacecolor=user_colors[i], markersize=10, markeredgecolor='black') for i in range(n_users)]

ax.set_title("Centroid Graph Construction", fontsize=16, pad=25)
ax.legend(handles=legend_centroids, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Centroids", fontsize=11)

plt.tight_layout()
plt.savefig("3_centroid_graph_3d.png", dpi=300, bbox_inches='tight')
print("Image 3 generated: 3_centroid_graph_3d.png")

# ==========================================
# IMAGE 4 : Mutual KNN at Claim Level (K=3)
# ==========================================
from collections import defaultdict

K_mknn = 3
nbrs_mknn = NearestNeighbors(n_neighbors=K_mknn+1, algorithm='ball_tree').fit(points)
distances_mknn, indices_mknn = nbrs_mknn.kneighbors(points)

# Precompute neighbors for O(1) lookup
neighbors_dict = {i: set(indices_mknn[i, 1:]) for i in range(n_claims)}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False
ax.view_init(elev=20, azim=-35)

setup_axes_and_grid(ax)

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes_3d, edgecolors='black', alpha=1, depthshade=False, zorder=10)

# Track which pairs of users have edges
user_edges = defaultdict(float)

# First, draw faint lines for standard KNN (non-mutual or same-user)
for i in range(n_claims):
    for k in range(1, K_mknn+1):
        j = indices_mknn[i, k]
        if i < j:
            is_mutual = i in neighbors_dict[j]
            user_i = i // claims_per_user
            user_j = j // claims_per_user
            if not (is_mutual and user_i != user_j):
                ax.plot([points[i, 0], points[j, 0]], 
                        [points[i, 1], points[j, 1]], 
                        [points[i, 2], points[j, 2]], 
                        color='#bdc3c7', linestyle=':', linewidth=1.0, alpha=0.5, zorder=4)

# Then draw highlighted lines for mutual KNN across different users
for i in range(n_claims):
    user_i = i // claims_per_user
    for k in range(1, K_mknn+1):
        j = indices_mknn[i, k]
        user_j = j // claims_per_user
        
        is_mutual = i in neighbors_dict[j]
        
        if is_mutual and i < j and user_i != user_j:
            ax.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    [points[i, 2], points[j, 2]], 
                    color='#e74c3c', linestyle='-', linewidth=2.5, zorder=8)
            
            # Accumulate weight (using inverse distance for visualization logic)
            dist = distances_mknn[i, k]
            sim = 1.0 / (1.0 + dist)
            pair = tuple(sorted([user_i, user_j]))
            user_edges[pair] += sim

ax.set_title(f"Mutual KNN Claims Connections (K={K_mknn})", fontsize=16, pad=25)

# Add custom legend entry for mutual edges
from matplotlib.lines import Line2D
custom_mutual_line = Line2D([0], [0], color='#e74c3c', lw=2.5, label='Mutual KNN Edge')
ax.legend(handles=legend_elements + [custom_mutual_line], loc='center left', bbox_to_anchor=(1.05, 0.5), title="Users & Connections", fontsize=11)

plt.tight_layout()
plt.savefig("4_mknn_claims_3d.png", dpi=300, bbox_inches='tight')
print("Image 4 generated: 4_mknn_claims_3d.png")

# ==========================================
# IMAGE 5 : Aggregated mKNN User Graph
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False
ax.view_init(elev=20, azim=-35)

setup_axes_and_grid(ax)

# Draw claims faintly
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=40, 
           edgecolors='gray', alpha=0.2, depthshade=False, zorder=5)

# We use centroids as user nodes for visualization
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           c=user_colors, s=250, edgecolors='black', linewidths=2,
           marker='D', alpha=1, depthshade=False, zorder=15)

# Draw edges between centroids based on user_edges
if user_edges:
    max_weight = max(user_edges.values())
else:
    max_weight = 1.0

for (u1, u2), weight in user_edges.items():
    thickness = 1.0 + 4.0 * (weight / max_weight)
    alpha_edge = 0.4 + 0.6 * (weight / max_weight)
    ax.plot([centroids[u1, 0], centroids[u2, 0]], 
            [centroids[u1, 1], centroids[u2, 1]], 
            [centroids[u1, 2], centroids[u2, 2]], 
            color='#c0392b', linestyle='-', linewidth=thickness, alpha=alpha_edge, zorder=8)

ax.set_title("Aggregated mKNN User Graph", fontsize=16, pad=25)

# Update legend
custom_agg_line = Line2D([0], [0], color='#c0392b', lw=3, label='Aggregated mKNN Edge')
ax.legend(handles=legend_centroids + [custom_agg_line], loc='center left', bbox_to_anchor=(1.05, 0.5), title="User Nodes & Edges", fontsize=11)

plt.tight_layout()
plt.savefig("5_mknn_user_graph_3d.png", dpi=300, bbox_inches='tight')
print("Image 5 generated: 5_mknn_user_graph_3d.png")

