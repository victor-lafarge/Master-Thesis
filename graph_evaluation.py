"""
Evaluation metrics for semantic user graphs.

Implements three complementary metrics based on known political landmarks
to assess and compare graph construction methods:

1. Neighborhood Purity @k  — fraction of same-group neighbors among the
   k strongest connections of each labeled node.
2. Weight Ratios            — intra-group vs inter-group mean edge weights.
3. Conductance              — cut-based separation quality on the labeled subgraph.
"""

import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import combinations
import itertools


# ─────────────────────────────────────────────
# 1. NEIGHBORHOOD PURITY @ k
# ─────────────────────────────────────────────

def neighborhood_purity(G: nx.Graph, labels: dict, k: int = 10) -> dict:
    """
    For each labeled node, find its k strongest neighbors (by edge weight)
    and compute the fraction belonging to the same political group.

    Returns:
        {
            "per_node":  {node: purity_score},
            "per_group": {group: mean_purity},
            "global":    float
        }
    """
    labeled_nodes = set(labels.keys()) & set(G.nodes())
    per_node = {}

    for node in labeled_nodes:
        # Filter to labeled neighbors first, then sort and take top-k
        labeled_neighbors = sorted(
            [(n, d) for n, d in G[node].items() if n in labeled_nodes],
            key=lambda x: x[1].get("weight", 1.0),
            reverse=True
        )
        top_k = [n for n, _ in labeled_neighbors[:k]]

        if not top_k:
            per_node[node] = 0.0
            continue

        same_group = sum(1 for n in top_k if labels[n] == labels[node])
        per_node[node] = same_group / len(top_k)

    per_group = defaultdict(list)
    for node, score in per_node.items():
        per_group[labels[node]].append(score)
    per_group = {g: np.mean(scores) for g, scores in per_group.items()}

    global_purity = np.mean(list(per_node.values())) if per_node else 0.0

    return {
        "per_node": per_node,
        "per_group": per_group,
        "global": global_purity
    }


# ─────────────────────────────────────────────
# 2. WEIGHT RATIOS (intra / inter group)
# ─────────────────────────────────────────────

def weight_ratios(G: nx.Graph, labels: dict) -> dict:
    """
    Computes within-group vs cross-group mean edge weights.
    Missing edges are considered to have a weight of 0 (no similarity).

    Returns:
        {
            "intra_group_mean_weight":  {group: float},
            "inter_group_mean_weight":  {(g1, g2): float},
            "ratio_intra_inter":        {(g1, g2): float}, # Asymmetric: intra(g1) / inter(g1, g2)
            "ratio_vs_all":             {group: float}     # intra(group) / inter(group, ALL_OTHERS)
        }
    """
    labeled_nodes = list(set(labels.keys()) & set(G.nodes()))

    def edge_weight(u, v):
        if G.has_edge(u, v):
            return G[u][v].get("weight", 1.0)
        return 0.0

    groups = defaultdict(list)
    for node in labeled_nodes:
        groups[labels[node]].append(node)

    # Intra-group mean weights
    intra_group_weights = {}
    for group, members in groups.items():
        weights = [edge_weight(u, v) for u, v in combinations(members, 2)]
        intra_group_weights[group] = np.mean(weights) if weights else np.nan

    # Inter-group mean weights
    inter_group_weights = {}
    group_names = list(groups.keys())
    for g1, g2 in combinations(group_names, 2):
        weights = [edge_weight(u, v) for u in groups[g1] for v in groups[g2]]
        mean_w = np.mean(weights) if weights else np.nan
        inter_group_weights[(g1, g2)] = mean_w
        inter_group_weights[(g2, g1)] = mean_w

    # Ratio intra / inter  (higher = better separation)
    ratios = {}
    for g1, g2 in itertools.permutations(group_names, 2):
        intra = intra_group_weights[g1]
        inter = inter_group_weights[(g1, g2)]
        ratios[(g1, g2)] = intra / inter if inter and inter > 0 else np.nan

    # Ratio group vs ALL others
    ratio_vs_all = {}
    for group, members in groups.items():
        other_members = [n for n in labeled_nodes if labels[n] != group]
        inter_all_weights = [edge_weight(u, v) for u in members for v in other_members]
        inter_all_mean = np.mean(inter_all_weights) if inter_all_weights else np.nan
        intra = intra_group_weights[group]
        ratio_vs_all[group] = intra / inter_all_mean if inter_all_mean and inter_all_mean > 0 else np.nan

    return {
        "intra_group_mean_weight": intra_group_weights,
        "inter_group_mean_weight": inter_group_weights,
        "ratio_intra_inter": ratios,
        "ratio_vs_all": ratio_vs_all
    }


# ─────────────────────────────────────────────
# 3. CONDUCTANCE ON LABELED SUBGRAPH
# ─────────────────────────────────────────────

def conductance_on_labeled_subgraph(G: nx.Graph, labels: dict) -> dict:
    """ 
    Computes conductance for each political group, restricted to labeled nodes only.
    
    Concretely: extracts the subgraph induced by labeled nodes, then computes
    conductance of each group within that subgraph.
    
    conductance(S) = cut(S, labeled\S) / min(vol(S), vol(labeled\S))
    
    Args:
        G:      weighted nx.Graph
        labels: dict {node_id: political_group}
    
    Returns:
        {
            "per_group": {group: conductance},
            "mean": float
        }
    """
    labeled_nodes = set(labels.keys()) & set(G.nodes())
    subgraph = G.subgraph(labeled_nodes)

    groups = defaultdict(set)
    for node in labeled_nodes:
        groups[labels[node]].add(node)

    results = {}
    for group, members in groups.items():
        complement = labeled_nodes - members

        cut = sum(
            d.get("weight", 1.0)
            for u in members
            for v, d in subgraph[u].items()
            if v in complement
        )

        vol_S = sum(
            d.get("weight", 1.0)
            for u in members
            for _, d in subgraph[u].items()
        )
        vol_comp = sum(
            d.get("weight", 1.0)
            for u in complement
            for _, d in subgraph[u].items()
        )

        denom = min(vol_S, vol_comp)
        results[group] = cut / denom if denom > 0 else np.nan

    return {
        "per_group": results,
        "mean": np.nanmean(list(results.values()))
    }


# ─────────────────────────────────────────────
# EVALUATE GRAPH (combined report)
# ─────────────────────────────────────────────

def evaluate_graph(G: nx.Graph, labels: dict, k: int = 100) -> dict:
    """
    Runs all graph-native evaluation metrics and returns a combined report.

    Args:
        G:      weighted nx.Graph
        labels: dict {node_id: political_group}
        k:      neighborhood size for purity

    Returns:
        Nested dict with all metrics.
    """
    print("Running neighborhood purity...")
    purity = neighborhood_purity(G, labels, k=k)

    print("Running weight ratios...")
    ratios = weight_ratios(G, labels)

    conductance = conductance_on_labeled_subgraph(G, labels)

    report = {
        "neighborhood_purity": purity,
        "weight_ratios": ratios,
        "conductance": conductance
    }

    # Pretty summary
    print("\n── EVALUATION SUMMARY ──")
    print(f"Global neighborhood purity @{k}: {purity['global']:.3f}")
    print(f"Purity per group: { {g: f'{v:.3f}' for g, v in purity['per_group'].items()} }")
    print(f"Mean conductance (lower is better): {conductance['mean']:.3f}")
    print(f"Conductance per group: { {g: f'{v:.3f}' for g, v in conductance['per_group'].items()} }")
    print("Weight ratios (intra/inter) [> 1 is good]:")
    for pair, ratio in ratios["ratio_intra_inter"].items():
        print(f"  {pair[0]} vs {pair[1]}: {ratio:.3f}")

    return report
