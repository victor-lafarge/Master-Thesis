"""
Graph construction methods for platform-independent semantic community modeling.

This module implements multiple graph construction strategies that transform
claim-level embeddings into weighted user graphs. Each method produces a graph
where nodes represent authors and edge weights reflect semantic proximity of
their discourse.

Methods implemented:
    - centroid_simple:  Fully-connected graph from pairwise centroid similarities.
    - claim_knn:        Directed KNN graph built at the claim level, then aggregated
                        to users with sqrt-normalization.
    - centroid_knn:     Directed KNN graph on author centroids.
    - claim_mknn:       Mutual KNN at the claim level (edges require reciprocal
                        neighborhood membership).
    - centroid_mknn:    Mutual KNN on author centroids.
    - claim_snn:        Shared Nearest Neighbors at the claim level (Jaccard
                        similarity of KNN neighborhoods).
    - centroid_snn:     Shared Nearest Neighbors on author centroids.

All embedding-based methods use cosine distance and assume L2-normalized vectors.
"""

import networkx as nx
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import math

def build_centroid_simple_graph(df_all_claims):
    """Build a fully-connected undirected graph from pairwise centroid cosine similarities.

    Each author is represented by the mean of their claim embeddings. Every pair
    of authors is connected with weight = 1 - cosine_distance.

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.

    Returns:
        nx.Graph with 'weight' and 'distance' edge attributes.
    """
    author_centroids = (
        df_all_claims
        .groupby("author")["embedding"]
        .apply(lambda vecs: np.mean(np.stack(vecs.values), axis=0))
    )

    authors = author_centroids.index.tolist()
    centroid_matrix = np.stack(author_centroids.values)
    
    dist_matrix = cosine_distances(centroid_matrix)

    G = nx.Graph()
    G.add_nodes_from(authors)

    for i, j in combinations(range(len(authors)), 2):
        d = dist_matrix[i, j]
        sim = max(0.0, 1.0 - d)
        G.add_edge(authors[i], authors[j], weight=sim, distance=d)
        
    return G

def build_claim_knn_graph(df_all_claims, k_neighbors=2):
    """Build a directed KNN graph at the claim level, aggregated to users.

    For each claim, its K nearest neighbors are found. Similarities between
    claims of different authors are accumulated, then normalized by
    sqrt(n_claims_a * n_claims_b) to avoid bias toward prolific authors.

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.
        k_neighbors: Number of nearest neighbors per claim.

    Returns:
        nx.DiGraph with 'weight' (normalized) and 'raw_weight' edge attributes.
    """
    EMB_MATRIX = np.stack(df_all_claims["embedding"].values)
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine", n_jobs=-1)
    knn.fit(EMB_MATRIX)
    distances, indices = knn.kneighbors(EMB_MATRIX)

    authors_array = df_all_claims["author"].values
    raw_edges = defaultdict(float)

    for i in range(len(EMB_MATRIX)):
        author_i = authors_array[i]
        for k in range(1, k_neighbors + 1):
            j = indices[i, k]
            dist = distances[i, k]
            author_j = authors_array[j]
            if author_i != author_j:
                sim = max(0.0, 1.0 - dist)
                raw_edges[(author_i, author_j)] += sim

    claims_count = df_all_claims.groupby("author").size().to_dict()

    G_knn = nx.DiGraph()
    G_knn.add_nodes_from(claims_count.keys())

    for (a, b), total_weight in raw_edges.items():
        n_a = claims_count[a]
        n_b = claims_count[b]
        normalized_weight = total_weight / math.sqrt(n_a * n_b)
        if G_knn.has_edge(a, b):
            G_knn[a][b]['weight'] += normalized_weight
            G_knn[a][b]['raw_weight'] += total_weight
        else:
            G_knn.add_edge(a, b, weight=normalized_weight, raw_weight=total_weight)

    return G_knn

def build_centroid_knn_graph(df_all_claims, k_neighbors=5):
    """Build a directed KNN graph on author centroids.

    Each author's centroid is connected to its K nearest centroid neighbors.
    Edge weight is the cosine similarity between the two centroids.

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.
        k_neighbors: Number of nearest neighbors per centroid.

    Returns:
        nx.DiGraph with 'weight' edge attribute.
    """
    author_centroids = (
        df_all_claims
        .groupby("author")["embedding"]
        .apply(lambda vecs: np.mean(np.stack(vecs.values), axis=0))
    )
    
    authors = author_centroids.index.to_numpy()
    if len(authors) < 2:
        G = nx.Graph()
        G.add_nodes_from(authors)
        return G
        
    centroid_matrix = np.stack(author_centroids.values)
    
    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(authors)), 
                           metric="cosine", n_jobs=-1)
    knn.fit(centroid_matrix)
    distances, indices = knn.kneighbors(centroid_matrix)
    
    G = nx.DiGraph()
    G.add_nodes_from(authors)
    
    for i in range(len(authors)):
        for k in range(1, min(k_neighbors + 1, len(authors))):
            j = indices[i, k]
            dist = distances[i, k]
            sim = max(0.0, 1.0 - dist)

            if G.has_edge(authors[i], authors[j]):
                G[authors[i]][authors[j]]['weight'] += sim
            else:
                G.add_edge(authors[i], authors[j], weight=sim)
    
    return G

def build_claim_mknn_graph(df_all_claims, k_neighbors=5):
    """Build a directed mutual KNN graph at the claim level, aggregated to users.

    Similar to claim_knn, but a claim-pair similarity is only counted when both
    claims appear in each other's K-nearest neighborhood (mutual condition).
    Weights are normalized by sqrt(n_claims_a * n_claims_b).

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.
        k_neighbors: Number of nearest neighbors per claim.

    Returns:
        nx.DiGraph with 'weight' (normalized) and 'raw_weight' edge attributes.
    """
    EMB_MATRIX = np.stack(df_all_claims["embedding"].values)
    n_claims = len(EMB_MATRIX)
    
    n_neighbors = min(k_neighbors + 1, n_claims)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    knn.fit(EMB_MATRIX)
    distances, indices = knn.kneighbors(EMB_MATRIX)

    authors_array = df_all_claims["author"].values
    raw_edges = defaultdict(float)
    
    neighbors_dict = {i: set(indices[i, 1:]) for i in range(n_claims)}

    for i in range(n_claims):
        author_i = authors_array[i]
        for k in range(1, n_neighbors):
            j = indices[i, k]
            author_j = authors_array[j]
            
            if i in neighbors_dict[j] and i < j and author_i != author_j:
                dist = distances[i, k]
                sim = max(0.0, 1.0 - dist)
                raw_edges[(author_i, author_j)] += sim
                raw_edges[(author_j, author_i)] += sim

    claims_count = df_all_claims.groupby("author").size().to_dict()

    G_mknn = nx.DiGraph()
    G_mknn.add_nodes_from(claims_count.keys())

    for (a, b), total_weight in raw_edges.items():
        n_a = claims_count[a]
        n_b = claims_count[b]
        normalized_weight = total_weight / math.sqrt(n_a * n_b)
        G_mknn.add_edge(a, b, weight=normalized_weight, raw_weight=total_weight)

    return G_mknn

def build_centroid_mknn_graph(df_all_claims, k_neighbors=5):
    """Build an undirected mutual KNN graph on author centroids.

    Two authors are connected if and only if each appears in the other's
    K-nearest centroid neighborhood. Edge weight is the cosine similarity.

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.
        k_neighbors: Number of nearest neighbors per centroid.

    Returns:
        nx.Graph with 'weight' edge attribute.
    """
    author_centroids = (
        df_all_claims
        .groupby("author")["embedding"]
        .apply(lambda vecs: np.mean(np.stack(vecs.values), axis=0))
    )
    
    authors = author_centroids.index.to_numpy()
    if len(authors) < 2:
        G = nx.Graph()
        G.add_nodes_from(authors)
        return G
        
    centroid_matrix = np.stack(author_centroids.values)
    
    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(authors)), metric="cosine", n_jobs=-1)
    knn.fit(centroid_matrix)
    distances, indices = knn.kneighbors(centroid_matrix)
    
    G = nx.Graph()
    G.add_nodes_from(authors)
    
    neighbors_dict = {i: set(indices[i, 1:]) for i in range(len(authors))}
    
    for i in range(len(authors)):
        for j in neighbors_dict[i]:
            if i in neighbors_dict[j] and i < j:  # Mutual KNN condition and avoid duplicates
                dist = cosine_distances([centroid_matrix[i]], [centroid_matrix[j]])[0][0]
                weight = max(0.0, 1.0 - dist)
                G.add_edge(authors[i], authors[j], weight=weight)
                
    return G

def build_claim_snn_graph(df_all_claims, k_neighbors=10, min_shared=1):
    """Build an undirected Shared Nearest Neighbors graph at the claim level.

    For each pair of claims belonging to different authors, the Jaccard
    similarity of their KNN neighborhoods is computed and accumulated as
    edge weight, then normalized by sqrt(n_claims_a * n_claims_b).

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.
        k_neighbors: Number of nearest neighbors per claim.
        min_shared: Minimum shared neighbors required to create a link.

    Returns:
        nx.Graph with 'weight' (normalized) and 'raw_weight' edge attributes.
    """
    EMB_MATRIX = np.stack(df_all_claims["embedding"].values)
    n_claims = len(EMB_MATRIX)
    
    n_neighbors = min(k_neighbors + 1, n_claims)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    knn.fit(EMB_MATRIX)
    _, indices = knn.kneighbors(EMB_MATRIX)

    authors_array = df_all_claims["author"].values
    claims_count = df_all_claims.groupby("author").size().to_dict()
    
    neighbors_dict = {i: set(indices[i, 1:]) for i in range(n_claims)}
    
    raw_edges = defaultdict(float)

    for i, j in combinations(range(n_claims), 2):
        author_i = authors_array[i]
        author_j = authors_array[j]
        if author_i == author_j:
            continue
        
        shared = neighbors_dict[i].intersection(neighbors_dict[j])
        if len(shared) >= min_shared:
            union = neighbors_dict[i].union(neighbors_dict[j])
            jaccard = len(shared) / len(union)
            pair = tuple(sorted([author_i, author_j]))
            raw_edges[pair] += jaccard

    G_snn = nx.Graph()
    G_snn.add_nodes_from(claims_count.keys())

    for (a, b), total_weight in raw_edges.items():
        n_a = claims_count[a]
        n_b = claims_count[b]
        normalized_weight = total_weight / math.sqrt(n_a * n_b)
        G_snn.add_edge(a, b, weight=normalized_weight, raw_weight=total_weight)

    return G_snn

def build_centroid_snn_graph(df_all_claims, k_neighbors=10, min_shared=1):
    """Build an undirected Shared Nearest Neighbors graph on author centroids.

    Edge weight is the Jaccard similarity of two authors' centroid KNN
    neighborhoods. An edge is created only if the number of shared neighbors
    meets the min_shared threshold.

    Args:
        df_all_claims: DataFrame with 'author' and 'embedding' columns.
        k_neighbors: Number of nearest neighbors per centroid.
        min_shared: Minimum shared neighbors required to create a link.

    Returns:
        nx.Graph with 'weight' (Jaccard similarity) edge attribute.
    """
    author_centroids = (
        df_all_claims
        .groupby("author")["embedding"]
        .apply(lambda vecs: np.mean(np.stack(vecs.values), axis=0))
    )
    
    authors = author_centroids.index.to_numpy()
    if len(authors) < 2:
        G = nx.Graph()
        G.add_nodes_from(authors)
        return G
        
    centroid_matrix = np.stack(author_centroids.values)
    
    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(authors)), metric="cosine", n_jobs=-1)
    knn.fit(centroid_matrix)
    _, indices = knn.kneighbors(centroid_matrix)
    
    G = nx.Graph()
    G.add_nodes_from(authors)
    
    neighbors_dict = {i: set(indices[i, 1:]) for i in range(len(authors))}
    
    for i, j in combinations(range(len(authors)), 2):
        shared_neighbors = neighbors_dict[i].intersection(neighbors_dict[j])
        if len(shared_neighbors) >= min_shared:
            union_neighbors = neighbors_dict[i].union(neighbors_dict[j])
            weight = len(shared_neighbors) / len(union_neighbors)
            G.add_edge(authors[i], authors[j], weight=weight)
            
    return G

def save_graph_parquet(G, output_path, pruning_percent=0):
    """Save a graph to a Parquet file with optional weight-based pruning.

    Edges below the given weight percentile are removed before saving.

    Args:
        G: NetworkX graph with 'weight' edge attribute.
        output_path: Path for the output Parquet file.
        pruning_percent: Percentile (0-100) of weakest edges to remove.

    Returns:
        Tuple of (total_edges, edges_after_pruning, weight_threshold).
    """
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "Source": u,
            "Target": v,
            "Weight": data["weight"] if data["weight"] != float("inf") else 999999
        })

    df_edges = pd.DataFrame(edges_data)
    if len(df_edges) == 0:
        return 0, 0, 0
        
    threshold = df_edges["Weight"].quantile(pruning_percent / 100.0)
    df_edges_filtered = df_edges[df_edges["Weight"] >= threshold].copy()
    
    df_edges_filtered.to_parquet(output_path, index=False)
    
    return len(df_edges), len(df_edges_filtered), threshold
