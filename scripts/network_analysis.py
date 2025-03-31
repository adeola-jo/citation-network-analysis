import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Convert to networkx graph for analysis
G = to_networkx(data, to_undirected=True)
print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Calculate basic network metrics
print("\nCalculating network metrics...")
# Degree centrality
degree_centrality = nx.degree_centrality(G)
# Betweenness centrality (this can be slow for large networks)
betweenness_centrality = nx.betweenness_centrality(G, k=100)  # Use k for approximation
# Closeness centrality
closeness_centrality = nx.closeness_centrality(G)
# PageRank
pagerank = nx.pagerank(G, alpha=0.85)

# Get top 10 nodes by different centrality measures
def get_top_nodes(centrality_dict, n=10):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

print("\nTop 10 nodes by degree centrality:")
for i, (node, centrality) in enumerate(get_top_nodes(degree_centrality)):
    print(f"{i+1}. Node {node}: {centrality:.4f}")

print("\nTop 10 nodes by betweenness centrality:")
for i, (node, centrality) in enumerate(get_top_nodes(betweenness_centrality)):
    print(f"{i+1}. Node {node}: {centrality:.4f}")

print("\nTop 10 nodes by PageRank:")
for i, (node, rank) in enumerate(get_top_nodes(pagerank)):
    print(f"{i+1}. Node {node}: {rank:.4f}")

# Community detection
print("\nDetecting communities...")
communities = nx.community.greedy_modularity_communities(G)
print(f"Detected {len(communities)} communities")
print(f"Sizes of the 5 largest communities: {[len(c) for c in communities[:5]]}")

# Visualize community structure (for a subset of nodes)
def visualize_communities(G, communities, max_nodes=500):
    # Create a smaller graph for visualization if needed
    if G.number_of_nodes() > max_nodes:
        # Get nodes from the largest communities
        nodes_to_keep = []
        for community in communities:
            nodes_to_keep.extend(list(community)[:int(max_nodes/len(communities))])
            if len(nodes_to_keep) >= max_nodes:
                break
        G_small = G.subgraph(nodes_to_keep)
    else:
        G_small = G
    
    # Map each node to its community
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    # Get node colors based on community
    node_colors = [community_map.get(node, 0) for node in G_small.nodes()]
    
    # Plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_small, seed=42)
    nx.draw_networkx_nodes(G_small, pos, node_color=node_colors, cmap='tab20', node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G_small, pos, alpha=0.2)
    plt.title("Community Structure in Citation Network")
    plt.axis('off')
    plt.savefig('community_structure.png')
    plt.close()

# Visualize communities (this might take some time)
visualize_communities(G, communities)
print("Community visualization saved as 'community_structure.png'")

# Calculate and visualize degree distribution
degrees = [d for n, d in G.degree()]
plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=30, alpha=0.7)
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distribution')
plt.savefig('degree_distribution.png')
plt.close()

# Fit a power law to check for scale-free property
from scipy import stats
degrees_for_fit = np.array([d for d in degrees if d > 0])
log_degrees = np.log10(degrees_for_fit)
# Count frequency of each degree
unique_degrees, counts = np.unique(degrees_for_fit, return_counts=True)
log_unique_degrees = np.log10(unique_degrees)
log_counts = np.log10(counts)
# Linear fit on log-log scale
slope, intercept, r_value, p_value, std_err = stats.linregress(log_unique_degrees, log_counts)

plt.figure(figsize=(10, 6))
plt.scatter(log_unique_degrees, log_counts, alpha=0.7)
plt.plot(log_unique_degrees, intercept + slope * log_unique_degrees, 'r')
plt.xlabel('Log(Degree)')
plt.ylabel('Log(Count)')
plt.title(f'Power Law Fit: γ = {-slope:.2f}')
plt.savefig('power_law_fit.png')
plt.close()

print(f"\nPower law exponent: γ = {-slope:.2f}")
if -slope > 2 and -slope < 3:
    print("The network appears to be scale-free (power law exponent between 2 and 3)")
else:
    print("The network may not be strictly scale-free")

# Calculate clustering coefficient
avg_clustering = nx.average_clustering(G)
print(f"\nAverage clustering coefficient: {avg_clustering:.4f}")

# Calculate average shortest path length
# This can be slow, so we'll use a sample of nodes
try:
    sample_nodes = np.random.choice(list(G.nodes()), size=min(100, G.number_of_nodes()), replace=False)
    sample_G = G.subgraph(sample_nodes)
    avg_path_length = nx.average_shortest_path_length(sample_G)
    print(f"Average shortest path length (on sampled subgraph): {avg_path_length:.4f}")
except nx.NetworkXError:
    print("Graph is not connected, cannot compute average shortest path length")

print("\nNetwork analysis complete!")
