import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community.community_louvain as community_louvain # pip install python-louvain
import math
import random

# LOAD Wiki-Vote Dataset
df = pd.read_csv('Wiki-Vote.txt', sep='\t', comment='#', names=['FromNodeId', 'ToNodeId'])
G = nx.from_pandas_edgelist(df, 'FromNodeId', 'ToNodeId', create_using=nx.DiGraph())

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)

# BASIC STATS
print(f"Graph Created: Wiki-Vote")
print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Density: {density:.6f}")

# DEGREE DISTRIBUTION
in_degrees = [d for n, d in G.in_degree()]
out_degrees = [d for n, d in G.out_degree()]

plt.figure(figsize=(12,5))

# In-Degree
plt.subplot(1,2,1)
plt.hist(in_degrees, bins=50, color='skyblue', edgecolor='black', log=True)
plt.title("Distribution of Votes Received (In-Degree)")
plt.xlabel("Votes")
plt.ylabel("Frequency (Log)")

# Out-Degree
plt.subplot(1,2,2)
plt.hist(out_degrees, bins=50, color='salmon', edgecolor='black', log=True)
plt.title("Distribution of Votes Cast (Out-Degree)")
plt.xlabel("Votes")
plt.ylabel("Frequency (Log)")
plt.tight_layout()
plt.savefig('1_degree_distribution.png', dpi=150)
print("Saved 1_degree_distribution.png")

# EGO-NET
degree_dict = dict(G.degree())
top_node = max(degree_dict, key=degree_dict.get)
print(f"Ego-Net for top node: {top_node} (Degree: {degree_dict[top_node]})")

# EGO GRAPH RADIUS 1
G_ego = nx.ego_graph(G, top_node, radius=1)
plt.figure(figsize=(8, 8))
pos_ego = nx.spring_layout(G_ego, seed=42)
nx.draw(G_ego, pos_ego, node_color='orange', node_size=50, with_labels=False, alpha=0.7)
nx.draw_networkx_nodes(G_ego, pos_ego, nodelist=[top_node], node_color='red', node_size=300) 
plt.title(f"Ego-Net of Node {top_node}")
plt.savefig('1b_ego_net.png', dpi=150)

# NETWORK METRICS & COMPONENTS
wcc = list(nx.weakly_connected_components(G))
scc = list(nx.strongly_connected_components(G))
print(f"Weakly Connected Components: {len(wcc)}")
print(f"Strongly Connected Components: {len(scc)}")

# COMPONENT SIZE
wcc_sizes = sorted([len(c) for c in wcc], reverse=True)
print(f"Top 5 Largest Component Sizes: {wcc_sizes[:5]}")
unique_sizes, counts = np.unique(wcc_sizes, return_counts=True)
print(f"Size distribution head: Size {unique_sizes[:3]} -> Count {counts[:3]}")

# NODES IN LSCC
G_scc_largest = G.subgraph(max(scc, key=len)).copy()
print(f"Nodes in LSCC: {G_scc_largest.number_of_nodes()}")

# PATH LENGTH & DIAMETER (FOR STRONGLY CONNECTED COMPONENT)
avg_path = nx.average_shortest_path_length(G_scc_largest)
diameter = nx.diameter(G_scc_largest)
print(f"Average Path Length (LSCC): {avg_path:.4f}")
print(f"Diameter (LSCC): {diameter}")

# CLUSTERING & ASSORTATIVITY
global_clustering = nx.transitivity(G) 
avg_local_clustering = nx.average_clustering(G) 
assortativity = nx.degree_assortativity_coefficient(G)
reciprocity = nx.reciprocity(G)

print(f"Global Clustering: {global_clustering:.4f}")
print(f"Avg Local Clustering: {avg_local_clustering:.4f}")
print(f"Assortativity: {assortativity:.4f}")
print(f"Reciprocity: {reciprocity:.4f}")

# CENTRALITY ANALYSIS
deg_cent = nx.degree_centrality(G)
# 2. PageRank
pagerank = nx.pagerank(G)
# 3. Betweenness (Approximated k=500)
btwn_cent = nx.betweenness_centrality(G, k=500, normalized=True)
# 4. Closeness (Calculated on LSCC)
close_cent = nx.closeness_centrality(G_scc_largest)

# CREATE DATAFRAME IN PAGERANK ORDER
cent_df = pd.DataFrame({
    'Degree': pd.Series(deg_cent),
    'PageRank': pd.Series(pagerank),
    'Betweenness': pd.Series(btwn_cent),
    'Closeness': pd.Series(close_cent)
}).fillna(0)

top_10 = cent_df.sort_values('PageRank', ascending=False).head(10)
top_10.to_csv('centrality_top10.csv')
print("Correlation Matrix:")
print(cent_df.corr())

# SMALL WORLD ANALYSIS
G_rand = nx.gnm_random_graph(num_nodes, num_edges, directed=True, seed=42)
C_rand = nx.average_clustering(G_rand)
# Estimate Random Path Length 
L_rand = np.log(num_nodes) / np.log(np.mean([d for n,d in G.degree()]))

sigma = (avg_local_clustering / C_rand) / (avg_path / L_rand)
print(f"Real C: {avg_local_clustering:.4f} vs Random C: {C_rand:.4f}")
print(f"Real L: {avg_path:.4f} vs Random L (approx): {L_rand:.4f}")
print(f"Small-World Sigma: {sigma:.4f}")

# COMMUNITIES
G_undir = G.to_undirected()

# Method A: Louvain (Optimization based)
partition_louvain = community_louvain.best_partition(G_undir)
modularity_louvain = community_louvain.modularity(partition_louvain, G_undir)
num_comms_louvain = len(set(partition_louvain.values()))

print(f"1. Louvain found {num_comms_louvain} communities.")
print(f"   Modularity: {modularity_louvain:.4f}")

# Conductance for Louvain
conductance_values = []
for comm_id in set(partition_louvain.values()):
    nodes_in_comm = [n for n in G_undir.nodes() if partition_louvain[n] == comm_id]
    if len(nodes_in_comm) < 2 or len(nodes_in_comm) == G_undir.number_of_nodes():
        continue
    try:
        val = nx.conductance(G_undir, nodes_in_comm)
        conductance_values.append(val)
    except:
        pass
avg_conductance = np.mean(conductance_values) if conductance_values else 0
print(f"   Avg Conductance (Louvain): {avg_conductance:.4f}")


# Method B: Label Propagation (Flow based)
communities_lpa = list(nx.community.label_propagation_communities(G_undir))
num_comms_lpa = len(communities_lpa)

# Modularity for LPA
partition_lpa = {node: i for i, comm in enumerate(communities_lpa) for node in comm}
modularity_lpa = community_louvain.modularity(partition_lpa, G_undir)

print(f"    Label Propagation found {num_comms_lpa} communities.")
print(f"   Modularity: {modularity_lpa:.4f}")

conductance_values_lpa = []

for comm in communities_lpa:
    if len(comm) < 2 or len(comm) == G_undir.number_of_nodes():
        continue
    try:
        val = nx.conductance(G_undir, list(comm))
        conductance_values_lpa.append(val)
    except Exception as e:
        pass

avg_conductance_lpa = np.mean(conductance_values_lpa) if conductance_values_lpa else 0
print(f"   Avg Conductance (LPA): {avg_conductance_lpa:.4f}")

nx.set_node_attributes(G, partition_louvain, 'community')

# Sample Top 1000 Nodes
degrees = dict(G.degree())
top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:1000]
G_viz = G.subgraph(top_nodes).copy()

colors = [partition_louvain.get(n) for n in G_viz.nodes()]

# Calculate Node Sizes based on Degree
d_vals = [G_viz.degree(n) for n in G_viz.nodes()]
# Normalize sizes between 10 and 100
min_size, max_size = 10, 150
if max(d_vals) > min(d_vals):
    sizes = [((d - min(d_vals)) / (max(d_vals) - min(d_vals))) * (max_size - min_size) + min_size for d in d_vals]
else:
    sizes = [20 for _ in d_vals]

plt.figure(figsize=(16, 16))

# Styling Parameters
edge_width = 0.05

plt.subplot(2, 2, 1)
plt.title("1. Random Layout")
nx.draw(G_viz, pos=nx.random_layout(G_viz), 
        node_size=sizes, node_color=colors, cmap='tab20', 
        width=edge_width, arrowsize=2)

plt.subplot(2, 2, 2)
plt.title("2. Circular Layout")
nx.draw(G_viz, pos=nx.circular_layout(G_viz), 
        node_size=sizes, node_color=colors, cmap='tab20', 
        width=edge_width, arrowsize=2)

plt.subplot(2, 2, 3)
plt.title("3. Grid Layout")
side = math.ceil(math.sqrt(G_viz.number_of_nodes()))
pos_grid = {n: (i % side, i // side) for i, n in enumerate(G_viz.nodes())}
nx.draw(G_viz, pos=pos_grid, 
        node_size=sizes, node_color=colors, cmap='tab20', 
        width=edge_width, arrowsize=2)

plt.subplot(2, 2, 4)
plt.title("4. Force-Directed (Spring) - Sized by Degree")
pos_spring = nx.spring_layout(G_viz, k=0.15, iterations=40, seed=42)
nx.draw(G_viz, pos=pos_spring, 
        node_size=sizes, node_color=colors, cmap='tab20', 
        width=edge_width, arrowsize=3)

plt.tight_layout()
plt.savefig('2_layout_comparison.png', dpi=300)

# K-Core Filter (k=20)
G_core = nx.k_core(G_undir, k=20) 

print(f"K-Core (k=20) nodes: {len(G_core)}")

core_partition = {n: partition_louvain[n] for n in G_core.nodes()}
core_colors = [core_partition[n] for n in G_core.nodes()]

plt.figure(figsize=(10, 10))
pos_core = nx.spring_layout(G_core, k=0.2, seed=42)
core_sizes = [d * 3 for n, d in G_core.degree()]

nx.draw_networkx_nodes(G_core, pos_core, node_size=core_sizes, node_color=core_colors, cmap='tab20')
nx.draw_networkx_edges(G_core, pos_core, alpha=0.3, width=0.2) 
plt.title(f"Filtered View: K-Core (k=20) | Nodes: {len(G_core)}")
plt.axis('off')
plt.savefig('3_filtered_view_kcore.png', dpi=300)

def run_sir_simulation(G, seed_nodes, beta=0.1, gamma=1.0, steps=20):
    status = {n: 'S' for n in G.nodes()}
    for n in seed_nodes:
        status[n] = 'I'
    infected_counts = []
    for t in range(steps):
        curr_infected = [n for n, s in status.items() if s == 'I']
        infected_counts.append(len(curr_infected))
        if not curr_infected: break
        next_status = status.copy()
        for n in curr_infected:
            for neighbor in G.neighbors(n):
                if status[neighbor] == 'S' and random.random() < beta:
                    next_status[neighbor] = 'I'
            if random.random() < gamma:
                next_status[n] = 'R'
        status = next_status
    return infected_counts

steps_sim = 15
beta_sim = 0.2

# SIR SIMULATION FOR TOP DEGREE SEEDS
top_degree_nodes = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]]
history_degree = run_sir_simulation(G, top_degree_nodes, beta=beta_sim, steps=steps_sim)

# SIR SIMULATION FOR TOP PAGERANK SEEDS
top_pagerank_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:5]
history_pagerank = run_sir_simulation(G, top_pagerank_nodes, beta=beta_sim, steps=steps_sim)

# SIR SIMULATION FOR RANDOM SEEDS
random_nodes = random.sample(list(G.nodes()), 5)
history_random = run_sir_simulation(G, random_nodes, beta=beta_sim, steps=steps_sim)

plt.figure(figsize=(10, 6))
plt.plot(history_degree, marker='o', label='Seed: Top Degree')
plt.plot(history_pagerank, marker='s', label='Seed: Top PageRank')
plt.plot(history_random, marker='x', linestyle='--', label='Seed: Random')

plt.title(f"Diffusion Comparison (SIR Model, beta={beta_sim})")
plt.xlabel("Time Steps")
plt.ylabel("Number of Infected Nodes")
plt.legend()
plt.grid(True)
plt.savefig('4_diffusion_simulation.png', dpi=150)
