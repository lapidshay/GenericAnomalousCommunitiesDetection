from networkx.generators.random_graphs import erdos_renyi_graph

EXPERIMENT_SETTINGS = {
	'anom_comm_alg': erdos_renyi_graph,

	'k_min': 1,
	'k_max': 1,

	'min_edge_weight': 3,

	'anom_m': [0.05, 0.1, 0.2, 0.4, 0.8],  # Rows
	'anom_inter_p': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],  # X-Axis
}

