from networkx.generators.random_graphs import barabasi_albert_graph, erdos_renyi_graph

EXPERIMENT_SETTINGS = {
	'norm_comm_alg': barabasi_albert_graph,
	'anom_comm_alg': erdos_renyi_graph,

	'k_min': 1,
	'k_max': 1,

	'norm_m': 1,
	'norm_inter_p': 0.075,

	'anom_m': [0.01, 0.02, 0.04, 0.08, 0.16],  # Rows

	'anom_inter_p': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # X-Axis
}

