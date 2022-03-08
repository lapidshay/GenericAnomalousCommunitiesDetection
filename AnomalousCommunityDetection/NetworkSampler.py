__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# Imports
########################################

import networkx as nx
import random
import warnings


########################################
# Network Sampler
########################################

class NetworkSampler:

	def __init__(self, community_part_label: str, vertex_part_label: str):
		"""
		Parameters
		----------
		G: nx.Graph, graph to sample edges from.
		community_part_label: string, community-representing-vertices part attribute value.
		vertex_part_label: string, regular vertices part attribute value.
		"""

		self._community_part_label = community_part_label
		self._vertex_part_label = vertex_part_label

	def sample_network_edges(self, G: nx.Graph, max_edges: int = None, generate_negative_edges: bool = False):
		"""
		Returns 2 lists - (1) sampled positive edges and (2) negative edges \ an empty list.

		Extracts community part vertices (community-representing-vertices).
		Samples positive (existing) edges and generates same number of negative (non-existing) edges,
		depending on generate_negative_edges boolean.

		Parameters
		----------
		G: nx.Graph, graph to sample edges from.
		max_edges: int, maximum edges to sample.
		generate_negative_edges: a boolean, determines whether to create negative edges.

		Returns
		-------
		Two lists of edges (2nd might be empty, depending on generate_negative_edges parameter).
		"""

		# Obtain community part vertices
		community_partite_nodes = [
			node for (node, data)
			in G.nodes(data=True)
			if data['partite'] == self._community_part_label
		]

		# Select random max_edges or all positive edges
		positive_edges = self._select_existing_edges(G, nodes_to_include=community_partite_nodes, max_edges=max_edges)

		# Determine whether to create negative edges (for training)
		if generate_negative_edges:

			# Maintain balanced train set by choosing same number of negative edges as positive edges
			negative_edges_num = len(positive_edges)

			# Generate random non existing links
			negative_edges = self._select_non_existing_edges(G, n=negative_edges_num)

		else:
			negative_edges = []

		return positive_edges, negative_edges

	def _select_existing_edges(self, G, nodes_to_include: list, max_edges=None):
		"""Returns a list of all existing links or random max_edges existing links."""
		selected_edges = set()

		# Add all nodes_to_include edges
		for node in nodes_to_include:
			selected_edges |= set(G.edges(node))

		# Number of possible edges to select
		possible_edges = len(selected_edges)

		# Random choose max_edges of them if given
		if max_edges is not None:
			if max_edges > possible_edges:
				warnings.warn(
					f'Argument "max_edges" is larger than the number of possible edges, {possible_edges}.'
					f'All possible edges were selected.'
				)
			max_edges = min(possible_edges, max_edges)
		else:
			max_edges = possible_edges

		return set(random.sample(list(selected_edges), k=max_edges))

	def _select_non_existing_edges(self, G, n, nodes_to_exclude: list = None):
		"""Returns a list of random n non-existing edges."""
		selected_edges = set()

		# Create lists of community part vertices and regular vertices
		comm_part_nodes = [v for (v, data) in G.nodes(data=True) if data['partite'] == self._community_part_label]
		vertex_part_nodes = [v for (v, data) in G.nodes(data=True) if data['partite'] == self._vertex_part_label]

		# Remove vertices from comm_part_nodes if nodes_to_exclude is given
		if nodes_to_exclude is not None:
			comm_part_nodes = list(set(comm_part_nodes) - set(nodes_to_exclude))

		while len(selected_edges) < n:

			# Randomly choose 2 nodes
			community_node = random.choice(comm_part_nodes)
			vertex_node = random.choice(vertex_part_nodes)

			# Add edge if they not exists
			if G.has_edge(community_node, vertex_node) or (vertex_node, community_node) in selected_edges:
				continue
			else:
				selected_edges.add((community_node, vertex_node))

		return list(selected_edges)

