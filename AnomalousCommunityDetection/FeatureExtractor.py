__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import networkx as nx
from tqdm.autonotebook import tqdm
import pandas as pd
from itertools import product


########################################
# Feature Extractor
########################################

class FeatureExtractor:
	def __init__(self, g):
		self._g = g

	########################################
	# edge topological features
	########################################

	def _friends_measure(self, neighborhood_1: set, neighborhood_2: set):
		"""
		Returns the friend measure.

		Counter adds 1 for:
			each vertex which is shared by both neighborhoods,
			each edge existing between the neighborhoods
		"""

		# instantiate a counter
		output = 0

		# count each shared vertex and inter-neighborhoods edges
		for edge in product(neighborhood_1, neighborhood_2):
			if edge in self._g.edges:
				output += 1

		return output

	def _get_edge_topological_features(self, u, v):
		"""
		Returns a dictionary containing edge (u, v) topological features.

		:param u: one of the edge's vertices.
		:param v: one of the edge's vertices.
		:return: a dictionary.

		"""

		# If edge exists, remove it and maintain a boolean to add it back later
		edge_removed = False
		if self._g.has_edge(u, v):
			self._g.remove_edge(u, v)
			edge_removed = True

		# vertices' neighborhoods
		u_neighborhood = set(self._g.neighbors(u))
		v_neighborhood = set(self._g.neighbors(v))

		# vertices' degrees
		u_deg = len(u_neighborhood)
		v_deg = len(v_neighborhood)

		# preferential attachment score
		preferential_attachment_score = u_deg * v_deg

		# friends measure
		friends_measure = self._friends_measure(u_neighborhood, v_neighborhood)

		# join vertices' neighborhoods
		total_friends = len(u_neighborhood | v_neighborhood)

		# shortest path
		if nx.has_path(self._g, u, v):
			shortest_path = len(nx.shortest_path(self._g, u, v)) - 1
		else:
			shortest_path = -1

		# instantiate a dictionary to contain edge topological features
		output_dict = {
			'total_friends': total_friends,
			'preferential_attachment_score': preferential_attachment_score,
			'friends_measure': friends_measure,
			'shortest_path': shortest_path,
			'vertex_1_degree': u_deg,
			'vertex_2_degree': v_deg
		}

		# if edge was removed, add it back
		if edge_removed:
			self._g.add_edge(u, v)

		return output_dict

	########################################
	# edge lists topological features
	########################################

	def _get_all_topological_features(self, pos_edges: list, neg_edges: list):
		"""
		Iterates through 2 lists of edges and extract theirs topological features.

		Creates a dictionary of form {(u,v): {edge_features... , u features... , v features... }}.

		:param pos_edges: a list of tuples, each indicating an existing edge.
		:param neg_edges: a list of tuples, each indicating a non-existing edge.
		:return: a dictionary.
		"""

		output = {}
		print('\nExtracting positive edges features...\n')
		for (u, v) in tqdm(pos_edges):
			edge_dict = self._get_edge_topological_features(u, v)
			edge_dict.update({'edge_exist': 1})
			output[f'({u}, {v})'] = edge_dict

		print('\nExtracting negative edges features...\n')
		for (u, v) in tqdm(neg_edges):
			edge_dict = self._get_edge_topological_features(u, v)
			edge_dict.update({'edge_exist': 0})
			output[f'({u}, {v})'] = edge_dict

		return output

	########################################
	# create train and test sets of edges' topological features
	########################################

	def create_topological_features_df(
			self, positive_edges: list, negative_edges: list, save: bool = False, save_dir_path: str = None):
		"""
		Extracts topological features of all given edge lists and returns as DataFrame.

		Operates on a single graph.
		One can provide both positive_edges list and negative_edges list or just positive edges.
		"""

		edges_dict = None
		if negative_edges is not None and len(negative_edges) > 0:
			edges_dict = self._get_all_topological_features(positive_edges, negative_edges)

		elif negative_edges is None or len(negative_edges) == 0:
			edges_dict = self._get_all_topological_features(positive_edges, [])

		edges_df = pd.DataFrame.from_dict(edges_dict, orient='index')

		if save:
			edges_df.to_csv(save_dir_path, index=True, encoding='UTF-8')

		return edges_df
