__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import networkx as nx
import itertools
from tqdm.notebook import tqdm
import numpy as np
from datetime import datetime
import json


##################################
# UnAttributed AMEN
##################################

class UnAttributedAMEN:
	def __init__(self, G):
		self._G = G  # Undirected networkx.Graph()
		self._m = len(self._G.edges())  # Num edges

	##################################
	# Helper methods
	##################################

	def _edge_surprisingness(self, v_i, v_j):
		"""Return edge surprisingness."""

		k_i = self._G.degree(v_i)
		k_j = self._G.degree(v_j)

		return (k_i * k_j) / (2 * self._m)

	def _is_edge_exist(self, v_i, v_j):
		"""Returns 1 if edge exists and 0 otherwise."""

		return 1 if (v_i, v_j) in self._G.edges() else 0

	##################################
	# Inner consistency
	##################################

	@staticmethod
	def _inner_edge_surprisingness_contribution(a_ij, surp_vi_vj):
		"""Returns surprisingness of an inner edge."""

		return a_ij - surp_vi_vj

	##################################
	# External separability
	##################################

	def _get_boundary_vertices(self, community_vertices: list):
		"""Returns a set of boundary vertices given a community."""

		boundary_edges = nx.edge_boundary(self._G, community_vertices)
		boundary_edges_nodes = set(itertools.chain(*boundary_edges))
		boundary_nodes = boundary_edges_nodes.difference(set(community_vertices))

		return boundary_nodes

	@staticmethod
	def _cross_boundary_edge_surprisingness_contribution(surp_vi_vb):
		"""Returns surprisingness of a cross-boundary edge."""

		return 1 - min(1, surp_vi_vb)

	##################################
	# Normality
	##################################

	def community_normality_measure(self, community_vertices: list):
		"""Returns normality measure of a community."""

		# Compute community consistency and inner edges surprisingness
		consistency = 0
		comm_edges_surprisingness = []
		for v_i in community_vertices:
			for v_j in community_vertices:

				surp_vi_vj = self._edge_surprisingness(v_i=v_i, v_j=v_j)
				a_ij = self._is_edge_exist(v_i=v_i, v_j=v_j)
				consistency += self._inner_edge_surprisingness_contribution(a_ij=a_ij, surp_vi_vj=surp_vi_vj)
				comm_edges_surprisingness.append(surp_vi_vj)

		# Normalize consistency in range [0, 1]
		cnss_norm_max_factor = np.power(len(community_vertices), 2)
		cnss_norm_min_factor = -np.sum(comm_edges_surprisingness)
		normalized_consistency = (consistency - cnss_norm_min_factor) / (cnss_norm_max_factor - cnss_norm_min_factor)

		# Compute separability
		boundary_vertices = self._get_boundary_vertices(community_vertices)
		separability = 0
		for v_i in community_vertices:
			for v_b in boundary_vertices:

				surp_vi_vb = self._edge_surprisingness(v_i=v_i, v_j=v_b)
				separability -= self._cross_boundary_edge_surprisingness_contribution(surp_vi_vb=surp_vi_vb)

		# Normalize separability in range [-1, 0]
		separability_norm_factor = np.sum(1 - np.minimum(1, comm_edges_surprisingness))
		normalized_separability = separability / (separability_norm_factor - separability)

		# Compute normality
		normality = normalized_consistency + normalized_separability

		return normality

	def rank_by_normality(self, partitions_map: dict, save=False, save_path=None):
		"""Ranks all partitions by normality."""

		comm_normalities = [
			(comm, self.community_normality_measure(comm_vertices))
			for comm, comm_vertices
			in tqdm(partitions_map.items())
		]

		comm_normalities.sort(key=lambda x: x[1], reverse=False)

		# Save rank
		if save:
			self._save_ranks(comm_normalities, save_path)

		return comm_normalities

	##################################
	# Utility methods
	##################################

	@staticmethod
	def _save_ranks(comm_normalities, save_path):
		"""Saves a sorted normality ranking list."""

		if save_path is None:
			save_path = f'comm_normality_ranking__{datetime.now().strftime("%m.%d_%H.%M")}.json'

		with open(save_path, 'w', encoding='UTF-8') as file:
			json.dump(comm_normalities, file)

	@staticmethod
	def load_ranks(path):
		"""Loads and returns a sorted normality ranking list, and convert back to list of tuples."""

		with open(path, 'r', encoding='UTF-8') as file:
			return [tuple(element) for element in json.load(file)]
