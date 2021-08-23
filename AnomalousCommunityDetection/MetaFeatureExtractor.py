"""
TODO: add module docstring.

"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import networkx as nx
import numpy as np


########################################
# MetaFeatureExtractor
########################################

class MetaFeatureExtractor:
	def __init__(self, edges_existence_prob_dict):

		self.edge_probs = edges_existence_prob_dict

		# if graph was sampled before creating edges_existence_prob_dict
		# create graph only from these edges:
		self.g = nx.from_edgelist(self.edge_probs.keys())

		# get community-representing vertices set
		# community needs to be first in the index tuples
		self.comm_vertices = self._get_community_repr_vertices_set(self.edge_probs)

	@ staticmethod
	def _get_community_repr_vertices_set(dic):
		"""Returns a set of recipe vertices which are involved in edges within the input."""

		# extract all edges, which are the keys of the dictionary
		edges = dic.keys()

		# extract only the first element of the edge-tuple (partite 1 node) and create a set
		return set(list(zip(*edges))[0])

	########################################
	# Community-representing vertices meta-features extraction
	########################################

	@ staticmethod
	def _binary_label_by_thresh(prob, thresh):
		return 1 if prob >= thresh else 0

	def _vertex_meta_features(self, u, thresh):
		"""
		Return a dictionary containing vertex's meta-features.

		It is assumed the graph g contains edge existing probabilities as edge attributes called 'exist_prob',
		for each edge that contains vertex u.

		:param u:
		:param thresh:
		:return:
		"""

		# extract edge existing probabilities
		vertex_edges_probs = [self.edge_probs[edge] for edge in self.g.edges(u)]

		# edges existing probability meta-features
		normality_prob_mean = np.mean(vertex_edges_probs)

		######
		######
		# leave it ????
		######
		######

		normality_prob_std = 1 - np.std(vertex_edges_probs)

		normality_prob_median = np.median(vertex_edges_probs)

		# label edges by the given threshold
		labels_by_thresh = [self._binary_label_by_thresh(prob, thresh=thresh) for prob in vertex_edges_probs]

		# labeled edges meta-features
		predicted_label_mean = np.mean(labels_by_thresh)

		######
		######
		# leave it ????
		######
		######

		predicted_label_std = 1 - np.std(labels_by_thresh)

		weighted_sum = self._weighted_sum(
			normality_prob_mean,
			normality_prob_std,
			normality_prob_median,
			predicted_label_mean,
			predicted_label_std)

		return {
			'normality_prob_mean__score': normality_prob_mean,
			'normality_prob_std__score': normality_prob_std,
			'normality_prob_median__score': normality_prob_median,
			'predicted_label_mean__score': predicted_label_mean,
			'predicted_label_std__score': predicted_label_std,
			'weighted_sum__score': weighted_sum
		}

	@staticmethod
	def _weighted_sum(npmean, npstd, npmed, prlmean, prlstd):
		####
		# Reddit
		####

		#weights = np.array([9.22421127, -3.89997392, 0.11203075, 10.08882187, -3.48322422])  # 1-std
		#weights = np.array([0.34535817, 0.14617009, 0.00369713, 0.38446181, 0.12031281])  # std

		#weights = np.array([-11.00535, 2.09727, -0.01981, -11.2241, 2.31862])
		#weights = np.array([-20.38115, 3.51572, -0.1253, -18.01393, 11.02389])  # 21.04 try
		#intercept = 14.78863
		#intercept = 24.10015  # 21.04 try

		weights = np.array([0.15183381, -0.03859362, 0.00116014, 0.14657687, -0.0609772])  # 29.04.21 reddit

		meta_feats_arr = np.vstack((npmean, npstd, npmed, prlmean, prlstd)).T
		return np.squeeze(meta_feats_arr.dot(weights))

	def get_comm_repr_vertices_meta_features(self, thresh=0.8):
		"""
		TODO: document
		Extract meta-features and returns as dictionary of form {comm_name: {meta-feat_1: x_1, ... meta-feat_n: x_n}}.

		Parameters
		----------
		thresh: A float to...

		Returns
		-------
		Dictionary containing each community-representing meta-features.

		Examples
		--------
		...
		"""

		return {
			comm: self._vertex_meta_features(comm, thresh)
			for comm
			in self.comm_vertices
		}

