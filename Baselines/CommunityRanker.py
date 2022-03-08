__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import networkx as nx
from networkx import NetworkXError
import json
from os import path
from datetime import datetime
from tqdm.autonotebook import tqdm
from Baselines.UnAttributedAMEN import UnAttributedAMEN


##################################
# Community Ranker
##################################

class CommunityRanker:
	def __init__(self, G: nx.Graph, file_name_prefix: str = None):
		self._G = G
		self._file_name_prefix = file_name_prefix
		self._unattributed_amen = UnAttributedAMEN(G)

	##################################
	# Measures scores
	##################################

	def _community_average_degree(self, community_vertices: list):
		"""
		Computes community's average degree score, given a list of vertices.
		The computation is done over the sub network induced from the community's vertices.
		Based on internal consistency only.
		Communities ranked at the boundaries are anomalous (normal)? which way to look?
		"""

		comm_edges = nx.subgraph(self._G, community_vertices).edges()
		return 2 * len(comm_edges) / len(community_vertices)

	def _community_cut_ratio(self, community_vertices: list):
		"""
		Computes community's cut ratio score, given a list of vertices.
		Based on external separability only.
		The fraction of existing cut edges out of all possible edges.
		The lower the more high-quality (normal), the higher the worse quality (anomalous)?
		"""

		return \
			nx.cuts.cut_size(self._G, community_vertices) / \
			(len(community_vertices) * (len(self._G.nodes()) - len(community_vertices)))

	def _community_conductance(self, community_vertices: list):
		"""
		Computes community's conductance score, given a list of vertices.
		Based on both internal consistency and external separability.
		The fraction of total edge volume that points outside the community.
		The lower the more high-quality (normal), the higher the worse quality (anomalous).
		"""

		conductance = nx.algorithms.cuts.conductance(self._G, community_vertices)
		return conductance

	def _community_flake_odf(self, community_vertices: list):
		"""
		Computes community's Flake-ODF score, given a list of vertices.
		Based on both internal consistency and external separability.
		The fraction of community vertices that have fewer edges pointing inside the community than to the outside.
		The lower the more high-quality (normal), the higher the worse quality (anomalous)?
		"""

		sub_network = self._G.subgraph(community_vertices)
		odf_vertices = []
		for v in community_vertices:
			try:
				kv = self._G.degree(v)
			except NetworkXError:
				return -1

			nam_inside_neighbors = len(list(sub_network.neighbors(v)))
			if nam_inside_neighbors < kv / 2:
				odf_vertices.append(v)

		flake_odf = len(odf_vertices) / len(community_vertices)
		return flake_odf

	def _community_avg_odf(self, community_vertices: list):
		"""
		Computes community's Average-ODF score, given a list of vertices.
		Based on external separability.
		The average fraction of the community cut (over size of community)
		The lower the more high-quality (normal), the higher the worse quality (anomalous)?
		"""
		return nx.cuts.cut_size(self._G, community_vertices) / len(community_vertices)

	##################################
	# Main methods
	##################################

	def rank_communities_by(self, partitions_map: dict, by: str, save: bool = False, save_dir: str = None):
		"""Creates a sorted list of tuples of communities' ranking by a given measure."""

		assert by in ['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf', 'unattr_amen'], \
			"You must choose between ['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf', 'unattr_amen]"

		if by == 'avg_degree':
			score_func = self._community_average_degree
		elif by == 'cut_ratio':
			score_func = self._community_cut_ratio
		elif by == 'conductance':
			score_func = self._community_conductance
		elif by == 'flake_odf':
			score_func = self._community_flake_odf
		elif by == 'avg_odf':
			score_func = self._community_avg_odf
		elif by == 'unattr_amen':
			score_func = self._unattributed_amen.community_normality_measure

		sort_reverse = False if by in ['avg_degree', 'unattr_amen'] else True

		scores = [
			(comm_name, score_func(comm_vertices))
			for comm_name, comm_vertices
			in tqdm(partitions_map.items())
		]

		scores.sort(key=lambda x: x[1], reverse=sort_reverse)

		if save:
			self._save_ranking(scores, save_dir, by)

		return scores

	def rank_communities_by_all_measures(self, partitions_map: dict, rank_by: list = None, exclude_amen: bool = False):
		"""Ranks communities by all measures and returns a dictionary of the rankings and scores."""

		assert (
				rank_by is None or
				set(rank_by).issubset(['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf', 'unattr_amen'])
		), "Choose None or any of ['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf', 'unattr_amen]"

		if rank_by is None:
			rank_by = ['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf', 'unattr_amen']

		output = {}
		for measure in rank_by:
			if exclude_amen and measure == 'unattr_amen':
				print('Skipping AMEN\n')
				continue
			print(f'Ranking by "{measure}"...')
			measure_ranking = self.rank_communities_by(partitions_map=partitions_map, by=measure)
			ranking, scores = zip(*measure_ranking)
			output[f'{measure}__ranking'] = ranking
			output[f'{measure}__score'] = scores

		return output

	##################################
	# Utility functions
	##################################

	def _save_ranking(self, ranking, dir_path, by):
		"""Saves a list of tuples as json file."""

		file_name = f'{by}_ranking__{datetime.now().strftime("%m.%d_%H.%M")}.json'
		if self._file_name_prefix is not None:
			file_name = f'{self._file_name_prefix}__{file_name}'
		file_path = path.join(dir_path, file_name)

		with open(file_path, 'w', encoding='UTF-8') as file:
			json.dump(ranking, file)

	@staticmethod
	def load_ranking(file_path: str):
		"""Loads and returns a sorted list json file, and convert back to list of tuples."""

		with open(file_path, 'r', encoding='UTF-8') as file:
			return [tuple(element) for element in json.load(file)]
