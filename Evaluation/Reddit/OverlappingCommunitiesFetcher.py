"""
TODO: add module docstring.

"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import pandas as pd
from os.path import join
from os import listdir
import numpy as np
import json
from tqdm.autonotebook import tqdm


##################################
# OverlappingCommunitiesFetcher
##################################

class OverlappingCommunitiesFetcher:
	def __init__(self, dir_path: str):
		self._dir_path = dir_path

		self._list_dir = list(listdir(self._dir_path))
		self._partitions = dict()

	##################################
	# Main method
	##################################

	def create_overlapping_communities_partitions(
			self,
			random_try: int,
			max_comms: int,
			min_edge_weight: int,
			min_common_vertices: int,
			min_co_occurrences: int,
			min_comm_size: int,
			max_comm_size: int,
			random_seed: int = None):

		"""
		Creates a dictionary of form {community_name: vertices}, composed of overlapping communities.

		:param random_try: number of candidate communities to try and find overlapping communities.
		:param max_comms: number of overlapping communities to return.
		:param min_edge_weight: min. edge weight to consider.
		:param min_common_vertices: min. common vertices between communities.
		:param min_co_occurrences: min. common occurrences of communities together.
		:param min_comm_size: min. community size to consider.
		:param max_comm_size: max. community size to consider.
		:return: a dictionary of form {community_name: vertices}, composed of overlapping communities.
		"""

		if random_seed is not None:
			self._set_random_seed(random_seed)

		# Sample random community names
		comm_candidate_names = list(np.random.choice(self._list_dir, random_try))

		# Create partitions dictionary of form {community_name: vertices}
		partitions_candidates_dict = self._create_partitions(
			comm_candidate_names, min_edge_weight, min_comm_size, max_comm_size)

		# Create an upper-triangular (above the diagonal) matrix,
		# containing number of joint vertices between each couple of communities
		joint_vertices_candidates = self._create_common_vertices_upper_diagonal_matrix(partitions_candidates_dict)

		# Create a list of tuples of communities with at least min_common_vertices and min_common_vertices
		overlapping_communities_pairs = self._find_overlapping_communities_pairs(
			joint_vertices=joint_vertices_candidates,
			partitions_map=partitions_candidates_dict,
			min_common_vertices=min_common_vertices,
			min_co_occurrences=min_co_occurrences)

		# Create a list of all overlapping communities names
		overlapping_comm_names = self._unite_overlapping_communities_names(overlapping_communities_pairs)

		# Randomly choose max_comms of them
		overlapping_comm_names = np.random.choice(overlapping_comm_names, size=max_comms, replace=False).tolist()

		# Create a dictionary of overlapping partitions
		overlapping_partitions = {
			comm_name: vertices
			for comm_name, vertices
			in partitions_candidates_dict.items()
			if comm_name in overlapping_comm_names
		}

		return overlapping_partitions

	##################################
	# Utility methods
	##################################

	@staticmethod
	def _set_random_seed(random_seed):
		np.random.seed(random_seed)

	##################################
	# Partitions creating methods
	##################################

	def _create_partitions(self, comm_candidates_names, min_edge_weight, min_comm_size, max_comm_size):
		"""Creates a partition dictionary, of form {community_name: vertices}."""

		print('Reading edges csv files...')
		output = dict()
		for comm_name in tqdm(comm_candidates_names):
			try:
				# Extract vertices from edges csv
				comm_vertices = self._extract_vertices_from_edges_csv(comm_name, min_edge_weight=min_edge_weight)

				# Extract number of vertices
				comm_size = len(list(comm_vertices.values())[0])

				# Update partitions only if comm size is within boundaries
				if min_comm_size <= comm_size <= max_comm_size:
					output.update(comm_vertices)
				else:
					continue

			# Skip if file not found
			except FileNotFoundError:
				continue

		return output

	def _extract_vertices_from_edges_csv(self, comm_name: str, min_edge_weight: int = None):
		"""Reads an edges csv file, and returns the set of participating vertices as a dictionary."""

		# Memoization
		if comm_name in self._partitions:
			return {comm_name: self._partitions[comm_name]}

		# Create full file path
		edges_full_path = join(self._dir_path, comm_name, f'{comm_name}.edges.csv')

		# Create an edges DataFrame from CSV
		edges_df = pd.read_csv(edges_full_path)

		# Drop unnecessary columns (to avoid memory overflow)
		edges_df.drop(['maxdate', 'mindate'], axis=1, inplace=True)

		# Leave only edges with min_weight if given
		if min_edge_weight:
			edges_df = edges_df[edges_df['weight'] >= min_edge_weight]

		# Remove rows which will create self loops (as nx.Graph.selfloop_edges() does not work in networkx version 2)
		edges_df = edges_df[edges_df['__src_id'] != edges_df['__dst_id']]

		# Create a set of vertices from all edges source and destination vertices
		vertices_set = set(edges_df['__src_id']).union(set(edges_df['__dst_id']))

		# Populate a dictionary with the vertices set
		comm_vertices = {comm_name: list(vertices_set)}

		# Update partitions dictionary
		self._partitions.update(comm_vertices)

		return comm_vertices

	##################################
	# Overlapping communities methods
	##################################

	@staticmethod
	def _create_common_vertices_upper_diagonal_matrix(partitions_map):
		"""
		Creates an upper-triangular (above the diagonal) matrix,
		containing number of joint vertices between each couple of communities.
		"""

		# Extract community names
		comm_names = list(partitions_map.keys())

		# Create 0-like matrix with shape (num. communities, num. communities)
		joint_vertices_matrix = np.zeros((len(comm_names), len(comm_names)), dtype=int)

		# Fill upper triangular matrix (above the diagonal)
		for i, comm1 in enumerate(comm_names):
			for j, comm2 in enumerate(comm_names[i:]):

				# Skip self-joint vertices
				if comm1 == comm2:
					continue

				# Calculate number of joint vertices within two communities
				joint_vertices_matrix[i, j + i] = len(
					set(partitions_map[comm1]) &
					set(partitions_map[comm2])
				)

		return joint_vertices_matrix

	@staticmethod
	def _find_overlapping_communities_pairs(
			joint_vertices: np.array,
			partitions_map: dict,
			min_common_vertices: int,
			min_co_occurrences: int):
		"""Creates a list of tuples of communities with at least min_common_vertices and min_common_vertices."""

		# Extract comm names
		comm_names = list(partitions_map.keys())

		# An array with rows containing co-occurrences, which are more than min_common_vertices
		co_occur_rows = np.where(joint_vertices >= min_common_vertices)[0]

		# Unique rows and their counts
		unique, counts = np.unique(co_occur_rows, return_counts=True)

		# An array of indices indicating rows with more than min_co_occurrences
		num_co_occurrences_by_indices = np.where(counts >= min_co_occurrences)

		# Reduced joint_vertices matrix,
		# with rows with more than min_co_occurrences and more than min_common_vertices
		reduced_joint_vertices = joint_vertices[num_co_occurrences_by_indices]

		pairs_with_common_users = []
		pairs_row_col = list(zip(*np.where(reduced_joint_vertices >= min_common_vertices)))

		for pair in pairs_row_col:
			pairs_with_common_users.append((comm_names[pair[0]], comm_names[pair[1]]))

		return pairs_with_common_users

	@staticmethod
	def _unite_overlapping_communities_names(overlapping_communities_pairs):
		# union all overlapping community names
		comms_1, comms_2 = list(zip(*overlapping_communities_pairs))
		overlapping_comm_names = list(set(comms_1) | set(comms_2))
		return overlapping_comm_names

	##################################
	# Development sanity check method
	##################################

	def _get_reduced_common_vertices_matrix_and_comm_names(self, partitions_map):
		common_vertices_matrix = self._create_common_vertices_upper_diagonal_matrix(partitions_map)

		# rows which are not all zeros
		non_all_zero_rows_indices = common_vertices_matrix.any(axis=1)

		# reduced common vertices matrix and comm names
		reduced_common_vertices_matrix = common_vertices_matrix[non_all_zero_rows_indices]
		reduced_partition_names = np.array(list(partitions_map.keys()))[non_all_zero_rows_indices].tolist()

		return reduced_common_vertices_matrix, reduced_partition_names


def set_random_seed(random_seed):
	np.random.seed(random_seed)


if __name__ == '__main__':
	REDDIT_MAIN_PATH = 'E:\\Datasets\\reddit'
	OUTPUT_PATH = 'Data\\RawOverlappingPartitionMaps'
	CONFIG = {
		'random_try': 1500,
		'max_comms': 110,
		'min_edge_weight': 3,
		'min_common_vertices': 3,
		'min_co_occurrences': 3,
		'min_comm_size': 30,
		'max_comm_size': 1500
	}

	fetcher = OverlappingCommunitiesFetcher(REDDIT_MAIN_PATH)

	for random_seed in tqdm(range(101, 111)):
		set_random_seed(random_seed)
		overlapping_partitions = fetcher.create_overlapping_communities_partitions(**CONFIG)
		file_path = join(OUTPUT_PATH, f'partitions_map_{random_seed}.json')
		with open(file_path, 'w') as file:
			json.dump(overlapping_partitions, file)
