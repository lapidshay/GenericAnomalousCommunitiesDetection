"""
TODO: Check this code again!
TODO: add module docstring.
TODO: support other normal and anomaly algorithms
TODO: fix random state and choose np or random random state
"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import networkx as nx
from numpy.random import choice
import numpy as np
from copy import deepcopy
import json
import pandas as pd
import os


##################################
# ...
##################################

class RedditGraphGenerator:

	def __init__(self, reddit_dir_path, anom_comm_alg, anom_inter_p , k_min, k_max):

		self._reddit_dir_path = reddit_dir_path
		self._anom_comm_alg = anom_comm_alg
		self._anom_inter_p = anom_inter_p
		self._k_min = k_min
		self._k_max = k_max

		self._start = 1
		self._G = None
		self._partitions = dict()
		self._updated_partitions = dict()

	##################################
	# ...
	##################################

	@staticmethod
	def _set_random_seed(random_seed):
		np.random.seed(random_seed)

	def generate_network(
			self,
			real_partitions_map: dict, min_edge_weight: int,
			anom_comm_sizes, anom_m, random_seed: int = None, avoid_anomlies: bool = False):

		if random_seed is not None:
			self._set_random_seed(random_seed)

		# Reset previously created networks
		self._reset_network()

		# create a DataFrame of all subreddits' edges
		all_edges_df, all_vertices_list = self._create_edges_df_and_vertices_list(
			partitions_map=real_partitions_map, min_edge_weight=min_edge_weight)

		# create a graph
		self._G = self._create_network_from_edges_df_and_vertices_list(all_edges_df, all_vertices_list)

		self._updated_partitions = deepcopy(self._partitions)

		if not avoid_anomlies:
			# Create and add anomalous communities to the network
			self.add_anomalous_comms(anom_comm_sizes, anom_m)

		# return a deep copy of the graph
		return deepcopy(self._G)

	def _reset_network(self):
		self._start = 1
		self._G = None
		self._partitions = dict()
		self._updated_partitions = dict()

	##################################
	# Reproduce Reddit Graph
	##################################

	def _read_edges_csv(self, sub_reddit_name: str, min_edge_weight: int = None):

		# Create an edges DataFrame from CSV
		edges_full_path = os.path.join(self._reddit_dir_path, sub_reddit_name, f'{sub_reddit_name}.edges.csv')
		edges_df = pd.read_csv(edges_full_path)

		# Drop unnecessary columns (to avoid memory overflow)
		edges_df = edges_df[['__src_id', '__dst_id', 'weight']]

		# Leave only edges with min_weight
		if min_edge_weight:
			edges_df = edges_df[edges_df['weight'] >= min_edge_weight]

		# remove rows which will create self loops
		# (as nx.Graph.selfloop_edges does not work in networkx version 2)
		edges_df = edges_df[edges_df['__src_id'] != edges_df['__dst_id']]

		edges_df['community'] = sub_reddit_name

		community_vertices = list(set(edges_df['__src_id']).union(set(edges_df['__dst_id'])))

		return edges_df, community_vertices

	def _create_edges_df_and_vertices_list(self, partitions_map: dict, min_edge_weight: int = None):

		all_edges_dfs = []
		all_vertices_list = []
		for sub in partitions_map:
			sub_edges_df, sub_vertices = self._read_edges_csv(sub, min_edge_weight)
			all_edges_dfs.append(sub_edges_df)

			all_vertices_list += sub_vertices
			self._partitions[sub] = sub_vertices

		all_edges_df = pd.concat(all_edges_dfs)

		return all_edges_df, all_vertices_list

	@staticmethod
	def _create_network_from_edges_df_and_vertices_list(edges_df: pd.DataFrame, vertices_list: list):

		# create graph from edges DataFrame
		G = nx.from_pandas_edgelist(edges_df, source='__src_id', target='__dst_id', create_using=nx.Graph)

		assert set(G.nodes) == set(vertices_list)
		# add vertices without edges
		#G.add_nodes_from(vertices_list)

		return G

	##################################
	# Add anomalous communities methods
	##################################

	def _create_anomalous_node_names(self, anom_comm_sizes):
		# add enumerated-named nodes, according to normal anomalous community sizes and the existing normal community nodes
		total_node_number = sum(anom_comm_sizes)

		nodes_enumeration = range(self._start, self._start + total_node_number)

		node_num_length = len(str(total_node_number)) + 1

		node_mapping = {enum: f'GenNode{str(enum).zfill(node_num_length)}' for enum in nodes_enumeration}

		return node_mapping

	def _create_anomalous_community_names(self, anom_comm_sizes):
		com_num_length = len(anom_comm_sizes) + 1

		community_names = [f'AnomComm{str(enum).zfill(com_num_length)}' for enum in range(1, len(anom_comm_sizes) + 1)]

		return community_names

	def add_anomalous_comms(self, anom_comm_sizes, anom_m):
		"""
		Adds edges of anomalous communities, and edges connecting them to other communities to main graph.

		For each normal community, create a sub-graph using graph generator algorithm (self._norm_comm_alg),
		modify edge names to align with correct community's nodes, and add the edges to main graph.

		:param anom_comm_sizes: A list of integers, containing sizes of normal communities.
		:param anom_m: currently, for erdos_renyi_graph algorithm, it is p (Probability for edge creation).
		:return: A deep copy of the nx.Graph.
		TODO: allow input of different arguments to different anomalous community algorithm, other than anom_m
		"""

		# add enumerated-named nodes, according to normal anomalous community sizes and the existing normal community nodes
		node_mapping = self._create_anomalous_node_names(anom_comm_sizes)
		self._G.add_nodes_from(node_mapping.values())

		community_names = self._create_anomalous_community_names(anom_comm_sizes)

		for n, comm_name in zip(anom_comm_sizes, community_names):

			current_comm = self._anom_comm_alg(n, anom_m)

			anom_comm_mapping = {u: u + self._start for u in range(0, n)}
			current_comm = nx.relabel_nodes(current_comm, anom_comm_mapping)
			current_comm = nx.relabel_nodes(current_comm, node_mapping)

			# create edges for each of communities' sizes
			# self._start is used to modify edge names to align with correct community's nodes
			#edges = ((u + self._start, v + self._start) for u, v in current_comm.edges())

			# add edges and nodes to main graph
			self._G.add_edges_from(current_comm.edges())
			self._G.add_nodes_from(current_comm.nodes())

			# add current community's nodes to a partition, and updated partition
			self._partitions[comm_name] = list(current_comm.nodes())
			self._updated_partitions[comm_name] = list(current_comm.nodes())

			# update the counter, to enable edge names alignment
			self._start += n

		# create edges between anomalous communities and normal communities
		self._create_anomalous_comms_inter_edges(anom_comm_sizes)

		# return a deep copy of the graph
		return deepcopy(self._G)

	def _create_anomalous_comms_inter_edges(self, anom_comm_sizes):
		"""

		Add edges between anomalous communities and normal communities to the main graph.

		For each anomalous community, choose nodes (self.inter_p proportioned amount) to connect to normal communities.
		Then for each node, choose which normal community (randomly-weighted by size) to connect to, and then which
		k (between self._k_min and self._k_max) nodes (randomly-weighted by degree) to connect to.

		Create these edges and add to main graph.
		Add the newly connected nodes to updated partitions.
			It makes the assumption, that because a node is connected with high probability
			to central nodes in another community, then the node is now part ot the community.

		:param anom_comm_sizes: A list of integers, containing sizes of normal communities.
		"""

		# choose all partitions except for anomalous community partitions
		normal_partitions = {k: v for k, v in self._partitions.items() if 'AnomComm' not in k}
		anomalous_partitions = {k: v for i, (k, v) in enumerate(self._partitions.items()) if i >= len(normal_partitions)}

		# create an array of number of nodes in each anomalous community (partition), to connect to normal communities
		# self.p inter is the proportion of nodes in each community to be connected to other communities
		num_nodes_to_con = (np.array(anom_comm_sizes) * self._anom_inter_p).astype(int)

		# for each anomalous community
		for p, (anom_comm_name, anom_comm_vertices) in enumerate(anomalous_partitions.items()):

			# randomly choose nodes to connect to outer partitions
			nodes_to_connect_outer = choice(anom_comm_vertices, size=num_nodes_to_con[p], replace=False, p=None)

			# create a distribution from sizes of partitions
			norm_partition_weights = [len(part) for part in normal_partitions.values()]
			norm_partition_weights = [w / sum(norm_partition_weights) for w in norm_partition_weights]

			# for each node which was chosen to connect to other communities
			for node_to_connect_from in nodes_to_connect_outer:

				# randomly choose k (between k_min and k_max) edges to create
				for _ in range(choice(range(self._k_min, self._k_max + 1))):

					# randomly (weighted by partition size) choose another partition to connect to
					outer_partition_to_connect = choice(list(normal_partitions.keys()), size=1, replace=False, p=norm_partition_weights)[0]

					# create 2 lists, of nodes and their weights (their degree),
					# from the partition which was chosen to be connected to
					nodes, node_weights = zip(*(self._G.degree(self._partitions[outer_partition_to_connect])))

					# create a distribution from weights
					node_weights = [w / sum(node_weights) for w in node_weights]

					# choose a node to connect to, within the partition which was chosen to be connected to
					node_to_connect_to = choice(nodes, size=1, replace=False, p=node_weights)[0]

					# add edges between partitions
					self._G.add_edge(node_to_connect_from, node_to_connect_to)

					#print(f'node "{node_to_connect_from}" from "{anom_comm_name}" was connected to node "{node_to_connect_to}" from "{outer_partition_to_connect}"')

					# update that partition to contain the newly connected node
					self._updated_partitions[outer_partition_to_connect].append(node_to_connect_from)

	##################################
	# Partitions interface methods
	##################################

	def get_partitions(self):
		"""
		Returns a dictionary of the networks' partitions.

		A map indicating each community's vertices.

		Returns
		-------
		A dictionary, of partitions (a map)
		"""

		return self._updated_partitions

	def save_partitions(self, file_path: str):
		"""Saves a dictionary as a Json file in given file_path."""

		with open(f'{file_path}', 'w', encoding='UTF8') as file:
			json.dump(self.get_partitions(), file)

