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


##################################
# Anomaly-Infused Community-Structured Random Network Generator
##################################

class AnomalyInfusedCommunityStructuredRandomNetworkGenerator:
	"""
	A class that creates a community-structured graph which simulates
	a graph composed of normal communities and anomalous communities.
	The class object contains the data of the graph partitions, as created during the graph generation.
	"""

	def __init__(self, norm_comm_alg, anom_comm_alg, k_min, k_max, random_seed=None):
		"""
		Instantiates an object which creates the graph and holds partition data.

		Parameters
		----------
		norm_comm_alg: Graph generator algorithm to be used for normal communities creation.
		anom_comm_alg: Graph generator algorithm to be used for anomalous communities creation.
		p_inter: Proportion of nodes in each community to be connected to other communities.
		k_min: minimal number of edges to be created from each node.
		k_max: maximal number of edges to be created from each node.
		random_seed:
		TODO: support other normal and anomaly algorithms
		"""

		self._norm_comm_alg = norm_comm_alg
		self._anom_comm_alg = anom_comm_alg
		self._k_min = k_min
		self._k_max = k_max

		self._random_seed = random_seed

		self._start = 1
		self._G = None
		self._partitions = []
		self._updated_partitions = []

	##################################
	# Generate Anomaly-Infused Community-Structured Random Network
	##################################

	def generate_network(self, norm_comm_sizes, norm_m, norm_inter_p, anom_comm_sizes, anom_m, anom_inter_p):
		"""
		Generates an Anomaly-Infused Community-Structured Random Network.
		TODO: allow input of different arguments to different normal community algorithm, other than norm_m

		Parameters
		----------
		norm_comm_sizes: A list of integers, containing sizes of normal communities.
		norm_m: Number of edges to attach from a new node to existing nodes, within a community.
		anom_comm_sizes: A list of integers, containing sizes of normal communities.
		anom_m: currently, for erdos_renyi_graph algorithm, it is p (Probability for edge creation).

		Returns
		-------
		A deep copy of the nx.Graph network.

		Examples
		--------
		"""

		# Reset previously created networks
		self._reset_network()

		# Create normal communities
		self.create_normal_comms(norm_comm_sizes, norm_m, norm_inter_p)

		# Create and add anomalous communities to the network
		self.add_anomalous_comms(anom_comm_sizes, anom_m, anom_inter_p)

		# return a deep copy of the graph
		return deepcopy(self._G)

	def _reset_network(self):
		self._start = 1
		self._G = None
		self._partitions = []
		self._updated_partitions = []

	##################################
	# Create normal communities methods
	##################################

	def create_normal_comms(self, norm_comm_sizes, norm_m, norm_inter_p):
		"""
		Generates a community-structured graph.

		TODO: allow input of different arguments to different normal community algorithm, other than norm_m

		Parameters
		----------
		norm_comm_sizes: A list of integers, containing sizes of normal communities.
		norm_m: Number of edges to attach from a new node to existing nodes, within a community.

		Returns
		-------
		A deep copy of the nx.Graph.

		Examples
		--------

		"""

		# instantiate empty Graph
		self._G = nx.Graph()

		# add enumerated-named nodes, according to normal community sizes
		total_node_number = sum(norm_comm_sizes)
		self._G.add_nodes_from(range(1, total_node_number))

		# instantiate a counter, to help with correct edge creation
		self._start = 1

		# create edges of the normal communities
		self._create_normal_comms(norm_comm_sizes=norm_comm_sizes, norm_m=norm_m)

		# create edges between the normal communities
		self._create_normal_comms_inter_edges(norm_inter_p=norm_inter_p)

		# return a deep copy of the graph
		return deepcopy(self._G)

	def _create_normal_comms(self, norm_comm_sizes, norm_m):
		"""
		Add normal communities' edges to main graph.

		For each normal community, create a sub-graph using graph generator algorithm (self._norm_comm_alg),
		modify edge names to align with correct community's nodes, and add the edges to main graph.

		Parameters
		----------
		norm_comm_sizes: a list of integers, containing sizes of normal communities.
		norm_m: Number of edges to attach from a new node to existing nodes, within a community.
		"""

		for n in norm_comm_sizes:

			# create edges for each of communities' sizes
			# self._start is used to modify edge names to align with correct community's nodes
			# TODO: allow input of different arguments to different normal community algorithm
			edges = (
				(u + self._start, v + self._start)
				for u, v
				in self._norm_comm_alg(n, norm_m, seed=self._random_seed).edges()
			)

			# add edges to main graph
			self._G.add_edges_from(edges)

			# add current community's nodes to a partition
			current_community_nodes = range(self._start, self._start + n)
			self._partitions.append(set(current_community_nodes))

			# update the counter, to enable edge names alignment
			self._start += n

	def _create_normal_comms_inter_edges(self, norm_inter_p):
		"""
		Add edges between communities to the main graph.

		For each normal community, choose nodes (self.inter_p proportioned amount) to connect to other communities.
		Then for each node, choose which community (randomly-weighted by size) to connect to, and then which
		k (between self._k_min and self._k_max) nodes (randomly-weighted by degree) to connect to.

		Create these edges and add to main graph.
		Add the newly connected nodes to updated partitions.
			It makes the assumption, that because a node is connected with high probability
			to central nodes in another community, then the node is now part ot the community.
		"""

		# a list of edges to add to main graph
		inter_edges = []

		# create a copy of partitions in order to update them after creating the edges
		updated_partitions = deepcopy(self._partitions)

		# create an array of number of word in each community (partition), to connect to other communities
		# norm_inter_p is the proportion of nodes in each community to be connected to other communities
		num_nodes_to_con = np.round(
			(np.array([
				len(part)
				for part
				in self._partitions
			]) * norm_inter_p)).astype(int)

		# for each normal community
		for p, partition in enumerate(self._partitions):

			# randomly choose nodes to connect to outer partitions
			nodes_to_connect_outer = choice(tuple(self._partitions[p]), size=num_nodes_to_con[p], replace=False, p=None)

			# partitions to choose to connect to (all out of current partition)
			partitions_to_choose = self._partitions[:p] + self._partitions[p + 1:]

			# create a distribution from sizes of partitions
			partition_weights = [len(part) for part in partitions_to_choose]
			partition_weights = [w / sum(partition_weights) for w in partition_weights]

			# for each node which was chosen to connect to other communities
			for node_to_connect_from in nodes_to_connect_outer:

				# randomly choose k (between k_min and k_max) edges to create
				for _ in range(choice(range(self._k_min, self._k_max + 1))):

					# randomly (weighted by partition size) choose another partition to connect to
					outer_partition_to_connect = choice(
						partitions_to_choose, size=1, replace=False, p=partition_weights
					)[0]

					# create 2 lists, of nodes and their weights (their degree),
					# from the partition which was chosen to be connected to
					nodes, node_weights = zip(*(self._G.degree(outer_partition_to_connect)))

					# create a distribution from node weights
					node_weights = [w / sum(node_weights) for w in node_weights]

					# choose a node to connect to, within the partition which was chosen to be connected to
					node_to_connect_to = choice(nodes, size=1, replace=False, p=node_weights)[0]

					# create edge and add to output list
					inter_edges.append((node_to_connect_from, node_to_connect_to))

					# extract the index of the partition which was chosen to be connected to
					up_part_idx = self._partitions.index(outer_partition_to_connect)

					# update that partition to contain the newly connected node
					updated_partitions[up_part_idx].add(node_to_connect_from)

					#print(f'node {node_to_connect_from} from partition {p+1} is being connected to node {node_to_connect_to} from partition {outer_partition_to_connect}')

		# add edges between communities to the main graph
		self._G.add_edges_from(inter_edges)

		# update self._updated_partitions with new partitions
		self._updated_partitions = updated_partitions

	##################################
	# Add anomalous communities methods
	##################################

	def add_anomalous_comms(self, anom_comm_sizes, anom_m, anom_inter_p):
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
		total_node_number = sum(anom_comm_sizes)
		self._G.add_nodes_from(range(self._start, self._start+total_node_number))

		for n in anom_comm_sizes:

			# create edges for each of communities' sizes
			# self._start is used to modify edge names to align with correct community's nodes
			edges = ((u + self._start, v + self._start) for u, v in self._anom_comm_alg(n, anom_m).edges())

			# add edges to main graph
			self._G.add_edges_from(edges)

			# add current community's nodes to a partition, and updated partition
			current_community_nodes = range(self._start, self._start + n)
			self._partitions.append(set(current_community_nodes))
			self._updated_partitions.append(set(current_community_nodes))

			# update the counter, to enable edge names alignment
			self._start += n

		# create edges between anomalous communities and normal communities
		self._create_anomalous_comms_inter_edges(anom_comm_sizes, anom_inter_p)

		# return a deep copy of the graph
		return deepcopy(self._G)

	def _create_anomalous_comms_inter_edges(self, anom_comm_sizes, anom_inter_p):
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
		normal_partitions = self._partitions[:-len(anom_comm_sizes)]

		anomalous_partitions = self._partitions[-len(anom_comm_sizes):]

		# create an array of number of nodes in each anomalous community (partition), to connect to normal communities
		# self.p inter is the proportion of nodes in each community to be connected to other communities
		num_nodes_to_con = np.round(
			np.array([
				len(anom_part)
				for anom_part
				in anomalous_partitions
			]) * anom_inter_p).astype(int)
		# num_nodes_to_con = np.array([3 for _ in self._partitions])  fixed number of nodes to connect

		# for each anomalous community
		for p, anomaly_community in enumerate(anomalous_partitions):

			# randomly choose nodes to connect to outer partitions
			nodes_to_connect_outer = choice(tuple(anomaly_community), size=num_nodes_to_con[p], replace=False, p=None)

			# create a distribution from sizes of partitions
			partition_weights = [len(part) for part in normal_partitions]
			partition_weights = [w / sum(partition_weights) for w in partition_weights]

			# for each node which was chosen to connect to other communities
			for node_to_connect_from in nodes_to_connect_outer:

				# randomly choose k (between k_min and k_max) edges to create
				for _ in range(choice(range(self._k_min, self._k_max + 1))):

					# randomly (weighted by partition size) choose another partition to connect to
					outer_partition_to_connect = choice(normal_partitions, size=1, replace=False, p=partition_weights)[0]

					# create 2 lists, of nodes and their weights (their degree),
					# from the partition which was chosen to be connected to
					nodes, node_weights = zip(*(self._G.degree(outer_partition_to_connect)))

					# create a distribution from weights
					node_weights = [w / sum(node_weights) for w in node_weights]

					# choose a node to connect to, within the partition which was chosen to be connected to
					node_to_connect_to = choice(nodes, size=1, replace=False, p=node_weights)[0]

					# add edges between partitions
					self._G.add_edge(node_to_connect_from, node_to_connect_to)

					# extract the index of the partition which was chosen to be connected to
					up_part_idx = normal_partitions.index(outer_partition_to_connect)

					# update that partition to contain the newly connected node
					self._updated_partitions[up_part_idx].add(node_to_connect_from)

					#print(f'node {node_to_connect_from} from partition {p+len(normal_partitions)+1} is being connected to node {node_to_connect_to} from partition {self._updated_partitions[up_part_idx]}')

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

		com_num_length = len(str(len(self._updated_partitions))) + 1

		partitions_strings_lists = [[str(v) for v in c] for c in self._updated_partitions]

		return {f'comm{str(c+1).zfill(com_num_length)}': comm for c, comm in enumerate(partitions_strings_lists)}

	def save_partitions(self, file_path: str):
		"""Saves a dictionary as a Json file in given file_path."""

		with open(f'{file_path}', 'w') as file:
			json.dump(self.get_partitions(), file)

