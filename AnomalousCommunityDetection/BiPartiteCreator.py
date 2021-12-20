"""
TODO: add module docstring.

"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import networkx as nx
from copy import deepcopy
from .utils import print_bipartite_properties


##################################
# A class for creating a BiPartite Graph from a partitions dictionary
##################################

class BiPartiteCreator:
	"""
	A class for creating a BiPartite Graph from a partitions dictionary.

	Attributes:
		_unfiltered_partitions_dict: Full partitions dictionary, before filtering.
		_partitions_dict: Will hold the filtered partitions.
		_vertices: A set to hold the set of vertices.
		_community_partite_label: Label of the community-representing vertices part.
		_vertex_partite_label: Label of the vertices part.
		_BPG: The BiPartite graph.
	"""

	def __init__(self, partitions_map: dict):
		"""
		Instantiates an object to create the BiPartite graph and hold partition data.
		"""
		self._unfiltered_partitions_dict = partitions_map
		self._partitions_dict = None
		self._vertices = set()
		self._community_partite_label = 'Community'
		self._vertex_partite_label = 'Vertex'
		self._BPG = nx.Graph()

	##################################
	# Utility methods
	##################################

	def _filter_partitions(self, community_list):
		"""Filters in the wanted communities."""

		# filter partition dictionary according to input community list
		self._partitions_dict = {comm: self._unfiltered_partitions_dict[comm] for comm in community_list}

	def _setify_vertices(self):
		"""Creates a set of all vertices contained in the partitions."""

		for part in self._partitions_dict:
			self._vertices |= set(self._partitions_dict[part])

	def _set_partite_labels(self, community_partite_label, vertex_partite_label):
		"""Updates parts' labels attributes if given."""

		if community_partite_label is not None:
			self._community_partite_label = community_partite_label

		if community_partite_label is not None:
			self._vertex_partite_label = vertex_partite_label

	def print_properties(self, network: str = ''):
		print_bipartite_properties(BPG=self._BPG, network=network)

	##################################
	# Main methods
	##################################

	def create_bipartite_graph(
			self,
			community_list: list,
			community_partite_label=None,
			vertex_partite_label=None):

		"""
		Generates a BiPartite graph.

		Filters the partitions dictionary to contain only the given communities.
		Create a BiPartite network (graph) where one part is composed of community-representing
		vertices, and the other part is composed of the original vertices.
		Edges are created between vertices of the two parts if a "regular" vertex belongs to
		the community that corresponds to the community-representing vertex.

		Parameters
		----------
		community_list: A list of communities to be filtered in to create the BiPartite graph.
		community_partite_label: Optional; a string to label the vertices of the community part.
		vertex_partite_label: Optional; a string to label the vertices of the vertices part.

		Returns
		-------
		nx.Graph object containing the BiPartite graph.

		Examples
		--------
		The following will create a BiPartite graph called BPG,
		which contains 3 community-representing vertices - 'comm1', 'comm2', 'comm3',
		and all the vertices that belong to their corresponding communities in the partition dictionary input,
		and the parts will be named 'group' and 'user':

		>>> BPC = BiPartiteCreator(partitions_dict)
		>>> BPG = BPC.create_bipartite_graph(['comm1', 'comm2', 'comm3'], 'group', 'user')
		"""

		# Filter in the wanted communities
		self._filter_partitions(community_list)

		# Create a set of all vertices contained in the communities
		self._setify_vertices()

		# Update part labels attributes if given
		self._set_partite_labels(community_partite_label, vertex_partite_label)

		# Create community-representing vertices partite attribute (for nx.Graph)
		community_nodes = {comm: {'partite': self._community_partite_label} for comm in self._partitions_dict.keys()}

		# Create "regular" vertices nodes partite attribute (for nx.Graph)
		vertex_nodes = {vertex: {'partite': self._vertex_partite_label} for vertex in self._vertices}

		# Create BiPartite edges from partitions
		edges = []
		for comm, comm_vertices in self._partitions_dict.items():
			edges += [(comm, vertex) for vertex in comm_vertices]

		# Create BiPartie graph with vertices' attributes
		self._BPG.add_nodes_from(community_nodes.items())
		self._BPG.add_nodes_from(vertex_nodes.items())
		self._BPG.add_edges_from(edges)

		# Return a deep copy of the graph
		return deepcopy(self._BPG)

	def create_bipartite_edges_df(
			self,
			save_path: str = None,
			save_csv: bool = False):
		"""Creates a DataFrame of edges from the BiPartite graph.

		Parameters
		----------
		save_path: Optional; A string indicating a path to save a CSV file.
		save_csv: Optional; A boolean whether to save a CSV file (in save_path).

		Returns
		---------
		A DataFrame containing the edges of the BiPartite graph.

		Examples
		--------
		The following will create a BiPartite graph, and then create a DataFrame that contains its edges:

		>>> BPC = BiPartiteCreator(partitions_dict)
		>>> BPC.create_bipartite_graph(['comm1', 'comm2', 'comm3'], 'group', 'user')
		>>> BPG_edges_df = BPC.create_bipartite_edges_df()
		"""

		# Create a DataFrame of edge list from the BiPartite graph
		df = nx.convert_matrix.to_pandas_edgelist(
			G=self._BPG,
			source=self._community_partite_label,
			target=self._vertex_partite_label
		)

		# Save DataFrame
		if save_csv:
			df.to_csv(save_path)

		return df
