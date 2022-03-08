__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import numpy as np
import os
import json
from tqdm.autonotebook import tqdm
import itertools
import networkx as nx
from ExperimentSettings import EXPERIMENT_SETTINGS
# import AnomalousCommunityDetector and AnomalyInfusedCommunityStructuredRandomNetworkGenerator from parent directory
original_cur_dir = os.getcwd()
os.chdir('..')
from SingleExperimentSettingDirCreator import SingleExperimentSettingDirCreator
os.chdir('..')
from AnomalyInfusedCommunityStructuredRandomNetworkGenerator.AnomalyInfusedCommunityStructuredRandomNetworkGenerator \
	import AnomalyInfusedCommunityStructuredRandomNetworkGenerator
os.chdir(original_cur_dir)


##################################
# Anomalous Community Sizes Generator
##################################

class AnomalousCommunitySizesGenerator:
	def __init__(self, community_sizes_list: list):
		self._community_sizes_list = community_sizes_list
		self._num_samples = 10
		self._size_groups = ['min', 'quantile10', 'quartile1', 'median', 'random']

	@staticmethod
	def _print(sizes, size_group):
		print(f'Generated community sizes sample by "{size_group}":')
		print(f'\tSizes: {sizes}')
		print(f'\tNum. sizes: {len(sizes)}')
		print(f'\tMin size: {np.min(sizes)}')
		print(f'\tMax size: {np.max(sizes)}')
		print(f'\tMean size: {np.mean(sizes):.3f}')
		print(f'\tMedian size: {np.median(sizes):.3f}')
		print(f'\tSTDV: {np.std(sizes):.3f}')

	@staticmethod
	def _set_random_seed(random_seed):
		np.random.seed(random_seed)

	def generate_community_sizes(
			self,
			size_group: str, rng: int,
			random_seed: int, num_samples: int, verbose: bool):

		if num_samples is not None:
			self._num_samples = num_samples

		if random_seed is not None:
			self._set_random_seed(random_seed)

		if size_group not in self._size_groups:
			raise TypeError(f"Expected 'size_group' argument to be one of {str(self._size_groups)}.")

		if size_group == 'random':
			# Sample random community sizes
			output = sorted(np.random.choice(self._community_sizes_list, self._num_samples, replace=True).tolist())
			if verbose:
				self._print(output, size_group)
			return output

		if size_group == 'min':
			val_1 = np.min(self._community_sizes_list)
			val_2 = val_1 + rng + 1

		elif size_group == 'median':
			val_1 = np.median(self._community_sizes_list) - int(rng / 2)
			val_2 = val_1 + rng + 1

		elif size_group == 'mean':
			val_1 = np.mean(self._community_sizes_list) - int(rng / 2)
			val_2 = val_1 + rng + 1

		elif size_group == 'quantile10':
			val_1 = np.quantile(self._community_sizes_list, q=0.1) - int(rng / 2)
			val_2 = val_1 + rng + 1

		elif size_group == 'quartile1':
			val_1 = np.percentile(self._community_sizes_list, q=25) - int(rng / 2)
			val_2 = val_1 + rng + 1

		output = sorted(np.random.randint(val_1, val_2, size=self._num_samples).tolist())
		if verbose:
			self._print(output, size_group)
		return output


##################################
# Generated Network AnomalyInfuser
##################################

class GenNetAnomalyInfuser:
	def __init__(
			self,
			raw_comm_sizes_dir_path: str,
			output_dir_path: str,
			train_test_split_num: int,
			network_generator_config: dict):

		# Paths
		self._raw_comm_sizes_file_paths = self._create_full_file_paths(raw_comm_sizes_dir_path)
		self._output_dir_path = output_dir_path
		self._partition_maps_output_dir = os.path.join(self._output_dir_path, 'PartitionMaps')
		self._networks_output_dir = os.path.join(self._output_dir_path, 'Networks')

		# Network generator configuration
		self._network_generator_config = network_generator_config
		self._log_configuration()  # Create a json log file of the configuration

		# Configuration for Network Generator generate_network() method
		self._network_generation_config = dict()
		self._network_generation_config['norm_m'] = self._network_generator_config.pop('norm_m')
		self._network_generation_config['norm_inter_p'] = self._network_generator_config.pop('norm_inter_p')
		self._network_generation_config['anom_m'] = self._network_generator_config.pop('anom_m')
		self._network_generation_config['anom_inter_p'] = self._network_generator_config.pop('anom_inter_p')

		# Train / test split configuration
		self._train_test_split_num = train_test_split_num

	##################################
	# Utility methods
	##################################

	def _log_configuration(self):
		file_path = os.path.join(self._output_dir_path, 'Configuration_Log.json')
		net_gen_config_log = {k: v for k, v in self._network_generator_config.items() if k != 'norm_comm_alg'}
		net_gen_config_log['anom_comm_alg'] = net_gen_config_log[
			'anom_comm_alg'].__name__  # random graph generator name
		with open(file_path, 'w', encoding='UTF8') as file:
			json.dump(net_gen_config_log, file)

	@staticmethod
	def _set_random_seed(random_seed):
		np.random.seed(random_seed)

	@staticmethod
	def _create_full_file_paths(raw_partitions_map_dir_path):
		file_names = os.listdir(raw_partitions_map_dir_path)
		return [os.path.join(raw_partitions_map_dir_path, fn) for fn in file_names]

	@staticmethod
	def _read_raw_comm_sizes_list(file_path):
		with open(file_path, 'r') as file:
			comm_sizes_list = json.load(file)
		return comm_sizes_list

	def _partitions_train_test_split(self, partitions_map: dict, num_anom_comms: int, random_seed: int = None):
		"""Returns 2 dictionaries of partitions maps for train and test sets"""

		if random_seed is not None:
			self._set_random_seed(random_seed)

		# Create a list of normal communities to sample from
		normal_partition_names = list(partitions_map.keys())[:-num_anom_comms]

		# Resample normal communities' names
		train_set_comms_names = np.random.choice(
			normal_partition_names, size=self._train_test_split_num, replace=False).tolist()

		# Create train set partitions map
		train_partitions_map = {k: v for k, v in partitions_map.items() if k in train_set_comms_names}

		# Create test set communities from the rest of normal communities and the anomalous communities
		test_partitions_map = {k: v for k, v in partitions_map.items() if k not in train_set_comms_names}

		return train_partitions_map, test_partitions_map

	@staticmethod
	def _partitions_map_sub_network(G, partitions_map):
		"""Returns sub network of given network and partitions map."""

		# get partitions map vertices and edges
		partitions_map_vertices = set(itertools.chain(*partitions_map.values()))
		partitions_map_vertices = {int(i) for i in partitions_map_vertices}
		partitions_map_edges = nx.subgraph(G, partitions_map_vertices).edges()

		# Create sub network (returning a nx.subgraph will return a view of the original graph)
		sub_betwork = nx.Graph()
		sub_betwork.add_nodes_from(partitions_map_vertices)
		sub_betwork.add_edges_from(partitions_map_edges)

		return sub_betwork

	##################################
	# Output utility methods
	##################################

	def save_partition_map(
			self, partitions_map: dict,
			anom_comm_size_group: str,
			raw_file_path: str,
			train_or_test: str):

		file_name = os.path.split(raw_file_path)[-1][:-5]  # last in split, without '.json' extension
		file_path = os.path.join(
			self._partition_maps_output_dir, anom_comm_size_group, f'{file_name}_{train_or_test}.json')
		with open(file_path, 'w', encoding='UTF8') as file:
			json.dump(partitions_map, file)

	def save_network(self, network: nx.Graph, anom_comm_size_group: str, raw_file_path: str, train_or_test: str):
		file_name = os.path.split(raw_file_path)[-1][:-5]  # last in split, without '.json' extension
		file_path = os.path.join(
			self._networks_output_dir, anom_comm_size_group, f'{file_name}_{train_or_test}_network.adjlist')
		with open(file_path, 'w', encoding='UTF8'):
			nx.readwrite.adjlist.write_adjlist(network, file_path)

	##################################
	# Main methods
	##################################

	def create_single_network(
			self,
			comm_sizes_file_path: str,
			anom_comm_size_group: str,
			rng: int,
			random_seed:
			int,
			num_anom_comms: int = 10,
			verbose: bool = False):
		"""
		Creates an anomalous-communities-infused network and returns train and test partitions maps and test sub-network.

		Works as follows:
			Creates a (real-world) network based on Reddit comments dataset, according to partitions map.
			Creates anomalous communities' sizes.
			Creates and infuse anomalous communities according to sizes.
			Creates train and test sets partitions map.
			Creates a sub-network induced from test set partitions map.
		"""

		# Fetch normal communities sizes
		normal_communties_sizes = self._read_raw_comm_sizes_list(comm_sizes_file_path)

		# Create anomalous communities sizes
		anom_comm_size_generator = AnomalousCommunitySizesGenerator(normal_communties_sizes)
		anomalous_communties_sizes = anom_comm_size_generator.generate_community_sizes(
			size_group=anom_comm_size_group,
			rng=rng,
			random_seed=random_seed,
			num_samples=num_anom_comms,
			verbose=verbose)

		# Create anomaly-infused network
		network_generator = AnomalyInfusedCommunityStructuredRandomNetworkGenerator(**self._network_generator_config)
		G = network_generator.generate_network(
			norm_comm_sizes=normal_communties_sizes,
			anom_comm_sizes=anomalous_communties_sizes,
			**self._network_generation_config)

		# Get partitions map (containing also infused anomalies)
		partitions_map = network_generator.get_partitions()

		# Split to train and test partitions maps
		train_partitions_map, test_partitions_map = self._partitions_train_test_split(
			partitions_map=partitions_map,
			num_anom_comms=num_anom_comms,
			random_seed=random_seed)

		# partitions_map_vertices = set(itertools.chain(*test_partitions_map.values()))

		# Create sub network induced from test partitions map vertices
		test_sub_network = self._partitions_map_sub_network(G=G, partitions_map=test_partitions_map)

		return train_partitions_map, test_partitions_map, test_sub_network

	def create_networks_with_anom_comm_size_group(
			self,
			anom_comm_size_group: str,
			rng: int,
			num_anom_comms: int = 10,
			verbose:
			bool = False):

		for rand_seed, fp in tqdm(enumerate(self._raw_comm_sizes_file_paths)):
			train_partitions_map, test_partitions_map, test_sub_network = self.create_single_network(
				comm_sizes_file_path=fp,
				anom_comm_size_group=anom_comm_size_group, rng=rng, random_seed=rand_seed,
				num_anom_comms=num_anom_comms,
				verbose=verbose
			)

			self.save_partition_map(
				partitions_map=train_partitions_map,
				anom_comm_size_group=anom_comm_size_group, raw_file_path=fp, train_or_test='train')
			self.save_partition_map(
				partitions_map=test_partitions_map,
				anom_comm_size_group=anom_comm_size_group, raw_file_path=fp, train_or_test='test')
			self.save_network(
				network=test_sub_network,
				anom_comm_size_group=anom_comm_size_group, raw_file_path=fp, train_or_test='test')


def create_experiment_networks(
		base_dir: str,
		raw_comm_sizes_dir_path: str,
		train_test_split_num: int,
		experiment_settings: dict,
		num_anom_comms=10,
		verbose=False):
	for p in experiment_settings['anom_inter_p']:
		for m in experiment_settings['anom_m']:

			# Create current network generator configuration
			cur_network_generator_config = {k: v for k, v in EXPERIMENT_SETTINGS.items()}
			cur_network_generator_config['anom_inter_p'] = p
			cur_network_generator_config['anom_m'] = m

			# Create paths
			current_experiment_dir = SingleExperimentSettingDirCreator(
				base_dir, cur_network_generator_config,
				['min', 'quantile10', 'quartile1', 'median', 'random'])._experiment_main_dir_name
			output_dir_path = os.path.join(current_experiment_dir, 'Data')

			# Create Anomaly infuser
			anomaly_infuser = GenNetAnomalyInfuser(
				raw_comm_sizes_dir_path=raw_comm_sizes_dir_path,
				output_dir_path=output_dir_path,
				train_test_split_num=train_test_split_num,
				network_generator_config=cur_network_generator_config)

			# Infuse anomalies of different group sizes and ranges
			for group, rng in zip(
					['min', 'quantile10', 'quartile1', 'median', 'random'],
					[5, 8, 16, 24, 30, 0]):

				anomaly_infuser.create_networks_with_anom_comm_size_group(
					anom_comm_size_group=group, rng=rng, num_anom_comms=num_anom_comms, verbose=verbose)

			print(f'Finished infusing anomalies to p={p}, m={m}')
