"""
TODO: add module docstring.
"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import os
import json
from datetime import datetime
import networkx as nx
import pandas as pd
from os.path import join
from os import listdir

# import AnomalousCommunityDetector and AnomalyInfusedCommunityStructuredRandomNetworkGenerator from parent directory
original_cur_dir = os.getcwd()
os.chdir('..')
#os.chdir('..')  # Comment this row if using jupyter notebook
from AnomalousCommunityDetection.AnomalousCommunityDetector import AnomalousCommunityDetector
from BaselineComparison.CommunityRanker import CommunityRanker

os.chdir(original_cur_dir)


##################################
# Experiment
##################################

class Experiment:
	def __init__(self, experiment_main_dir_path, detector_config, detection_config):
		"""
		A class that perform a single experiment

		Parameters
		----------
		experiment_main_dir_path: directory path of specific experiment.
		detector_config: a dict with detector configuration kwargs.
		detection_config: a dict with detectoion configuration kwargs.
		"""

		# Paths
		self._partition_maps_dir_path = join(experiment_main_dir_path, 'Data', 'PartitionMaps')
		self._networks_dir_path = join(experiment_main_dir_path, 'Data', 'Networks')
		self._results_logs_dir_path = join(experiment_main_dir_path, 'ResultsAndLogs')

		# Detector and Detection configuration dictionaries
		self._detector_config = detector_config
		self._detection_config = detection_config

	##################################
	# Utility Functions
	##################################

	def _partitions_map_file_paths_by_size_group(self, size_group):
		"""Returns full paths of corresponding train and test partitions maps (for our algorithm)."""

		size_group_dir_path = join(self._partition_maps_dir_path, size_group)
		file_names = listdir(size_group_dir_path)
		train_partitions_maps_file_paths = [join(size_group_dir_path, fn) for fn in file_names if 'train' in fn]
		test_partitions_maps_file_paths = [join(size_group_dir_path, fn) for fn in file_names if 'test' in fn]

		return train_partitions_maps_file_paths, test_partitions_maps_file_paths

	def _network_file_paths_by_size_group(self, size_group):
		"""Returns full paths of networks (for baseline's comparison)."""

		size_group_dir_path = join(self._networks_dir_path, size_group)
		file_names = os.listdir(size_group_dir_path)
		test_networks_file_paths = [join(size_group_dir_path, fn) for fn in file_names]

		return test_networks_file_paths

	@staticmethod
	def _read_partition_map(file_path, verbose: bool = False):
		if verbose:
			print(f'Reading file {file_path}')
		with open(file_path, 'r') as file:
			return json.load(file)

	@staticmethod
	def _read_edge_list(file_path, verbose: bool = False):
		if verbose:
			print(f'Reading file {file_path}')
		with open(file_path, 'rb') as file:
			G = nx.readwrite.adjlist.read_adjlist(file)
		return G

	@staticmethod
	def _log_single_network(
			train_partitions_map: dict, test_partitions_map: dict, num_anom_comms: int, enumeration: int):
		# TODO: Tbink of it
		train_comm_sizes = [len(comms) for comms in train_partitions_map.values()]
		test_comm_sizes = [len(comms) for comms in test_partitions_map.values()]

		norm_comm_sizes = train_comm_sizes + test_comm_sizes[:-num_anom_comms]
		anom_comm_sizes = test_comm_sizes[-num_anom_comms:]
		log_dict = {
			'enumeration': enumeration,
			'norm_comm_sizes': norm_comm_sizes,
			'anom_comm_sizes': anom_comm_sizes,
			'bipart_train_part': train_partitions_map,
			'bipart_test_part': test_partitions_map
		}
		return log_dict

	def _save_single_exp_results_and_log(
			self, our_results: pd.DataFrame, baseline_results: pd.DataFrame,
			log_dict: dict, size_group: str, enumeration: int):

		if our_results is not None:
			our_results_fp = join(self._results_logs_dir_path, size_group, f'OurResults_{str(enumeration):.2}.csv')
			our_results.to_csv(our_results_fp)

		if baseline_results is not None:
			# TODO: Remove this
			#basline_results_file_path = join(self._results_logs_dir_path, size_group, f'AMEN_AvgODF_BaseLineResults_{str(index):.2}.csv')
			#basline_results_file_path = join(self._results_logs_dir_path, size_group, f'AMEN_BaseLineResults_{str(index):.2}.csv')

			basline_results_fp = join(self._results_logs_dir_path, size_group, f'BaselineResults_{str(enumeration):.2}.csv')
			baseline_results.to_csv(basline_results_fp)

		# TODO: Remove comment, think of it
		"""
		"""
		network_log_file_path = join(self._results_logs_dir_path, size_group, f'NetworkLog_{str(enumeration):.2}.json')
		with open(network_log_file_path, 'w') as file:
			json.dump(log_dict, file)

	##################################
	# Main Experiment Functions
	##################################

	def _detect_anomalies(self, train_partitions_map, test_partitions_map):
		detector = AnomalousCommunityDetector(
			train_partitions_map=train_partitions_map,
			test_partitions_map=test_partitions_map,
			**self._detector_config
		)

		results = detector.detect_anomalous_communities(**self._detection_config)
		return results

	@staticmethod
	def _baseline_community_rank(G: nx.Graph, test_partitions_map: dict, rank_by: list, exclude_amen: bool):
		"""Ranks communities by baseline algorithms (including AMEN)"""

		community_ranker = CommunityRanker(G)
		ranking_scores = community_ranker.rank_communities_by_all_measures(
			partitions_map=test_partitions_map, rank_by=rank_by, exclude_amen=exclude_amen)
		return pd.DataFrame.from_dict(ranking_scores)

	def perform_single_experiment(
			self,
			size_group: str,
			train_partitions_maps_file_path: str,
			test_partitions_maps_file_path: str,
			test_network_file_path: str,
			num_anom_comms: int,
			rank_by: list,
			exclude_amen: bool,
			enumeration: int):
		"""
		Performs a single experiment - our algorithm, baseline algorithms, and saves results.

		:param size_group:
		:param train_partitions_maps_file_path:
		:param test_partitions_maps_file_path:
		:param test_network_file_path:
		:param num_anom_comms:
		:param rank_by:
		:param exclude_amen:
		:param enumeration:
		:return:
		"""

		# Load partition maps
		train_partitions_map = self._read_partition_map(train_partitions_maps_file_path)
		test_partitions_map = self._read_partition_map(test_partitions_maps_file_path)

		# Detect anomalies (Our algorithm)
		my_results = self._detect_anomalies(train_partitions_map, test_partitions_map)

		# Rank communities (Baseline algorithms)
		G = self._read_edge_list(test_network_file_path)  # load network
		basline_results = self._baseline_community_rank(
			G=G,
			test_partitions_map=test_partitions_map,
			rank_by=rank_by,
			exclude_amen=exclude_amen)

		# Log experiment
		log_dict = self._log_single_network(
			train_partitions_map=train_partitions_map,
			test_partitions_map=test_partitions_map,
			num_anom_comms=num_anom_comms,
			enumeration=enumeration)

		# Save results
		self._save_single_exp_results_and_log(
			our_results=my_results,
			baseline_results=basline_results,
			log_dict=log_dict,
			size_group=size_group,
			enumeration=enumeration)

	def perform_single_size_group_experiments(
			self, size_group: str, num_anom_comms: int, rank_by: list, exclude_amen: bool, max_experiments: int):
		"""
		Performs a single size group's all experiments.

		:param size_group:
		:param train_partitions_maps_file_path:
		:param test_partitions_maps_file_path:
		:param test_network_file_path:
		:param num_anom_comms:
		:param rank_by:
		:param exclude_amen:
		:param enumeration:
		:return:
		"""
		# Get all file paths in size group (of partitions maps and networks)
		train_maps_fps, test_maps_fps = self._partitions_map_file_paths_by_size_group(size_group=size_group)
		test_networks_fps = self._network_file_paths_by_size_group(size_group=size_group)

		begin_time = datetime.now()

		# Perform experiments for all files
		for idx, (train_map_fp, test_map_fp, network_fp) in enumerate(zip(train_maps_fps, test_maps_fps, test_networks_fps)):
			if idx == max_experiments:
				break

			self.perform_single_experiment(
				enumeration=idx,
				size_group=size_group,
				train_partitions_maps_file_path=train_map_fp,
				test_partitions_maps_file_path=test_map_fp,
				test_network_file_path=network_fp,
				num_anom_comms=num_anom_comms,
				rank_by=rank_by,
				exclude_amen=exclude_amen
			)
			print(f'Finished experiment {idx}. Elpased time: {datetime.now() - begin_time}')

	def perform_all_size_groups_experiments(
			self,
			num_anom_comms: int=10,
			rank_by: list = None, exclude_amen: bool = False,
			max_experiments: int=10):
		"""
		Performs a all size groups' experiments.

		:param size_group:
		:param train_partitions_maps_file_path:
		:param test_partitions_maps_file_path:
		:param test_network_file_path:
		:param num_anom_comms:
		:param rank_by:
		:param exclude_amen:
		:param enumeration:
		:return:
		"""
		for size_group in ['min', 'quartile1', 'median', 'mean', 'random']:
			print(f'\n########\n\tBeginning size group "{size_group}" experiment...\n########\n')
			self.perform_single_size_group_experiments(
				size_group=size_group,
				num_anom_comms=num_anom_comms,
				rank_by=rank_by,
				exclude_amen=exclude_amen,
				max_experiments=max_experiments)


def perform_all_experiments(
		experiment_dir: str,
		detector_config: dict,
		detection_config: dict,
		num_anom_comms:int = 10,
		baselines: list = None,
		exclude_amen: bool = False,
		max_experiments: int = 10,
		dir_idx_tuple: tuple = None):

	exp_dirs = [d for d in listdir(experiment_dir) if d != 'Raw_data']
	if dir_idx_tuple:
		exp_dirs = [exp_dirs[idx] for idx in range(dir_idx_tuple[0], dir_idx_tuple[1])]
		print(exp_dirs)
	#exp_dirs = exp_dirs[17:-1]  # Starting from 'ErdosRenyi_p0.20_k[1,1]_m0.20' and excluding RawData dir
	# to complete - 'ErdosRenyi_p0.20_k[1,1]_m0.10' random

	for exp_dir in exp_dirs:
		print(f'\n################\n################\n\tBeginning "{exp_dir}" experiments directory...\n################\n################\n')
		exp_dir_path = join(experiment_dir, exp_dir)
		exp = Experiment(
			experiment_main_dir_path=exp_dir_path, detector_config=detector_config, detection_config=detection_config)
		exp.perform_all_size_groups_experiments(
			num_anom_comms=num_anom_comms, rank_by=baselines, exclude_amen=exclude_amen, max_experiments=max_experiments)


def perform_specific_experiments(
		exp_dir_path: str, size_groups: list,
		detector_config: dict, detection_config: dict, num_anom_comms:int = 10,
		baselines: list=None, exclude_amen:bool=False, max_experiments: int=10):

	exp_dirs = listdir(experiment_dir)

	#exp_dirs = exp_dirs[4:-1]  # Starting from 'ErdosRenyi_p0.30_k[1,1]_m0.40' and excluding RawData dir
	#exp_dirs = exp_dirs[24:-1]  # Starting from 'ErdosRenyi_p0.25_k[1,1]_m0.80' and excluding RawData dir
	#exp_dirs = exp_dirs[29:-1]  # Starting from 'ErdosRenyi_p0.30_k[1,1]_m0.80' and excluding RawData dir
	#exp_dirs = exp_dirs[35:-1]  # Starting from 'ErdosRenyi_p0.40_k[1,1]_m0.05' and excluding RawData dir
	# to complete - second or third folder, and p0.30_k[1,1]_m0.40 random


	print(f'\n################\n################\n\tBeginning "{exp_dir_path}" experiments directory...\n################\n################\n')
	exp = Experiment(experiment_main_dir_path=exp_dir_path, detector_config=detector_config, detection_config=detection_config)
	for size_group in size_groups:
		exp.perform_size_group_experiment(
			size_group=size_group,
			num_anom_comms=num_anom_comms,
			rank_by=baselines,
			exclude_amen=exclude_amen,
			max_experiments=max_experiments)


if __name__ == '__main__':

	experiment_dir = EXPERIMENT_DIR
	detector_config = DETECTOR_CONFIG
	detection_config = DETECTION_CONFIG
	num_anom_comms = NUM_ANOM_COMMS
	baselines = ['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf']#, 'unattr_amen']
	#baselines = ['unattr_amen']

	exclude_amen = False

	"""
	"""
	perform_all_experiments(
		experiment_dir=experiment_dir,
		detector_config=detector_config,
		detection_config=detection_config,
		num_anom_comms=num_anom_comms,
		baselines=baselines,
		exclude_amen=exclude_amen,
		max_experiments=5)

	#exp_dir_path = 'Experiment\ErdosRenyi_p0.35_k[1,1]_m0.80'
	#size_groups = ['median', 'mean', 'random']

	#exp_dir_path = 'Experiment\ErdosRenyi_p0.30_k[1,1]_m0.40'
	#size_groups = ['random']

	#exp_dir_path = 'Experiment\ErdosRenyi_p0.05_k[1,1]_m0.20'
	#size_groups = ['random']

	"""
	#exp_dir_path = 'Experiment_different_inter_edges\\ErdosRenyi_p0.40_k[1,1]_m0.40'
	exp_dir_path = 'Experiment_different_inter_edges\\ErdosRenyi_p0.20_k[1,1]_m0.10'
	size_groups = ['random']
	perform_specific_experiments(
		exp_dir_path=exp_dir_path,
		size_groups=size_groups,
		detector_config=detector_config,
		detection_config=detection_config,
		num_anom_comms=num_anom_comms,
		baselines=baselines,
		exclude_amen=exclude_amen,
		max_experiments=7)
	"""




