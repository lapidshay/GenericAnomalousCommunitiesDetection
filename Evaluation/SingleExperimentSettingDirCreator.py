"""
TODO: add module docstring.
"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

from os import mkdir
from os.path import join, isdir


##################################
# Dir Creator
##################################

class SingleExperimentSettingDirCreator:
	# TODO: remove checkpoint
	def __init__(self, base_dir: str, network_generator_config: dict):

		self._base_dir = base_dir
		self._network_generator_config = network_generator_config

		# Create main directory name, composed of network generator configuration
		self._experiment_main_dir_name = self._create_experiment_main_dir_name()

		self._sub_dir_names = ['Data', 'ResultsAndLogs']  # 'Checkpoint'
		self._group_sizes = ['min', 'quartile1', 'median', 'mean', 'random']

	def _create_experiment_main_dir_name(self):
		"""Creates main directory name, composed of network generator configuration."""

		output = ''

		if self._network_generator_config['anom_comm_alg'].__name__ == 'gnp_random_graph':
			output = 'ErdosRenyi'

		elif self._network_generator_config['anom_comm_alg'].__name__ == 'barabasi_albert_graph':
			output = 'BarabasiAlbert'

		output += f'_p{self._network_generator_config["anom_inter_p"]:.2f}'
		output += f'_k[{self._network_generator_config["k_min"]},{self._network_generator_config["k_max"]}]'
		output += f'_m{self._network_generator_config["anom_m"]:.2f}'

		return join(self._base_dir, output)

	def create_directories(self):
		"""Create all directories (at all levels) needed for a single experiment, determined by configuration."""

		# Guard from overriding an already conducted experiment
		if isdir(self._experiment_main_dir_name):
			print('Watch out! directories already exist! Terminating program!')
			exit()

		# Create experiment main dir
		mkdir(self._experiment_main_dir_name)

		##################################
		# Create sub-dirs
		##################################

		[
			mkdir(join(self._experiment_main_dir_name, dir_name))
			for dir_name
			in self._sub_dir_names
		]

		##################################
		# Create sub-sub dirs
		##################################

		# Create sub-sub-dirs in Data sub-dir
		mkdir(join(self._experiment_main_dir_name, 'Data', 'PartitionMaps'))
		mkdir(join(self._experiment_main_dir_name, 'Data', 'Networks'))
		#mkdir(join(self._experiment_main_dir_name, 'Data', 'RawOverlappingPartitionMaps'))

		"""
		# Create sub-sub-dirs - group sizes in Checkpoint sub-dir
		[
			mkdir(join(self._experiment_main_dir_name, 'Checkpoint', group))
			for group
			in self._group_sizes
		]
		"""

		# Create sub-sub-dirs - group sizes in ResultsAndLogs sub-dir
		[
			mkdir(join(self._experiment_main_dir_name, 'ResultsAndLogs', group))
			for group
			in self._group_sizes
		]

		##################################
		# Create sub-sub-sub dirs
		##################################

		# Create sub-sub-sub-dirs - group sizes in Data/PartitionMaps sub-sub-dir
		[
			mkdir(join(self._experiment_main_dir_name, 'Data', 'PartitionMaps', group))
			for group
			in self._group_sizes
		]

		# Create sub-sub-sub-dirs - group sizes in Data/Networks sub-sub-dir
		[
			mkdir(join(self._experiment_main_dir_name, 'Data', 'Networks', group))
			for group
			in self._group_sizes
		]

	def add_single_group_size_sub_dir(self, group_size_name: str):
		"""Add single group size needed directories (at all levels)."""

		# Create sub-sub-dirs
		mkdir(join(self._experiment_main_dir_name, 'Checkpoint', group_size_name))
		mkdir(join(self._experiment_main_dir_name, 'ResultsAndLogs', group_size_name))

		# Create sub-sub-sub-dirs
		mkdir(join(self._experiment_main_dir_name, 'Data', 'PartitionMaps', group_size_name))
		mkdir(join(self._experiment_main_dir_name, 'Data', 'Networks', group_size_name))


def create_experiment_directories(base_dir, network_generator_config):
	for p in network_generator_config['anom_inter_p']:
		for m in network_generator_config['anom_m']:

			temp_dict = {k: v for k, v in network_generator_config.items()}
			temp_dict['anom_inter_p'] = p
			temp_dict['anom_m'] = m
			SingleExperimentSettingDirCreator(base_dir, temp_dict).create_directories()


if __name__ == '__main__':
	#TODO: Delete
	"""
	from ExperimentSettings import EXPERIMENT_SETTINGS
	base_dir = 'ExperimentBarabasi'
	for p in EXPERIMENT_SETTINGS['anom_inter_p']:
		for m in EXPERIMENT_SETTINGS['anom_m']:
			temp_dict = {k: v for k, v in EXPERIMENT_SETTINGS.items()}
			temp_dict['anom_inter_p'] = p
			temp_dict['anom_m'] = m
			SingleExperimentSettingDirCreator(base_dir, temp_dict).create_directories()
			#DirCreator(base_dir, temp_dict).add_group_size_dir('quartile1')
	"""
	pass
