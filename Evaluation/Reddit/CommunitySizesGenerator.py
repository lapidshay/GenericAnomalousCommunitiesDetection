__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import numpy as np


##################################
# CommunitySizesGenerator
##################################

class CommunitySizesGenerator:
	def __init__(self, partitions_map: dict):
		"""
		A Class for generating community sizes for a generated network experiment.

		Parameters
		----------
		partitions_map: a partition map to base community sizes upon.
		"""

		self._partitions_sizes = sorted([len(val) for val in partitions_map.values()])
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

	def generate_single_community_sizes(
			self,
			size_group: str, rng: int = 30, num_samples: int = 10,
			random_seed: int = None, verbose: bool = False):
		"""
		Generates a sorted list of single size group's communities' sizes.

		Parameters
		----------
		size_group: a keyword, to determine sizes. should be either of ['min', 'quantile10', 'quartile1', 'median', 'random'].
		rng: range of sizes to create around the given measure.
		num_samples: number of samples to create.
		random_seed: ...
		verbose: print.
		"""

		# Set random seed if given
		if random_seed is not None:
			self._set_random_seed(random_seed)

		if size_group == 'random':
			# Sample random community sizes
			output = sorted(np.random.choice(self._partitions_sizes, num_samples, replace=True).tolist())
			if verbose:
				self._print(output, size_group)
			return output

		if size_group == 'min':
			val_1 = np.min(self._partitions_sizes)
			val_2 = val_1 + rng + 1

		elif size_group == 'median':
			val_1 = np.median(self._partitions_sizes) - int(rng/2)
			val_2 = val_1 + rng + 1

		elif size_group == 'quantile10':
			val_1 = np.quantile(self._partitions_sizes, q=0.1) - int(rng/2)
			val_2 = val_1 + rng + 1

		elif size_group == 'quartile1':
			val_1 = np.percentile(self._partitions_sizes, q=25) - int(rng/2)
			val_2 = val_1 + rng + 1

		# Assertion
		else:
			raise TypeError(f"Expected 'size_group' argument to be one of {str(self._size_groups)}.")

		output = sorted(np.random.randint(val_1, val_2, size=num_samples).tolist())
		if verbose:
			self._print(output, size_group)
		return output
