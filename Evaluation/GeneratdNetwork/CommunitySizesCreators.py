__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import pandas as pd
import os
import numpy as np
import json
from tqdm.autonotebook import tqdm
from os.path import join


##################################
# Reddit Community Sizes Fetcher
##################################

class RedditCommunitySizesFetcher:

	def __init__(self, dir_path):
		"""
		A Class for generating community sizes for a generated network experiment based on Reddit network.

		Parameters
		----------
		dir_path: a path to Reddit netowrk directories.
		"""

		self._dir_path = dir_path
		self._comm_sizes = {}

	def _fetch_subreddit_num_users(self, subreddit_name: str, use_edges_csv: bool):
		"""
		Counts and returns unique users ios a single subreddit.


		Parameters
		----------
		subreddit_name: subreddit name.
		use_edges_csv: a boolean to determine wheter to use edges ot vertices file.
		"""

		# Determine whether to use edges or vertices file
		typ = 'edges' if use_edges_csv else 'vertices'

		# Create a full path of single subreddit edges / vertices csv file
		edges_full_path = join(self._dir_path, subreddit_name, f'{subreddit_name}.{typ}.csv')

		try:
			df = pd.read_csv(edges_full_path)

		except FileNotFoundError:
			print(f'Could not open {subreddit_name}.{typ}.csv... Skipped')
			return None

		# Count unique users in subreddit
		if use_edges_csv:
			num_users = len(set(df['__src_id']).union(set(df['__dst_id'])))
		else:
			num_users = len(df['__id'])

		# free memory
		del df

		return num_users

	def create_comm_size_dict(self, use_edges_csv: bool, subreddits_to_exclude: list = None):
		"""
		Creates and returns a dictionary of form {subreddit_name: num_users}.

		Parameters
		----------
		use_edges_csv: a boolean to determine wheter to use edges ot vertices file.
		subreddits_to_exclude: a lst of subreddits to exclude.
		"""

		list_dir = list(os.listdir(self._dir_path))

		for subreddit_name in tqdm(list_dir):
			# Excluding subreddits_to_exclude, if given
			if subreddits_to_exclude and subreddit_name in subreddits_to_exclude:
				continue

			# Fetch subreddit's num. users and populate dict
			self._comm_sizes[subreddit_name] = self._fetch_subreddit_num_users(
				subreddit_name=subreddit_name, use_edges_csv=use_edges_csv)

		return self._comm_sizes

	def save_comm_size_dict_json(self, path):
		"""
		Saves the community sizes map in given path.

		Parameters
		----------
		path: a path to sace the dictionary.
		"""

		with open(path, 'w') as file:
			json.dump(self._comm_sizes, file)


class NormalCommunitySizesGenerator:
	def __init__(self, comm_sizes_file_path: str):
		"""
		A Class for generating sample of normal community sizes.

		Parameters
		----------
		comm_sizes_file_path: a path to a dictionary of form {subreddit_name: num_users}.
		"""

		self._comm_sizes_file_path = comm_sizes_file_path
		self._partitions = dict()

	def generate_community_sizes_from_reddit(
			self,
			num_comms: int, min_comm_size: int, max_comm_size: int,
			random_seed: int = None):
		"""
		Creates a list of community sizes, sampled (with replace) from real Reddit community sizes.


		Parameters
		----------
		num_comms: number of normal communities to sample.
		min_comm_size: minimal size of communities to consider.
		max_comm_size: maximal size of communities to consider.
		random_seed:
		"""
		if random_seed is not None:
			self._set_random_seed(random_seed)

		# Read community: sizes dictionary created from Subreddit
		with open(self._comm_sizes_file_path, 'r', encoding='utf8') as file:
			subreddits_sizes_dict = json.load(file)

		# Create a list of sizes, truncated by min and max values
		sorted_truncated_sizes = sorted(
			[
				size for size
				in subreddits_sizes_dict.values()
				if (size is not None) and (min_comm_size <= size <= max_comm_size)
			])

		# Sample community sizes
		output = np.random.choice(sorted_truncated_sizes, num_comms, replace=True)

		# Sort
		output = np.sort(output)

		# Convert to list
		output = output.tolist()

		return output

	##################################
	# Utility methods
	##################################

	@staticmethod
	def _set_random_seed(random_seed):
		np.random.seed(random_seed)
