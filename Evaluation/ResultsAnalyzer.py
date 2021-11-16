"""
TODO: add module docstring.

"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import pandas as pd
import os
import json
import numpy as np
from os.path import join
from os import listdir
from PrecisionRecallEvaluator import PrecisionRecallEvaluator as Evaluator
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


##################################
# Constants
##################################

matplotlib.rcParams['legend.loc'] = 'lower right'
matplotlib.rcParams['legend.fontsize'] = 9

_ALL_SIZE_GROUPS = ['min', 'quartile1', 'median', 'mean', 'random']
_ALL_PS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
_ALL_MS = [0.05, 0.1, 0.2, 0.4, 0.8]

_META_FEATURES = [
	'normality_prob_mean',
	'normality_prob_std',
	'normality_prob_median',
	'predicted_label_mean',
	'predicted_label_std',
	'weighted_sum'
]

_BASELINES = ['avg_degree', 'cut_ratio', 'conductance', 'flake_odf', 'avg_odf', 'unattr_amen']


##################################
# Results Analyzer
##################################

class ResultsAnalyzer:
	def __init__(
			self,
			experiment_dir_path: str,
			experiment_settings: dict,
			anomalous_comm_names: list):

		self._column_pairs = self._create_column_pairs(meta_features=_META_FEATURES, baselines=_BASELINES)

		# Paths
		self._ordered_dir_paths = self._order_dir_paths(experiment_dir_path, experiment_settings)

		self._anomalous_comm_names = anomalous_comm_names

	########################################
	########################################
	# Utility methods
	########################################
	########################################

	@staticmethod
	def _create_column_pairs(meta_features, baselines):
		"""Creates a dictionary with fitting columns names tuples of form (algo__ranking, algo__score)."""
		output = {
			'OurResults': [[f'{r}__ranking', f'{r}__score'] for r in meta_features],
			'BaselineResults': [[f'{r}__ranking', f'{r}__score'] for r in baselines]

		}
		return output

	@staticmethod
	def _order_dir_paths(experiment_dir_path, experiment_settings):
		"""Creates a dictionary with all paths, of form {anom_inter_p: {size_group: {anom_m: path}}}."""

		inter_ps = experiment_settings['anom_inter_p']
		anom_ms = experiment_settings['anom_m']

		sub_dirs_names = listdir(experiment_dir_path)
		result_and_logs_dir_paths = [
			join(experiment_dir_path, sub_dir_name, 'ResultsAndLogs')
			for sub_dir_name
			in sub_dirs_names
		]

		output = dict()
		for p in inter_ps:
			output[f'p{p:.2f}'] = dict()
			cur_p_sub_dir_names = [sdpth for sdpth in result_and_logs_dir_paths if f'_p{p:.2f}' in sdpth]
			for size_group in ['min', 'quartile1', 'median', 'mean', 'random']:  # Excluded 'max', 'prec75'
				output[f'p{p:.2f}'][size_group] = dict()
				for m, res_log_path in zip(anom_ms, cur_p_sub_dir_names):
					output[f'p{p:.2f}'][size_group][f'm{m:.2f}'] = join(res_log_path, size_group)

		return output

	########################################
	########################################
	# Single Experiment Log
	########################################
	########################################

	# TODO: Think if needed
	def create_single_experiment_log(self, dir_path):
		"""Creates single experiment log DataFrame (constant inter_p and anom_m, over several networks)."""

		raw_log_dict = self._load_log_files(dir_path=dir_path)
		log_dict_for_df = self._create_log_dict_for_df(raw_log_dict)
		log_df = pd.DataFrame.from_dict(log_dict_for_df, orient='index')

		return log_df

	@staticmethod
	def _find_log_file_paths(dir_path: str):
		return [join(dir_path, fn) for fn in listdir(dir_path) if 'NetworkLog' in fn]

	def _load_log_files(self, dir_path: str):
		log_file_paths = self._find_log_file_paths(dir_path)
		raw_log_dict = {}

		for log_file_path in log_file_paths:
			log_file_name = os.path.split(log_file_path)[-1].strip('.json')
			print(log_file_name)
			with open(log_file_path, 'r') as file:
				raw_log_dict[f'{log_file_name}'] = json.load(file)
		return raw_log_dict

	def _create_log_dict_for_df(self, raw_log_dict):
		log_dict_for_df = {}
		for idx, content in raw_log_dict.items():
			log_dict_for_df[idx] = {}
			log_dict_for_df[idx].update(self._describe('norm_comm', content['norm_comm_sizes']))
			log_dict_for_df[idx].update(self._describe('anom_comm', content['anom_comm_sizes']))
			log_dict_for_df[idx].update(
				{k: v for k, v in content.items() if k in ['num_nodes', 'num_edges', 'avg_degree']})
		return log_dict_for_df

	@staticmethod
	def _describe(name, comm_sizes):
		output = dict()
		output[f'{name}_size'] = len(comm_sizes)
		output[f'{name}_sum'] = np.sum(comm_sizes)
		output[f'{name}_min'] = np.min(comm_sizes)
		output[f'{name}_max'] = np.max(comm_sizes)
		output[f'{name}_mean'] = np.mean(comm_sizes)
		output[f'{name}_median'] = np.median(comm_sizes)
		output[f'{name}_perc75'] = np.percentile(comm_sizes, q=75)
		return output

	########################################
	########################################
	# Single Experiment Results Analysis
	########################################
	########################################

	def create_single_experiment_results_analysis_df(self, dir_path, algo: str, reverse: bool = False):
		"""
		Creates a single experiment results analysis DataFrame.

		A single experiment means a set of anom_inter_p, anom_m and size_group values, and several networks.
		"""
		results_file_paths = self._find_results_file_paths(dir_path=dir_path, algo=algo)

		# Determine correct columns pairs ('MyResults' or 'BaseLineResults')
		column_pairs = self._column_pairs[algo]

		res_dict_for_df = {}
		for file_path in results_file_paths:
			results_file_name = os.path.split(file_path)[1].strip('.csv')
			df = pd.read_csv(file_path, index_col=0)

			res_dict_for_df[results_file_name] = {}

			for (ranking_col, score_col) in column_pairs:

				# Reversing AMEN and avg. degree results due their different behavior
				if score_col in ['unattr_amen__score', 'avg_degree__score']:
					col_reverse = not reverse

				else:
					col_reverse = reverse

				evaluator = Evaluator(
					ranking_and_score_df=df,
					ranking_col=ranking_col,
					score_col=score_col,
					anom_comm_names=self._anomalous_comm_names)
				col_name = score_col.replace('__score', '')
				res_dict_for_df[results_file_name][f'{col_name}'] = evaluator.get_avg_precision(reverse=col_reverse)

		results_analysis_df = pd.DataFrame.from_dict(res_dict_for_df, orient='index')
		return results_analysis_df

	@staticmethod
	def _find_results_file_paths(dir_path: str, algo: str):
		"""Returns a list of all file paths of files of given type in a given directory."""

		return [join(dir_path, fn) for fn in listdir(dir_path) if fn.startswith(algo)]

	@staticmethod
	def get_single_experiment_mean_average_precision_dict(results_analysis_df):
		"""Returns Mean Average Precision of each meta-feature over several experiments."""

		output = results_analysis_df.describe().loc[['mean', 'std']].to_dict()
		return output

	########################################
	########################################
	# Meta-Features comparison
	########################################
	########################################

	def compare_meta_features_single_p_diff_size_groups(self, p, size_groups):
		"""
		Creates a dictionary of mean average precision scores.

		Dictionary of form {group_size: {meta_feature: mean avg. precision, ...}, ...}
		for each of the input group sizes, for each of the meta-features, and for a single anom_inter_p.
		"""
		output = dict()

		for size_group in size_groups:
			output[size_group] = {}
			# Find corresponding single experiment (p, size_group, anom_m) path
			exp_dir_path = self._ordered_dir_paths[f'p{p:.2f}'][size_group]['m0.80']  # Representative anom_m

			# Create results analysis DataFrame
			results_analysis_df = self.create_single_experiment_results_analysis_df(
				dir_path=exp_dir_path, algo='OurResults')

			# Get meta-features' mean average precision dictionary
			mean_avg_prec_dict = self.get_single_experiment_mean_average_precision_dict(results_analysis_df)

			# Extract only the mean average precision values (and not the stdv)
			for meta_feature, avg_prec in mean_avg_prec_dict.items():
				output[size_group][meta_feature] = avg_prec['mean']

		return output

	@staticmethod
	def plot_meta_features_comparison_single_p_diff_size_groups(comp_df, p, skip_median: bool):

		# TODO: solve this
		if skip_median:
			comp_df = comp_df.drop('normality_prob_median', axis=1)
			comp_df = comp_df.drop('weighted_sum', axis=1)

		fig, axes = plt.subplots(1, 2, figsize=(17, 6), tight_layout=True)
		sns.lineplot(data=comp_df.T, ax=axes[0], dashes=False)
		sns.lineplot(data=comp_df, ax=axes[1], dashes=False)

		axes[0].set_xlabel('Meta-Features')
		axes[0].set_ylabel('Average Precision (AP)')
		axes[0].set_title(f'AP vs. meta-features for different size groups', fontsize=14)
		#axes[0].set_ylim(0, 1)
		xticks_labels = [col.replace('_', '\n', 1).replace('_', ' ') for col in comp_df.columns]
		axes[0].set_xticklabels(xticks_labels)

		axes[1].set_xlabel('Anomalous Community Sizes')
		axes[1].set_ylabel('Average Precision (AP)')
		axes[1].set_title(f'AP vs. size groups for different meta-features', fontsize=14)
		#axes[1].set_ylim(0, 1)

		fig.suptitle(f"inter_p = {p}", fontsize=18)
		plt.show()

	def plot_meta_features_comparison_grouped_by_ps_size_groups(self, skip_median: bool):
		for p in _ALL_PS:
			map_scores_dict = self.compare_meta_features_single_p_diff_size_groups(p=p, size_groups=_ALL_SIZE_GROUPS)
			map_scores_df = pd.DataFrame.from_dict(map_scores_dict, orient='index')
			self.plot_meta_features_comparison_single_p_diff_size_groups(comp_df=map_scores_df, p=p, skip_median=skip_median)

	def get_meta_features_comparison_df(self):
		temp = {
			p: self.compare_meta_features_single_p_diff_size_groups(p=p, size_groups=_ALL_SIZE_GROUPS)
			for p
			in _ALL_PS
		}
		temp = {
			p: pd.DataFrame.from_dict(comparison_dict, orient='index').mean()
			for p, comparison_dict
			in temp.items()
		}
		output = pd.DataFrame.from_dict(temp, orient='index')
		output.index.rename('anom_inter_p', inplace=True)
		return output

	########################################
	########################################
	# Evaluation of our algorithm (single meta-feature)
	########################################
	########################################

	# Single inter_p - different ms and size groups
	# /////////////////////////////////////////////

	def compare_single_p_diff_ms_and_size_groups(self, meta_feature, p, size_groups):
		"""
		Creates a dictionary of mean average precision scores.

		Dictionary of form {group_size: {anom_m: mean avg. precision, ...}, ...}
		for each of the input group sizes, for each of the anom_m values,
		for a single anom_inter_p, and for a single meta-feature.
		"""

		output = {}

		for size_group in size_groups:
			output[size_group] = {}
			# For each corresponding single experiment (p, size_group, anom_m)
			for m, exp_path in self._ordered_dir_paths[f'p{p:.2f}'][size_group].items():

				# Create results analysis DataFrame
				results_analysis_df = self.create_single_experiment_results_analysis_df(exp_path, algo='OurResults')

				# Get all meta-features' mean average precision dictionary
				temp = self.get_single_experiment_mean_average_precision_dict(results_analysis_df)

				# Extract input meta-feature's mean average precision score
				output[size_group][m] = temp[meta_feature]['mean']

		return output

	@staticmethod
	def plot_single_p_diff_ms_and_size_groups(comp_df, p):
		fig, axes = plt.subplots(1, 2, figsize=(17, 6), tight_layout=True)
		sns.lineplot(data=comp_df.T, ax=axes[0], dashes=False)
		sns.lineplot(data=comp_df, ax=axes[1], dashes=False)
		axes[0].set_xlabel('anom_m')
		axes[0].set_ylabel('Average Precision')
		axes[0].set_title(f'AP vs. anom_m for different size groups', fontsize=14)
		axes[0].set_ylim(0, 1)

		axes[1].set_xlabel('Anomalous Community Sizes')
		axes[1].set_ylabel('Average Precision')
		axes[1].set_title(f'AP vs. size groups for different anom_m values', fontsize=14)
		axes[1].set_ylim(0, 1)
		fig.suptitle(f"inter_p = {p}", fontsize=18)
		plt.show()

	def plot_grouped_by_ps_ms_and_size_groups(self, meta_feature: str):
		for p in _ALL_PS:
			comparison_dict = self.compare_single_p_diff_ms_and_size_groups(
					meta_feature=meta_feature, p=p, size_groups=_ALL_SIZE_GROUPS)
			comparison_df = pd.DataFrame.from_dict(comparison_dict, orient='index')
			self.plot_single_p_diff_ms_and_size_groups(comparison_df, p)

	def get_ps_comparison_df(self, meta_feature):
		temp = {
			p: self.compare_single_p_diff_ms_and_size_groups(meta_feature=meta_feature, p=p, size_groups=_ALL_SIZE_GROUPS)
			for p
			in _ALL_PS
		}
		temp = {
			p: pd.DataFrame.from_dict(comparison_dict, orient='columns').mean()
			for p, comparison_dict
			in temp.items()
		}
		output = pd.DataFrame.from_dict(temp, orient='columns')
		output.index.rename('size_group', inplace=True)
		return output

	# Single m - different inter_ps and size groups
	# /////////////////////////////////////////////

	def compare_single_m_diff_ps_and_size_groups(self, meta_feature, m, ps, size_groups):
		"""
		Creates a dictionary of mean average precision scores.

		Dictionary of form {anom_inter_p: {group_size: mean avg. precision, ...}, ...}
		for each of the input anom_inter_p values, for each of the group sizes,
		for a single anom_m, and for a single meta-feature.
		"""

		output = {}

		for p in ps:
			output[f'p{p:.2f}'] = {}
			for size_group in size_groups:

				# Find corresponding single experiment (p, size_group, anom_m)
				exp_path = self._ordered_dir_paths[f'p{p:.2f}'][size_group][f'm{m:.2f}']

				# Create results analysis DataFrame
				results_analysis_df = self.create_single_experiment_results_analysis_df(exp_path, algo='OurResults')

				# Get all meta-features' mean average precision dictionary
				temp = self.get_single_experiment_mean_average_precision_dict(results_analysis_df)

				# Extract input meta-feature's mean average precision score
				output[f'p{p:.2f}'][size_group] = temp[meta_feature]['mean']

		return output

	@staticmethod
	def plot_single_m_diff_ps_and_size_groups(comp_df, m):
		fig, axes = plt.subplots(1, 2, figsize=(17, 6), tight_layout=True)
		sns.lineplot(data=comp_df.T, ax=axes[0], dashes=False)
		sns.lineplot(data=comp_df, ax=axes[1], dashes=False)

		axes[0].set_xlabel('Anomalous Community Sizes')
		axes[0].set_ylabel('Average Precision')
		axes[0].set_title(f'AP vs. anom_inter_p for different size groups', fontsize=14)
		axes[0].set_ylim(0, 1)

		axes[1].set_xlabel('anom_inter_p')
		axes[1].set_ylabel('Average Precision')
		axes[1].set_title(f'AP vs. size groups for different anom_inter_p values', fontsize=14)
		axes[1].set_ylim(0, 1)

		fig.suptitle(f"anom_m = {m}", fontsize=18)
		plt.show()

	def plot_grouped_by_m_ps_and_size_groups(self, meta_feature: str):
		for m in _ALL_MS:
			comparison_dict = self.compare_single_m_diff_ps_and_size_groups(
					meta_feature=meta_feature, m=m, ps=_ALL_PS, size_groups=_ALL_SIZE_GROUPS)
			comparison_df = pd.DataFrame.from_dict(comparison_dict, orient='index')
			self.plot_single_m_diff_ps_and_size_groups(comparison_df, m)

	def get_ms_comparison_df(self, meta_feature):
		temp = {
			m: self.compare_single_m_diff_ps_and_size_groups(
				meta_feature=meta_feature, m=m, ps=_ALL_PS, size_groups=_ALL_SIZE_GROUPS)
			for m
			in _ALL_MS
		}
		temp = {
			m: pd.DataFrame.from_dict(comparison_dict, orient='columns').mean()
			for m, comparison_dict
			in temp.items()
		}
		output = pd.DataFrame.from_dict(temp, orient='columns')
		output.index.rename('anom_inter_p', inplace=True)
		return output
	# Single size group - different ps and ms
	# /////////////////////////////////////////////

	def compare_single_size_group_diff_ps_and_ms(self, meta_feature, size_group, ps, ms):
		"""
		Creates a dictionary of mean average precision scores.

		Dictionary of form {anom_inter_p: {anom_m: mean avg. precision, ...}, ...}
		for each of the input anom_inter_p values, for each of the input anom_m values,
		for a single size_group, and for a single meta-feature.
		"""
		output = dict()

		for p in ps:
			output[f'p{p:.2f}'] = {}
			for m in ms:
				exp_path = self._ordered_dir_paths[f'p{p:.2f}'][size_group][f'm{m:.2f}']

				# Create results analysis DataFrame
				results_analysis_df = self.create_single_experiment_results_analysis_df(exp_path, algo='OurResults')

				# Get all meta-features' mean average precision dictionary
				temp = self.get_single_experiment_mean_average_precision_dict(results_analysis_df)

				# Extract input meta-feature's mean average precision score
				output[f'p{p:.2f}'][f'm{m:.2f}'] = temp[meta_feature]['mean']

		return output

	@staticmethod
	def plot_single_size_group_diff_ps_and_ms(comp_df, size_group):
		fig, axes = plt.subplots(1, 2, figsize=(17, 6), tight_layout=True)
		sns.lineplot(data=comp_df.T, ax=axes[0], dashes=False)
		sns.lineplot(data=comp_df, ax=axes[1], dashes=False)

		axes[0].set_xlabel('anom_m')
		axes[0].set_ylabel('Average Precision')
		axes[0].set_title(f'AP vs. anom_m for different anom_inter_p values', fontsize=14)
		axes[0].set_ylim(0, 1)

		axes[1].set_xlabel('anom_inter_p')
		axes[1].set_ylabel('Average Precision')
		axes[1].set_title(f'AP vs. anom_inter_p for different anom_m values', fontsize=14)
		axes[1].set_ylim(0, 1)

		fig.suptitle(f"Anomalous Community Size Group = '{size_group}'", fontsize=18)
		plt.show()

	def plot_grouped_by_size_groups_ps_and_ms(self, meta_feature):
		for size_group in _ALL_SIZE_GROUPS:
			comparison_dict = self.compare_single_size_group_diff_ps_and_ms(
				meta_feature=meta_feature, size_group=size_group, ps=_ALL_PS, ms=_ALL_MS)
			comparison_df = pd.DataFrame.from_dict(comparison_dict, orient='index')
			self.plot_single_size_group_diff_ps_and_ms(comparison_df, size_group)

	def get_size_groups_comparison_df(self, meta_feature):
		temp = {
			size_group: self.compare_single_size_group_diff_ps_and_ms(
				meta_feature=meta_feature, size_group=size_group, ps=_ALL_PS, ms=_ALL_MS)
			for size_group
			in _ALL_SIZE_GROUPS
		}
		temp = {
			size_group: pd.DataFrame.from_dict(comparison_dict, orient='columns').mean()
			for size_group, comparison_dict
			in temp.items()
		}
		output = pd.DataFrame.from_dict(temp, orient='columns')
		output.index.rename('anom_m', inplace=True)
		return output

	########################################
	########################################
	# MyAlgo and Baseline Comparison
	########################################
	########################################

	def baseline_compare_single_m_and_size_group_diff_p(self, m, size_group, ps, meta_feature, reverse: bool):

		temp = dict()

		for p in ps:
			temp[p] = {}

			exp_path = self._ordered_dir_paths[f'p{p:.2f}'][size_group][f'm{m:.2f}']

			# My results
			my_res_df = self.create_single_experiment_results_analysis_df(exp_path, algo='OurResults', reverse=False)
			temp[p]['OurAlgorithm'] = self.get_single_experiment_mean_average_precision_dict(my_res_df)[meta_feature]['mean']

			# Baseline results
			baseline_res_df = self.create_single_experiment_results_analysis_df(exp_path, algo='BaselineResults', reverse=reverse)
			for alg, score in self.get_single_experiment_mean_average_precision_dict(baseline_res_df).items():
				temp[p][alg] = score['mean']

		comp_df = pd.DataFrame.from_dict(temp, orient='index')
		comp_df = comp_df.rename({col: col.replace('__score__AP', '') for col in comp_df.columns}, axis=1)

		return comp_df

	def plot_baseline_comparison(self, meta_feature: str, reverse: bool):
		# create horizontal line of subplots, using with len(size_groups) subplots
		fig, axes = plt.subplots(len(_ALL_MS), len(_ALL_SIZE_GROUPS), figsize=(17, 20), tight_layout=True)

		for i, m in enumerate(_ALL_MS):
			for j, size_group in enumerate(_ALL_SIZE_GROUPS):

				comp_df = self.baseline_compare_single_m_and_size_group_diff_p(
					m=m, size_group=size_group, ps=_ALL_PS, meta_feature=meta_feature, reverse=reverse)
				legend = 'full' if j == len(_ALL_SIZE_GROUPS) - 1 else False
				sns.lineplot(data=comp_df, ax=axes[i, j], dashes=True, legend=legend)

				if j == 0:
					axes[i, j].set_ylabel(f"anom_m = {m}", fontsize=16)

				if i == 0:
					axes[i, j].set_title(f"size group = '{size_group}'", fontsize=16)

		plt.setp(axes, ylim=(0, 1.05))

		plt.show()

	def __get_baseline_comparison_df(self, meta_feature, reverse: bool):
		# TODO: develop
		for m in _ALL_MS:
			for size_group in _ALL_SIZE_GROUPS:
				temp = self.baseline_compare_single_m_and_size_group_diff_p(
					meta_feature=meta_feature, size_group=size_group, ps=_ALL_PS, m=m, reverse=reverse)


	"""
	########################################
	# Experiment Results Analysis - Precison@K - maybe later
	########################################

	def _precision_at_k_all_cols(self, ranking_columns, k, columns, boundary):
		output = {}
		for comm_col, rank_col in columns:
			ranking = ranking_columns[comm_col]
	

			if boundary == 'top':
				ranking = self._invert_rank_by_ranking_index(ranking)

			# invert ranking if needed
			if self._determine_top_or_bottom_ranking(ranking) == 'bottom':
			if rank_col in self._invert_columns:
				ranking = self._invert_rank_by_ranking_index(ranking)

			output[f'{rank_col}_prec_@_{k}'] = self._precision_at_k(ranking, k)
		return output

	@staticmethod
	def _precision_at_k(ranking, k):
		true_at_k = np.where(ranking < k)[0]
		prec_at_k = true_at_k.shape[0] / k
		return prec_at_k
	"""