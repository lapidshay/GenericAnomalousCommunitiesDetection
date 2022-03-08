__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import pandas as pd


########################################
# Meta-Feature Ranker
########################################

class MetaFeatureRanker:
	def __init__(self, meta_feature_scores_dict: dict):

		# Create a DataFrame from dict
		self._meta_feature_scores_df = pd.DataFrame.from_dict(meta_feature_scores_dict, orient='index')

	def rank_columns(self):
		"""Creates new columns which contain column ranking, and changes DataFrame inplace."""

		ranked_features_mini_dfs = []
		for col in self._meta_feature_scores_df.columns:

			# Sort series and convert to 1-column DataFrame with community names as index
			col_mini_df = self._meta_feature_scores_df[col].sort_values(ascending=False).to_frame()

			# Update index name to contain "ranking" instead of "score"
			col_mini_df.index.name = col.replace('score', 'ranking')

			# Put index as columns
			col_mini_df.reset_index(inplace=True)
			ranked_features_mini_dfs.append(col_mini_df)

		ranked_columns_df = pd.concat(ranked_features_mini_dfs, sort=False, axis=1)
		return ranked_columns_df
