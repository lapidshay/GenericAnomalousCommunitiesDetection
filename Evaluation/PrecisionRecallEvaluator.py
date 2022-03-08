__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


class PrecisionRecallEvaluator:
	def __init__(self, ranking_and_score_df, ranking_col, score_col, anom_comm_names: list):

		# Create a deep copy of the input DataFrame slice
		self._df = ranking_and_score_df[[ranking_col, score_col]].copy(deep=True)

		self._anom_comm_names = anom_comm_names
		self._ranking_column = ranking_col
		self._score_column = score_col

		# Create a label column, and label anomalous communities
		self._label_anomalous_communities()

	def _label_anomalous_communities(self):
		"""Creates a label column, and labels communities with 1 (anomalous) and 0 (normal)."""

		self._df['anomalous'] = self._df[self._ranking_column].apply(lambda x: int(x in self._anom_comm_names))
		self._df[self._score_column] = - self._df[self._score_column]

	def get_df(self):
		return self._df

	def get_avg_precision(self, reverse=False):
		"""Return average precision."""

		if reverse:
			self._df[self._score_column] = - self._df[self._score_column]
		return average_precision_score(self._df['anomalous'], self._df[self._score_column])

	def plot_precision_recall(self, reverse=False):
		"""Plots precision-recall curve."""

		if reverse:
			self._df[self._score_column] = - self._df[self._score_column]

		precision, recall, thresholds = precision_recall_curve(self._df['anomalous'], self._df[self._score_column])
		plt.plot(recall, precision, label=f'AP = {self.get_avg_precision():.3f}')
		plt.legend()
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.show()
