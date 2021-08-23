"""
TODO: add module docstring.

"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

########################################
# imports
########################################

import pandas as pd
from .utils import \
	model_validation, print_scores_confusion_matrix, get_classifier_scores, convert_literal_tuple_string_index_to_tuple


########################################
# LinkPredictor
########################################

class LinkPredictor:
	"""
	TODO: document
	"""

	def __init__(self, classifier_model):
		"""
		TODO: document
		Inits a...

		Parameters
		----------
		classifier_model: ..
		label_col_name: ..
		"""

		self._model = classifier_model
		self._label_col_name = None

		# scores
		self._train_set_validation_scores = None
		self._test_set_prediction_summary = None

	########################################
	# Training
	########################################

	def fit(self, train_df: pd.DataFrame, label_col_name: str, val_size: float=0.1, verbose: bool=False):
		"""
		TODO: document
		Trains a classifier.

		First splits to train and validation set to evaluate a copy of the classifier's performance and report it.
		Then trains with all input data.

		Parameters
		----------
		train_df: A pandas.DataFrame to train on.
		label_col_name: A string to determine the label (target) column name in train_df.
		val_size: Optional; default 0.1
			a float to determine train/validation split for evaluation.
		verbose: Optional; default=False
			A boolean to determine whether to print the trained classifier evaluation scores.
		"""

		# set label column's name
		self._label_col_name = label_col_name

		# split data and label
		X_train_val = train_df.drop(self._label_col_name, axis=1)
		y_train_val = train_df[self._label_col_name].values

		# evaluate
		self._train_set_validation_scores = model_validation(self._model, X_train_val, y_train_val, val_size)

		# train classifier on ol of the input data
		self._model.fit(X_train_val, y_train_val)

		# print evaluation performance
		if verbose:
			if val_size == 0:
				raise ValueError('Argument \'val_size\' is 0. Can not perform evaluation.')
			print_scores_confusion_matrix(self._train_set_validation_scores, data_name='validation')

	########################################
	# Inference
	########################################

	def _test_set_edges_prediction_summary(self, X_test, y_test, verbose):

		scores = get_classifier_scores(self._model, X_test, y_test, data_name='test')

		self._test_set_prediction_summary = {
			'predicted_exist': scores['test_tp'],
			'predicted_not_exist': scores['test_fn'],
			'predicted_ratio': scores['test_acc'],
		}

		# Prints scores and confusion matrix on of given classifier on test DataFrame
		if verbose:
			print(f'Test set edge existence predictions:')
			[print(f'\t{str(k).ljust(10)}: {str(v)[:5]}') for k, v in self._test_set_prediction_summary.items()]

	def get_edges_existence_prob(
			self, test_df: pd.DataFrame, comm_before_user: bool=True, vertex_to_int: bool=False, verbose=False):
		"""
		TODO: document

		Returns a dictionary of form {(node_1, node_2): edge_existence_probability} created from input DataFrame.

		Parameters
		----------
		test_df: A list of communties to be fiterd in to create the BiPartite graph.
		comm_before_user: Optional; a string to label the vertices of the community part.
		index_str_before_digit: Optional; a string to label the vertices of the vertices part.
		verbose: Optional; default=False
			A boolean to determine whether to print the trained classifier evaluation scores.

		Returns
		-------
		nx.Graph object contating the BiPartite graph.

		Examples
		--------
		The following will create a BiPartite graph called BPG,
		which contains 3 communitey-representing vertices - 'comm1', 'comm2', 'comm3',
		and all the vertices that belong to their corresponding communities in the partition dictionary input,
		and the partites will be names 'group' and 'user':

		"""

		# split to data and labels
		X_test = test_df.drop(self._label_col_name, axis=1)
		y_test = test_df[self._label_col_name].values

		# get prediction statistics
		self._test_set_edges_prediction_summary(X_test, y_test, verbose)

		# get all edges existence probabilities
		probs = self._model.predict_proba(X_test)[:, 1]

		# convert DataFrame's literal tuple string index to a tuple index
		if type(test_df.index[0]) == str:
			convert_literal_tuple_string_index_to_tuple(
				test_df, comm_before_user=comm_before_user, vertex_to_int=vertex_to_int)

		# create a dictionary
		return {idx: pr for idx, pr in zip(test_df.index, probs)}

# TODO: verify not needed and delete
"""

# instantiate a DataFrame with only the indices of input DataFrame, to populate with edges existence probabilities
df_to_dict = df[[]]

# get all edges existence probabilities
df_to_dict['existence_prob'] = self._get_edges_exist_probability(df)

# infer index type
index_type = type(df_to_dict.index[0])

# if index is of str type
if index_type == str:

	# converts DataFrame's literal string index to a tuple (int, str) index
	link_prediction_utils.convert_literal_string_index_to_tuple(
		df_to_dict, comm_before_user=comm_before_user, index_str_before_digit=index_str_before_digit)

# create a dictionary of edge and it's existing probability
edges_exist_prob_dict = df_to_dict.to_dict(orient='index')

return edges_exist_prob_dict

"""
