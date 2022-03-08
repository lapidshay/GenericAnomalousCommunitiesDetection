__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

import os
import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn import metrics


##################################
# Anomalous Community Detector Utils
##################################

def checkpoint_paths(dir_path: str = None, save: bool = False):
	# file names
	train_path = 'Train_Topological_Features.csv'
	test_path = 'Test_Topological_Features.csv'

	# id save_dir_path is not given, use a default
	if dir_path is None:
		dir_path = os.path.join(os.getcwd(), 'Checkpoint')

	if save:
		# if dir path does not exist, create it
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)

	train_path = os.path.join(dir_path, train_path)
	test_path = os.path.join(dir_path, test_path)

	return train_path, test_path


def load_topological_features_df(dir_path: str):
	# get train and test file paths
	train_path, test_path = checkpoint_paths(dir_path=dir_path, save=False)

	# read CSV files to DataFrames
	train_df = pd.read_csv(train_path, index_col=0)
	test_df = pd.read_csv(test_path, index_col=0)

	return train_df, test_df


##################################
# BiPartite Creator Utils
##################################


def print_bipartite_properties(BPG, network: str = ''):
	"""Prints the properties of a bipartite graph."""

	props = get_bipartite_properties(BPG)
	partite_1 = props['partite_1_label']
	partite_2 = props['partite_2_label']

	print(f"{network} BiPartite network properties:")
	print(f"\tNumber of '{partite_1}'-partite vertices: {props['partite_1_num_vertices']}")
	print(f"\tNumber of '{partite_2}'-partite vertices: {props['partite_2_num_vertices']}")
	print(f"\tTotal number of vertices: {props['total_vertices']}")
	print(f"\tTotal number of edges: {props['total_edges']}")


def get_bipartite_properties(BPG):
	"""Returns a dictionary with bipartite graph properties."""

	# infer the 2 partites' labels
	partite_1, partite_2 = _infer_bipartite_partite_labels(BPG)

	# get each partites' vertices
	partite_1_vertices = _get_partite_vertices(BPG, partite_1)
	partite_2_vertices = _get_partite_vertices(BPG, partite_2)

	return {
		'partite_1_num_vertices': len(partite_1_vertices),
		'partite_2_num_vertices': len(partite_2_vertices),
		'partite_1_label': partite_1,
		'partite_2_label': partite_2,
		'total_vertices': len(BPG.nodes()),
		'total_edges': len(BPG.edges())
	}


def _infer_bipartite_partite_labels(BPG):
	"""Returns a list of 2 strings, the labels of the partites."""
	return list(set(nx.get_node_attributes(BPG, 'partite').values()))


def _get_partite_vertices(BPG, partite_label):
	"""Returns a list containing partite's vertices."""
	return [vertx for vertx in BPG.nodes(data="partite") if vertx[1] == partite_label]


##################################
# LinkPredictor Utils
##################################

def model_validation(model, X, y, val_size):
	"""Model performance evaluation"""
	# split to train and validation sets, and split data and labels
	train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=val_size)

	# create a deep copy of classifier
	model_copy = deepcopy(model)

	# train model copy
	model_copy.fit(train_X, train_y)

	# calculate validation set scores
	validation_scores = get_classifier_scores(model_copy, val_X, val_y, 'validation')
	return validation_scores


def get_classifier_scores(clf, X, y_true, data_name: str):
	"""Returns dictionary with scores."""

	# predict X using classifier
	y_preds = clf.predict(X)

	# scores
	prc = metrics.precision_score(y_true, y_preds)
	acc = metrics.accuracy_score(y_true, y_preds)
	f1 = metrics.f1_score(y_true, y_preds)
	auc = None
	if len(np.unique(y_true)) == 2:
		auc = metrics.roc_auc_score(y_true, y_preds)

	# confusion metrics
	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_preds).ravel()

	# create a dictionary with all scores
	output = {
		f'{data_name}_prc': prc,
		f'{data_name}_acc': acc,
		f'{data_name}_f1': f1,
		f'{data_name}_auc': auc,
		f'{data_name}_tn': tn,
		f'{data_name}_fp': fp,
		f'{data_name}_fn': fn,
		f'{data_name}_tp': tp
	}

	return output


def print_scores_confusion_matrix(scores, data_name):
	"""Prints scores of a trained classifier given data to predict, and corresponding ground truth labels."""

	cnf_str = f"""
                  Predicted
                   0     1   
                ------------- 
            0  | {str(scores[f'{data_name}_tn']).ljust(4)} | {str(scores[f'{data_name}_fp']).ljust(4)} |
     True      |-------------|
            1  | {str(scores[f'{data_name}_fn']).ljust(4)} | {str(scores[f'{data_name}_tp']).ljust(4)} |
                ------------- 
            """

	scores_str = {
		f'Precision': scores[f'{data_name}_prc'],
		'Accuracy': scores[f'{data_name}_acc'],
		'F1': scores[f'{data_name}_f1'],
		'ROC AUC': scores[f'{data_name}_auc']
	}

	print(f'{str(data_name).capitalize()} scores:')
	[print(f'\t{str(k).ljust(10)}: {str(v)[:5]}') for k, v in scores_str.items()]
	print(cnf_str)


def _index_tuple_literal_eval(string: str):
	"""Evaluates a string literal of form 'recipe_num, malt', and returns a tuple (recipe_num(int), malt(str))."""

	rec_malt = string[1:-1].split(', ')
	rec = int(rec_malt[0])
	malt = ', '.join(rec_malt[1:])
	return rec, malt


def _index_tuple_literal_eval_with_ordering(string: str, comm_before_user: bool, vertex_to_int: bool):
	# split string of form "(aa, xx, .., zz)" to "aa" and ["xx", .., "zz"]
	community, *vertex = string[1:-1].split(', ')

	# join vertex name components back to one string
	vertex = ', '.join(vertex)

	# if vertex name is a number, convert it to integer
	if vertex_to_int and vertex.isdigit():
		vertex = int(vertex)

	# determine the order of the tuple
	return (community, vertex) if comm_before_user else (vertex, community)


def convert_literal_tuple_string_index_to_tuple(
		df: pd.DataFrame, comm_before_user: bool = True, vertex_to_int: bool = False):
	"""
	Converts a DataFrame's literal string index of form '(community, vertex)' to a tuple (community, vertex) index.

	Changes input DataFrame inplace.
	"""

	# convert index to column
	df.reset_index(level=0, inplace=True)

	# evaluate tuple literal
	df['evaluated_index'] = [
		_index_tuple_literal_eval_with_ordering(
			string=tup.index,
			comm_before_user=comm_before_user,
			vertex_to_int=vertex_to_int)
		for tup
		in df.itertuples()
	]

	# convert back to index
	df.set_index('evaluated_index', inplace=True)
