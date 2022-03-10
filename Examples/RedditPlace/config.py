import os
from xgboost import XGBClassifier

DATA_PATHS = {
	'PARTITIONS_MAP': 'Data\\partition_map.json',
	'TRAIN_SET_COMMS': 'Data\\train_set_comms.json',
	'TEST_SET_COMMS': 'Data\\test_set_comms.json'
}

CONFIG = {
	'community_partite_label': 'Subreddit',
	'vertex_partite_label': 'User',
	'classifer_obj': XGBClassifier(),
	'max_edges_to_sample': None,
	'label_thresh': 0.5,
	'val_size': 0.2,
	'save_topological_features': True,
	'save_dir_path': 'Checkpoint',
	'verbose': True
}


OUTPUT_PATH = os.path.join(os.getcwd(), 'Results.csv')
