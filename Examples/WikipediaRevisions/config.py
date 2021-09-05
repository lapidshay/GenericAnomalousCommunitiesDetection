import os
from xgboost import XGBClassifier
from os.path import join


DATA_PATHS = {
	'example_1': {
		'TRAIN_SET_COMMS': join('Data', 'NoCategory__TrainSet__2019-01-01_2019-07-01.json'),
		'TEST_SET_COMMS': join('Data', 'NoCategory__TestSet__2019-01-01_2019-07-01.json')
	},
	'example_2': {
		'TRAIN_SET_COMMS': join('Data', 'NoCategory__TrainSet__2016-01-01_2020-07-15.json'),
		'TEST_SET_COMMS': join('Data', 'NoCategory__TestSet__2016-01-01_2020-07-15.json')
	}
}


CONFIG = {
	'community_partite_label': 'Page',
	'vertex_partite_label': 'User',
	'classifer_obj': XGBClassifier(),
	'max_edges_to_sample': None,
	'label_thresh': 0.5,
	'val_size': 0.2,
	'save_topological_features': True,
	'save_dir_path': 'Checkpoint',
	'verbose': True
}


OUTPUT_PATH = join(os.getcwd(), 'Results.csv')
