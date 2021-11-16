from xgboost import XGBClassifier


##################################
# Anomaly Detector Config
##################################

DETECTOR_CONFIG = {
	'community_partite_label': 'Community',
	'vertex_partite_label': 'Vertex',
	'classifer_obj': XGBClassifier(),
}

DETECTION_CONFIG = {
	'max_edges_to_sample': None,
	'label_thresh': 0.5,
	'val_size': 0.2,
	'save_topological_features': False,
	'save_dir_path': 'Checkpoint',
	'verbose': True
}

NUM_ANOM_COMMS = 10
