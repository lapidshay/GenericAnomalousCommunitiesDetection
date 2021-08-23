"""
TODO: add module docstring.

"""

__author__ = 'Shay Lapid'
__email__ = 'lapidshay@gmail.com'

##################################
# Imports
##################################

from xgboost import XGBClassifier
from .BiPartiteCreator import BiPartiteCreator
from .NetworkSampler import NetworkSampler
from .FeatureExtractor import FeatureExtractor
from .LinkPredictor import LinkPredictor
from .MetaFeatureExtractor import MetaFeatureExtractor
from .MetaFeatureRanker import MetaFeatureRanker
from .utils import checkpoint_paths, load_topological_features_df


##################################
# A class for creating a BiPartite Graph from a partitions dictionary
##################################

class AnomalousCommunityDetector:
	"""
	...
	"""

	def __init__(
			self,
			train_partitions_map: dict, test_partitions_map: dict,
			community_partite_label: str='Community', vertex_partite_label: str='Vertex',
			classifer_obj=XGBClassifier()):
		"""
		...
		"""

		self._train_partitions_map = train_partitions_map
		self._test_partitions_map = test_partitions_map

		self._community_partite_label = community_partite_label
		self._vertex_partite_label = vertex_partite_label

		self._link_predictor = LinkPredictor(classifer_obj)

		self._BPG_train = None
		self._BPG_test = None
		self._train_topo_feat_df = None
		self._test_topo_feat_df = None
		self._sorted_ranked = None

	def detect_anomalous_communities(
			self,
			max_edges_to_sample: int=None,
			label_thresh: float=0.5,
			val_size: float=0.1,
			save_topological_features=False,
			save_dir_path=None,
			verbose: bool=False):
		"""
		TODO: document

		Parameters
		----------
		max_edges_to_sample: Int; default None.
			maximal number of edges to sample from train BiPartite network.
		label_thresh: Float; default 0.5.
			A float to determine the classification threshold of the label-based meta-features.
		val_size: Optional; default 0.1
			A float to determine train/validation split for the link-prediction classifier evaluation.
		save_topological_features:
		save_dir_path:
		verbose: Optional; default=False
			A boolean to determine whether to print some properties and progress.

		Returns
		---------
		A DataFrame of community-representing vertices, sorted and ranked by the meta-features.
		"""

		# Create BiPartite networks
		self._create_bi_partite_networks(verbose=verbose)

		# Sample edges
		train_pos_edges, train_neg_edge, test_pos_edges = self._sample_edges(max_edges_to_sample=max_edges_to_sample)

		# Extract topological features
		self._extract_topological_features(
			train_pos_edges, train_neg_edge, test_pos_edges, save_topological_features, save_dir_path)

		# Train Link-Prediction classifier
		self._fit_link_prediction_classifer(val_size=val_size, verbose=verbose)

		# Extract meta-features extraction
		meta_feats_dict = self._extract_meta_features(label_thresh=label_thresh, verbose=verbose)

		# Rank and sort meta-feature
		self._rank_sort_meta_features(meta_feats_dict)

		return self._sorted_ranked

	def detect_anomalous_communities_from_topological_features(
			self,
			dir_path: str='Checkpoint',
			label_thresh: float=0.5,
			val_size: float=0.1,
			verbose: bool=False):
		"""
		TODO: document

		Parameters
		----------
		dir_path: String

		label_thresh: Float; default 0.5.
			A float to determine the classification threshold of the label-based meta-features.
		val_size: Optional; default 0.1
			A float to determine train/validation split for the link-prediction classifier evaluation.
		verbose: Optional; default=False
			A boolean to determine whether to print some properties and progress.

		Returns
		---------
		A DataFrame of community-representing vertices, sorted and ranked by the meta-features.
		"""

		# Load topological features DataFrames
		self._train_topo_feat_df, self._test_topo_feat_df = load_topological_features_df(dir_path=dir_path)

		# Train Link-Prediction classifier
		self._fit_link_prediction_classifer(val_size=val_size, verbose=verbose)

		# Extract meta-features extraction
		meta_feats_dict = self._extract_meta_features(label_thresh=label_thresh, verbose=verbose)

		# Rank and sort meta-feature
		self._rank_sort_meta_features(meta_feats_dict)

		return self._sorted_ranked

	def _create_bi_partite_networks(self, verbose):
		BPG_train_generator = BiPartiteCreator(self._train_partitions_map)
		BPG_test_generator = BiPartiteCreator(self._test_partitions_map)

		self._BPG_train = BPG_train_generator.create_bipartite_graph(
			list(self._train_partitions_map.keys()),
			community_partite_label=self._community_partite_label,
			vertex_partite_label=self._vertex_partite_label)
		self._BPG_test = BPG_test_generator.create_bipartite_graph(
			list(self._test_partitions_map.keys()),
			community_partite_label=self._community_partite_label,
			vertex_partite_label=self._vertex_partite_label)

		if verbose:
			BPG_train_generator.print_properties('Train')
			BPG_test_generator.print_properties('Test')

	def _sample_edges(self, max_edges_to_sample):
		sampler = NetworkSampler(self._community_partite_label, self._vertex_partite_label)

		train_pos_edges, train_neg_edge = sampler.sample_network_edges(
			G=self._BPG_train,
			max_edges=max_edges_to_sample,
			generate_negative_edges=True)

		test_pos_edges, _ = sampler.sample_network_edges(
			G=self._BPG_test,
			max_edges=max_edges_to_sample,
			generate_negative_edges=False)

		return train_pos_edges, train_neg_edge, test_pos_edges

	def _extract_topological_features(self, train_pos_edges, train_neg_edge, test_pos_edges, save, save_dir_path):
		train_feat_extractor = FeatureExtractor(self._BPG_train)
		test_feat_extractor = FeatureExtractor(self._BPG_test)

		train_path, test_path = checkpoint_paths(dir_path=save_dir_path, save=save)

		self._train_topo_feat_df = train_feat_extractor.create_topological_features_df(
			positive_edges=train_pos_edges, negative_edges=train_neg_edge, save=save, save_dir_path=train_path)
		self._test_topo_feat_df = test_feat_extractor.create_topological_features_df(
			positive_edges=test_pos_edges, negative_edges=[], save=save, save_dir_path=test_path)

	def _fit_link_prediction_classifer(self, val_size, verbose):
		self._link_predictor.fit(
			train_df=self._train_topo_feat_df, label_col_name='edge_exist', val_size=val_size, verbose=verbose)

	def _extract_meta_features(self, label_thresh, verbose):
		edges_exist_prob_dict = self._link_predictor.get_edges_existence_prob(self._test_topo_feat_df, verbose=verbose)
		meta_feat_extractor = MetaFeatureExtractor(edges_exist_prob_dict)
		meta_feats_dict = meta_feat_extractor.get_comm_repr_vertices_meta_features(thresh=label_thresh)
		return meta_feats_dict

	def _rank_sort_meta_features(self, meta_feats_dict):
		meta_feat_ranker = MetaFeatureRanker(meta_feats_dict)
		self._sorted_ranked = meta_feat_ranker.rank_columns()
