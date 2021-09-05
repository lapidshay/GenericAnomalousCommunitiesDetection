__author__ = "Shay Lapid"
__email__ = "lapidshay@gmail.com"

import requests
import json
import functools
import time
from datetime import datetime


def timer(func):
	"""Decorator. Prints the runtime of the decorated function."""

	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.perf_counter()
		value = func(*args, **kwargs)
		print(f'Finished function "{func.__name__}" in {round(time.perf_counter() - start_time, 3)} seconds.')

		return value

	return wrapper_timer


class WikipediaRevisionsFetcher:
	def __init__(self):
		pass

	#################################
	# Helper functions
	#################################

	@staticmethod
	def _basic_revision_query_string(wiki_lang, include_rev_content):
		"""Returns basic revisions query prefix."""

		# base URL of Wikipedia API
		api_str = f'https://{wiki_lang}.wikipedia.org/w/api.php'

		# some query properties
		query_props_str = f'action=query&prop=revisions&rvslots=*&format=json&formatversion=2'

		# revision props to show
		revisions_props = '|'.join(['ids', 'timestamp', 'user', 'flags', 'comment'])
		if include_rev_content:
			revisions_props += '|content'

		rev_props_str = f'rvprop={revisions_props}'

		return f'{api_str}?{query_props_str}&{rev_props_str}'

	def _create_page_title_query_string(self, wiki_lang, num_revisions, include_rev_content, page_title):
		"""Returns a query string for finding revisions of specific page."""

		# create basic revision query string prefix
		basic_rev_str = self._basic_revision_query_string(wiki_lang, include_rev_content)

		# num. revisions limit
		limit_str = f'rvlimit={num_revisions}'

		# page title
		title_str = f'titles={page_title}'

		# concatenate to a full query
		return f'{basic_rev_str}&{limit_str}&{title_str}'

	def _create_rev_id_query_string(self, wiki_lang, rev_id, include_rev_content):
		"""Returns a query string for finding a specific revision by its ID."""

		# create basic revision query string prefix
		basic_rev_str = self._basic_revision_query_string(wiki_lang, include_rev_content)

		# revision id
		rev_id_str = f'revids={rev_id}'

		# concatenate for a full query
		return f'{basic_rev_str}&{rev_id_str}'

	@staticmethod
	def _pad_timestamp(timestamp_int: [int, str]):
		return int(str(timestamp_int).ljust(14, str(0)))

	@staticmethod
	def _time_stamp_string(timestamp):
		return datetime.strptime(str(timestamp), '%Y%m%d%H%M%S').strftime('%d/%m/%Y %H:%M:%S')

	#################################
	# Main functions
	#################################

	@timer
	def create_revision_dict_by_page_title(self, wiki_lang, num_revisions, page_title, include_rev_content):
		"""
		Fetches revisions of a Wikipedia page and returns page namespace, id and revisions dictionary.

		The function takes a relatively long time to be executed, due to the request time.

		Parameters
		----------
		wiki_lang: Language of Wikipedia, str
			Langauge is denoted with 2 letters, i.e, 'en'
		num_revisions: Number of revisions to fetch, int
		page_title: Page title to fetch revisions for, str
		include_rev_content: Determines whether to include revisions' content, bool

		Returns
		-------
		namespace: Namespace of page
		pageid: Page id
		revisions: A dictionary of form {
			revision_id: {'minor', 'user', 'timestamp', 'comment', 'content' (optional)}
			}

		Notes
		-----
		The returned revisions dictionary contains the following items:
			'minor': bool, corresponds to minor or major revision
			'user': str, corresponds to username who made the revision
			'timestamp':str, corresponds to revision's timestamp
			'comment':str, corresponds to user's comment for the revision
			'content' (optional):str, corresponds to the content of the revision
		The returned revisions dictionary is suitable for creating a pandas.DataFrame,
			using pandas.DataFrame.from_dict(revision_dictionary, orient='index')

		Example
		-------
		>>> WikipediaRevisionsFetcher().create_revision_dict_by_page_title('en', 3, 'Homebrewing', False)
		(0,
		 146918,
		 {959365983: {'parentid': 945881523,
			'minor': True,
			'user': 'FrescoBot',
			'timestamp': '2020-05-28T12:25:40Z',
			'comment': 'Bot: [[User:FrescoBot/Links|link syntax]] and minor changes'},
		  945881523: {'parentid': 936049601,
			'minor': True,
			'user': 'MacRellik',
			'timestamp': '2020-03-16T18:44:36Z',
			'comment': 'added carbonation tabs as an alternative to to priming sugar'},
		  936049601: {'parentid': 916713102,
			'minor': False,
			'user': 'InternetArchiveBot',
			'timestamp': '2020-01-16T11:38:49Z',
			'comment': 'Rescuing 1 sources and tagging 0 as dead.) #IABot (v2.0'}})
		"""

		# create a query string
		query_string = self._create_page_title_query_string(wiki_lang, num_revisions, include_rev_content, page_title)

		# request content
		req = requests.request('GET', query_string)

		# find the dictionary string within the request text
		#relevant_text = re.search(r'(?<=<pre>)(.*)(?=</pre>)', req.text, flags=re.M | re.S)[0]

		# extract relevant dictionary from text
		request_dictionary = json.loads(req.text)
		request_dictionary = request_dictionary['query']['pages'][0]

		# extract namespace and page id
		namespace = request_dictionary['ns']
		pageid = request_dictionary['pageid']

		# extract revisions' details
		temp_revisions = request_dictionary['revisions']
		revisions = {}
		for rev in temp_revisions:
			cur_rev_id = rev['revid']
			revisions[cur_rev_id] = {
				k: v
				for k, v in rev.items()
				if k in ['parentid', 'minor', 'user', 'timestamp', 'comment']
			}

			# include content
			if include_rev_content:
				revisions[cur_rev_id].update({'content': rev['slots']['main']['content']})

		return namespace, pageid, revisions

	@timer
	def create_revision_dict_by_revision_id(self, wiki_lang, revision_id, include_rev_content):
		# TODO: write docstring!

		# create a query string
		query_string = self._create_rev_id_query_string(wiki_lang, revision_id, include_rev_content)

		# request content
		req = requests.request('GET', query_string)

		# extract relevant dictionary from text
		request_dictionary = json.loads(req.text)
		request_dictionary = request_dictionary['query']['pages'][0]

		# extract namespace and page id
		namespace = request_dictionary['ns']
		pageid = request_dictionary['pageid']
		page_title = request_dictionary['title']

		# extract revisions' details
		temp_revisions = request_dictionary['revisions']
		revisions = {}
		for rev in temp_revisions:
			cur_rev_id = rev['revid']
			revisions[cur_rev_id] = {
				k: v
				for k, v in rev.items()
				if k in ['parentid', 'minor', 'user', 'timestamp', 'comment']
			}
			# include content
			if include_rev_content:
				revisions[cur_rev_id].update({'content': rev['slots']['main']['content']})

		return namespace, pageid, page_title, revisions

	@timer
	def get_revision_content_by_revision_id(self, wiki_lang, revision_id):
		"""
		Fetches revision's content by revision ID.

		The function takes a relatively long time to be executed, due to the request time.

		Parameters
		----------
		wiki_lang: Language of Wikipedia, str
			Language is denoted with 2 letters, i.e, 'en'
		revision_id: revision id to fetch, int

		Returns
		-------
		content: Content of input revision, str

		Example
		-------
		>>> WikipediaRevisionsFetcher().create_revision_dict_by_page_title('en', 1212312)
		"History of [[Pittsburgh, Pennsylvania]]\r\n\r\nOn [[July 21]], [[1877]], a day after bloody
		[[riot]]ing in [[Baltimore]] from [[Baltimore and Ohio Railroad]] workers and the deaths of 9
		rail workers at the hands of the [[Maryland]] militia, workers in Pittsburgh staged a sympathy
		[[strike]] that was met with an assault by the state militia - Pittsburgh then erupted into widespread
		rioting. \r\n\r\nWith the recessions of the [[1970s]] and the advent of cheap foreign labor,
		Pittsburgh's steel mills found themselves unable to compete with foreign steel mills, and most closed down.
		This created a ripple effect that decimated the local economy, as railroads, mines, and factories across the
		region shut down, one by one.\r\n\r\nThe [[1980s]] and [[1990s]] brought a remarkable transformation.
		The smoke filled skies and sooty buildings gave way to a new position as a major technology center.
		High technology and biomedical firms grew up around the area's colleges."
		"""

		# create a query string
		query_string = self._create_rev_id_query_string(wiki_lang, revision_id, include_rev_content=True)

		# request content
		req = requests.request('GET', query_string)

		# extract relevant dictionary from text
		request_dictionary = json.loads(req.text)

		# extract revision content
		revision_content = request_dictionary['query']['pages'][0]['revisions'][0]['slots']['main']['content']

		return revision_content

	@timer
	def continous_create_revision_dict_by_page_title(
			self,
			wiki_lang, page_title,
			start_time=None, end_time=None, num_revisions=50,
			include_rev_content=True,
			verbose=False):

		"""
		Fetches revisions of a Wikipedia page and returns page namespace, id and revisions dictionary.

		The function takes a relatively long time to be executed, due to the request time.

		Parameters
		----------
		wiki_lang: Language of Wikipedia, str
			Langauge is denoted with 2 letters, i.e, 'en'
		num_revisions: Number of revisions to fetch, int
		page_title: Page title to fetch revisions for, str
		include_rev_content: Determines whether to include revisions' content, bool

		Returns
		-------
		namespace: Namespace of page
		pageid: Page id
		revisions: A dictionary of form {
			revision_id: {'minor', 'user', 'timestamp', 'comment', 'content' (optional)}
			}

		Notes
		-----
		The returned revisions dictionary contains the following items:
			'minor': bool, corresponds to minor or major revision
			'user': str, corresponds to username who made the revision
			'timestamp':str, corresponds to revision's timestamp
			'comment':str, corresponds to user's comment for the revision
			'content' (optional):str, corresponds to the content of the revision
		The returned revisions dictionary is suitable for creating a pandas.DataFrame,
			using pandas.DataFrame.from_dict(revision_dictionary, orient='index')

		Example
		-------
		>>> WikipediaRevisionsFetcher().create_revision_dict_by_page_title('en', 3, 'Homebrewing', False)
		(0,
		 146918,
		 {959365983: {'parentid': 945881523,
			'minor': True,
			'user': 'FrescoBot',
			'timestamp': '2020-05-28T12:25:40Z',
			'comment': 'Bot: [[User:FrescoBot/Links|link syntax]] and minor changes'},
		  945881523: {'parentid': 936049601,
			'minor': True,
			'user': 'MacRellik',
			'timestamp': '2020-03-16T18:44:36Z',
			'comment': 'added carbonation tabs as an alternative to to priming sugar'},
		  936049601: {'parentid': 916713102,
			'minor': False,
			'user': 'InternetArchiveBot',
			'timestamp': '2020-01-16T11:38:49Z',
			'comment': 'Rescuing 1 sources and tagging 0 as dead.) #IABot (v2.0'}})
		"""

		start_time = self._pad_timestamp(start_time)
		end_time = self._pad_timestamp(end_time)

		# create a query string without times
		query_string_no_time = self._create_page_title_query_string(wiki_lang, num_revisions, include_rev_content, page_title)

		# add starting time (earliest time, in API called rvend) - Remains constant throughout execution
		query_string_start_time = f'{query_string_no_time}&rvend={start_time}'


		# add ending time (latest time, in the API called rvstart)
		if end_time:
			query_string = f'{query_string_start_time}&rvstart={end_time}'
		else:
			query_string = f'{query_string_start_time}&rvstart=now'

		if verbose:
			print(f'{datetime.now().strftime("%H:%M:%S")}:')
			print(f'\tProcessing query: "{query_string}"')
			print(f'\tFirst query min time (constant): {self._time_stamp_string(start_time)}')
			print(f'\tFirst query ending time:         {self._time_stamp_string(end_time)}')

		# instantiate a dictionary to hold output
		revisions_output = {}

		while True:
			# request content
			req = requests.request('GET', query_string)

			# extract relevant dictionary from text
			request_dictionary = json.loads(req.text)

			# extract pages dictionary
			page_dictionary = request_dictionary['query']['pages'][0]

			# extract namespace and page id
			namespace = page_dictionary['ns']
			pageid = page_dictionary['pageid']

			# extract revisions' details
			revisions_dictionary = page_dictionary['revisions']

			for rev in revisions_dictionary:
				cur_rev_id = rev['revid']
				revisions_output[cur_rev_id] = {
					k: v
					for k, v in rev.items()
					if k in ['parentid', 'minor', 'user', 'timestamp', 'comment']
				}

				# include content
				if include_rev_content:

					# if content is hidden, return None
					revisions_output[cur_rev_id].update({'content': rev['slots']['main'].get('content', None)})

			if verbose:
				print(f'{datetime.now().strftime("%H:%M:%S")}:')
				print(f'\tProcessed {len(revisions_dictionary)} revisions.')

			# finishing term
			finished = request_dictionary.get('batchcomplete', False)
			if finished:
				break

			# extract next query ending time
			# add starting time (earliest time, in API called rvend)
			next_request_end_time_full_str = request_dictionary['continue']['rvcontinue']
			next_request_end_time = int(next_request_end_time_full_str.split('|')[0])

			if verbose:
				print(f'\tNext query min. time (constant): {self._time_stamp_string(start_time)}')
				print(f'\tNext query ending time:          {self._time_stamp_string(next_request_end_time)}')

			query_string = f'{query_string_start_time}&rvstart={next_request_end_time}'

		return namespace, pageid, revisions_output
