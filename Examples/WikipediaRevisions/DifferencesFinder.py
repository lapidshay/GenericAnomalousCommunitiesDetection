from difflib import Differ
import numpy as np
import re
import functools
import time


def timer(func):
	"""Decorator. Prints the runtime of the decorated function."""

	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.perf_counter()
		value = func(*args, **kwargs)
		print(f'Finished function "{func.__name__}" in {round(time.perf_counter() - start_time, 3)} seconds.')

		return value

	return wrapper_timer


class DifferencesFinder:
	def __init__(self):
		pass

	@staticmethod
	def _expand_indices(span, exp_length, max_text_length):
		"""Returns a span tuple, expanded by exp_length, and clipped by [0, max_text_length]."""

		a = np.clip(span[0] - exp_length, a_min=0, a_max=max_text_length)
		b = np.clip(span[1] + exp_length, a_min=0, a_max=max_text_length)
		return a, b

	@staticmethod
	def _merge_overlapping_spans(spans):
		"""Returns a list of spans, where overlapping spans are merged to one span."""

		output = [spans.pop(0)]
		for i in range(len(spans)):
			cur_span = spans.pop(0)

			# check if current span overlaps last span in output
			if cur_span[0] <= output[-1][1]:

				# remove last span from output
				span_to_expand = output.pop()

				# merge
				output.append((span_to_expand[0], cur_span[1]))
			else:
				output.append(cur_span)
		return output

	def _change_indices(self, dif_string, exp_length):
		"""Returns expanded span indices of changes (denoted by [+,-,^])."""

		# remove '? ' prefix
		pruned_dif_string = dif_string[2:]

		# match change characters
		change_matches = re.finditer(r'[\+|\-|\^]+', pruned_dif_string)

		# extract match indices
		match_spans = [match.span() for match in change_matches]

		# smart expand indices with exp_length
		match_spans = [self._expand_indices(span, exp_length, len(pruned_dif_string)) for span in match_spans]

		# merge overlapping spans
		match_spans = self._merge_overlapping_spans(match_spans)

		return match_spans

	def difference(self, line_1, line_2, exp_length=10):
		"""Return a dictionary of expanded deletions and additions of 2 input string."""

		output = {'deletions_and_changes': None, 'additions_and_change': None}

		if type(line_1) == np.float:
			if np.isnan(line_1):
				return output

		if type(line_2) == np.float:
			if np.isnan(line_2):
				return output

		if line_1 is None or line_2 is None:
			return output

		# compare 2 string
		dif = list(Differ().compare([line_1], [line_2]))

		# find the lines that describe the changes
		# options are:
		# 1 (only deletions), 2 (only additions), 1 and 3 (deletions and editions)
		dif_marks_lines = np.where([line.startswith('? ') for line in dif])

		if 1 in dif_marks_lines[0]:
			change_indices = self._change_indices(dif[1], exp_length)
			output['deletions_and_changes'] = [line_1[span[0]:span[1]] for span in change_indices]

		if 3 in dif_marks_lines[0]:
			change_indices = self._change_indices(dif[3], exp_length)
			output['additions_and_change'] = [line_2[span[0]:span[1]] for span in change_indices]

		elif 2 in dif_marks_lines[0]:
			change_indices = self._change_indices(dif[2], exp_length)
			output['additions_and_change'] = [line_2[span[0]:span[1]] for span in change_indices]

		return output
