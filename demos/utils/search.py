# Copyright (c) 2025 Janea Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

from utils.embeddings_container import EmbeddingsContainer
from utils.models import ClosestLabels, Index

import faiss

search_k_default = 10
use_forgiving_metric = True


def get_common_counts(
    closest_actual_lists: ClosestLabels, closest_ref_lists: ClosestLabels
) -> list[int]:
    def _common_count(closest_actual, closest_ref):
        set_actual = set(closest_actual)
        set_ref = set(closest_ref)
        return len(set(set_actual) & set(set_ref))

    return [
        _common_count(actual, ref) for actual, ref in zip(closest_actual_lists, closest_ref_lists)
    ]


def get_search_accuracy(common_counts: list[int], k: int) -> float:
    modified_counts = (
        common_counts
        if not use_forgiving_metric
        else [k if count >= k / 2 else count for count in common_counts]
    )

    return float(np.array(modified_counts).mean() / k)


def get_search_closest_labels(
    index: Index, container: EmbeddingsContainer, k: int
) -> ClosestLabels:
    if not index.is_trained:
        index.train(container.train)

    if index.ntotal == 0:
        index.add(container.add)

    (_, I) = index.search(container.search, k)
    return I.tolist()


def get_ref_search_closest_labels(container: EmbeddingsContainer, k: int) -> ClosestLabels:
    ref_index = faiss.IndexFlat(container.d, faiss.METRIC_INNER_PRODUCT)  # type: ignore[attr-defined] # noqa
    return get_search_closest_labels(ref_index, container, k)
