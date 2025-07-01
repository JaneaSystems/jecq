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

"""
This script compares the search accuracy of different vector indices
for a given dataset.

For each member of the dataset, it searches for the k-closest vectors
(nearest neighbors) and compares the results against the `IndexFlat` index.
"""

import logging
from utils.common import df_to_str, log_info
import numpy as np
import pandas as pd
from utils.embeddings_container import EmbeddingsContainer
from utils.example_embeddings import get_example_embeddings
from utils.vector_indices import get_name, get_default_indices, validate_index
from utils.search import (
    get_common_counts,
    get_search_accuracy,
    search_k_default as k,
    use_forgiving_metric,
)

logger = logging.getLogger(__name__)

logger.info(f"Starting {__file__}...")
log_info()


logger.info("Loading dataset...")
text, data = get_example_embeddings()
embeddings = np.asarray(data)

container = EmbeddingsContainer(
    full=embeddings, add=embeddings, train=embeddings, search=embeddings
)

logger.info(f"Loaded {container}")

n, d = container.add.shape
indices = get_default_indices(n=n, d=d)

for i, index in enumerate(indices):
    index_name = get_name(index)

    logger.info(f"[{index_name}] Training...")
    index.train(container.train)
    validate_index(index)

    logger.info(f"[{index_name}] Adding...")
    index.add(container.add)

    logger.info(f"[{index_name}] Searching...")
    (D, I) = index.search(container.search, k)

    logger.info(f"[{index_name}] Scoring...")
    non_first_matches = [(a, I[a]) for a in range(n) if a != I[a][0]]
    if non_first_matches:
        logger.info(f"[{index_name}] Non-first matches: {non_first_matches}")
    assert not [
        (a, I[a]) for a in range(n) if a not in I[a]
    ], f"[{index_name}] Query not within top {k} results"

    data[f"{index_name}_closest"] = I.tolist()
    data[f"{index_name}_distances"] = D.tolist()
    data[f"{index_name}_common"] = get_common_counts(I.tolist(), data["IndexFlat_closest"].tolist())

index_names = [get_name(index) for index in indices]
accuracies = [
    get_search_accuracy(data[f"{index_name}_common"].tolist(), k) for index_name in index_names
]
df_accuracies = pd.DataFrame({"index": index_names, "search_accuracy": accuracies}).round(4)
logger.info(
    f"Accuracy summary with use_forgiving_metric={use_forgiving_metric}:\n"
    f"{df_to_str(df_accuracies)}"
)
