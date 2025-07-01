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
This script compares the resource usage of different vector
indices for a given dataset.

It measures the memory usage and time taken for training,
adding, and searching with each index.
"""

from utils.resources import get_memory_usage, log_memory_usage, time_it
from utils.search import search_k_default as k
from utils.vector_indices import get_name, get_default_indices, validate_index
from utils.example_embeddings import get_example_embeddings
from utils.embeddings_container import EmbeddingsContainer
import pandas as pd
import numpy as np
from utils.common import df_to_str, log_info
import logging

logger = logging.getLogger(__name__)

logger.info(f"Starting {__file__}...")
log_info()


duplication_factor = 100

log_memory_usage("Before data load")
logger.info("Loading dataset...")
text, data_original = get_example_embeddings()
data = pd.concat([data_original] * duplication_factor, ignore_index=True)

data_full = np.asarray(data, copy=True)

container = EmbeddingsContainer(
    full=data_full,
    add=data_full,
    train=np.asarray(data_original, copy=True),
    search=np.asarray(data_original)[0::100].copy(),
)
del data
del data_original
del data_full
del text
logger.info(f"Loaded {container}")
log_memory_usage("After data load")

n, d = container.add.shape
indices = get_default_indices(n=n, d=d)

metrics = []
columns = ["index", "memory_usage_mb", "train_time_sec", "add_time_sec", "search_time_sec"]

for index in indices:
    index_name = get_name(index)
    before_memory = get_memory_usage()

    logger.info(f"[{index_name}] Training...")
    train_time = time_it(lambda: index.train(container.train))  # noqa: F821
    validate_index(index)

    logger.info(f"[{index_name}] Adding...")
    add_time = time_it(lambda: index.add(container.add))  # noqa: F821
    after_memory = get_memory_usage()
    usage_diff = after_memory - before_memory

    logger.info(f"[{index_name}] Searching...")
    search_time = time_it(lambda: index.search(container.search, k))  # noqa: F821

    row = [index_name, usage_diff, train_time, add_time, search_time]
    metrics.append(row)
    row_dict = {column: value for (column, value) in zip(columns[1:], row[1:])}
    logger.info(f"[{index_name}] Metrics: {row_dict}")

logger.info("Deleting source data...")
del container
log_memory_usage("After data delete")

df_usage = pd.DataFrame(metrics, columns=columns).round(2)
logger.info(f"Metrics summary:\n{df_to_str(df_usage)}")
