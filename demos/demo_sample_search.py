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
This script compares the resource usage of different vector indices
indices for a given dataset.

It measures the memory usage and time taken for training,
adding, and searching with each index.
"""

from utils.search import get_search_closest_labels
from utils.vector_indices import get_name, get_default_indices, is_ivf
from utils.example_embeddings import get_example_embeddings
from utils.embeddings_container import EmbeddingsContainer
import numpy as np
from utils.common import log_info
import random

import logging

logger = logging.getLogger(__name__)

logger.info(f"Starting {__file__}...")
log_info()


logger.info("Loading dataset...")
text, data = get_example_embeddings()
embeddings = np.asarray(data)
embeddings_train = embeddings[1::2]
embeddings_add = embeddings[::2]
i = random.randint(0, len(embeddings_add) - 1)
search_str = text[i]

container = EmbeddingsContainer(
    full=embeddings,
    train=embeddings_train,
    add=embeddings_add,
    search=np.atleast_2d(embeddings_add[i]),
)

logger.info(f"Loaded {container}")

n, d = container.add.shape
indices = [index for index in get_default_indices(n=n, d=d) if not is_ivf(index)]

logger.info(f"Searching for: \n'{search_str}'")

for index in indices:
    index.verbose = False

    k = 5
    closest_labels = get_search_closest_labels(index, container, k)[0]
    matches = str.join("\n", [f"'{text[label]}'" for label in closest_labels])
    logger.info(f"[{get_name(index)}] Top {k} Matches:\n'{matches}'")
