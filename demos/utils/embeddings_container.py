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


class EmbeddingsContainer:
    def __init__(self, full: np.ndarray, train: np.ndarray, add: np.ndarray, search: np.ndarray):
        self.full = full
        self.train = train
        self.add = add
        self.search = search
        self.d = full.shape[1]

        assert all(
            arr.shape[1] == self.d for arr in [train, add, search]
        ), "All embeddings must have the same dimensionality"

    def __repr__(self):
        return (
            f"Embeddings(full={self.full.shape}, "
            f"train={self.train.shape}, "
            f"add={self.add.shape}, "
            f"search={self.search.shape})"
        )
