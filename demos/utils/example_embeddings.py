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

import os
import os.path
import numpy as np
import pandas as pd
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_example_embeddings(
    path: str = "qiaojin/PubMedQA",
    name: str = "pqa_artificial",
    feature_name: str = "long_answer",
    num_texts: int = 20000,
    cache_dir: str = "./dataset_cache",
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[str], pd.DataFrame]:
    cache_dir = os.path.abspath(cache_dir)

    logger.info(
        f"Loading dataset={path}, "
        f"name={name}, "
        f"feature_name={feature_name}, "
        f"num_texts={num_texts}, "
        f"cache_dir={cache_dir}..."
    )

    os.makedirs(cache_dir, exist_ok=True)

    ds = load_dataset(path, name, split=f"train[:{num_texts}]", cache_dir=cache_dir)

    texts = ds[feature_name]

    sanitized_path = path.replace("/", "_")
    normalize = True
    cache_name = f"embeddings_{sanitized_path}_{num_texts}_normalize_{normalize}.npy"
    emb_cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(emb_cache_path):
        logger.info(f"Loading embeddings from cache: {emb_cache_path}")
        embeddings = np.load(emb_cache_path)
    else:
        logger.info(f"Computing embeddings with {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=normalize)
        np.save(emb_cache_path, embeddings)
        logger.info(f"Saved embeddings to {emb_cache_path}")

    return texts, pd.DataFrame(embeddings)


if __name__ == "__main__":
    texts, df = get_example_embeddings()

    print(f"Texts head: {texts[:10]}")
    print(f"Embeddings head: {df.head()}")
    print(f"Embeddings shape: {df.shape}")
