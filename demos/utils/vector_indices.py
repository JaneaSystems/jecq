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

# mypy: disable-error-code="attr-defined"

from utils.models import Index
import faiss
import jecq

import math
import logging

logger = logging.getLogger(__name__)


def get_name(index: Index) -> str:
    return type(index).__name__


def is_jecq_index(index: Index) -> bool:
    module_name = type(index).__module__
    return "jecq" in module_name or "jecq" in get_name(index).lower()


def get_default_indices(n, d) -> list[Index]:
    # Values are from running demos/demo_parameter_optimization.py
    pq_multiplier = 625.823
    th_high = 0.0079425
    th_mid = 8.56425e-05
    nlist = math.ceil(math.sqrt(n))
    nprobe_factor = 0.1
    nprobe = math.ceil(nprobe_factor * nlist)

    return get_indices(
        d=d, pq_multiplier=pq_multiplier, th_high=th_high, th_mid=th_mid, nlist=nlist, nprobe=nprobe
    )


def get_indices(
    d, pq_multiplier: float, th_high: float, th_mid: float, nlist: int = 1, nprobe: int = 1
) -> list[Index]:
    metric = faiss.METRIC_INNER_PRODUCT

    indices = [
        faiss.IndexFlat(d, metric),
        faiss.IndexPQ(d, d, 8, metric),
        faiss.IndexIVFPQ(faiss.IndexFlat(d, metric), d, nlist, d, 8, metric),
        jecq.IndexJecq(d, pq_multiplier, th_high, th_mid),
        jecq.IndexIVFJecq(d, nlist, pq_multiplier, th_high, th_mid),
    ]

    for index in indices:
        index.verbose = True

        if is_ivf(index):
            index.nprobe = nprobe

    index_names = [get_name(index) for index in indices]

    logger.info(f"Loading indices {index_names}...")
    logger.info(
        f"Parameters: pq_multiplier={pq_multiplier}, "
        f"th_high={th_high}, "
        f"th_mid={th_mid}, "
        f"nlist={nlist}, "
        f"nprobe={nprobe}"
    )

    return indices


def is_ivf(index: Index) -> bool:
    return "IVF" in get_name(index)


def validate_index(index: Index):
    if not is_jecq_index(index):
        return

    def validate_features(feature_ratio, threshold, name):
        if feature_ratio < threshold:
            message = (
                f"[{get_name(index)}] " f"Too few {name} features: {feature_ratio} < {threshold}"
            )
            logger.error(message)
            raise ValueError(message)

    pq_ratio, itq_ratio, discarded_ratio = get_feature_classification_ratios(index)

    validate_features(pq_ratio, 0.05, "PQ")
    validate_features(itq_ratio, 0.1, "ITQ")
    validate_features(discarded_ratio, 0.05, "Discarded")


def get_feature_classification_ratios(index: Index) -> tuple[float, float, float]:
    if not is_jecq_index(index):
        return (1.0, 0.0, 0.0)

    d = index.d

    if d <= 0:
        message = f"[{get_name(index)}] Invalid dimension count: {d}"
        logger.error(message)
        raise ValueError(message)

    pq_features = get_pq_features(index)
    itq_features = get_itq_features(index)
    discarded_features = set(range(d)).difference(pq_features + itq_features)

    pq_ratio = len(pq_features) / float(d)
    itq_ratio = len(itq_features) / float(d)
    discarded_ratio = len(discarded_features) / float(d)

    return pq_ratio, itq_ratio, discarded_ratio


def get_theoretical_mem_usage_ratio(index: Index) -> float:
    if not is_jecq_index(index):
        return 1

    d = index.d

    if d <= 0:
        message = f"[{get_name(index)}] Invalid dimension count: {d}"
        logger.error(message)
        raise ValueError(message)

    pq_features = len(get_pq_features(index))
    itq_features = len(get_itq_features(index))
    return (pq_features + math.ceil(itq_features / 8)) / d


def get_itq_features(index: Index) -> list[int]:
    return faiss.vector_to_array(index.itq_features).tolist()


def get_pq_features(index: Index) -> list[int]:
    return faiss.vector_to_array(index.pq_features).tolist()


def get_feature_variances(index: Index) -> list[float]:
    return faiss.vector_to_array(index.feature_variances).tolist()
