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

from utils.models import ClosestLabels, Params
from utils.vector_indices import get_feature_classification_ratios, get_theoretical_mem_usage_ratio
from utils.search import get_search_accuracy, get_common_counts, get_search_closest_labels
from utils.embeddings_container import EmbeddingsContainer
import jecq
from scipy.optimize import minimize_scalar
import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_params_fitness(
    container: EmbeddingsContainer,
    k: int,
    ref_closest_labels: ClosestLabels,
    params: Params,
    search_accuracy_fitness: bool,
) -> tuple[float, Any]:
    pq_multiplier, th_high, th_mid = params
    jecq_index = jecq.IndexJecq(container.d, pq_multiplier, th_high, th_mid)  # type: ignore[attr-defined] # noqa

    index_closest_labels = get_search_closest_labels(jecq_index, container, k)
    common_counts = get_common_counts(index_closest_labels, ref_closest_labels)
    search_accuracy = get_search_accuracy(common_counts, k)

    pq_ratio, itq_ratio, discarded_ratio = get_feature_classification_ratios(jecq_index)
    mem_usage_ratio = get_theoretical_mem_usage_ratio(jecq_index)
    extra_data = (search_accuracy, mem_usage_ratio, pq_ratio, itq_ratio, discarded_ratio)

    fitness = (
        search_accuracy if search_accuracy_fitness else (search_accuracy**3) / mem_usage_ratio
    )

    params_str = [f"{i:.5f}" for i in params]

    logger.info(
        f"Evaluated params: {params_str}, "
        f"fitness={fitness:.4f}, "
        f"search_accuracy={search_accuracy:.4f}, "
        f"mem_usage_ratio={mem_usage_ratio:.2f} "
        f"pq_ratio={pq_ratio:.2f}, "
        f"itq_ratio={itq_ratio:.2f}, "
        f"discarded_ratio={discarded_ratio:.2f}"
    )

    return fitness, extra_data


def get_optimal_pq_multiplier(
    container: EmbeddingsContainer,
    k: int,
    ref_closest_labels: ClosestLabels,
    th_high: float,
    th_mid: float,
) -> float:
    logger.info("Calculating optimal pq_multiplier using brent's method...")

    def evaluate_pq_multiplier(pq_multiplier):
        return -get_params_fitness(
            container,
            k,
            ref_closest_labels,
            [float(pq_multiplier), th_high, th_mid],
            search_accuracy_fitness=True,
        )[0]

    opt_result = minimize_scalar(
        fun=evaluate_pq_multiplier,
        method="brent",
        tol=0.01,
        bracket=(1, 1500),
        options={"maxiter": 20},
    )

    if not opt_result.success:
        message = f"pq_multiplier optimization failed: {opt_result.message}"
        logger.error(message)
        raise RuntimeError(message)

    return float(opt_result.x)
