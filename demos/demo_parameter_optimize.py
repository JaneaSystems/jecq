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
This script optimizes hyper-parameters for Jecq indices against a data-set
using a genetic algorithm. Currently, the fitness function for a parameter
set is computed as:
`(search_accuracy ** 3) / mem_usage_ratio`

This script may take several hours to complete depending on various factors;
you can adjust the `population_size` and `generations` variables below to
control the optimization duration.
"""

from utils.genetic_algorithm import run_ga
from utils.params_evaluate import get_params_fitness, get_optimal_pq_multiplier

from utils.search import get_ref_search_closest_labels, search_k_default as k, use_forgiving_metric

from utils.example_embeddings import get_example_embeddings
from utils.stats import get_feature_variances
from utils.embeddings_container import EmbeddingsContainer
import pandas as pd
import numpy as np
from utils.common import log_info, df_to_str, df_unpack_tuple_column
import logging

logger = logging.getLogger(__name__)

logger.info(f"Starting {__file__}...")
log_info()

pd.set_option("display.float_format", lambda x: "%.5f" % x)


logger.info("Loading dataset...")
_, data = get_example_embeddings()
embeddings_full = np.asarray(data)

container = EmbeddingsContainer(
    full=embeddings_full,
    add=embeddings_full,
    train=embeddings_full,
    search=embeddings_full[0::4].copy(),
)

logger.info(f"Loaded {container}")

logger.info("Creating reference index results...")
ref_closest = get_ref_search_closest_labels(container, k)

logger.info("Calculating feature variances...")
feature_variances = get_feature_variances(container)
logger.debug(f"Feature variances: \n{pd.DataFrame(feature_variances).describe()}")


def variance_percentile(p: float) -> float:
    return float(np.percentile(feature_variances, p))


logger.info("Calculating pq_multiplier_bounds...")
sample_th_high = variance_percentile(35)
sample_th_mid = variance_percentile(20)
opt_pq_multiplier = get_optimal_pq_multiplier(
    container, k, ref_closest, th_high=sample_th_high, th_mid=sample_th_mid
)
pq_multiplier_bounds = (opt_pq_multiplier / 6, opt_pq_multiplier * 3)
logger.info(
    f"Optimal pq_multiplier: {opt_pq_multiplier:.2f} "
    f"with sample_th_high={sample_th_high:.4f}, "
    f"sample_th_mid={sample_th_mid:.4f}, "
    f"using pq_multiplier_bounds={pq_multiplier_bounds}"
)

# GA parameters
population_size = 40
generations = 10

logger.info(f"Optimizing all parameters with " f"use_forgiving_metric={use_forgiving_metric}...")

mutation_rate = 0.1
crossover_rate = 0.8

th_centre = variance_percentile(25)
th_high_bounds = (th_centre, variance_percentile(95))
th_mid_bounds = (variance_percentile(5), th_centre)
bounds = [pq_multiplier_bounds, th_high_bounds, th_mid_bounds]


def evaluate_fn(params):
    return get_params_fitness(
        container=container,
        k=k,
        ref_closest_labels=ref_closest,
        params=params,
        search_accuracy_fitness=False,
    )


all_individuals = run_ga(
    population_size=population_size,
    generations=generations,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    bounds=bounds,
    evaluate_fn=evaluate_fn,
)

df_results = pd.DataFrame(all_individuals, columns=["params", "score", "extra_data"])

df_results = df_unpack_tuple_column(df_results, "params", ["pq_multiplier", "th_high", "th_mid"])

extra_data_columns = [
    "search_accuracy",
    "mem_usage_ratio",
    "pq_ratio",
    "itq_ratio",
    "discarded_ratio",
]
df_results = df_unpack_tuple_column(df_results, "extra_data", extra_data_columns)

df_results.drop_duplicates(subset=["pq_multiplier", "th_high", "th_mid"], inplace=True)
df_results.sort_values(by="score", ascending=False, inplace=True)

logger.info(f"All params by fitness:\n{df_to_str(df_results)}")
logger.info(f"Top params by fitness:\n{df_to_str(df_results.head(10))}")

df_top_accuracy = df_results.sort_values(by="search_accuracy", ascending=False)
logger.info(f"Top params by search_accuracy:\n{df_to_str(df_top_accuracy.head(10))}")
