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

import random
import logging
from typing import Any
import time

from utils.models import BoundsList, Params

logger = logging.getLogger(__name__)


def _create_individual(bounds: BoundsList) -> Params:
    return [random.uniform(lb, ub) for lb, ub in bounds]


def _tournament_selection(population: list[Params], k: int, evaluate_fn):
    selected = random.sample(population, min(k, len(population)))
    selected.sort(key=evaluate_fn, reverse=True)
    return selected[0]


def _crossover(parent1: Params, parent2: Params) -> tuple[Params, Params]:
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i], child2[i] = parent2[i], parent1[i]
    return child1, child2


def _mutate(individual: Params, mutation_rate: float, bounds: BoundsList) -> Params:
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            lb, ub = bounds[i]
            individual[i] = random.uniform(lb, ub)
    return individual


def run_ga(
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    bounds: BoundsList,
    evaluate_fn,
) -> list[tuple[Params, float, Any]]:
    logger.info(
        f"Running genetic algorithm with "
        f"population_size={population_size}, "
        f"generations={generations}, "
        f"mutation_rate={mutation_rate}, "
        f"crossover_rate={crossover_rate}, "
        f"bounds={bounds}..."
    )

    best_individual = None
    best_score = float("-inf")
    all_individuals = []
    seen_params = set()

    # Initialize the population.
    population = [_create_individual(bounds) for _ in range(population_size)]

    for gen in range(generations):
        gen_start_time = time.perf_counter()
        new_population: list[Params] = []
        while len(new_population) < population_size:
            # Select two parents.
            parent1 = _tournament_selection(population, k=3, evaluate_fn=evaluate_fn)
            parent2 = _tournament_selection(population, k=3, evaluate_fn=evaluate_fn)
            # Crossover.
            if random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Mutation.
            child1 = _mutate(child1, mutation_rate, bounds)
            child2 = _mutate(child2, mutation_rate, bounds)
            new_population.extend([child1, child2])
        # Ensure the new population size is maintained.
        population = new_population[:population_size]

        # Evaluate population and track the best individual.
        for individual in population:
            params_str = str(individual)
            if params_str in seen_params:
                # Skip if this individual has already been evaluated.
                continue

            seen_params.add(params_str)

            score, extra_data = evaluate_fn(individual)
            all_individuals.append((individual, score, extra_data))
            if score > best_score:
                best_score = score
                best_individual = individual

        gen_end_time = time.perf_counter()
        gen_duration_mins = (gen_end_time - gen_start_time) / 60
        time_left_mins = (generations - gen - 1) * gen_duration_mins

        logger.info(
            f"Generation completed; "
            f"generation={gen}, "
            f"best_score={best_score:.4f}, "
            f"best_individual={best_individual}, "
            f"gen_duration_mins={gen_duration_mins:.2f}, "
            f"time_left_mins={time_left_mins:.2f}"
        )

    return all_individuals
