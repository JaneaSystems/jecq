// Copyright (c) 2025 Janea Systems
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "index_jecq.h"
#include "feature_classifier.h"

#include <faiss/utils/utils.h>

#include <cinttypes>
#include <memory>
#include <numeric>
#include <queue>

namespace {

void fill_with_labels(faiss::idx_t ntotal, std::vector<faiss::idx_t>* vec) {
    assert(ntotal == 0 ? vec->empty() : (vec->size() % ntotal == 0));

    for (faiss::idx_t i = 0; i < vec->size(); ++i) {
        (*vec)[i] = i % ntotal;
    }
}
} // namespace

namespace jecq {
std::vector<float> IndexJecq::get_pq_vector(faiss::idx_t n, const float* x)
        const {
    return get_filtered_features(n, this->d, x, this->pq_features);
}

std::vector<float> IndexJecq::get_itq_vector(faiss::idx_t n, const float* x)
        const {
    return get_filtered_features(n, this->d, x, this->itq_features);
}

IndexJecq::IndexJecq() : IndexJecq(0, 10.0, 0.05, 0.005) {}

IndexJecq::IndexJecq(
        faiss::idx_t d,
        float pq_multiplier,
        float th_high,
        float th_mid)
        : IndexJecqBase(pq_multiplier, th_high, th_mid),
          Index(d, faiss::MetricType::METRIC_INNER_PRODUCT) {
    this->is_trained = false;
}

void IndexJecq::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    ntotal += n;

    const auto t0 = faiss::getmillisecs();

    if (!pq_features.empty()) {
        if (this->index_pq.codes.is_owned) {
            this->index_pq.codes.owned_data.reserve(
                    ntotal * this->index_pq.code_size);
        }

        const faiss::idx_t max_bs = faiss::product_quantizer_compute_codes_bs;
        const auto usual_bs = std::min(max_bs, n);

        std::vector<float> pq_data(pq_features.size() * usual_bs);

        for (faiss::idx_t i = 0; i < n; i += usual_bs) {
            const auto actual_bs = std::min(usual_bs, n - i);

            filter_by_features(
                    actual_bs,
                    d,
                    x + this->d * i,
                    this->pq_features,
                    pq_data.data());

            this->index_pq.add(actual_bs, pq_data.data());
        }
    }

    const auto t1 = faiss::getmillisecs();

    if (!itq_features.empty()) {
        if (this->index_itq.codes.is_owned) {
            this->index_itq.codes.owned_data.reserve(
                    ntotal * this->index_itq.code_size);
        }

        std::vector<float> itq_row(itq_features.size());

        for (faiss::idx_t i = 0; i < n; ++i) {
            filter_by_features(
                    x + this->d * i, this->itq_features, itq_row.data());
            this->index_itq.add(1, itq_row.data());
        }
    }

    const auto t2 = faiss::getmillisecs();

    if (verbose && n > 0) {
        printf("Adding IndexJecq complete; total_ms = %.1f, pq_ms=%.1f, itq_ms=%.1f\n",
               t2 - t0,
               t1 - t0,
               t2 - t1);
    }
}

void IndexJecq::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    std::vector<float> pq_distances(n * index_pq.ntotal, 0.0);
    std::vector<faiss::idx_t> pq_labels(n * index_pq.ntotal, -1);

    if (index_pq.ntotal > 0) {
        const auto pq_vector = get_pq_vector(n, x);
        index_pq.search(
                n,
                pq_vector.data(),
                index_pq.ntotal,
                pq_distances.data(),
                pq_labels.data(),
                nullptr);
    } else {
        fill_with_labels(this->ntotal, &pq_labels);
    }

    std::vector<float> itq_distances(n * index_itq.ntotal, 0.0);
    std::vector<faiss::idx_t> itq_labels(n * index_itq.ntotal, -1);

    if (index_itq.ntotal > 0) {
        const auto itq_vector = get_itq_vector(n, x);
        index_itq.search(
                n,
                itq_vector.data(),
                index_itq.ntotal,
                itq_distances.data(),
                itq_labels.data(),
                nullptr);
    } else {
        fill_with_labels(this->ntotal, &itq_labels);
    }

    std::vector<float> total_distances(this->ntotal, 0.0);

    const auto distance_comparer = [&total_distances](
                                           faiss::idx_t a, faiss::idx_t b) {
        return total_distances[b] < total_distances[a];
    };

    std::priority_queue<
            faiss::idx_t,
            std::vector<faiss::idx_t>,
            decltype(distance_comparer)>
            best_labels(distance_comparer);

    for (faiss::idx_t i = 0; i < n; ++i) {
        std::fill(total_distances.begin(), total_distances.end(), 0.0);
        assert(best_labels.empty());

        for (faiss::idx_t j = 0; j < this->index_pq.ntotal; ++j) {
            const auto pq_index = this->index_pq.ntotal * i + j;

            auto& total_distance = total_distances[pq_labels[pq_index]];

            assert(total_distance == 0.0);

            total_distance = pq_distances[pq_index] * this->pq_multiplier;
        }

        for (faiss::idx_t j = 0; j < this->index_itq.ntotal; ++j) {
            const auto itq_index = this->index_itq.ntotal * i + j;
            total_distances[itq_labels[itq_index]] += itq_distances[itq_index];
        }

        for (faiss::idx_t t = 0; t < this->ntotal; ++t) {
            if (best_labels.size() < k) {
                best_labels.push(t);
            }

            else if (distance_comparer(t, best_labels.top())) {
                best_labels.pop();
                best_labels.push(t);
            }
        }

        const auto available_count = best_labels.size();

        for (faiss::idx_t t = 0; t < available_count; ++t) {
            const auto top = best_labels.top();
            const auto label_index = k * i + available_count - 1 - t;

            labels[label_index] = top;
            distances[label_index] = total_distances[top];
            best_labels.pop();
        }

        for (faiss::idx_t t = available_count; t < k; ++t) {
            labels[k * i + t] = -1;
            distances[k * i + t] = -std::numeric_limits<float>::infinity();
        }
    }
}

void IndexJecq::reset() {
    index_pq.reset();
    index_itq.reset();
}

void IndexJecq::train(faiss::idx_t n, const float* x) {
    const auto t0 = faiss::getmillisecs();

    if (verbose) {
        printf("Training IndexJecq with %d features on %" PRId64 " vectors\n ",
               this->d,
               n);
    }

    if (this->reclassify_features_when_training) {
        this->reclassify_features(n, x);
    }

    if (verbose) {
        printf("Classified IndexJecq features; pq_features.size()=%zu, itq_features.size()=%zu, discarded_features.size()=%zu\n",
               pq_features.size(),
               itq_features.size(),
               d - (pq_features.size() + itq_features.size()));
    }

    const auto t1 = faiss::getmillisecs();

    if (!pq_features.empty()) {
        const int pq_nbits = 8;
        index_pq = faiss::IndexPQ(
                pq_features.size(),
                pq_features.size(),
                pq_nbits,
                this->metric_type);

        const auto pq_data = get_pq_vector(n, x);
        index_pq.train(n, pq_data.data());
    } else {
        index_pq = faiss::IndexPQ();
    }

    const auto t2 = faiss::getmillisecs();

    if (!itq_features.empty()) {
        index_itq = IndexITQFlat(itq_features.size());

        const auto itq_data = get_itq_vector(n, x);
        index_itq.train(n, itq_data.data());
    } else {
        index_itq = IndexITQFlat();
    }

    const auto t3 = faiss::getmillisecs();

    this->is_trained = true;

    if (verbose) {
        printf("Training IndexJecq complete; total_ms = %.1f, classification_ms=%.1f, pq_ms=%.1f, itq_ms=%.1f\n",
               t3 - t0,
               t1 - t0,
               t2 - t1,
               t3 - t2);
    }
}

} // namespace jecq