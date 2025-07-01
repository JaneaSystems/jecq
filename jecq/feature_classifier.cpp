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

#include "feature_classifier.h"

#include <faiss/VectorTransform.h>

#include <cassert>

namespace {
std::vector<float> get_variances(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x) {
    // Build PCA transformer that keeps all components (no whitening, no
    // random rotation)
    faiss::PCAMatrix pca(d, d, /*eigen_power=*/0.0f, /*random_rotation=*/false);

    pca.train(n, x);

    return pca.eigenvalues;
}

} // namespace

namespace jecq {

void classify_features(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x,
        float th_high,
        float th_mid,
        std::vector<faiss::idx_t>* high_var_features,
        std::vector<faiss::idx_t>* mid_var_features,
        std::vector<float>* feature_variances) {
    if (n <= 1) {
        return;
    }

    assert(x != nullptr);
    assert(high_var_features != nullptr);
    assert(mid_var_features != nullptr);
    assert(feature_variances != nullptr);

    *feature_variances = get_variances(n, d, x);

    for (faiss::idx_t i = 0; i < d; ++i) {
        const auto scaled_var = (*feature_variances)[i] / (n - 1);
        (*feature_variances)[i] = scaled_var;

        if (scaled_var > th_high) {
            high_var_features->push_back(i);
        } else if (scaled_var > th_mid) {
            mid_var_features->push_back(i);
        }
    }
}
std::vector<float> get_filtered_features(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x,
        const std::vector<faiss::idx_t>& features) {
    std::vector<float> result;
    result.resize(features.size() * n);
    filter_by_features(n, d, x, features, result.data());
    return result;
}

void filter_by_features(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x,
        const std::vector<faiss::idx_t>& features,
        float* output) {
    for (faiss::idx_t i = 0; i < n; ++i) {
        const float* row = x + d * i;
        for (const auto feature : features) {
            *output = row[feature];
            ++output;
        }
    }
}

void filter_by_features(
        const float* x,
        const std::vector<faiss::idx_t>& features,
        float* output) {
    for (const auto feature : features) {
        *output = x[feature];
        ++output;
    }
}

} // namespace jecq
