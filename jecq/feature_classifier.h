/*
 * Copyright (c) 2025 Janea Systems
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <faiss/MetricType.h>
#include <vector>

namespace jecq {

void classify_features(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x,
        float th_high,
        float th_mid,
        std::vector<faiss::idx_t>* high_var_features,
        std::vector<faiss::idx_t>* mid_var_features,
        std::vector<float>* feature_variances);

std::vector<float> get_filtered_features(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x,
        const std::vector<faiss::idx_t>& features);

void filter_by_features(
        faiss::idx_t n,
        faiss::idx_t d,
        const float* x,
        const std::vector<faiss::idx_t>& features,
        float* output);

void filter_by_features(
        const float* x,
        const std::vector<faiss::idx_t>& features,
        float* output);
} // namespace jecq
