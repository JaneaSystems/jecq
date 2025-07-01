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

#include "datasets.h"
#include "index_helpers.h"
#include "utils.h"

#include <faiss/MetricType.h>

#include <gtest/gtest.h>

#include <cassert>
#include <iostream>

namespace {
template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    if (!v.empty()) {
        out << '[';
        for (const auto& val : v) {
            out << val << ", ";
        }

        out << "\b\b]";
    }
    return out;
}
} // namespace

namespace jecq_test {

std::vector<float> get_standard_dataset(int db_size, int d) {
    std::vector<float> xdb;
    xdb.reserve(db_size);

    for (int i = 0; i < db_size; i++) {
        for (int j = 0; j < d; j++) {
            xdb.push_back(i == 2 ? j : (i == 1 || i == 3) ? 1 : -i * j);
        }
    }

    return xdb;
}

std::vector<float> get_standard_query(const std::vector<float>& xdb, int d) {
    auto xq = get_row(xdb, d, 2);
    auto xq_perturbed = xq;
    perturb(&xq_perturbed);

    xq.insert(xq.end(), xq_perturbed.begin(), xq_perturbed.end());
    normalize(&xq, d);

    return xq;
}
void verify_with_standard_query(
        const std::vector<float>& xdb,
        const faiss::Index& index) {
    const auto d = index.d;
    const auto db_size = xdb.size() / d;

    const auto xq = get_standard_query(xdb, d);
    const faiss::idx_t nq = xq.size() / d;
    const faiss::idx_t k = 3;

    const auto [distances, labels] = search(index, xq, k);

#if _DEBUG
    std::cout << "Labels: " << labels << std::endl;
    std::cout << "Distances: " << distances << std::endl;
#endif

    const std::vector<faiss::idx_t> expected_labels1{2, 1, 3};
    const std::vector<faiss::idx_t> expected_labels2{2, 3, 1};

    for (int i = 0; i < nq; ++i) {
        const auto begin = labels.begin() + i * k;

        const std::vector<faiss::idx_t> labels_subset(
                begin, begin + expected_labels1.size());

        EXPECT_TRUE(
                expected_labels1 == labels_subset ||
                expected_labels2 == labels_subset)
                << "Actual labels: " << ::testing::PrintToString(labels_subset)
                << "\nExpect labels: "
                << ::testing::PrintToString(expected_labels1) << " OR "
                << ::testing::PrintToString(expected_labels2);
    }
}
} // namespace jecq_test