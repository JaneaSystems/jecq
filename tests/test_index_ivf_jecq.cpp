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
#include "index_factory.h"
#include "index_helpers.h"
#include "utils.h"

#include <jecq/index_ivf_jecq.h>
#include <jecq/index_jecq.h>

#include <gtest/gtest.h>

namespace jecq_test {

TEST(TestIVFJecq, TestCompareWithIndexJecq) {
    const int d = DEFAULT_DIMENSIONS;
    const int db_size = DEFAULT_DB_SIZE;

    auto index1 = create_index_jecq(
            IndexType::IndexJecq, DEFAULT_DIMENSIONS, 1, 0.05, 0.005, true);
    auto index2 = create_index_jecq(
            IndexType::IndexIVFJecq, DEFAULT_DIMENSIONS, 1, 0.05, 0.005, true);

    const std::vector<jecq::IndexJecqBase*> indices = {
            index1.get(), index2.get()};
    const auto xdb = get_standard_dataset(db_size, d);

    for (auto* index : indices) {
        index->reclassify_features_when_training = false;
        index->pq_features = {0, 1, 2};
        index->itq_features = {3, 5};

        auto& faiss_index = index->as_faiss_index();
        faiss_index.verbose = true;
        train(&faiss_index, xdb);
        add(&faiss_index, xdb);
    }

    const auto xq = get_standard_query(xdb, d);

    std::vector<std::vector<float>> sorted_distance_list;

    for (auto* index : indices) {
        const faiss::idx_t k = db_size;
        const auto [distances, labels] = search(index->as_faiss_index(), xq, k);

        std::vector<float> sorted_distances(distances.size());

        for (int i = 0; i < labels.size(); ++i) {
            sorted_distances[labels[i]] = distances[i];
        }
        sorted_distance_list.push_back(sorted_distances);
    }

    assert(sorted_distance_list.size() == 2);

    EXPECT_EQ(sorted_distance_list[0].size(), sorted_distance_list[1].size());

    for (int i = 0; i < sorted_distance_list[0].size(); ++i) {
        EXPECT_EQ(sorted_distance_list[0][i], sorted_distance_list[1][i])
                << "Label = " << i;
    }
}
} // namespace jecq_test