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

#include <algorithm>
#include <limits>
#include <memory>

namespace jecq_test {

class TestIndexCommonTestFixture : public ::testing::TestWithParam<IndexType> {
};

INSTANTIATE_TEST_CASE_P(
        TestIndexCommon,
        TestIndexCommonTestFixture,
        ::testing::Values(IndexType::IndexJecq, IndexType::IndexIVFJecq),
        ::testing::PrintToStringParamName());

TEST_P(TestIndexCommonTestFixture, TestBadPQMultiplier) {
    EXPECT_ANY_THROW(create_index_jecq(GetParam(), 6, -1, 5, 4));
    EXPECT_ANY_THROW(create_index_jecq(GetParam(), 6, 0, 5, 4));
}

TEST_P(TestIndexCommonTestFixture, TestBadThresholds) {
    const auto index_type = GetParam();
    EXPECT_ANY_THROW(create_index_jecq(index_type, 6, 10, 5, -1));
    EXPECT_ANY_THROW(create_index_jecq(index_type, 6, 10, 5, 0));
    EXPECT_ANY_THROW(create_index_jecq(index_type, 6, 10, 5, 5));
    EXPECT_ANY_THROW(create_index_jecq(index_type, 6, 10, 4, 5));
}

TEST_P(TestIndexCommonTestFixture, TestAddWithoutTrainFails) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    EXPECT_FALSE(faiss_index.is_trained);

    const auto xdb = get_standard_dataset();
    EXPECT_ANY_THROW(faiss_index.add(xdb.size(), xdb.data()));
}

TEST_P(TestIndexCommonTestFixture, TestAdd) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    train(&faiss_index, get_standard_dataset());

    const auto n = 5;
    const auto data = random_vector_float(n * faiss_index.d);

    faiss::idx_t ntotal = 0;
    for (int i = 0; i < n; ++i) {
        faiss_index.add(i + 1, data.data());
        ntotal += (i + 1);

        EXPECT_EQ(ntotal, faiss_index.ntotal);
    }
}

TEST_P(TestIndexCommonTestFixture, TestReclassifyFeaturesOnByDefault) {
    EXPECT_TRUE(
            create_index_jecq(GetParam())->reclassify_features_when_training);
}

TEST_P(TestIndexCommonTestFixture, TestReclassifyFeaturesOffDoesNotReclassify) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    const std::vector<faiss::idx_t> pq_features{3, 4};
    const std::vector<faiss::idx_t> itq_features{1, 5};

    index.reclassify_features_when_training = false;
    index.pq_features = pq_features;
    index.itq_features = itq_features;

    train(&faiss_index, get_standard_dataset());

    EXPECT_EQ(pq_features, index.pq_features);
    EXPECT_EQ(itq_features, index.itq_features);
}

TEST_P(TestIndexCommonTestFixture, TestReclassifyFeaturesOnReclassifies) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    const std::vector<faiss::idx_t> pq_features{3, 4};
    const std::vector<faiss::idx_t> itq_features{1, 5};

    index.reclassify_features_when_training = true;
    index.pq_features = pq_features;
    index.itq_features = itq_features;

    train(&faiss_index, get_standard_dataset());

    EXPECT_NE(pq_features, index.pq_features);
    EXPECT_NE(itq_features, index.itq_features);
    EXPECT_NE(std::vector<float>(faiss_index.d, 0.0f), index.feature_variances);
}

TEST_P(TestIndexCommonTestFixture,
       TestTrainIsIdempotentWithRespectToFeatureClassification) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    const auto xdb = get_standard_dataset();

    train(&faiss_index, xdb);
    const auto pq_features = index.pq_features;
    const auto itq_features = index.itq_features;

    train(&faiss_index, xdb);
    EXPECT_EQ(pq_features, index.pq_features);
    EXPECT_EQ(itq_features, index.itq_features);
}

TEST_P(TestIndexCommonTestFixture, TestSearchWhenEmpty) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    train(&faiss_index, get_standard_dataset());

    const auto k = 5;
    const auto nq = 3;
    const auto [distances, labels] =
            search(faiss_index, random_vector_float(nq * faiss_index.d), k);

    EXPECT_TRUE(std::all_of(distances.begin(), distances.end(), [](float d) {
        return d < -10000;
    }));

    EXPECT_EQ(std::vector<faiss::idx_t>(nq * k, -1), labels);
}

TEST_P(TestIndexCommonTestFixture, TestSearchMoreThanDBSize) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    const auto n = 4;
    train(&faiss_index, get_standard_dataset());
    add(&faiss_index, random_vector_float(n * faiss_index.d));

    const auto k = n + 2;
    const auto nq = 3;
    const auto [distances, labels] =
            search(faiss_index, random_vector_float(n * faiss_index.d), k);

    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < k; ++j) {
            const auto offset = i * k + j;
            if (j < n) {
                EXPECT_GT(
                        distances[offset], -std::numeric_limits<float>::max());
                EXPECT_NE(-1, labels[offset]);
            } else {
                EXPECT_LE(distances[offset], -10000.0);
                EXPECT_EQ(-1, labels[offset]);
            }
        }
    }
}

TEST_P(TestIndexCommonTestFixture,
       TestSearchMustBeLikePQWhenFlatAndAllFeaturesUsePQ) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr = create_index_jecq(
            GetParam(), DEFAULT_DIMENSIONS, 1, 0.05, 0.005, true);

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    const auto d = faiss_index.d;

    faiss::IndexPQ index_ref(d, d, 8, faiss_index.metric_type);

    index.reclassify_features_when_training = false;
    for (int i = 0; i < faiss_index.d; ++i) {
        index.pq_features.push_back(i);
    }

    const auto train_data = random_vector_float(10000 * d);
    train(&index_ref, train_data);
    train(&faiss_index, train_data);

    const auto xdb = random_vector_float(20000 * d);
    add(&index_ref, xdb);
    add(&faiss_index, xdb);

    const auto sdb = 500;
    const auto xs = random_vector_float(sdb * d);

    for (int i = 0; i < sdb; ++i) {
        const auto k = 5;

        const auto xq = get_row(train_data, d, i);

        const auto [distances_ref, labels_ref] = search(index_ref, xq, k);
        const auto [distances_idx, labels_idx] = search(faiss_index, xq, k);

        EXPECT_EQ(labels_ref, labels_idx);

        EXPECT_EQ(distances_ref.size(), distances_idx.size());
        for (size_t i = 0; i < distances_ref.size(); ++i) {
            EXPECT_FLOAT_EQ(distances_ref[i], distances_idx[i])
                    << "mismatch at index " << i
                    << " expected: " << distances_ref[i]
                    << " actual: " << distances_idx[i];
        }
    }
}

TEST_P(TestIndexCommonTestFixture, TestSearchComplex) {
    std::unique_ptr<jecq::IndexJecqBase> index_ptr =
            create_index_jecq(GetParam());

    auto& index = *index_ptr;
    auto& faiss_index = index.as_faiss_index();

    index.reclassify_features_when_training = false;
    index.pq_features = {0, 1, 2};
    index.itq_features = {3, 5};

    const auto xdb = get_standard_dataset();
    train(&faiss_index, xdb);
    add(&faiss_index, xdb);
    verify_with_standard_query(xdb, faiss_index);
}

} // namespace jecq_test