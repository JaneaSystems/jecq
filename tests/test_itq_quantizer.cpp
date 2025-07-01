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
#include "utils.h"

#include <jecq/index_jecq.h>
#include <jecq/itq_quantizer.h>

#include <gtest/gtest.h>

namespace {

std::vector<float> get_itq_dataset(size_t db_size) {
    std::vector<float> xdb(db_size);

    for (int i = 0; i < db_size; ++i) {
        xdb[i] = (i < db_size / 2) ? 0.0 : 10000.0;
    }

    return xdb;
}

uint8_t compute_byte(const jecq::ITQQuantizer& itq, float x) {
    uint8_t byte;
    itq.compute_codes(&x, &byte, 1);

    return byte;
}

std::vector<uint8_t> compute_codes(
        const jecq::ITQQuantizer& itq,
        const std::vector<float>& x) {
    std::vector<uint8_t> output(itq.code_size * x.size());

    itq.compute_codes(x.data(), output.data(), x.size());
    return output;
}

} // namespace

namespace jecq_test {

TEST(TestItqQuantizer, TestOneDimensionEncodesCorrectly) {
    const size_t db_size = 1000;

    jecq::ITQQuantizer itq(1);

    const auto xdb = get_itq_dataset(db_size);

    itq.train(db_size, xdb.data());

    const uint8_t first_encoded = compute_byte(itq, xdb[0]);
    const uint8_t last_encoded = compute_byte(itq, xdb.back());

    for (int i = 0; i < db_size; ++i) {
        const auto reference = (i < db_size / 2) ? first_encoded : last_encoded;

        EXPECT_EQ(reference, compute_byte(itq, xdb[i])) << "failed for i=" << i;
    }
}

TEST(TestItqQuantizer,
     TestOneDimensionSingleMultipleEncodeWorksLikeMultipleSingleEncodes) {
    const size_t db_size = 1000;

    jecq::ITQQuantizer itq(1);

    const auto xdb = get_itq_dataset(db_size);

    itq.train(db_size, xdb.data());

    const auto multi_encoded = compute_codes(itq, xdb);

    for (int i = 0; i < xdb.size(); ++i) {
        const auto byte = compute_byte(itq, xdb[i]);
        EXPECT_EQ(byte, multi_encoded[i]);
    }
}

} // namespace jecq_test