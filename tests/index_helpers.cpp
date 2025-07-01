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

#include "index_factory.h"

#include <cassert>

namespace jecq_test {

void train(faiss::Index* index, const std::vector<float>& data) {
    assert(index);
    assert(data.size() % index->d == 0);

    index->train(data.size() / index->d, data.data());
}

void add(faiss::Index* index, const std::vector<float>& data) {
    assert(index);
    assert(data.size() % index->d == 0);

    index->add(data.size() / index->d, data.data());
}

std::vector<uint8_t> encode(
        faiss::Index* index,
        const std::vector<float>& data) {
    assert(index);
    assert(data.size() % index->d == 0);

    const faiss::idx_t n = data.size() / index->d;

    std::vector<uint8_t> result(n * index->sa_code_size());

    index->sa_encode(n, data.data(), result.data());

    return result;
}

std::pair<std::vector<float>, std::vector<faiss::idx_t>> search(
        const faiss::Index& index,
        const std::vector<float>& sq,
        faiss::idx_t k) {
    const auto nq = sq.size() / index.d;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    index.search(nq, sq.data(), k, distances.data(), labels.data());

    return std::pair(distances, labels);
}

} // namespace jecq_test