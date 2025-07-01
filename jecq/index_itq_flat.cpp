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

#include "index_itq_flat.h"
#include "itq_quantizer.h"

#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>

namespace jecq {

IndexITQFlat::IndexITQFlat(faiss::idx_t d, int itq_iters)
        : IndexFlatCodes(ITQQuantizer::get_code_size(d), d),
          itq(d, itq_iters) {}

IndexITQFlat::IndexITQFlat() : IndexITQFlat(0, 50) {}

void IndexITQFlat::train(faiss::idx_t n, const float* x) {
    this->itq.train(n, x);
    is_trained = true;
}

inline void IndexITQFlat::sa_encode(
        faiss::idx_t n,
        const float* x,
        uint8_t* bytes) const {
    this->itq.compute_codes(x, bytes, n);
}

void IndexITQFlat::sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x)
        const {
    this->itq.decode(bytes, x, n);
}

void IndexITQFlat::search(
        faiss::idx_t n,
        const float* xq,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "ITQ must be trained before search");

    std::vector<uint8_t> q_codes(n * code_size);
    sa_encode(n, xq, q_codes.data());

    std::vector<int> hamming_distances(n * k);

    faiss::int_maxheap_array_t res = {
            size_t(n), size_t(k), labels, hamming_distances.data()};

    hammings_knn_hc(
            &res, q_codes.data(), codes.data(), ntotal, code_size, true);

    for (faiss::idx_t i = 0; i < hamming_distances.size(); ++i) {
        distances[i] =
                this->itq.get_inner_product_distance(hamming_distances[i]);
    }
}

void IndexITQFlat::reset() {
    codes.clear();
    ntotal = 0;
}

} // namespace jecq