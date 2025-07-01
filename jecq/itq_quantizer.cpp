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

#include "itq_quantizer.h"

#include <faiss/utils/hamming.h>
#include <faiss/utils/hamming_distance/common.h>

namespace jecq {
float ITQQuantizer::get_inner_product_distance(
        const uint8_t* a,
        const uint8_t* b) const {
    size_t hamming_distance = 0;

    for (size_t i = 0; i < this->code_size; ++i) {
        hamming_distance += faiss::popcount32(a[i] ^ b[i]);
    }

    return get_inner_product_distance(hamming_distance);
}

ITQQuantizer::ITQQuantizer(faiss::idx_t d, int itq_iters)
        : Quantizer(d, get_code_size(d)), itq_transform(d, d, true) {
    this->itq_transform.itq.max_iter = itq_iters;
}

ITQQuantizer::ITQQuantizer() : ITQQuantizer(0, 50) {}

void ITQQuantizer::train(size_t n, const float* x) {
    this->itq_transform.train(n, x);
}

void ITQQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    const int dim = static_cast<int>(this->d);
    std::vector<float> x_proj(n * dim);

    itq_transform.apply_noalloc(n, x, x_proj.data());
    faiss::fvecs2bitvecs(x_proj.data(), codes, dim, n);
}

void ITQQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
    std::vector<float> x_proj(n * this->code_size * 8);
    faiss::bitvecs2fvecs(code, x_proj.data(), this->code_size * 8, n);

    itq_transform.reverse_transform(n, x_proj.data(), x);
}
} // namespace jecq