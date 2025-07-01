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

#include <faiss/VectorTransform.h>
#include <faiss/impl/Quantizer.h>

namespace jecq {
class ITQQuantizer : public faiss::Quantizer {
   private:
    faiss::ITQTransform itq_transform;

   public:
    static constexpr size_t get_code_size(faiss::idx_t d) {
        return (d == 0) ? 0 : ((d + 7) >> 3);
    }

    explicit ITQQuantizer(faiss::idx_t d, int itq_iters = 50);
    explicit ITQQuantizer();

    float get_inner_product_distance(const uint8_t* a, const uint8_t* b) const;

    template <class THamming>
    float get_inner_product_distance(THamming hamming_distance) const {
        return this->d - static_cast<float>(2 * hamming_distance);
    }

    /** Train the quantizer
     *
     * @param x       training vectors, size n * d
     */
    void train(size_t n, const float* x) override;

    /** Quantize a set of vectors
     *
     * @param x        input vectors, size n * d
     * @param codes    output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    /** Decode a set of vectors
     *
     * @param codes    input codes, size n * code_size
     * @param x        output vectors, size n * d
     */
    void decode(const uint8_t* code, float* x, size_t n) const override;
};
} // namespace jecq