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

#include "itq_quantizer.h"

#include <faiss/Index.h>
#include <faiss/IndexFlatCodes.h>

namespace jecq {

class IndexITQFlat : public faiss::IndexFlatCodes {
   public:
    ITQQuantizer itq;

    explicit IndexITQFlat(faiss::idx_t d, int itq_iters = 50);

    explicit IndexITQFlat();

    void train(faiss::idx_t n, const float* x) override;

    void sa_encode(faiss::idx_t n, const float* x, uint8_t* bytes)
            const override;

    void sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x)
            const override;

    void search(
            faiss::idx_t n,
            const float* xq,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;

    void reset() override;
};
} // namespace jecq