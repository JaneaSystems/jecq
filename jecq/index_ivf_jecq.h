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

#include "index_jecq_base.h"
#include "itq_quantizer.h"

#include <faiss/IndexIVF.h>
#include <faiss/impl/ProductQuantizer.h>

#include <memory>
#include <vector>

namespace jecq {

class IndexIVFJecq : public IndexJecqBase, public faiss::IndexIVF {
   private:
    int itq_iters;
    faiss::ProductQuantizer pq_quantizer;
    ITQQuantizer itq_quantizer;

   public:
    IndexIVFJecq();

    IndexIVFJecq(
            faiss::idx_t d,
            faiss::idx_t nlist,
            float pq_multiplier,
            float th_high,
            float th_mid,
            int itq_iters = 50);

    void encode_vectors(
            faiss::idx_t n,
            const float* x,
            const faiss::idx_t* list_nos,
            uint8_t* code,
            bool include_listno = false) const override;

    faiss::InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const faiss::IDSelector* sel = nullptr,
            const faiss::IVFSearchParameters* params = nullptr) const override;

    void train(faiss::idx_t n, const float* x) override;

    faiss::Index& as_faiss_index() override {
        return *this;
    }

    const faiss::Index& as_faiss_index() const override {
        return *this;
    }

    void reconstruct_from_offset(
            faiss::idx_t list_no,
            faiss::idx_t offset,
            float* recons) const override;

    friend class IVFJecqScanner;
};

} // namespace jecq
