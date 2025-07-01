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

#include "index_itq_flat.h"
#include "index_jecq_base.h"

#include <faiss/IndexPQ.h>
#include <faiss/faiss/Index.h>

#include <vector>

namespace jecq {

class IndexJecq : public IndexJecqBase, public faiss::Index {
   private:
    faiss::IndexPQ index_pq;
    IndexITQFlat index_itq;

    std::vector<float> get_pq_vector(faiss::idx_t n, const float* x) const;
    std::vector<float> get_itq_vector(faiss::idx_t n, const float* x) const;

   public:
    /** Constructor.
     *
     * @param d                    dimensionality of the input vectors
     * @param pq_multiplier        PQ Multiplier
     * @param th_high              threshold for high variance features
     * @param th_mid               threshold for mid variance features
     */
    explicit IndexJecq(
            faiss::idx_t d,
            float pq_multiplier,
            float th_high,
            float th_mid);

    IndexJecq();

    void add(faiss::idx_t n, const float* x) override;

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params) const override;

    void reset() override;

    void train(faiss::idx_t n, const float* x) override;

    faiss::Index& as_faiss_index() override {
        return *this;
    }

    const faiss::Index& as_faiss_index() const override {
        return *this;
    }
};
} // namespace jecq