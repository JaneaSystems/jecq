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

#include "index_jecq_base.h"
#include "feature_classifier.h"

#include <faiss/impl/FaissAssert.h>

namespace jecq {

IndexJecqBase::IndexJecqBase(float pq_multiplier, float th_high, float th_mid)
        : pq_multiplier(pq_multiplier), th_high(th_high), th_mid(th_mid) {
    FAISS_THROW_IF_NOT_MSG(
            pq_multiplier > 0.0f, "pq_multiplier must be positive");
    FAISS_THROW_IF_NOT_MSG(th_mid > 0.0f, "th_high must be positive");
    FAISS_THROW_IF_NOT_MSG(
            th_high > th_mid, "th_high must be greater than th_mid");
}

void IndexJecqBase::reclassify_features(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= 0, "n must be non-negative");

    pq_features.clear();
    itq_features.clear();
    classify_features(
            n,
            this->as_faiss_index().d,
            x,
            this->th_high,
            this->th_mid,
            &pq_features,
            &itq_features,
            &feature_variances);
}
} // namespace jecq