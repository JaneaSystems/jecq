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

#include <jecq/index_ivf_jecq.h>
#include <jecq/index_jecq.h>

namespace jecq_test {

std::ostream& operator<<(std::ostream& os, const IndexType& index_type) {
    switch (index_type) {
        case IndexType::IndexJecq: {
            return os << "IndexJecq";
        }

        case IndexType::IndexIVFJecq: {
            return os << "IndexIVFJecq";
        }

        default:
            throw std::runtime_error("Unknown IndexType");
    }
}

std::unique_ptr<jecq::IndexJecqBase> create_index_jecq(
        IndexType index_type,
        faiss::idx_t d) {
    return create_index_jecq(index_type, d, 10, 0.05, 0.005, false);
}

std::unique_ptr<jecq::IndexJecqBase> create_index_jecq(
        IndexType index_type,
        faiss::idx_t d,
        float pq_multiplier,
        float th_high,
        float th_mid,
        bool force_flat) {
    switch (index_type) {
        case IndexType::IndexJecq: {
            auto index_jecq = std::make_unique<jecq::IndexJecq>(
                    d, pq_multiplier, th_high, th_mid);
            index_jecq->verbose = true;
            return index_jecq;
        }

        case IndexType::IndexIVFJecq: {
            const faiss::idx_t nlist = force_flat ? 1 : 10;
            const faiss::idx_t nprobe = force_flat ? 1 : 4;
            auto index_ivf_jecq = std::make_unique<jecq::IndexIVFJecq>(
                    d, nlist, pq_multiplier, th_high, th_mid);
            index_ivf_jecq->verbose = true;
            return index_ivf_jecq;
        }

        default:
            throw std::runtime_error("Unknown IndexType");
    }
}

} // namespace jecq_test