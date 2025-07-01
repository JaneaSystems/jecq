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

#include "index_ivf_jecq.h"
#include "feature_classifier.h"

#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <cinttypes>

namespace jecq {
IndexIVFJecq::IndexIVFJecq() : IndexIVFJecq(0, 1, 10, 0.05, 0.005, 50) {}

IndexIVFJecq::IndexIVFJecq(
        faiss::idx_t d,
        faiss::idx_t nlist,
        float pq_multiplier,
        float th_high,
        float th_mid,
        int itq_iters)
        : IndexJecqBase(pq_multiplier, th_high, th_mid),
          IndexIVF(
                  new faiss::IndexFlat(
                          d,
                          faiss::MetricType::METRIC_INNER_PRODUCT),
                  d,
                  nlist,
                  /*code_size=*/0,
                  faiss::MetricType::METRIC_INNER_PRODUCT),
          itq_iters(itq_iters) {}

void IndexIVFJecq::encode_vectors(
        faiss::idx_t n,
        const float* x,
        const faiss::idx_t* list_nos,
        uint8_t* code,
        bool include_listno) const {
    std::vector<float> pq_data(pq_features.size());
    std::vector<float> itq_data(itq_features.size());

    for (faiss::idx_t i = 0; i < n; i++) {
        const float* row = x + i * d;
        uint8_t* const cp = code + i * code_size;

        if (!pq_features.empty()) {
            filter_by_features(row, pq_features, pq_data.data());
            pq_quantizer.compute_code(pq_data.data(), cp);
        }

        if (!itq_features.empty()) {
            filter_by_features(row, itq_features, itq_data.data());
            itq_quantizer.compute_codes(
                    itq_data.data(), cp + pq_quantizer.code_size, 1);
        }
    }
}

struct IVFJecqScanner : faiss::InvertedListScanner {
    const IndexIVFJecq* parent;
    std::vector<float> q_pq;
    std::vector<uint8_t> q_itq;
    const float* q = nullptr;

    IVFJecqScanner(const IndexIVFJecq* p, bool store_pairs)
            : InvertedListScanner(store_pairs), parent(p) {
        this->keep_max = true;
    }

    void set_query(const float* query) override {
        this->q = query;
        this->code_size = parent->code_size;

        if (!parent->pq_features.empty()) {
            q_pq.resize(parent->pq_features.size());
            filter_by_features(query, parent->pq_features, q_pq.data());
        }

        if (!parent->itq_features.empty()) {
            std::vector<float> itq_data(parent->itq_features.size());
            filter_by_features(query, parent->itq_features, itq_data.data());

            q_itq.resize(parent->itq_quantizer.code_size);
            parent->itq_quantizer.compute_codes(
                    itq_data.data(), q_itq.data(), 1);
        }
    }

    void set_list(faiss::idx_t list_no, float coarse_dis) {}

    float distance_to_code(const uint8_t* code) const override {
        float pq_distance = 0, itq_distance = 0;

        if (!parent->pq_features.empty()) {
            std::vector<float> recon(parent->pq_features.size());
            parent->pq_quantizer.decode(code, recon.data(), 1);
            pq_distance = faiss::fvec_inner_product(
                    q_pq.data(), recon.data(), recon.size());
        }

        code += parent->pq_quantizer.code_size;

        if (!parent->itq_features.empty()) {
            itq_distance = parent->itq_quantizer.get_inner_product_distance(
                    code, q_itq.data());
        }

        return (parent->pq_multiplier * pq_distance + itq_distance);
    }
};

faiss::InvertedListScanner* IndexIVFJecq::get_InvertedListScanner(
        bool store_pairs,
        const faiss::IDSelector* sel,
        const faiss::IVFSearchParameters* params) const {
    return new IVFJecqScanner(this, store_pairs);
}

void IndexIVFJecq::train(faiss::idx_t n, const float* x) {
    const auto t0 = faiss::getmillisecs();

    if (verbose) {
        printf("Training IndexIVFJecq with %d features on %" PRId64
               " vectors\n ",
               this->d,
               n);
    }

    if (this->reclassify_features_when_training) {
        this->reclassify_features(n, x);
    }

    if (verbose) {
        printf("Classified IndexIVFJecq features; pq_features.size()=%zu, itq_features.size()=%zu, discarded_features.size()=%zu\n",
               pq_features.size(),
               itq_features.size(),
               d - (pq_features.size() + itq_features.size()));
    }

    const auto t1 = faiss::getmillisecs();

    if (!pq_features.empty()) {
        const int nbits = 8;
        pq_quantizer = faiss::ProductQuantizer(
                pq_features.size(), pq_features.size(), nbits);
        const auto pq_data = get_filtered_features(n, d, x, pq_features);
        pq_quantizer.train(n, pq_data.data());
    } else {
        pq_quantizer = faiss::ProductQuantizer();
    }

    const auto t2 = faiss::getmillisecs();

    if (!itq_features.empty()) {
        itq_quantizer = ITQQuantizer(itq_features.size(), itq_iters);
        const auto itq_data = get_filtered_features(n, d, x, itq_features);
        itq_quantizer.train(n, itq_data.data());
    } else {
        itq_quantizer = ITQQuantizer();
    }

    const auto t3 = faiss::getmillisecs();

    // Train base
    this->code_size = pq_quantizer.code_size + itq_quantizer.code_size;
    assert(this->own_invlists);
    delete this->invlists;
    this->invlists = new faiss::ArrayInvertedLists(nlist, code_size);
    IndexIVF::train(n, x);

    const auto t4 = faiss::getmillisecs();

    if (verbose) {
        printf("Training IndexIVFJecq complete; total_ms = %.1f, classification_ms=%.1f, pq_ms=%.1f, itq_ms=%.1f, IndexIVF_ms=%.1f\n",
               t4 - t0,
               t1 - t0,
               t2 - t1,
               t3 - t2,
               t4 - t3);
    }
}

void IndexIVFJecq::reconstruct_from_offset(
        faiss::idx_t list_no,
        faiss::idx_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_codes(list_no) + offset * code_size;

    std::fill_n(recons, d, 0.0f);

    if (!pq_features.empty()) {
        std::vector<float> pq_data(pq_features.size());
        pq_quantizer.decode(code, pq_data.data(), 1);
        filter_by_features(pq_data.data(), pq_features, recons);

        code += pq_quantizer.code_size;
    }

    if (!itq_features.empty()) {
        std::vector<float> itq_data(itq_features.size());
        itq_quantizer.decode(code, itq_data.data(), 1);
        filter_by_features(itq_data.data(), itq_features, recons);
    }
}

} // namespace jecq
