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

#include "datasets.h"

#include <jecq/index_jecq_base.h>

#include <memory>
#include <ostream>

namespace jecq_test {

enum class IndexType { IndexJecq, IndexIVFJecq };

std::ostream& operator<<(std::ostream&, const IndexType&);

std::unique_ptr<jecq::IndexJecqBase> create_index_jecq(
        IndexType,
        faiss::idx_t d,
        float pq_multiplier,
        float th_high,
        float th_mid,
        bool force_flat = false);

std::unique_ptr<jecq::IndexJecqBase> create_index_jecq(
        IndexType,
        faiss::idx_t d = DEFAULT_DIMENSIONS);
} // namespace jecq_test