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

#include <faiss/Index.h>

#include <vector>

namespace jecq_test {

constexpr const size_t DEFAULT_DB_SIZE = 300;

constexpr const size_t DEFAULT_DIMENSIONS = 6;

std::vector<float> get_standard_dataset(
        int db_size = DEFAULT_DB_SIZE,
        int d = DEFAULT_DIMENSIONS);

std::vector<float> get_standard_query(const std::vector<float>& xdb, int d);

void verify_with_standard_query(
        const std::vector<float>& xdb,
        const faiss::Index&);
} // namespace jecq_test