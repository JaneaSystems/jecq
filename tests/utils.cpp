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

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

namespace jecq_test {

std::vector<float> get_row(
        const std::vector<float>& data,
        faiss::idx_t d,
        int i) {
    return {data.begin() + d * i, data.begin() + d * (i + 1)};
}

std::vector<float> random_vector_float(size_t s) {
    std::vector<float> v(s, 0);
    for (size_t i = 0; i < s; ++i) {
        v[i] = std::rand();
    }

    return v;
}

void normalize(std::vector<float>* input_ptr, int d) {
    auto& input = *input_ptr;

    assert(input.size() % d == 0);

    int rows = input.size() / d;

    for (int i = 0; i < rows; ++i) {
        float sum_square = 0;

        for (int j = 0; j < d; ++j) {
            sum_square += input[i * d + j] * input[i * d + j];
        }

        float root_sum_square = sum_square == 0 ? 1 : std::sqrt(sum_square);

        for (int j = 0; j < d; ++j) {
            input[i * d + j] /= root_sum_square;
        }
    }
}

void perturb(std::vector<float>* input_ptr) {
    for (auto& x : *input_ptr) {
        x += 0.01;
    }
}
} // namespace jecq_test
