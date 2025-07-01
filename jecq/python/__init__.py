# Copyright (c) 2025 Janea Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# mypy: disable-error-code="attr-defined, name-defined, has-type"
# flake8: noqa

import logging
import sys
import inspect

# We import * so that the symbol foo can be accessed as jecq.foo.
from .loader import *  # NOQA

# additional wrappers
from jecq import class_wrappers
from faiss.array_conversions import *  # NOQA
from faiss import add_ref_in_constructor, add_ref_in_method, add_ref_in_method_explicit_own

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

shard_ivf_index_centroids = class_wrappers.handle_shard_ivf_index_centroids(
    shard_ivf_index_centroids
)  # NOQA

this_module = sys.modules[__name__]

# handle sub-classes
for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    if inspect.isclass(obj):
        the_class = obj
        base_names = {base.__name__ for base in inspect.getmro(the_class)}

        def is_sub(the_class, parent_class):
            return issubclass(the_class, parent_class) or parent_class.__name__ in base_names

        if is_sub(the_class, Index):  # NOQA: F405
            class_wrappers.handle_Index(the_class)

        if is_sub(the_class, IndexBinary):  # NOQA: F405
            class_wrappers.handle_IndexBinary(the_class)

        if is_sub(the_class, VectorTransform):  # NOQA: F405
            class_wrappers.handle_VectorTransform(the_class)

        if is_sub(the_class, Quantizer):  # NOQA: F405
            class_wrappers.handle_Quantizer(the_class)

        if is_sub(the_class, IndexRowwiseMinMax) or is_sub(
            the_class, IndexRowwiseMinMaxFP16
        ):  # NOQA: F405
            class_wrappers.handle_IndexRowwiseMinMax(the_class)

        if is_sub(the_class, SearchParameters):  # NOQA: F405
            class_wrappers.handle_SearchParameters(the_class)

        if is_sub(the_class, CodePacker):  # NOQA: F405
            class_wrappers.handle_CodePacker(the_class)

add_ref_in_constructor(IndexIVFFlat, 0)
add_ref_in_constructor(IndexIVFFlatDedup, 0)
add_ref_in_constructor(IndexPreTransform, {2: [0, 1], 1: [0]})
add_ref_in_method(IndexPreTransform, "prepend_transform", 0)
add_ref_in_constructor(IndexIVFPQ, 0)
add_ref_in_constructor(IndexIVFPQR, 0)
add_ref_in_constructor(IndexIVFPQFastScan, 0)
add_ref_in_constructor(IndexIVFResidualQuantizer, 0)
add_ref_in_constructor(IndexIVFLocalSearchQuantizer, 0)
add_ref_in_constructor(IndexIVFResidualQuantizerFastScan, 0)
add_ref_in_constructor(IndexIVFLocalSearchQuantizerFastScan, 0)
add_ref_in_constructor(IndexIVFSpectralHash, 0)
add_ref_in_method_explicit_own(IndexIVFSpectralHash, "replace_vt")

add_ref_in_constructor(Index2Layer, 0)
add_ref_in_constructor(Level1Quantizer, 0)
add_ref_in_constructor(IndexIVFScalarQuantizer, 0)
add_ref_in_constructor(IndexRowwiseMinMax, 0)
add_ref_in_constructor(IndexRowwiseMinMaxFP16, 0)
add_ref_in_constructor(IndexIDMap, 0)
add_ref_in_constructor(IndexIDMap2, 0)
add_ref_in_constructor(IndexHNSW, 0)
add_ref_in_method(IndexShards, "add_shard", 0)
add_ref_in_method(IndexBinaryShards, "add_shard", 0)
add_ref_in_constructor(IndexRefineFlat, {2: [0], 1: [0]})
add_ref_in_constructor(IndexRefine, {2: [0, 1]})

add_ref_in_constructor(IndexBinaryIVF, 0)
add_ref_in_constructor(IndexBinaryFromFloat, 0)
add_ref_in_constructor(IndexBinaryIDMap, 0)
add_ref_in_constructor(IndexBinaryIDMap2, 0)

add_ref_in_method(IndexReplicas, "addIndex", 0)
add_ref_in_method(IndexBinaryReplicas, "addIndex", 0)

add_ref_in_constructor(BufferedIOWriter, 0)
add_ref_in_constructor(BufferedIOReader, 0)

add_ref_in_constructor(IDSelectorNot, 0)
add_ref_in_constructor(IDSelectorAnd, slice(2))
add_ref_in_constructor(IDSelectorOr, slice(2))
add_ref_in_constructor(IDSelectorXOr, slice(2))
add_ref_in_constructor(IDSelectorTranslated, slice(2))

add_ref_in_constructor(IDSelectorXOr, slice(2))
add_ref_in_constructor(IndexIVFIndependentQuantizer, slice(3))
