# Jecq

Jecq (Just Enough Compression & Quantization) is an open-source C++ library by Janea Systems for efficient similarity search of dense vectors. It is designed as a drop-in extension for [Faiss](https://github.com/facebookresearch/faiss), Meta’s popular library for fast vector similarity search. Jecq introduces advanced, dimension-aware compression techniques that significantly reduce memory footprint while maintaining high search accuracy. Complete wrappers for Python/numpy are provided.

## Key Features
### Dimension-Aware Compression
Jecq analyzes the statistical relevance of each vector dimension and applies varying levels of quantization, achieving 
high levels of compression while retaining high search accuracy. See [STATISTICS.md](STATISTICS.md) for an example of this in action against a sample dataset. 

### Faiss Compatibility
Provides two Faiss-compatible indices:
* jecq::IndexJecq – drop-in replacement for faiss::IndexPQ
* jecq::IndexIVFJecq – drop-in replacement for faiss::IndexIVFPQ

### Hyper-Parameter Optimization
Includes a bundled optimizer to help users select hyper-parameters that best balance compression ratio and search accuracy for their data.

### CPU Implementation
Written in C++ for CPUs; currently no GPU support.

## Why Use Jecq?
### Reduced Storage + High Accuracy
In our tests against a sample dataset, using IndexPQ as a benchmark, we achieved a compression ratio of 15.9% (~6x compression) while retaining ~85% search accuracy.

### Easy Integration
Seamlessly integrates with existing Faiss-based pipelines and vector databases.

## How Jecq Compression Works
Jecq’s approach is based on the observation that not all vector dimensions contribute equally to search relevance. Instead of applying uniform compression, Jecq:

* Analyzes variance by computing the eigenvalues of the covariance matrix from training data to measure the statistical relevance (variance) of each dimension.

* Encodes dimensions according to three categories:
    1. High Variance features are encoded with Product Quantization (PQ), using as many sub-quantizers as dimensions, with 8 bits per dimension.
    2. Medium Variance features are encoded with Iterative Quantization ([ITQ](https://slazebni.cs.illinois.edu/publications/ITQ.pdf)), with 1 bit per dimension.
    3. Low Variance features are discarded (0 bits per dimension).

* Stores compressed vectors in a custom, compact format accessible via a lightweight API.

This non-uniform, relevance-based quantization enables aggressive compression without sacrificing mission-critical search signals.

## Search Functionality
Distance Metric: Supports [inner product distance](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_inner_product) only.

```math
\mathrm{search\_distance}(q, v) = \mathrm{ip\_distance}_{\mathrm{pq\_features}}(q, v)\,\times\,\mathrm{pq\_multiplier}
\;+\;
\mathrm{ip\_distance}_{\mathrm{itq\_features}}(q, v)
```

## Hyper-parameters:

* `pq_multiplier`: Weight for PQ features in search distance calculation.
* `th_high`: Variance threshold above which features are PQ-encoded.
* `th_mid`: Variance threshold below which features are discarded.

Note: "Variance" here refers to eigenvalues from the covariance matrix, not naive sample variance.

## Installation
Jecq is distributed with precompiled Python libraries. The core is implemented in C++ and requires only a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. Compiles with CMake. See [INSTALL.md](INSTALL.md) for step-by-step instructions.

## Use Cases
* Retrieval-augmented generation (RAG)
* Recommendation engines
* Semantic search
* Edge AI and IoT deployments
* Cost-sensitive enterprise AI search