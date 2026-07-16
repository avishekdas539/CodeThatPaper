# TurboQuant

> A pure NumPy implementation of **TurboQuant**, a vector quantization algorithm based on optimal scalar quantization of Gaussian-distributed embeddings.

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
<img src="https://img.shields.io/badge/Numpy-Latest-orange.svg">
<img src="https://img.shields.io/badge/Scipy-Latest-green.svg">
<img src="https://img.shields.io/badge/License-MIT-success.svg">
</p>

---

## Overview

TurboQuant is a lightweight implementation of a **minimum mean squared error (MMSE)** vector quantizer for dense embeddings.

The algorithm performs the following steps:

1. Normalize each embedding to have norm **√d**
2. Rotate the embedding using a random orthogonal matrix
3. Quantize each coordinate using **Lloyd-Max optimized centroids** for a standard Gaussian distribution
4. Pack the quantization indices into compact bit streams
5. Reconstruct the original vector by unpacking, inverse rotation, and reverse normalization

The implementation is written entirely using **NumPy** and **SciPy**, making it easy to understand and extend.

---

## Features

- ✅ Lloyd-Max centroid optimization
- ✅ Gaussian MMSE quantization
- ✅ Random orthogonal rotation
- ✅ Bit packing / unpacking
- ✅ Vector reconstruction
- ✅ Relative MSE benchmarking
- ✅ Rich console benchmark report

---

## Installation

```bash
git clone https://github.com/avishekdas539/CodeThatPaper.git
cd TurboQuant

pip install numpy scipy rich
```

---

## Usage

```python
import numpy as np
from turboquant import TurboQuantMSE

quantizer = TurboQuantMSE(bits=4)

vectors = np.random.normal(size=(1024, 1536)).astype(np.float32)

quantized = quantizer.encode(vectors)

decoded = quantizer.decode(quantized)
```

---

## Quantized Object

The encoder returns a `QuantizedVector` object containing

- Packed bitstream
- Vector norms
- Rotation matrix
- Original tensor shape
- Original dtype

Example

```python
QuantizedVector(
    shape=(1024, 1536),
    bits=4,
    dtype=float32,
    packed_dtype=uint8,
    packed_bytes=786432,
    original_bytes=6291456,
    compressed_bytes=...
)
```

---

# Algorithm

For each input vector **x**

```text
Normalize
        │
        ▼
Random Orthogonal Rotation
        │
        ▼
Gaussian Lloyd-Max Quantization
        │
        ▼
Bit Packing
```

Decoding performs

```text
Bit Unpacking
        │
        ▼
Centroid Lookup
        │
        ▼
Inverse Rotation
        │
        ▼
Reverse Normalization
```

---

# Benchmark

![Benchmark](assets/TEST%20Results.png)
---

## Key Observations

- Increasing the quantization precision from **1 bit** to **4 bits** reduces reconstruction error by nearly **38×**.
- Compression decreases linearly as the number of quantization bits increases.
- Reconstruction quality is largely independent of embedding dimensionality.
- Encoding time grows approximately linearly with vector dimension.

---

# Running the Benchmark

```bash
python turboquant-benchmark.py
```

Produces a Rich console report similar to

```
╭── TurboQuant MSE Benchmark ──╮
│ Vectors/Test : 4096          │
│ Dimensions   : 1024 1536 ... │
│ Bit Widths   : 1 2 3 4       │
╰──────────────────────────────╯

dim = 1024

Bits   MSE     Compression ...
1      0.363   32x
2      0.117   16x
3      0.034    8x
4      0.009    8x
```

---

# Current Limitations

This implementation is primarily intended for **research and educational purposes**.

Current limitations include:

- Rotation matrix is generated and stored per encoding.
- Lloyd-Max centroid computation is performed at initialization.
- Encoding uses Python loops for centroid lookup.
- CPU-only implementation.
- No SIMD or GPU acceleration.

---

# Future Improvements

- Vectorized centroid search
- AVX/SIMD implementation
- GPU implementation (CUDA)
- Shared rotation matrices
- Learned rotations
- Product Quantization hybrid
- Optimized bit packing
- Streaming encoder

---

# References

- TurboQuant: *Ultra-Fast Quantization of Large Language Model Embeddings*
- Lloyd, S. (1982). Least Squares Quantization in PCM.
- Max, J. (1960). Quantizing for Minimum Distortion.

---

# License

MIT License.