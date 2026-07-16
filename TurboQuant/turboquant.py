import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def compute_centroids(bits : int, tol = 1e-7, max_iter = 100):
    buckets = 2**bits

    centroids = np.linspace(-10.0, 10.0, buckets)
    boundaries = np.zeros(buckets+1)


    for n in range(max_iter):
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        for i in range(1, buckets):
            boundaries[i] = (centroids[i-1] + centroids[i])/2
        
        new_centroids = np.zeros_like(centroids)
        for i in range(buckets):
            b_low = boundaries[i]
            b_high = boundaries[i+1]

            neumerator, err1 = quad(lambda x : x * norm.pdf(x), b_low, b_high)
            denominator, err2 = quad(lambda x : norm.pdf(x), b_low, b_high) # this should be always positive
            if denominator>0:
                new_centroids[i] = neumerator / denominator
            else:
                new_centroids[i] = centroids[i]
        
        if np.allclose(centroids, new_centroids, tol):
            print(f'{bits} Bits Boundary Converged in {n} Iteration.')
            break
        
        centroids = new_centroids
    
    return centroids, boundaries

class QuantizedVector:
    def __init__(self, packed_bits : np.ndarray, bits : int, norm_ : np.ndarray, Q : np.ndarray, shape, last_bit_paked_count : int, source_dtype : np.dtype) -> None:
        self.packed_bits = packed_bits
        self.norm_ = norm_
        self.Q = Q
        self.shape = shape
        self.last_bit_paked_count = last_bit_paked_count
        self.source_dtype = source_dtype
        self.original_bytes = np.prod(shape) * np.dtype(source_dtype).itemsize
        self.bits = bits

    def __repr__(self) -> str:
        packed_bytes = self.packed_bits.nbytes if self.packed_bits is not None else 0
        norm_bytes = self.norm_.nbytes if self.norm_ is not None else 0
        q_bytes = self.Q.nbytes if self.Q is not None else 0
        compressed_bytes = packed_bytes + norm_bytes + q_bytes

        ratio = (
            self.original_bytes / compressed_bytes
            if compressed_bytes > 0
            else float("inf")
        )

        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"bits={self.bits}, "
            f"dtype={self.source_dtype}, "
            f"packed_dtype={self.packed_bits.dtype}, "
            f"packed_bytes={packed_bytes}, "
            f"original_bytes={self.original_bytes}, "
            f"compressed_bytes={compressed_bytes}, "
            f"compression_ratio={ratio:.2f}x"
            f")"
        )

class TurboQuantMSE:
    def __init__(self, bits : int) -> None:
        self.bits = bits
        self.centroids, self.boundaries = compute_centroids(bits=bits)
    
    def generate_rotation_matrix(self, d : int, seed : int = 42):
        np.random.seed(seed)
        SIGMA = (np.random.randn(d, d))
        Q, R = np.linalg.qr(SIGMA)
        return Q

    def rotate_vector(self, Q : np.ndarray, vector : np.ndarray):
        # ([d,d] @ [B,d].T).T = ([d,d] @ [d, B]).T = [d, B].T = [B, d]
        return (Q @ vector.T).T

    def normalize_vector(self, vector : np.ndarray, d : int):
        norm_ = np.linalg.norm(vector, axis=-1, keepdims=True)
        vector = vector / norm_ * np.sqrt(d)
        return vector, norm_
    
    def reverse_normalize(self, vector : np.ndarray, d : int, norm_ : np.ndarray) -> np.ndarray:
        return vector * norm_ / np.sqrt(d)
    
    def search_idx(self, value, centroids : np.ndarray):
        """Atctual Implementation in Paper considering centroid is not sorted"""
        return np.argmin(np.abs(centroids - value))

    def search_idx_optimized(self, value, centroids: np.ndarray):
        # Find where 'value' would fit in the sorted array
        idx = np.int64(np.searchsorted(centroids, value))
        
        # Handle edge cases where value is out of bounds
        if idx == 0:
            return 0
        if idx == len(centroids):
            return len(centroids) - 1
            
        # Check which neighbor is closer: idx or idx-1
        if abs(centroids[idx] - value) < abs(centroids[idx - 1] - value):
            return idx
        else:
            return idx - 1

    def get_indexes(self, vector_flat : np.ndarray):
        idxs = [self.search_idx_optimized(value=i, centroids=self.centroids) for i in vector_flat]
        return idxs

    def pack_bits(self, values: list, bits: int = 4):
        max_per_element = 8 // bits
        packed_values = []
        i = 0
        last_element_packed_elem = 0
        while i < len(values):
            j = 0
            value = 0
            while i < len(values) and j < max_per_element:
                value = (value << bits) | values[i]
                i += 1
                j += 1
            
            last_element_packed_elem = j # Just assign j directly
            packed_values.append(value)
            
        return np.array(packed_values, dtype=np.uint8), last_element_packed_elem
    
    def unpack_bits(self, packed_bits : np.ndarray, last_bit_packed_count : int):
        idxs = []
        bit_mask = (1 << self.bits) - 1 # More efficient than 2**self.bits-1
        max_elements = 8 // self.bits
        
        for i in range(len(packed_bits)):
            current_element = int(packed_bits[i]) # Ensure standard python int
            
            if i == len(packed_bits) - 1:
                n_packed_elements = last_bit_packed_count
            else:
                n_packed_elements = max_elements
                
            chunk = []
            for _ in range(n_packed_elements):
                new_element = current_element & bit_mask
                chunk.append(new_element)
                current_element >>= self.bits
                
            # KEY FIX: Reverse the chunk because we extract from right-to-left
            # but we packed them left-to-right!
            idxs.extend(reversed(chunk))
            
        return idxs

    def encode(self, vector : np.ndarray) -> QuantizedVector:
        shape, vec_bytes, dtype = vector.shape, vector.nbytes, vector.dtype
        d = vector.shape[-1]
        Q = self.generate_rotation_matrix(d=d)

        vector, norm_ = self.normalize_vector(vector=vector, d=d)
        vector = self.rotate_vector(Q=Q, vector=vector)
        vector = vector.flatten()

        centroid_ids = self.get_indexes(vector_flat=vector)

        packed_bits, last_elem_packed_elem = self.pack_bits(values=centroid_ids, bits=self.bits)

        quantized_vector = QuantizedVector(
            packed_bits = packed_bits,
            bits = self.bits,
            norm_=norm_,
            Q = Q,
            last_bit_paked_count = last_elem_packed_elem,
            shape=shape,
            source_dtype=dtype
        )

        return quantized_vector

    def decode(self, quantized_vector : QuantizedVector):
        uppacked_bits = self.unpack_bits(quantized_vector.packed_bits, last_bit_packed_count=quantized_vector.last_bit_paked_count)
        vector = np.array([self.centroids[i] for i in uppacked_bits], dtype=quantized_vector.source_dtype)
        vector = np.reshape(vector, quantized_vector.shape)

        # Inverse Rotate
        # Q @ Q.T = I => Q.T = Q^-1
        # Inverse Rotate = Inv(Q) @ Vector = Q.T @ Vector
        vector = self.rotate_vector(Q = quantized_vector.Q.T, vector = vector)

        # Reverse Normalize
        vector = self.reverse_normalize(vector = vector, d = quantized_vector.shape[-1], norm_ = quantized_vector.norm_)

        return vector




# quantizer = TurboQuantMSE(bits=4)

# vectors = np.random.normal(scale=10,size=(10240, 1536))
# q_vetor = quantizer.encode(vector=vectors)
# print(q_vetor)
# dequantized_vector = quantizer.decode(q_vetor)

# MSE = np.average(
#     np.linalg.norm(dequantized_vector - vectors, axis=1)**2 / 
#     np.linalg.norm(vectors, axis=1)**2
# )
# shape, vec_bytes, dtype = vectors.shape, vectors.nbytes, vectors.dtype
# shape, vec_bytes, dtype
# d = vectors.shape[-1]
# np.random.seed(42)
# SIGMA = (np.random.randn(d, d))
# SIGMA
# Q, R = np.linalg.qr(SIGMA)
# Q
# vectors_rotated = (Q @ vectors.T).T # ([d,d] @ [B,d].T).T = ([d,d] @ [d, B]).T = [d, B].T = [B, d]
# vectors_rotated[0], vectors[0]
# norm_ = np.linalg.norm(vectors_rotated, axis=1, keepdims=True)
# norm_.shape, norm_[0]
# vectors_hat = vectors_rotated/norm_ * np.sqrt(d)
# vectors_hat = vectors_hat.flatten()
# vectors_hat.shape
# bits = 4
# centroids, boundary = compute_centroids(bits=bits, max_iter=1000)
# centroids, boundary
# import numpy as np

# def search_idx(value, centroids : np.ndarray):
#     """Atctual Implementation in Paper considering centroid is not sorted"""
#     return np.argmin(np.abs(centroids - value))

# def search_idx_optimized(value, centroids: np.ndarray):
#     # Find where 'value' would fit in the sorted array
#     idx = np.int64(np.searchsorted(centroids, value))
    
#     # Handle edge cases where value is out of bounds
#     if idx == 0:
#         return 0
#     if idx == len(centroids):
#         return len(centroids) - 1
        
#     # Check which neighbor is closer: idx or idx-1
#     if abs(centroids[idx] - value) < abs(centroids[idx - 1] - value):
#         return idx
#     else:
#         return idx - 1
# vectors_hat[:20]
# i = search_idx_optimized(vectors_hat[1], centroids=centroids)
# print(i, centroids[i], vectors_hat[1])
# idxs = [search_idx_optimized(value=i, centroids=centroids) for i in vectors_hat]
# idxs[:5]
# def pack_bits(values : list, bits : int = 4):
#     max_per_element = 8 // bits
#     packed_values = []
#     i = 0
#     while i < len(values):
#         j = 0
#         value = 0
#         while i< len(values) and j < max_per_element:
#             value = value << bits | values[i]
#             i += 1
#             j += 1
#         if j == max_per_element-1:
#             last_element_packed_elem = max_per_element
#         else:
#             last_element_packed_elem = j
#         packed_values.append(value)
#     return np.array(packed_values, dtype=np.uint8), last_element_packed_elem
# paked_bits, last_elem_pacck_elem = pack_bits(values=idxs, bits=bits)
# paked_bits
# len(idxs)
# vectors.nbytes/paked_bits.nbytes
# values_back = np.array([centroids[i] for i in idxs]).reshape(shape)
# values_back[0]
# values_back.shape
# rotate_back = (Q.T @ values_back.T).T / np.sqrt(d) * norm_
# rotate_back.shape
# rotate_back[0]
# vectors[0]
# np.average(np.linalg.norm(rotate_back-vectors, axis=1, keepdims=True)/norm_)
# np.average(norm_)
# mse_quant = np.average(
#     np.linalg.norm(rotate_back - vectors, axis=1)**2 / 
#     np.linalg.norm(vectors, axis=1)**2
# )
# mse_quant







import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ---------------- Configuration ---------------- #

BITS_LIST = [1, 2, 3, 4]
VECTOR_DIMS = [1024, 1536, 2048]
N_VECTORS = 4096
SCALE = 10
SEED = 42

np.random.seed(SEED)


@dataclass
class BenchResult:
    bits: int
    dim: int
    mse: float
    compression_ratio: float
    encode_time: float
    decode_time: float
    encode_throughput: float  # vectors/sec
    decode_throughput: float  # vectors/sec


def compute_mse(decoded: np.ndarray, original: np.ndarray) -> float:
    """Relative MSE, normalized per-vector by squared norm."""
    num = np.linalg.norm(decoded - original, axis=1) ** 2
    denom = np.linalg.norm(original, axis=1) ** 2
    return float(np.mean(num / denom))


def compute_compression_ratio(qvec) -> float:
    packed_size = (
        qvec.packed_bits.nbytes
        + qvec.norm_.nbytes
        + qvec.Q.nbytes
    )
    return qvec.original_bytes / packed_size


def run_benchmark(bits_list: List[int], dims: List[int], n_vectors: int, scale: float) -> List[BenchResult]:
    results = []

    for bits in bits_list:
        print(f"Testing {bits}-bit quantizer...")
        quantizer = TurboQuantMSE(bits)

        for dim in dims:
            vectors = np.random.normal(scale=scale, size=(n_vectors, dim)).astype(np.float32)

            start = time.perf_counter()
            qvec = quantizer.encode(vectors)
            encode_time = time.perf_counter() - start

            start = time.perf_counter()
            decoded = quantizer.decode(qvec)
            decode_time = time.perf_counter() - start

            mse = compute_mse(decoded, vectors)
            ratio = compute_compression_ratio(qvec)

            results.append(
                BenchResult(
                    bits=bits,
                    dim=dim,
                    mse=mse,
                    compression_ratio=ratio,
                    encode_time=encode_time,
                    decode_time=decode_time,
                    encode_throughput=n_vectors / encode_time if encode_time > 0 else float("inf"),
                    decode_throughput=n_vectors / decode_time if decode_time > 0 else float("inf"),
                )
            )

    return results


def print_report(results: List[BenchResult]) -> None:
    col_widths = {
        "bits": 4,
        "dim": 6,
        "mse": 12,
        "ratio": 11,
        "enc_time": 10,
        "dec_time": 10,
        "enc_thr": 12,
        "dec_thr": 12,
    }
    total_width = sum(col_widths.values()) + 3 * (len(col_widths) - 1) + 2

    header = (
        f"{'Bits':>{col_widths['bits']}} | "
        f"{'Dim':>{col_widths['dim']}} | "
        f"{'MSE':>{col_widths['mse']}} | "
        f"{'Compression':>{col_widths['ratio']}} | "
        f"{'Encode(s)':>{col_widths['enc_time']}} | "
        f"{'Decode(s)':>{col_widths['dec_time']}} | "
        f"{'Enc(vec/s)':>{col_widths['enc_thr']}} | "
        f"{'Dec(vec/s)':>{col_widths['dec_thr']}}"
    )

    print("\n")
    print("=" * total_width)
    print("TurboQuant MSE Benchmark Report".center(total_width))
    print("=" * total_width)
    print(header)
    print("-" * total_width)

    for r in results:
        print(
            f"{r.bits:{col_widths['bits']}d} | "
            f"{r.dim:{col_widths['dim']}d} | "
            f"{r.mse:{col_widths['mse']}.6e} | "
            f"{r.compression_ratio:{col_widths['ratio']-1}.2f}x | "
            f"{r.encode_time:{col_widths['enc_time']}.4f} | "
            f"{r.decode_time:{col_widths['dec_time']}.4f} | "
            f"{r.encode_throughput:{col_widths['enc_thr']}.1f} | "
            f"{r.decode_throughput:{col_widths['dec_thr']}.1f}"
        )

    print("=" * total_width)

    # Summary section
    best_mse = min(results, key=lambda r: r.mse)
    best_ratio = max(results, key=lambda r: r.compression_ratio)
    fastest_encode = max(results, key=lambda r: r.encode_throughput)

    print("\nSummary:")
    print(f"  Lowest MSE:          {best_mse.bits}-bit @ dim={best_mse.dim}  (MSE={best_mse.mse:.6e})")
    print(f"  Best compression:    {best_ratio.bits}-bit @ dim={best_ratio.dim}  ({best_ratio.compression_ratio:.2f}x)")
    print(f"  Fastest encode:      {fastest_encode.bits}-bit @ dim={fastest_encode.dim}  ({fastest_encode.encode_throughput:.1f} vec/s)")
    print()


if __name__ == "__main__":
    results = run_benchmark(BITS_LIST, VECTOR_DIMS, N_VECTORS, SCALE)
    print_report(results)