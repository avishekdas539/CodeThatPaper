"""
TurboQuant MSE Benchmark
-------------------------
Benchmarks a TurboQuant-style vector quantizer across bit-widths and vector
dimensions, reporting reconstruction error, compression ratio, and
encode/decode throughput as colored console tables.

Requires: numpy, rich  ->  pip install rich

Assumes `TurboQuantMSE` is importable from your own module, e.g.:
    from turboquant import TurboQuantMSE
"""

import time
from dataclasses import dataclass
from typing import List

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from turboquant import TurboQuantMSE

console = Console()

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


# ---------------- Core measurement ---------------- #

def compute_mse(decoded: np.ndarray, original: np.ndarray) -> float:
    """Mean relative squared error, normalized per-vector by squared norm."""
    num = np.linalg.norm(decoded - original, axis=1) ** 2
    denom = np.linalg.norm(original, axis=1) ** 2
    return float(np.mean(num / denom))


def compute_compression_ratio(qvec) -> float:
    packed_size = qvec.packed_bits.nbytes ## + qvec.norm_.nbytes + qvec.Q.nbytes
    return qvec.original_bytes / packed_size


def run_benchmark(
    bits_list: List[int], dims: List[int], n_vectors: int, scale: float
) -> List[BenchResult]:
    results = []
    total_steps = len(bits_list) * len(dims)

    progress_cols = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )

    with Progress(*progress_cols, console=console) as progress:
        task = progress.add_task("Starting...", total=total_steps)

        for bits in bits_list:
            quantizer = TurboQuantMSE(bits)

            for dim in dims:
                progress.update(task, description=f"[cyan]{bits}-bit[/cyan] quantizer @ dim={dim}")

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
                progress.advance(task)

    return results


# ---------------- Formatting helpers ---------------- #

def _gradient_color(value: float, all_values: List[float], higher_is_better: bool) -> str:
    """Rank `value` within `all_values` and map to red -> yellow -> green."""
    lo, hi = min(all_values), max(all_values)
    pct = 1.0 if hi == lo else (value - lo) / (hi - lo)
    if not higher_is_better:
        pct = 1.0 - pct
    if pct >= 0.66:
        return "green3"
    elif pct >= 0.33:
        return "yellow3"
    return "red3"


def _paint(text: str, color: str) -> str:
    return f"[{color}]{text}[/{color}]"


# ---------------- Report sections ---------------- #

def print_config_panel() -> None:
    cfg = (
        f"[bold]Vectors per test:[/bold] {N_VECTORS:,}\n"
        f"[bold]Dimensions:[/bold] {VECTOR_DIMS}\n"
        f"[bold]Bit widths:[/bold] {BITS_LIST}\n"
        f"[bold]Scale (std):[/bold] {SCALE}\n"
        f"[bold]Seed:[/bold] {SEED}"
    )
    console.print(
        Panel(cfg, title="[bold cyan]TurboQuant MSE Benchmark[/bold cyan]", box=box.ROUNDED, expand=False)
    )
    console.print(
        "[dim]Legend:[/dim] "
        + _paint("\u25a0", "green3") + " best   "
        + _paint("\u25a0", "yellow3") + " mid   "
        + _paint("\u25a0", "red3") + " worst"
        + " [dim](relative to other rows in the same table)[/dim]\n"
    )


def print_dimension_tables(results: List[BenchResult]) -> None:
    dims = sorted(set(r.dim for r in results))

    for dim in dims:
        rows = sorted((r for r in results if r.dim == dim), key=lambda r: r.bits)

        mse_vals = [r.mse for r in rows]
        ratio_vals = [r.compression_ratio for r in rows]
        enc_vals = [r.encode_throughput for r in rows]
        dec_vals = [r.decode_throughput for r in rows]

        table = Table(
            title=f"dim = {dim}",
            box=box.SIMPLE_HEAVY,
            header_style="bold white on dark_blue",
            title_style="bold",
            row_styles=["", "on grey11"],
        )
        table.add_column("Bits", justify="right")
        table.add_column("MSE", justify="right")
        table.add_column("Compression", justify="right")
        table.add_column("Encode (s)", justify="right")
        table.add_column("Decode (s)", justify="right")
        table.add_column("Enc (vec/s)", justify="right")
        table.add_column("Dec (vec/s)", justify="right")

        for r in rows:
            table.add_row(
                str(r.bits),
                _paint(f"{r.mse:.6e}", _gradient_color(r.mse, mse_vals, higher_is_better=False)),
                _paint(f"{r.compression_ratio:.2f}x", _gradient_color(r.compression_ratio, ratio_vals, higher_is_better=True)),
                f"{r.encode_time:.4f}",
                f"{r.decode_time:.4f}",
                _paint(f"{r.encode_throughput:,.0f}", _gradient_color(r.encode_throughput, enc_vals, higher_is_better=True)),
                _paint(f"{r.decode_throughput:,.0f}", _gradient_color(r.decode_throughput, dec_vals, higher_is_better=True)),
            )

        console.print(table)


def print_pivot_table(results: List[BenchResult]) -> None:
    """Compact bits x dim matrix of MSE, for scanning the rate-distortion trade-off at a glance."""
    dims = sorted(set(r.dim for r in results))
    bits_list = sorted(set(r.bits for r in results))
    by_key = {(r.bits, r.dim): r for r in results}

    all_mse = [r.mse for r in results]

    table = Table(
        title="MSE overview (bits x dim)",
        box=box.DOUBLE_EDGE,
        header_style="bold white on dark_magenta",
        title_style="bold",
    )
    table.add_column("Bits", justify="right")
    for dim in dims:
        table.add_column(f"dim={dim}", justify="right")

    for bits in bits_list:
        row = [str(bits)]
        for dim in dims:
            r = by_key.get((bits, dim))
            if r is None:
                row.append("-")
            else:
                color = _gradient_color(r.mse, all_mse, higher_is_better=False)
                row.append(_paint(f"{r.mse:.3e}", color))
        table.add_row(*row)

    console.print(table)


def print_summary_table(results: List[BenchResult]) -> None:
    best_mse = min(results, key=lambda r: r.mse)
    best_ratio = max(results, key=lambda r: r.compression_ratio)
    fastest_encode = max(results, key=lambda r: r.encode_throughput)
    fastest_decode = max(results, key=lambda r: r.decode_throughput)

    table = Table(title="Best Configurations", box=box.DOUBLE_EDGE, header_style="bold white on dark_green")
    table.add_column("Metric")
    table.add_column("Config", justify="right")
    table.add_column("Value", justify="right")

    table.add_row("Lowest MSE", f"{best_mse.bits}-bit @ dim={best_mse.dim}", f"{best_mse.mse:.6e}")
    table.add_row("Best compression", f"{best_ratio.bits}-bit @ dim={best_ratio.dim}", f"{best_ratio.compression_ratio:.2f}x")
    table.add_row("Fastest encode", f"{fastest_encode.bits}-bit @ dim={fastest_encode.dim}", f"{fastest_encode.encode_throughput:,.0f} vec/s")
    table.add_row("Fastest decode", f"{fastest_decode.bits}-bit @ dim={fastest_decode.dim}", f"{fastest_decode.decode_throughput:,.0f} vec/s")

    console.print(table)


# ---------------- Entry point ---------------- #

if __name__ == "__main__":
    print_config_panel()
    results = run_benchmark(BITS_LIST, VECTOR_DIMS, N_VECTORS, SCALE)
    console.print()
    print_dimension_tables(results)
    print_pivot_table(results)
    console.print()
    print_summary_table(results)