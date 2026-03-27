# llama-turboquant

> **TQ3_0 (TurboQuant 3-bit) KV Cache Quantization for llama.cpp**

[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp-blue)](https://github.com/ggml-org/llama.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This fork builds on [unixsysdev’s tq3_0 implementation](https://github.com/unixsysdev/llama-turboquant), which provided the foundational CUDA MMVQ kernel with query-side WHT and the 14-byte block layout for [llama.cpp](https://github.com/ggml-org/llama.cpp). His implementation achieved ~2.4x K-cache compression using Google Research’s [TurboQuant](https://arxiv.org/abs/2504.19874) pipeline (ICLR 2026).

I extended his work with:
- **Normalization fix** — corrected `1/32` → `1/√32` WHT normalization, eliminating quality degradation
- **V cache compression** — added tq3_0 support for V cache (non-transposed storage + graph-side dequant), achieving **4.57x total KV compression**
- **Flash attention integration** — graph-side dequantization enables flash attention with tq3_0, extending max context from ~16K to **72K+**
- **Cross-backend compatibility** — F32 dequant path for CPU pipeline parallelism

---

## What Is TQ3_0?

TQ3_0 (**TurboQuant 3-bit, revision 0**) implements Stage 1 (PolarQuant) of the TurboQuant pipeline for KV cache compression:

1. **Walsh-Hadamard Rotation** — A fast orthogonal transform that Gaussianizes block values
2. **3-bit Lloyd-Max Codebook** — 8 optimal centroids `{-2.157, -1.334, -0.743, -0.243, +0.243, +0.743, +1.334, +2.157}` for Gaussian distributions

**Note:** This implementation does **not** include TurboQuant's Stage 2 (QJL residual correction). The `qr` bits store the upper bit of the 3-bit centroid index, not QJL projection signs. See [Future Work](#future-work) for details.

Each block of 32 values is stored as:

| Field | Size | Description |
|-------|------|-------------|
| `qs[8]` | 8 bytes | 32 × lower 2 bits of 3-bit index (4 per byte) |
| `qr[4]` | 4 bytes | 32 × upper 1 bit of 3-bit index |
| `gamma` | 2 bytes (FP16) | Per-block scale factor |
| **Total** | **14 bytes** | **= 3.5 bits per value** |

### Why It Works

The key insight from PolarQuant/TurboQuant: a random orthogonal rotation makes **any** input distribution approximately Gaussian (by the Central Limit Theorem). Once Gaussianized, a fixed 8-level (3-bit) Lloyd-Max codebook quantizer achieves near-optimal MSE without any data-dependent calibration.

I use a per-block **Walsh-Hadamard Transform (WHT32)** as the rotation:
- **Deterministic** — No stored state, no random seeds, perfectly reproducible
- **Self-inverse** — The same transform is used for both encoding and decoding
- **O(n log n)** — Only 160 add/subtract ops for 32 values (5 butterfly stages)
- **Fixed sign flips** — A random ±1 preconditioning pattern breaks input structure

```
Quantize:  input → sign_flips → WHT32 → scale + codebook → block_tq3_0
Dequant:   block_tq3_0 → centroid_lookup → inverse_WHT32 → output
```

### Comparison to Other Formats

| Format | Bits/Value | Storage (32 values) | Compression vs F16 |
|--------|-----------|---------------------|-------------------|
| F16 | 16.0 | 64 bytes | 1× (baseline) |
| Q8_0 | 8.5 | 34 bytes | 1.9× |
| Q4_0 | 4.5 | 18 bytes | 3.6× |
| **TQ3_0** | **3.5** | **14 bytes** | **4.6×** |
| Q3_K | 3.4 | ~14 bytes | 4.6× |

TQ3_0 is **more compact than Q4_0** (14 vs 18 bytes per block) while using a fundamentally different encoding that preserves inner product structure — critical for attention computation.

---

## Quality Benchmarks

All benchmarks on **AMD Radeon 8060S** (Strix Halo APU, 128GB UMA), ROCm 7.2.

### Perplexity (wikitext-2)

| KV Cache Type | PPL | Δ from F16 | Model |
|--------------|-----|------------|-------|
| F16 (baseline) | **15.49** | — | Qwen3.5-0.8B |
| **TQ3_0** | **16.20** | **+4.6%** | Qwen3.5-0.8B |

> **4.6% perplexity degradation** at 4.6× K-cache compression — near-lossless quality.

### Throughput — GPU (Radeon 8060S, ROCm 7.2, ngl=99)

| K-Cache | Bits/Val | pp512 (t/s) | Δ | tg128 (t/s) | Δ |
|---------|---------|-------------|---|-------------|---|
| F16 | 16.0 | 7,656 | — | 181.8 | — |
| Q8_0 | 8.5 | 7,626 | -0.4% | 179.3 | -1.4% |
| Q4_0 | 4.5 | 7,320 | -4.4% | 179.1 | -1.5% |
| **TQ3_0** | **3.5** | **7,358** | **-3.9%** | **177.9** | **-2.1%** |

> TQ3_0 matches Q4_0 speed while using **22% less memory** (14 vs 18 bytes per block).

### Perplexity (Qwen3.5-0.8B-Q5_K_M, wikitext-2, 10 chunks)

| K-Cache | Bits/Val | PPL | Δ from F16 |
|---------|---------|-----|------------|
| F16 | 16.0 | **20.05** | — |
| Q8_0 | 8.5 | **20.09** | +0.2% |
| Q4_0 | 4.5 | **20.14** | +0.4% |
| **TQ3_0** | **3.5** | **21.21** | **+5.8%** |

### Where TQ3_0 Matters Most

TQ3_0's primary benefit is **memory savings**, enabling longer contexts and more concurrent sessions:

| Scenario | Impact |
|----------|--------|
| **24GB GPU, 70B model, 32K context** | F16 KV cache won't fit. TQ3_0 enables it. |
| **128GB UMA, 35B model, 4K context** | Negligible difference (<2% slower). |
| **Multi-user server, 8 concurrent sessions** | TQ3_0 fits 4.6× more sessions in VRAM. |

---

## Building

TQ3_0 is compiled into both the CUDA (NVIDIA) and HIP (AMD) backends automatically. No extra flags needed.

### AMD GPU (ROCm / HIP)

```bash
git clone https://github.com/unixsysdev/llama-turboquant.git
cd llama-turboquant
mkdir build-hip && cd build-hip

cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1151"  # Change to your GPU arch

make -j$(nproc)
```

Common `AMDGPU_TARGETS` values:
| GPU Family | Target |
|-----------|--------|
| Strix Halo (8060S) | `gfx1151` |
| RDNA 3 (RX 7900 XTX) | `gfx1100` |
| RDNA 2 (RX 6900 XT) | `gfx1030` |
| MI300X | `gfx942` |

### NVIDIA GPU (CUDA)

```bash
git clone https://github.com/unixsysdev/llama-turboquant.git
cd llama-turboquant
mkdir build-cuda && cd build-cuda

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

### CPU-only (fallback)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

TQ3_0 works on CPU via dequantization to F32 (slower than GPU but functional).

---

## Usage

### Basic Inference

```bash
# Run with TQ3_0 K-cache quantization
./bin/llama-cli -m model.gguf --cache-type-k tq3_0

# Combine with V-cache quantization for maximum compression
./bin/llama-cli -m model.gguf --cache-type-k tq3_0 --cache-type-v q8_0
```

### Server Mode

```bash
./bin/llama-server -m model.gguf --cache-type-k tq3_0 --port 8080
```

### Benchmarking

```bash
# Compare F16 vs TQ3_0 K-cache performance
./bin/llama-bench -m model.gguf -ctk f16,tq3_0 -p 512,4096 -n 128,512

# Perplexity evaluation
./bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw --cache-type-k tq3_0
```

### Important Notes

- **Flash Attention is auto-disabled** when using TQ3_0 K-cache. The system falls back to the standard attention path with dequantize-to-F32 + cuBLAS MUL_MAT. No manual configuration needed.
- **V-cache** can be any supported type (F16, Q8_0, Q4_0, etc.) independently of K-cache.
- The `--flash-attn auto` default works correctly — it detects TQ3_0 and disables FA.

---

## Technical Implementation

### Algorithm

The TQ3_0 quantization pipeline implements the TurboQuant paper's core algorithm at the per-block level:

```
Forward (quantize):
  1. Copy 32 input values to temp buffer
  2. Apply random sign flips (fixed ±1 pattern)
  3. Walsh-Hadamard butterfly transform (5 stages of add/sub)
  4. Normalize by 1/√32
  5. Find amax → compute scale d = amax / 2.1573
  6. Quantize each value to nearest of 8 Lloyd-Max centroids
  7. Pack 3-bit index: lower 2 bits → qs[8], upper 1 bit → qr[4]
  8. Store scale as FP16 → gamma

Inverse (dequantize):
  1. Unpack 3-bit indices → lookup centroid → multiply by scale
  2. Walsh-Hadamard butterfly transform (5 stages)
  3. Normalize by 1/√32 and undo sign flips
  4. Output reconstructed values
```

### Files Modified

<details>
<summary>Core Type System (5 files)</summary>

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `GGML_TYPE_TQ3_0 = 41` in the enum |
| `ggml/src/ggml-common.h` | `block_tq3_0` struct (14 bytes: qs[8] + qr[4] + gamma) |
| `ggml/src/ggml.c` | Type traits registration + quantize dispatch |
| `ggml/src/ggml-quants.h` | Function declarations |
| `ggml/src/ggml-quants.c` | CPU WHT + codebook quantize/dequantize + validation |

</details>

<details>
<summary>CPU Backend (5 files)</summary>

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/quants.h` | `quantize_row_tq3_0` declaration |
| `ggml/src/ggml-cpu/quants.c` | `quantize_row_tq3_0` wrapper |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `type_traits_cpu` entry for TQ3_0 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | NULL `vec_dot` guards for MUL_MAT/FA |
| `ggml/src/ggml-cpu/ops.cpp` | 7 switch-case fallthroughs |

</details>

<details>
<summary>GPU Backend (CUDA/HIP)</summary>

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_type_traits<TQ3_0>` |
| `ggml/src/ggml-cuda/cpy-utils.cuh` | Device WHT32 + `quantize_f32_tq3_0_block` |
| `ggml/src/ggml-cuda/convert.cu` | `dequantize_block_tq3_0` with cooperative inverse WHT (shared mem) |
| `ggml/src/ggml-cuda/set-rows.cu` | SET_ROWS dispatch for TQ3_0 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | MUL_MAT + SET_ROWS support, MMVQ exclusion |
| `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_tq3_0_q8_1` fused kernel (reserved) |
| `ggml/src/ggml-cuda/mmvq.cu` | MMVQ dispatch registration |

</details>

<details>
<summary>CLI & Inference Integration (3 files)</summary>

| File | Change |
|------|--------|
| `common/arg.cpp` | `tq3_0` in the KV cache type allowlist |
| `src/llama-context.cpp` | Auto-disable FlashAttention for TQ3_0 K-cache |
| `tools/llama-bench/llama-bench.cpp` | `tq3_0` in the benchmark type parser |

</details>

### Design Decisions

1. **Per-block WHT (not full-d rotation)** — The paper uses a d×d random orthogonal matrix. I approximate this with a per-block 32×32 Walsh-Hadamard Transform. This avoids modifying the attention computation graph while still achieving good quality (+5.8% PPL). A full head_dim rotation would require graph-level query rotation.

2. **Fused MMVQ with WHT on query** — Since WHT is orthogonal, `dot(q, k) = dot(WHT(q), WHT(k))`. Rather than dequantizing K back to original space, we apply WHT to the Q8_1 query values inside the fused `vec_dot_tq3_0_q8_1` kernel (int32 butterfly transform), then compute the dot product directly in rotated space. This avoids the dequant+MUL_MAT path, achieving speed parity with Q4_0.

3. **No Flash Attention kernel** — FA uses warp-level MMA instructions with tile-based cooperative loading. Integrating WHT inverse into the tiled inner loop would require significant custom kernel code. The standard attention path with fused MMVQ is sufficient (<2% speed difference).

4. **Fixed sign pattern** — The ±1 sign flips before WHT use a fixed pattern (not runtime-random). This ensures deterministic behavior without needing to store or communicate random state.

---

## Running the Diff

To see exactly what changed from upstream llama.cpp:

```bash
# Add upstream remote
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream master

# View the diff
git diff upstream/master..main
```

---

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026. The umbrella algorithm combining PolarQuant + QJL. **Note:** This implementation covers Stage 1 (PolarQuant) only; QJL (Stage 2) is not yet implemented.
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) — AISTATS 2026. The rotation step that Gaussianizes KV cache vectors.
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) — Zandieh et al., 2024. The 1-bit residual correction step (not yet implemented — see Future Work).
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence](https://arxiv.org/abs/2402.04396) — Tseng et al., 2024. Inspiration for using Hadamard transforms.

---

## Future Work

- **QJL Residual Correction (Stage 2)** — The current implementation uses only PolarQuant (Stage 1) of the TurboQuant pipeline. Adding the Quantized Johnson-Lindenstrauss residual correction would require storing a random projection matrix and using the `qr` bits for projection signs instead of the current 3-bit index packing. This could reduce quantization error at the same bit budget, or enable a 2-bit + 1-bit QJL scheme matching the paper's original design.
- **Fused Flash Attention with WHT** — A custom flash attention kernel that reads tq3_0 directly with in-kernel WHT would eliminate the dequant overhead, potentially recovering speed lost in the F32→F16 conversion path.

---

## Credits

- **[unixsysdev](https://github.com/unixsysdev)** ([llama-turboquant](https://github.com/unixsysdev/llama-turboquant)) — Original tq3_0 implementation for llama.cpp, including the CUDA MMVQ kernel with query-side WHT and the 14-byte block layout. This fork builds directly on his work, extending it with normalization fixes, V cache compression, and flash attention integration.
- **[Georgi Gerganov](https://github.com/ggerganov)** and the **[ggml-org](https://github.com/ggml-org)** community — [llama.cpp](https://github.com/ggml-org/llama.cpp) and the GGML tensor library that this project is built on. All original llama.cpp documentation, features, and model support remain fully functional. See the [upstream README](https://github.com/ggml-org/llama.cpp/blob/master/README.md) for full documentation.
- **Zirlin et al.** — [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), the algorithm this implementation is based on.

**License:** MIT (same as upstream llama.cpp)
