# CUDA voxel map for KISS-ICP

Replaces `tsl::robin_map<Voxel, std::vector<Eigen::Vector3d>>` in `VoxelHashMap` with a
`cuco::static_map` (cuCollections) in GPU memory, after FAR-LIO's cuVoxelMap
(TUMFTM/FAR-LIO, Apache-2.0). The ICP data association and the 6x6 normal equations run on the GPU
too, otherwise every correspondence query would be a host-device round trip.

## Build

```bash
cmake -S cpp/kiss_icp -B build -DKISS_ICP_USE_CUDA=ON          # needs CUDA >= 12, CMake >= 3.24
cmake --build build -j
./build/core/kiss_icp_cuda_check                                 # CPU-reference comparison + timings
```

| CMake option | default | meaning |
|---|---|---|
| `KISS_ICP_USE_CUDA` | `OFF` | build the GPU map; `OFF` keeps the original CPU code |
| `KISS_ICP_CUDA_MAX_POINTS_PER_VOXEL` | `20` | compile-time slot size; runtime `max_points_per_voxel` must be <= this |
| `KISS_ICP_CUDA_MAX_VOXELS` | `524288` | default voxel capacity (4th ctor argument overrides it) |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | e.g. `87` for Orin, `86` for RTX 30xx |

cuCollections is fetched at the commit FAR-LIO pins (it pulls rapids-cmake and CCCL from GitHub).
If your project already defines `cuco::cuco`, that target is used instead.

## Design

```
cuco::static_map<uint64 key, uint32 block>     key = 21 bits per axis (+-2^20 voxels)
block pool  key[N] | count[N] | float xyz[N][MAX_POINTS]
free list   stack of unused block indices
```

* **AddPoints**: keys computed on GPU, points stable-sorted by key, one thread per voxel appends its
  points in input order using the original rules (voxel full, or closer than
  `sqrt(voxel^2 / max_points)` to a stored point). Lock-free and deterministic, and it gives the
  same map as the CPU code (apart from float storage).
* **RemovePointsFarFromLocation**: one pass over the pool, freed blocks go back to the free list,
  keys are bulk-erased. Erasing leaves tombstones; at 25 % tombstones the map is rebuilt from the
  pool (only `(key, block)` pairs, no point data is copied).
* **Registration**: the source frame is uploaded once per `AlignPointsToMap`. Each iteration sends
  the pose (12 doubles) and reads back 28 doubles (JTJ upper triangle, JTr, count). NN search is
  float, residuals and the normal equations are double.

* **VoxelDownsample** (`DownsampleGpu.cu`, used automatically in the CUDA build for inputs of
  4096+ points): returns exactly the CPU result **including sampling order**. The CPU returns points in
  `tsl::robin_map` iteration order, and KISS-ICP depends on it: the second downsample and the map
  keep the first point per voxel, so scan order instead of hash order biases the representative
  points (grid pattern, more drift; v7-v10 had this bug). The GPU reproduces the robin_map layout:
  first point per voxel via a stable radix sort, home bucket = same hash & same table size as
  `reserve(n)`, slot positions by a prefix-max scan (with wrap-around), and a per-cluster replay of
  tsl's robin-hood insertions for entries sharing a home bucket. Pathological clusters
  (> 128 slots) fall back to the CPU (`gpu::DownsampleFallbackCount()`).
  `VoxelDownsampleCpu()` is the original, available in both builds.

* **Preprocess** (`PreprocessGpu.cu`, deskew + range crop, inputs of 4096+ points): one thread
  per point evaluates Sophus' `SE3::exp((s - 1) * omega)` step by step and applies it, then the
  points inside `(min_range, max_range)` are compacted in input order. Compiled with
  `-fmad=false`; crop-only output is bit-identical to the CPU, deskewed points differ by ~1e-15 m
  (GPU vs libm sin/cos). `Preprocessor::PreprocessCpu()` is the original.

Public API of `VoxelHashMap` is unchanged (`KissICP.cpp`, pybind sources compile as-is). Additions:
`GetClosestNeighbors(vector)` (batched), `NumVoxels()`, `Gpu()`.

## Differences to the CPU map

* Points are stored as `float`: ~0.24 mm ulp at 2 km, ~1 mm at 8 km from the map origin.
* Fixed capacity: when the pool is full, points of new voxels are dropped and counted
  (`Gpu().DroppedForCapacity()`). Size it with the 4th constructor argument or CMake.
* `GetClosestNeighbor(single point)` still works but is a full GPU round trip; use the batched call.
* Not safe to call `AlignPointsToMap` concurrently on the same map (shared device scratch).
* Requires compute capability >= 6.0 (double `atomicAdd`).
