// CUDA voxel downsampling for KISS-ICP (plain C++ header, no CUDA types).
#pragma once

#include <cstddef>
#include <functional>

namespace kiss_icp::gpu {

/// Inputs smaller than this go to the CPU version: for a few thousand points the kernel launches
/// and transfers cost more than the hash map.
inline constexpr std::size_t kDownsampleMinPoints = 4096;

/// Same result as kiss_icp::VoxelDownsample on the CPU, including the order of the output
/// (the CPU returns points in tsl::robin_map iteration order, which KISS-ICP depends on: the next
/// downsample and the map keep the first point per voxel). Voxel coordinates must stay within
/// +-2^20 per axis.
///
/// xyz: n interleaved points. alloc_out(m) must return storage for 3*m doubles.
/// Thread safe; concurrent calls are serialized.
void VoxelDownsample(const double *xyz,
                     std::size_t n,
                     double voxel_size,
                     const std::function<double *(std::size_t)> &alloc_out);

/// How often the GPU path handed over to the CPU implementation (pathological hash clusters).
std::size_t DownsampleFallbackCount();

}  // namespace kiss_icp::gpu
