// CUDA voxel downsampling for KISS-ICP (plain C++ header, no CUDA types).
#pragma once

#include <cstddef>
#include <functional>

namespace kiss_icp::gpu {

/// Inputs smaller than this go to the CPU version: for a few thousand points the kernel launches
/// and transfers cost more than the hash map.
inline constexpr std::size_t kDownsampleMinPoints = 4096;

/// Keeps the first point (in input order) of every voxel, like kiss_icp::VoxelDownsample on the
/// CPU, and returns the kept points in input order. Voxel assignment uses the same double
/// arithmetic as kiss_icp::PointToVoxel, so the kept set is identical to the CPU result
/// (the CPU returns it in hash-map order). Voxel coordinates must stay within +-2^20 per axis.
///
/// xyz: n interleaved points. alloc_out(m) must return storage for 3*m doubles.
/// Thread safe; concurrent calls are serialized.
void VoxelDownsample(const double *xyz,
                     std::size_t n,
                     double voxel_size,
                     const std::function<double *(std::size_t)> &alloc_out);

}  // namespace kiss_icp::gpu
