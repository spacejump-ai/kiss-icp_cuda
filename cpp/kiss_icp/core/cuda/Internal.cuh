// Device-to-device building blocks shared by the CUDA .cu files (not part of the public API).
#pragma once

#include <cstddef>

namespace kiss_icp::gpu::detail {

/// d_in, d_out: device arrays of n interleaved xyz doubles (d_out capacity >= n).
/// Return the number of points written to d_out. Synchronous; serialized internally.
std::size_t VoxelDownsampleDevice(const double *d_in, std::size_t n, double voxel_size, double *d_out);

std::size_t PreprocessDevice(const double *d_in,
                             const double *d_stamps,  // device, or nullptr (no deskew)
                             std::size_t n,
                             double t_min,
                             double t_max,
                             const double omega[6],
                             double min_range,
                             double max_range,
                             double *d_out);

}  // namespace kiss_icp::gpu::detail
