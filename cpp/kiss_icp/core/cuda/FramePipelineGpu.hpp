// CUDA: preprocess + two voxel downsamples with one upload. Plain C++ header.
#pragma once

#include <cstddef>
#include <functional>

namespace kiss_icp::gpu {

/// frame = Preprocess(xyz), frame_ds = VoxelDownsample(frame, voxel_ds),
/// source = VoxelDownsample(frame_ds, voxel_src); results identical to the separate GPU calls.
/// The scan is uploaded once and only the requested results are downloaded.
/// alloc_frame may be empty (frame is then not downloaded). Each alloc(m) returns 3*m doubles.
void PreprocessAndVoxelize(const double *xyz,
                           std::size_t n,
                           const double *timestamps,  // n values or nullptr (no deskew)
                           double t_min,
                           double t_max,
                           const double omega[6],
                           double min_range,
                           double max_range,
                           double voxel_ds,
                           double voxel_src,
                           const std::function<double *(std::size_t)> &alloc_frame,
                           const std::function<double *(std::size_t)> &alloc_frame_ds,
                           const std::function<double *(std::size_t)> &alloc_source);

}  // namespace kiss_icp::gpu
