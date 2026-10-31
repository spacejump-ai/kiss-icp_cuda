// MIT License
//
// CUDA voxel hash map for KISS-ICP.
// Replaces tsl::robin_map<Voxel, std::vector<Eigen::Vector3d>> with a cuco::static_map
// (cuCollections) living in device memory, following the cuVoxelMap idea from FAR-LIO
// (Leitenstern et al., 2026, https://github.com/TUMFTM/FAR-LIO, Apache-2.0).
//
// This header is deliberately free of CUDA, Eigen and Sophus so that it can be included from
// plain C++ translation units compiled with the host compiler (-Werror -Wconversion etc.).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kiss_icp::gpu {

/// Compile-time upper bound for points per voxel (storage is a fixed-size slot per voxel).
/// The runtime value (KISS-ICP's max_points_per_voxel) must be <= this.
#ifndef KISS_ICP_CUDA_MAX_POINTS_PER_VOXEL
#define KISS_ICP_CUDA_MAX_POINTS_PER_VOXEL 20
#endif
inline constexpr int kMaxPointsPerVoxel = KISS_ICP_CUDA_MAX_POINTS_PER_VOXEL;

/// Default capacity (number of live voxels). Can be overridden per map instance.
#ifndef KISS_ICP_CUDA_MAX_VOXELS
#define KISS_ICP_CUDA_MAX_VOXELS (1u << 19)
#endif
inline constexpr std::size_t kDefaultMaxVoxels = KISS_ICP_CUDA_MAX_VOXELS;

/// Sum of the ICP normal equations, as returned by VoxelMapGpu::LinearSystem.
struct LinearSystemSums {
    std::array<double, 21> JTJ_upper{};  ///< row-major upper triangle of the 6x6 JTJ
    std::array<double, 6> JTr{};
    std::size_t num_correspondences = 0;
};

/// "NVIDIA GeForce RTX 3060 (sm_86, 28 SMs)" for the current device.
std::string DeviceDescription();

class VoxelMapGpu {
public:
    VoxelMapGpu(double voxel_size,
                double max_distance,
                unsigned int max_points_per_voxel,
                std::size_t max_voxels = kDefaultMaxVoxels);
    ~VoxelMapGpu();
    VoxelMapGpu(const VoxelMapGpu &) = delete;
    VoxelMapGpu &operator=(const VoxelMapGpu &) = delete;
    VoxelMapGpu(VoxelMapGpu &&) noexcept;
    VoxelMapGpu &operator=(VoxelMapGpu &&) noexcept;

    void Clear();
    std::size_t NumVoxels() const;
    std::size_t Capacity() const;

    /// Insert points (world frame, interleaved xyz, n points). Same rules as the CPU map:
    /// a point is dropped if its voxel is full or if it is closer than
    /// sqrt(voxel_size^2 / max_points_per_voxel) to a point already in that voxel.
    /// Points of the same voxel are processed in input order, so the result is deterministic
    /// and identical to the CPU version (up to float storage).
    void AddPoints(const double *xyz, std::size_t n);

    /// Drop every voxel whose first point is >= max_distance away from origin.
    void RemoveFarFromLocation(const double origin[3]);

    /// All map points, interleaved xyz (order unspecified, like iterating the robin_map).
    std::vector<double> Pointcloud() const;

    /// Batched nearest neighbour over the 27-voxel neighbourhood.
    /// nn_xyz receives 3*n doubles, dist n doubles (DBL_MAX if nothing was found).
    void ClosestNeighbors(const double *xyz, std::size_t n, double *nn_xyz, double *dist) const;

    // ---- registration support (used by Registration.cpp in the CUDA build) -------------------
    /// Upload the source frame once per AlignPointsToMap call (sensor/deskewed frame).
    void SetSource(const double *xyz, std::size_t n) const;
    /// For T (row-major 3x4: R|t) apply T to the source, find correspondences closer than
    /// max_correspondence_distance and accumulate the Geman-McClure weighted normal equations.
    LinearSystemSums LinearSystem(const double T[12],
                                  double max_correspondence_distance,
                                  double kernel_scale) const;

    /// Number of points dropped because the voxel pool was exhausted (sticky counter).
    std::size_t DroppedForCapacity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kiss_icp::gpu
