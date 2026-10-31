// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <tuple>
#include <vector>

#include "VoxelUtils.hpp"

#ifdef KISS_ICP_CUDA
// ---------------------------------------------------------------------------------------------
// CUDA build: the local map lives on the GPU (cuco::static_map, see cuda/VoxelMapGpu.hpp).
// Public API is unchanged, so KissICP.cpp and the python bindings compile as-is.
// ---------------------------------------------------------------------------------------------
#include <memory>

#include "cuda/VoxelMapGpu.hpp"

namespace kiss_icp {
struct VoxelHashMap {
    explicit VoxelHashMap(double voxel_size,
                          double max_distance,
                          unsigned int max_points_per_voxel,
                          std::size_t max_voxels = gpu::kDefaultMaxVoxels);

    void Clear();
    bool Empty() const;
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;
    /// Single query, kept for API compatibility. Each call is a GPU round trip, so prefer
    /// GetClosestNeighbors() (batched) or Registration (fully on the GPU).
    std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d &query) const;
    std::vector<std::tuple<Eigen::Vector3d, double>> GetClosestNeighbors(
        const std::vector<Eigen::Vector3d> &queries) const;

    std::size_t NumVoxels() const { return gpu_->NumVoxels(); }
    const gpu::VoxelMapGpu &Gpu() const { return *gpu_; }

    double voxel_size_;
    double max_distance_;
    unsigned int max_points_per_voxel_;
    std::unique_ptr<gpu::VoxelMapGpu> gpu_;
};
}  // namespace kiss_icp

#else
// ---------------------------------------------------------------------------------------------
// CPU build (original implementation)
// ---------------------------------------------------------------------------------------------
#include <tsl/robin_map.h>

namespace kiss_icp {
struct VoxelHashMap {
    explicit VoxelHashMap(double voxel_size, double max_distance, unsigned int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    inline std::size_t NumVoxels() const { return map_.size(); }
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;
    std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d &query) const;
    /// Batched version (TBB parallel). Same call exists in the CUDA build, where it is one GPU
    /// launch instead of one round trip per point, so prefer it in user code.
    std::vector<std::tuple<Eigen::Vector3d, double>> GetClosestNeighbors(
        const std::vector<Eigen::Vector3d> &queries) const;

    double voxel_size_;
    double max_distance_;
    unsigned int max_points_per_voxel_;
    tsl::robin_map<Voxel, std::vector<Eigen::Vector3d>> map_;
};
}  // namespace kiss_icp
#endif
