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
#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <tuple>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include <cstdlib>
#include <vector>

#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/Threshold.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

/// Per-stage timing of RegisterFrame. Enabled with the environment variable KISS_ICP_PROFILE
/// (its value is the report interval in frames, e.g. KISS_ICP_PROFILE=100; any non-number = 50).
/// Reports go to stderr; stats() can also be read directly.
struct FrameTiming {
    double preprocess_ms = 0.0, voxelize_ms = 0.0, align_ms = 0.0, update_ms = 0.0,
           total_ms = 0.0;
    int icp_iterations = 0;
    std::size_t input_points = 0, source_points = 0, map_points_added = 0;
};

struct KISSConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 0.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // registration params
    int max_num_iterations = 500;
    double convergence_criterion = 0.0001;
    int max_num_threads = 0;

    // Motion compensation
    bool deskew = true;
};

class KissICP {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

public:
    explicit KissICP(const KISSConfig &config)
        : config_(config),
          preprocessor_(config.max_range, config.min_range, config.deskew, config.max_num_threads),
          registration_(
              config.max_num_iterations, config.convergence_criterion, config.max_num_threads),
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {
        if (const char *env = std::getenv("KISS_ICP_PROFILE")) {
            const int n = std::atoi(env);
            profile_every_ = n > 0 ? n : 50;
        }
    }

public:
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                      const std::vector<double> &timestamps);
    Vector3dVectorTuple Voxelize(const std::vector<Eigen::Vector3d> &frame) const;

    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); };

    const VoxelHashMap &VoxelMap() const { return local_map_; };
    VoxelHashMap &VoxelMap() { return local_map_; };

    const Sophus::SE3d &pose() const { return last_pose_; }
    Sophus::SE3d &pose() { return last_pose_; }

    const Sophus::SE3d &delta() const { return last_delta_; }
    Sophus::SE3d &delta() { return last_delta_; }
    void Reset();

    /// Timing of the most recent RegisterFrame call (always measured, cheap).
    const FrameTiming &LastTiming() const { return last_timing_; }

private:
    void ReportProfile();
    FrameTiming last_timing_;
    int profile_every_ = 0;  // 0 = no reports
    std::vector<FrameTiming> profile_window_;

    Sophus::SE3d last_pose_;
    Sophus::SE3d last_delta_;

    // KISS-ICP pipeline modules
    KISSConfig config_;
    Preprocessor preprocessor_;
    Registration registration_;
    VoxelHashMap local_map_;
    AdaptiveThreshold adaptive_threshold_;
};

}  // namespace kiss_icp::pipeline
