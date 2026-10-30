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

#include "KissICP.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <Eigen/Core>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t0 = Clock::now();

    // Preprocess the input cloud
    const auto &preprocessed_frame = preprocessor_.Preprocess(frame, timestamps, last_delta_);
    const auto t1 = Clock::now();

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(preprocessed_frame);
    const auto t2 = Clock::now();

    // Get adaptive_threshold
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // Compute initial_guess for ICP
    const auto initial_guess = last_pose_ * last_delta_;

    // Run ICP
    const auto new_pose = registration_.AlignPointsToMap(source,         // frame
                                                         local_map_,     // voxel_map
                                                         initial_guess,  // initial_guess
                                                         3.0 * sigma,    // max_correspondence_dist
                                                         sigma);         // kernel
    const auto t3 = Clock::now();

    // Compute the difference between the prediction and the actual estimate
    const auto model_deviation = initial_guess.inverse() * new_pose;

    // Update step: threshold, local map, delta, and the last pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
    last_delta_ = last_pose_.inverse() * new_pose;
    last_pose_ = new_pose;
    const auto t4 = Clock::now();

    last_timing_.preprocess_ms = ms(t0, t1);
    last_timing_.voxelize_ms = ms(t1, t2);
    last_timing_.align_ms = ms(t2, t3);
    last_timing_.update_ms = ms(t3, t4);
    last_timing_.total_ms = ms(t0, t4);
    last_timing_.icp_iterations = registration_.last_num_iterations_;
    last_timing_.input_points = frame.size();
    last_timing_.source_points = source.size();
    last_timing_.map_points_added = frame_downsample.size();
    if (profile_every_ > 0) {
        profile_window_.push_back(last_timing_);
        if (static_cast<int>(profile_window_.size()) >= profile_every_) ReportProfile();
    }

    // Return the (deskew) input raw scan (preprocessed_frame) and the points used for registration
    // (source)
    return {preprocessed_frame, source};
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const auto source = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}
void KissICP::ReportProfile() {
    auto median = [&](auto get) {
        std::vector<double> v;
        v.reserve(profile_window_.size());
        for (const auto &t : profile_window_) v.push_back(static_cast<double>(get(t)));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
        return v[v.size() / 2];
    };
    double mean_total = 0.0, max_total = 0.0;
    for (const auto &t : profile_window_) {
        mean_total += t.total_ms;
        max_total = std::max(max_total, t.total_ms);
    }
    mean_total /= static_cast<double>(profile_window_.size());
#ifdef KISS_ICP_CUDA
    const char *backend = "cuda";
#else
    const char *backend = "cpu";
#endif
    std::fprintf(stderr,
                 "[kiss_icp profile %s] %zu frames, median ms: preprocess %.1f | voxelize %.1f | "
                 "align %.1f (%.0f it) | update %.1f | total %.1f  (mean %.1f, max %.1f)  "
                 "points in %.0f, source %.0f, map-add %.0f\n",
                 backend, profile_window_.size(),
                 median([](const FrameTiming &t) { return t.preprocess_ms; }),
                 median([](const FrameTiming &t) { return t.voxelize_ms; }),
                 median([](const FrameTiming &t) { return t.align_ms; }),
                 median([](const FrameTiming &t) { return t.icp_iterations; }),
                 median([](const FrameTiming &t) { return t.update_ms; }),
                 median([](const FrameTiming &t) { return t.total_ms; }), mean_total, max_total,
                 median([](const FrameTiming &t) { return t.input_points; }),
                 median([](const FrameTiming &t) { return t.source_points; }),
                 median([](const FrameTiming &t) { return t.map_points_added; }));
    profile_window_.clear();
}

void KissICP::Reset() {
    last_pose_ = Sophus::SE3d();
    last_delta_ = Sophus::SE3d();

    // Clear the local map
    local_map_.Clear();

    // Reset adaptive threshold (it will start fresh)
    adaptive_threshold_ =
        AdaptiveThreshold(config_.initial_threshold, config_.min_motion_th, config_.max_range);
}

}  // namespace kiss_icp::pipeline
