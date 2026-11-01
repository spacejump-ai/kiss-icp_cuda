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
#include "Preprocessing.hpp"

#include "Profile.hpp"
#include "VoxelUtils.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#ifdef KISS_ICP_CUDA
#include "cuda/FramePipelineGpu.hpp"
#include "cuda/PreprocessGpu.hpp"
#endif

namespace kiss_icp {

#ifdef KISS_ICP_CUDA
namespace {
struct GpuArgs {
    const double *stamps = nullptr;  // nullptr: no deskew
    double t_min = 0.0, t_max = 0.0;
    Sophus::SE3d::Tangent omega;
};

GpuArgs MakeGpuArgs(bool deskew_enabled,
                    const std::vector<Eigen::Vector3d> &frame,
                    const std::vector<double> &timestamps,
                    const Sophus::SE3d &relative_motion) {
    GpuArgs g;
    g.omega = relative_motion.log();
    if (deskew_enabled && !timestamps.empty()) {
        if (timestamps.size() < frame.size()) {  // the CPU version throws from .at() too
            throw std::out_of_range("kiss_icp::Preprocess: fewer timestamps than points");
        }
        const auto [mn, mx] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
        g.stamps = timestamps.data();
        g.t_min = *mn;
        g.t_max = *mx;
    }
    return g;
}

// alloc callback writing into a std::vector<Eigen::Vector3d> (3 packed doubles per point)
std::function<double *(std::size_t)> Sink(std::vector<Eigen::Vector3d> &v) {
    static_assert(sizeof(Eigen::Vector3d) == 3 * sizeof(double), "unexpected Eigen layout");
    return [&v](std::size_t m) {
        v.resize(m);
        return m > 0 ? v.front().data() : nullptr;
    };
}
}  // namespace
#endif

Preprocessor::Preprocessor(const double max_range,
                           const double min_range,
                           const bool deskew,
                           const int max_num_threads)
    : max_range_(max_range),
      min_range_(min_range),
      deskew_(deskew),
      max_num_threads_(max_num_threads > 0 ? max_num_threads
                                           : tbb::this_task_arena::max_concurrency()) {
    // This global variable requires static duration storage to be able to manipulate the max
    // concurrency from TBB across the entire class
    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads_));
}

std::vector<Eigen::Vector3d> Preprocessor::Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                                      const std::vector<double> &timestamps,
                                                      const Sophus::SE3d &relative_motion) const {
    const profile::Scope profile_scope(profile::kPreprocess);
#ifdef KISS_ICP_CUDA
    if (frame.size() >= gpu::kPreprocessMinPoints) {
        const GpuArgs g = MakeGpuArgs(deskew_, frame, timestamps, relative_motion);
        std::vector<Eigen::Vector3d> out;
        gpu::Preprocess(frame.front().data(), frame.size(), g.stamps, g.t_min, g.t_max,
                        g.omega.data(), min_range_, max_range_, Sink(out));
        return out;
    }
#endif
    return PreprocessCpu(frame, timestamps, relative_motion);
}

PreprocessedFrame Preprocessor::PreprocessAndVoxelize(const std::vector<Eigen::Vector3d> &frame,
                                                       const std::vector<double> &timestamps,
                                                       const Sophus::SE3d &relative_motion,
                                                       double voxel_ds,
                                                       double voxel_src,
                                                       bool keep_frame) const {
    PreprocessedFrame out;
#ifdef KISS_ICP_CUDA
    if (frame.size() >= gpu::kPreprocessMinPoints) {
        const GpuArgs g = MakeGpuArgs(deskew_, frame, timestamps, relative_motion);
        gpu::PreprocessAndVoxelize(frame.front().data(), frame.size(), g.stamps, g.t_min, g.t_max,
                                   g.omega.data(), min_range_, max_range_, voxel_ds, voxel_src,
                                   keep_frame ? Sink(out.frame) : nullptr,
                                   Sink(out.frame_downsample), Sink(out.source));
        return out;
    }
#endif
    out.frame = Preprocess(frame, timestamps, relative_motion);
    out.frame_downsample = VoxelDownsample(out.frame, voxel_ds);
    out.source = VoxelDownsample(out.frame_downsample, voxel_src);
    if (!keep_frame) std::vector<Eigen::Vector3d>().swap(out.frame);
    return out;
}

std::vector<Eigen::Vector3d> Preprocessor::PreprocessCpu(const std::vector<Eigen::Vector3d> &frame,
                                                         const std::vector<double> &timestamps,
                                                         const Sophus::SE3d &relative_motion) const {
    const std::vector<Eigen::Vector3d> &deskewed_frame = [&]() {
        if (!deskew_ || timestamps.empty()) {
            return frame;
        } else {
            const auto &[min, max] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
            const double min_time = *min;
            const double max_time = *max;
            const auto normalize = [&](const double t) {
                return (t - min_time) / (max_time - min_time);
            };
            const auto &omega = relative_motion.log();
            std::vector<Eigen::Vector3d> deskewed_frame(frame.size());
            tbb::parallel_for(
                // Index Range
                tbb::blocked_range<size_t>{0, deskewed_frame.size()},
                // Parallel Compute
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t idx = r.begin(); idx < r.end(); ++idx) {
                        const auto &point = frame.at(idx);
                        const auto &stamp = normalize(timestamps.at(idx));
                        const auto pose = Sophus::SE3d::exp((stamp - 1.0) * omega);
                        deskewed_frame.at(idx) = pose * point;
                    };
                });
            return deskewed_frame;
        }
    }();
    std::vector<Eigen::Vector3d> preprocessed_frame;
    preprocessed_frame.reserve(deskewed_frame.size());
    std::for_each(deskewed_frame.cbegin(), deskewed_frame.cend(), [&](const auto &point) {
        const double point_range = point.norm();
        if (point_range < max_range_ && point_range > min_range_) {
            preprocessed_frame.emplace_back(point);
        }
    });
    preprocessed_frame.shrink_to_fit();
    return preprocessed_frame;
}
}  // namespace kiss_icp
