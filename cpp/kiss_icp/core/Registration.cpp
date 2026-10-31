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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include "Profile.hpp"
#include "VoxelHashMap.hpp"
#include "VoxelUtils.hpp"

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

using Correspondences = tbb::concurrent_vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
using LinearSystem = std::pair<Eigen::Matrix6d, Eigen::Vector6d>;

#ifndef KISS_ICP_CUDA
namespace {
inline double square(double x) { return x * x; }

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

Correspondences DataAssociation(const std::vector<Eigen::Vector3d> &points,
                                const kiss_icp::VoxelHashMap &voxel_map,
                                const double max_correspondance_distance) {
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    Correspondences correspondences;
    correspondences.reserve(points.size());
    tbb::parallel_for(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        [&](const tbb::blocked_range<points_iterator> &r) {
            std::for_each(r.begin(), r.end(), [&](const auto &point) {
                const auto &[closest_neighbor, distance] = voxel_map.GetClosestNeighbor(point);
                if (distance < max_correspondance_distance) {
                    correspondences.emplace_back(point, closest_neighbor);
                }
            });
        });
    return correspondences;
}

LinearSystem BuildLinearSystem(const Correspondences &correspondences, const double kernel_scale) {
    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto &[source, target] = correspondence;
        const Eigen::Vector3d residual = source - target;
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
        return std::make_tuple(J_r, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto GM_weight = [&](const double &residual2) {
        return square(kernel_scale) / square(kernel_scale + residual2);
    };

    using correspondence_iterator = Correspondences::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(),
                                                    correspondences.cend()},
        // Identity
        LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<correspondence_iterator> &r, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                r.begin(), r.end(), J, sum_linear_systems, [&](const auto &correspondence) {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(correspondence);
                    const double w = GM_weight(residual.squaredNorm());
                    return LinearSystem(J_r.transpose() * w * J_r,        // JTJ
                                        J_r.transpose() * w * residual);  // JTr
                });
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        sum_linear_systems);

    return {JTJ, JTr};
}
}  // namespace
#endif  // !KISS_ICP_CUDA

namespace kiss_icp {

#ifdef KISS_ICP_CUDA
namespace {
// Data association + normal equations on the GPU (VoxelMapGpu::LinearSystem).
LinearSystem BuildLinearSystemGpu(const VoxelHashMap &voxel_map,
                                  const Sophus::SE3d &T,
                                  const double max_correspondence_distance,
                                  const double kernel_scale) {
    const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> T34 = T.matrix3x4();
    const auto sums = voxel_map.Gpu().LinearSystem(T34.data(), max_correspondence_distance, kernel_scale);
    Eigen::Matrix6d JTJ;
    int idx = 0;
    for (int a = 0; a < 6; ++a) {
        for (int b = a; b < 6; ++b) {
            JTJ(a, b) = JTJ(b, a) = sums.JTJ_upper[static_cast<std::size_t>(idx++)];
        }
    }
    const Eigen::Vector6d JTr = Eigen::Map<const Eigen::Vector6d>(sums.JTr.data());
    return {JTJ, JTr};
}
}  // namespace
#endif

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      // Only manipulate the number of threads if the user specifies something greater than 0
      max_num_threads_(max_num_threads > 0 ? max_num_threads
                                           : tbb::this_task_arena::max_concurrency()) {
    // This global variable requires static duration storage to be able to manipulate the max
    // concurrency from TBB across the entire class
    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads_));
}

Sophus::SE3d Registration::AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                                            const VoxelHashMap &voxel_map,
                                            const Sophus::SE3d &initial_guess,
                                            const double max_distance,
                                            const double kernel_scale) {
    last_num_iterations_ = 0;
    profile::BeginFrame();
    const profile::Scope profile_scope(profile::kAlign);
    struct AlignInfo {  // reports iterations/source size on every return path
        const int &iterations;
        std::size_t points;
        ~AlignInfo() { profile::SetAlignInfo(iterations, points); }
    } align_info{last_num_iterations_, frame.size()};
    if (voxel_map.Empty()) return initial_guess;

#ifdef KISS_ICP_CUDA
    // The source frame is uploaded once; each iteration only sends the current pose (12 doubles)
    // and reads back the 6x6 system (28 doubles). Iterate exactly like the CPU version:
    // source_j = T_icp * initial_guess * frame.
    voxel_map.Gpu().SetSource(frame.empty() ? nullptr : frame.front().data(), frame.size());
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < max_num_iterations_; ++j) {
        // Equations (10) + (11)
        const auto &[JTJ, JTr] = BuildLinearSystemGpu(voxel_map, T_icp * initial_guess, max_distance, kernel_scale);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        T_icp = estimation * T_icp;
        last_num_iterations_ = j + 1;
        if (dx.norm() < convergence_criterion_) break;
    }
    return T_icp * initial_guess;
#else
    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < max_num_iterations_; ++j) {
        // Equation (10)
        const auto correspondences = DataAssociation(source, voxel_map, max_distance);
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(correspondences, kernel_scale);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        last_num_iterations_ = j + 1;
        // Termination criteria
        if (dx.norm() < convergence_criterion_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
#endif
}

}  // namespace kiss_icp
