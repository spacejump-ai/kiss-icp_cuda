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
#pragma once

#include <Eigen/Core>
#include <functional>
#include <sophus/se3.hpp>
#include <vector>

namespace kiss_icp {

struct PreprocessedFrame {
    std::vector<Eigen::Vector3d> frame;             ///< deskewed + range cropped (empty if not requested)
    std::vector<Eigen::Vector3d> frame_downsample;  ///< VoxelDownsample(frame, voxel_ds): goes into the map
    std::vector<Eigen::Vector3d> source;            ///< VoxelDownsample(frame_downsample, voxel_src): ICP input
};

struct Preprocessor {
    Preprocessor(const double max_range,
                 const double min_range,
                 const bool deskew,
                 const int max_num_threads);

    std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                            const std::vector<double> &timestamps,
                                            const Sophus::SE3d &relative_motion) const;
    /// The original CPU implementation (TBB). Preprocess() uses the GPU in the CUDA build for
    /// inputs >= gpu::kPreprocessMinPoints and this otherwise.
    std::vector<Eigen::Vector3d> PreprocessCpu(const std::vector<Eigen::Vector3d> &frame,
                                               const std::vector<double> &timestamps,
                                               const Sophus::SE3d &relative_motion) const;
    /// Preprocess + the two VoxelDownsample calls of KissICP::Voxelize in one go
    /// (voxel_ds = voxel_size * 0.5, voxel_src = voxel_size * 1.5 there).
    /// CUDA build: the scan is uploaded once and stays on the GPU between the steps; results are
    /// identical to calling Preprocess and VoxelDownsample separately. keep_frame = false skips
    /// downloading the full preprocessed scan.
    PreprocessedFrame PreprocessAndVoxelize(const std::vector<Eigen::Vector3d> &frame,
                                            const std::vector<double> &timestamps,
                                            const Sophus::SE3d &relative_motion,
                                            double voxel_ds,
                                            double voxel_src,
                                            bool keep_frame = true) const;
    double max_range_;
    double min_range_;
    bool deskew_;
    int max_num_threads_;
};
}  // namespace kiss_icp
