// CUDA version of kiss_icp::Preprocessor::Preprocess (deskew + range crop). Plain C++ header.
#pragma once

#include <cstddef>
#include <functional>

namespace kiss_icp::gpu {

/// Inputs smaller than this stay on the CPU.
inline constexpr std::size_t kPreprocessMinPoints = 4096;

/// Same as the CPU Preprocess:
///  - deskew (if timestamps != nullptr): p' = SE3::exp((s_i - 1) * omega) * p_i with
///    s_i = (t_i - t_min) / (t_max - t_min), mirroring Sophus' SE3::exp step by step;
///  - keep points with min_range < |p'| < max_range, in input order.
/// omega: relative_motion.log() (Sophus order: translation part, then rotation part).
/// Without deskew the result is bit-identical to the CPU version; with deskew the points can
/// differ by a few ulp (GPU vs libm sin/cos).
/// alloc_out(m) must return storage for 3*m doubles. Thread safe (calls are serialized).
void Preprocess(const double *xyz,
                std::size_t n,
                const double *timestamps,  // n values or nullptr (no deskew)
                double t_min,
                double t_max,
                const double omega[6],
                double min_range,
                double max_range,
                const std::function<double *(std::size_t)> &alloc_out);

}  // namespace kiss_icp::gpu
