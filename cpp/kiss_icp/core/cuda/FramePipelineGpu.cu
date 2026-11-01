// See FramePipelineGpu.hpp.
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

#include "../Profile.hpp"
#include "FramePipelineGpu.hpp"
#include "Internal.cuh"
#include "VoxelKey.cuh"  // InitDeviceOrThrow

#define KISS_FP_CHECK(expr)                                                                  \
    do {                                                                                     \
        const cudaError_t kiss_cuda_err_ = (expr);                                           \
        if (kiss_cuda_err_ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("[kiss_icp::gpu] ") + #expr + " failed: " + \
                                     cudaGetErrorString(kiss_cuda_err_));                    \
        }                                                                                    \
    } while (0)

namespace kiss_icp::gpu {
namespace {

// Leaked singleton (device memory must not be freed from a static destructor at exit).
class FramePipeline {
public:
    void Run(const double *xyz,
             std::size_t n,
             const double *timestamps,
             double t_min,
             double t_max,
             const double omega[6],
             double min_range,
             double max_range,
             double voxel_ds,
             double voxel_src,
             const std::function<double *(std::size_t)> &alloc_frame,
             const std::function<double *(std::size_t)> &alloc_frame_ds,
             const std::function<double *(std::size_t)> &alloc_source) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!device_ok_) {
            InitDeviceOrThrow();
            device_ok_ = true;
        }
        Reserve(n);

        std::size_t m_frame = 0, m_ds = 0, m_src = 0;
        {
            const profile::Scope scope(profile::kPreprocess);
            KISS_FP_CHECK(cudaMemcpy(in_, xyz, 3 * n * sizeof(double), cudaMemcpyHostToDevice));
            if (timestamps != nullptr) {
                KISS_FP_CHECK(cudaMemcpy(stamps_, timestamps, n * sizeof(double), cudaMemcpyHostToDevice));
            }
            m_frame = detail::PreprocessDevice(in_, timestamps != nullptr ? stamps_ : nullptr, n, t_min,
                                               t_max, omega, min_range, max_range, frame_);
            if (alloc_frame) Download(alloc_frame, frame_, m_frame);
        }
        {
            const profile::Scope scope(profile::kVoxelize);
            m_ds = detail::VoxelDownsampleDevice(frame_, m_frame, voxel_ds, ds_);
            m_src = detail::VoxelDownsampleDevice(ds_, m_ds, voxel_src, src_);
            Download(alloc_frame_ds, ds_, m_ds);
            Download(alloc_source, src_, m_src);
        }
    }

private:
    static void Download(const std::function<double *(std::size_t)> &alloc, const double *d, std::size_t m) {
        double *h = alloc(m);
        if (m > 0) KISS_FP_CHECK(cudaMemcpy(h, d, 3 * m * sizeof(double), cudaMemcpyDeviceToHost));
    }

    static void Grow(double *&ptr, std::size_t count) {
        if (ptr != nullptr) KISS_FP_CHECK(cudaFree(ptr));
        ptr = nullptr;
        KISS_FP_CHECK(cudaMalloc(&ptr, count * sizeof(double)));
    }

    void Reserve(std::size_t n) {
        if (n <= capacity_) return;
        const std::size_t cap = n + n / 4;
        Grow(in_, 3 * cap);
        Grow(frame_, 3 * cap);
        Grow(ds_, 3 * cap);
        Grow(src_, 3 * cap);
        Grow(stamps_, cap);
        capacity_ = cap;
    }

    std::mutex mutex_;
    bool device_ok_ = false;
    std::size_t capacity_ = 0;
    double *in_ = nullptr, *frame_ = nullptr, *ds_ = nullptr, *src_ = nullptr, *stamps_ = nullptr;
};

FramePipeline &Instance() {
    static FramePipeline *instance = new FramePipeline();  // intentionally leaked
    return *instance;
}

}  // namespace

void PreprocessAndVoxelize(const double *xyz,
                           std::size_t n,
                           const double *timestamps,
                           double t_min,
                           double t_max,
                           const double omega[6],
                           double min_range,
                           double max_range,
                           double voxel_ds,
                           double voxel_src,
                           const std::function<double *(std::size_t)> &alloc_frame,
                           const std::function<double *(std::size_t)> &alloc_frame_ds,
                           const std::function<double *(std::size_t)> &alloc_source) {
    if (n > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("[kiss_icp::gpu] PreprocessAndVoxelize: too many points");
    }
    if (n == 0) {
        if (alloc_frame) alloc_frame(0);
        alloc_frame_ds(0);
        alloc_source(0);
        return;
    }
    Instance().Run(xyz, n, timestamps, t_min, t_max, omega, min_range, max_range, voxel_ds, voxel_src,
                   alloc_frame, alloc_frame_ds, alloc_source);
}

}  // namespace kiss_icp::gpu
