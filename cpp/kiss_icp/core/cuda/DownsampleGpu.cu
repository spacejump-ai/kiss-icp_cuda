// CUDA voxel downsampling, see DownsampleGpu.hpp.
//
//   1. upload points, key[i] = voxel key of point i, idx[i] = i
//   2. stable radix sort (key, idx)          -> points of one voxel are contiguous, in input order
//   3. flag[idx] = 1 at the head of each run -> first point of each voxel
//   4. select flagged points (keeps input order), download
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

#include "DownsampleGpu.hpp"
#include "VoxelKey.cuh"

#define KISS_DS_CHECK(expr)                                                                  \
    do {                                                                                     \
        const cudaError_t kiss_cuda_err_ = (expr);                                           \
        if (kiss_cuda_err_ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("[kiss_icp::gpu] ") + #expr + " failed: " + \
                                     cudaGetErrorString(kiss_cuda_err_));                    \
        }                                                                                    \
    } while (0)

namespace kiss_icp::gpu {
namespace {
using detail::Key;

constexpr int kBlockSize = 256;

struct P3 {
    double x, y, z;
};

__global__ void KeysKernel(const P3 *__restrict__ pts,
                           int n,
                           double voxel_size,
                           Key *__restrict__ keys,
                           int *__restrict__ idx,
                           unsigned char *__restrict__ flags) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const P3 p = pts[i];
        keys[i] = detail::PackKey(detail::ToVoxel(p.x, voxel_size), detail::ToVoxel(p.y, voxel_size),
                                  detail::ToVoxel(p.z, voxel_size));
        idx[i] = i;
        flags[i] = 0;
    }
}

__global__ void FlagHeadsKernel(const Key *__restrict__ sorted_keys,
                                const int *__restrict__ sorted_idx,
                                int n,
                                unsigned char *__restrict__ flags) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        if (j == 0 || sorted_keys[j] != sorted_keys[j - 1]) flags[sorted_idx[j]] = 1;
    }
}

// Device buffers that only grow. Held by a leaked singleton: freeing device memory from a static
// destructor runs after the CUDA runtime has shut down.
class Downsampler {
public:
    void Run(const double *xyz,
             std::size_t n_points,
             double voxel_size,
             const std::function<double *(std::size_t)> &alloc_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (num_sms_ == 0) num_sms_ = InitDeviceOrThrow();
        const int n = static_cast<int>(n_points);
        Reserve(n_points);

        KISS_DS_CHECK(cudaMemcpy(pts_, xyz, n_points * sizeof(P3), cudaMemcpyHostToDevice));
        KeysKernel<<<Grid(n_points), kBlockSize>>>(pts_, n, voxel_size, keys_, idx_, flags_);
        KISS_DS_CHECK(cudaGetLastError());

        // Radix sort is stable: equal keys keep their input order.
        std::size_t bytes = 0;
        KISS_DS_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_, keys_sorted_, idx_,
                                                      idx_sorted_, n, 0, detail::kKeyTotalBits));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceRadixSort::SortPairs(temp_, bytes, keys_, keys_sorted_, idx_,
                                                      idx_sorted_, n, 0, detail::kKeyTotalBits));

        FlagHeadsKernel<<<Grid(n_points), kBlockSize>>>(keys_sorted_, idx_sorted_, n, flags_);
        KISS_DS_CHECK(cudaGetLastError());

        KISS_DS_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, pts_, flags_, pts_out_, num_out_, n));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceSelect::Flagged(temp_, bytes, pts_, flags_, pts_out_, num_out_, n));

        int m = 0;
        KISS_DS_CHECK(cudaMemcpy(&m, num_out_, sizeof(int), cudaMemcpyDeviceToHost));
        double *out = alloc_out(static_cast<std::size_t>(m));
        if (m > 0) {
            KISS_DS_CHECK(cudaMemcpy(out, pts_out_, static_cast<std::size_t>(m) * sizeof(P3),
                                     cudaMemcpyDeviceToHost));
        }
    }

private:
    int Grid(std::size_t work) {
        const std::size_t blocks = (work + kBlockSize - 1) / kBlockSize;
        const std::size_t cap = static_cast<std::size_t>(num_sms_) * 8;
        return static_cast<int>(blocks < cap ? blocks : cap);
    }

    template <typename T>
    static void Grow(T *&ptr, std::size_t count) {
        if (ptr != nullptr) KISS_DS_CHECK(cudaFree(ptr));
        ptr = nullptr;
        KISS_DS_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }

    void Reserve(std::size_t n) {
        if (n <= capacity_) return;
        const std::size_t cap = n + n / 4;  // headroom so slightly bigger scans don't reallocate
        Grow(pts_, cap);
        Grow(pts_out_, cap);
        Grow(keys_, cap);
        Grow(keys_sorted_, cap);
        Grow(idx_, cap);
        Grow(idx_sorted_, cap);
        Grow(flags_, cap);
        if (num_out_ == nullptr) KISS_DS_CHECK(cudaMalloc(&num_out_, sizeof(int)));
        capacity_ = cap;
    }

    void Temp(std::size_t bytes) {
        if (bytes <= temp_bytes_) return;
        Grow(temp_, bytes);
        temp_bytes_ = bytes;
    }

    std::mutex mutex_;
    int num_sms_ = 0;
    std::size_t capacity_ = 0, temp_bytes_ = 0;
    P3 *pts_ = nullptr, *pts_out_ = nullptr;
    Key *keys_ = nullptr, *keys_sorted_ = nullptr;
    int *idx_ = nullptr, *idx_sorted_ = nullptr, *num_out_ = nullptr;
    unsigned char *flags_ = nullptr, *temp_ = nullptr;
};

}  // namespace

void VoxelDownsample(const double *xyz,
                     std::size_t n,
                     double voxel_size,
                     const std::function<double *(std::size_t)> &alloc_out) {
    if (n == 0) {
        alloc_out(0);
        return;
    }
    if (n > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("[kiss_icp::gpu] VoxelDownsample: too many points");
    }
    static Downsampler *instance = new Downsampler();  // intentionally leaked, see class comment
    instance->Run(xyz, n, voxel_size, alloc_out);
}

}  // namespace kiss_icp::gpu
