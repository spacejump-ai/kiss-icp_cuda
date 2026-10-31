// CUDA deskew + range crop, see PreprocessGpu.hpp.
// Compiled with -fmad=false (CMake) so the arithmetic is evaluated like the CPU code (no FMA).
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

#include "PreprocessGpu.hpp"
#include "VoxelKey.cuh"  // InitDeviceOrThrow

#define KISS_PP_CHECK(expr)                                                                  \
    do {                                                                                     \
        const cudaError_t kiss_cuda_err_ = (expr);                                           \
        if (kiss_cuda_err_ != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("[kiss_icp::gpu] ") + #expr + " failed: " + \
                                     cudaGetErrorString(kiss_cuda_err_));                    \
        }                                                                                    \
    } while (0)

namespace kiss_icp::gpu {
namespace {

constexpr int kBlockSize = 256;

struct P3 {
    double x, y, z;
};

struct Twist {
    double v[6];  // Sophus order: upsilon (translation part), omega (rotation part)
};

// Sophus::SE3d::exp(a) * p, written out like Sophus 1.24 (SO3::expAndTheta, SO3::leftJacobian,
// Eigen's Quaternion::_transformVector).
__device__ inline P3 ExpTimesPoint(const double a[6], const P3 &p) {
    const double ux = a[0], uy = a[1], uz = a[2];
    const double wx = a[3], wy = a[4], wz = a[5];
    constexpr double kEps = 1e-10;

    // SO3::expAndTheta
    const double theta_sq = wx * wx + wy * wy + wz * wz;
    double theta, imag, real;
    if (theta_sq < kEps * kEps) {
        theta = 0.0;
        const double theta_po4 = theta_sq * theta_sq;
        imag = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4;
        real = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4;
    } else {
        theta = sqrt(theta_sq);
        const double half_theta = 0.5 * theta;
        imag = sin(half_theta) / theta;
        real = cos(half_theta);
    }
    const double qw = real, qx = imag * wx, qy = imag * wy, qz = imag * wz;

    // SO3::leftJacobian(omega, theta): theta is passed, so theta_sq is recomputed from it
    const double th_sq = theta * theta;
    // Omega = hat(w), Omega_sq = Omega * Omega
    const double O[3][3] = {{0.0, -wz, wy}, {wz, 0.0, -wx}, {-wy, wx, 0.0}};
    double V[3][3];
    if (th_sq < kEps * kEps) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) V[r][c] = (r == c ? 1.0 : 0.0) + 0.5 * O[r][c];
    } else {
        const double c1 = (1.0 - cos(theta)) / th_sq;
        const double c2 = (theta - sin(theta)) / (th_sq * theta);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const double o2 = O[r][0] * O[0][c] + O[r][1] * O[1][c] + O[r][2] * O[2][c];
                V[r][c] = ((r == c ? 1.0 : 0.0) + c1 * O[r][c]) + c2 * o2;
            }
        }
    }
    const double tx = V[0][0] * ux + V[0][1] * uy + V[0][2] * uz;
    const double ty = V[1][0] * ux + V[1][1] * uy + V[1][2] * uz;
    const double tz = V[2][0] * ux + V[2][1] * uy + V[2][2] * uz;

    // q * p (Eigen: uv = q.vec() x p; uv += uv; p + w * uv + q.vec() x uv), then + t
    double uvx = qy * p.z - qz * p.y;
    double uvy = qz * p.x - qx * p.z;
    double uvz = qx * p.y - qy * p.x;
    uvx += uvx;
    uvy += uvy;
    uvz += uvz;
    const double rx = p.x + qw * uvx + (qy * uvz - qz * uvy);
    const double ry = p.y + qw * uvy + (qz * uvx - qx * uvz);
    const double rz = p.z + qw * uvz + (qx * uvy - qy * uvx);
    return P3{rx + tx, ry + ty, rz + tz};
}

__global__ void PreprocessKernel(const P3 *__restrict__ in,
                                 const double *__restrict__ stamps,
                                 int n,
                                 double t_min,
                                 double t_span,
                                 Twist omega,
                                 double min_range,
                                 double max_range,
                                 P3 *__restrict__ out,
                                 unsigned char *__restrict__ keep) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        P3 p = in[i];
        if (stamps != nullptr) {
            const double s = (stamps[i] - t_min) / t_span;
            double a[6];
            for (int k = 0; k < 6; ++k) a[k] = (s - 1.0) * omega.v[k];
            p = ExpTimesPoint(a, p);
        }
        // Eigen's norm(): sqrt((x*x + y*y) + z*z), explicitly rounded (never fused)
        const double r = sqrt(__dadd_rn(__dadd_rn(__dmul_rn(p.x, p.x), __dmul_rn(p.y, p.y)),
                                        __dmul_rn(p.z, p.z)));
        out[i] = p;
        keep[i] = (r < max_range && r > min_range) ? 1 : 0;
    }
}

// Leaked singleton (device memory must not be freed from a static destructor at exit).
class Preprocessor {
public:
    void Run(const double *xyz,
             std::size_t n_points,
             const double *timestamps,
             double t_min,
             double t_max,
             const double omega[6],
             double min_range,
             double max_range,
             const std::function<double *(std::size_t)> &alloc_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (num_sms_ == 0) num_sms_ = InitDeviceOrThrow();
        const int n = static_cast<int>(n_points);
        Reserve(n_points);

        KISS_PP_CHECK(cudaMemcpy(in_, xyz, n_points * sizeof(P3), cudaMemcpyHostToDevice));
        if (timestamps != nullptr) {
            KISS_PP_CHECK(cudaMemcpy(stamps_, timestamps, n_points * sizeof(double),
                                     cudaMemcpyHostToDevice));
        }
        Twist w;
        for (int k = 0; k < 6; ++k) w.v[k] = omega[k];
        PreprocessKernel<<<Grid(n_points), kBlockSize>>>(
            in_, timestamps != nullptr ? stamps_ : nullptr, n, t_min, t_max - t_min, w, min_range,
            max_range, deskewed_, keep_);
        KISS_PP_CHECK(cudaGetLastError());

        std::size_t bytes = 0;
        KISS_PP_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, deskewed_, keep_, out_, num_out_, n));
        Temp(bytes);
        KISS_PP_CHECK(cub::DeviceSelect::Flagged(temp_, bytes, deskewed_, keep_, out_, num_out_, n));

        int m = 0;
        KISS_PP_CHECK(cudaMemcpy(&m, num_out_, sizeof(int), cudaMemcpyDeviceToHost));
        double *dst = alloc_out(static_cast<std::size_t>(m));
        if (m > 0) {
            KISS_PP_CHECK(cudaMemcpy(dst, out_, static_cast<std::size_t>(m) * sizeof(P3),
                                     cudaMemcpyDeviceToHost));
        }
    }

private:
    int Grid(std::size_t work) const {
        const std::size_t blocks = (work + kBlockSize - 1) / kBlockSize;
        const std::size_t cap = static_cast<std::size_t>(num_sms_) * 8;
        return static_cast<int>(blocks < cap ? blocks : cap);
    }

    template <typename T>
    static void Grow(T *&ptr, std::size_t count) {
        if (ptr != nullptr) KISS_PP_CHECK(cudaFree(ptr));
        ptr = nullptr;
        KISS_PP_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }

    void Reserve(std::size_t n) {
        if (n <= capacity_) return;
        const std::size_t cap = n + n / 4;
        Grow(in_, cap);
        Grow(deskewed_, cap);
        Grow(out_, cap);
        Grow(stamps_, cap);
        Grow(keep_, cap);
        if (num_out_ == nullptr) KISS_PP_CHECK(cudaMalloc(&num_out_, sizeof(int)));
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
    P3 *in_ = nullptr, *deskewed_ = nullptr, *out_ = nullptr;
    double *stamps_ = nullptr;
    unsigned char *keep_ = nullptr, *temp_ = nullptr;
    int *num_out_ = nullptr;
};

}  // namespace

void Preprocess(const double *xyz,
                std::size_t n,
                const double *timestamps,
                double t_min,
                double t_max,
                const double omega[6],
                double min_range,
                double max_range,
                const std::function<double *(std::size_t)> &alloc_out) {
    if (n == 0) {
        alloc_out(0);
        return;
    }
    if (n > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("[kiss_icp::gpu] Preprocess: too many points");
    }
    static Preprocessor *instance = new Preprocessor();  // intentionally leaked
    instance->Run(xyz, n, timestamps, t_min, t_max, omega, min_range, max_range, alloc_out);
}

}  // namespace kiss_icp::gpu
