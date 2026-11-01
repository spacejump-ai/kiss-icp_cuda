// CUDA voxel downsampling, see DownsampleGpu.hpp.
//
// Must return exactly what the CPU version returns, *including the order*: the CPU iterates a
// tsl::robin_map, so its output order is pseudo-random in space, and KISS-ICP depends on that
// order (the second downsample and the map insertion keep the first point per voxel). Returning
// points in scan order instead biases which point represents a voxel (grid-locked pattern, more
// drift). So the robin_map iteration order is reproduced:
//
//   1. key[i] = voxel of point i; stable radix sort (key, i); the head of each run is the first
//      point of its voxel (= the point the CPU inserts).
//   2. home bucket of each voxel = std::hash<Voxel> & (B - 1), with B the bucket count the CPU's
//      grid.reserve(n) produces (next power of two >= ceil(n / 0.5)).
//   3. sort voxels by (home, first index). Robin-hood hashing keeps entries ordered by home
//      bucket, so slot positions follow from a prefix max: pos_r = r + max_{j<=r}(home_j - j)
//      (with a rotation if the table wraps around its end).
//   4. inside each cluster of consecutive occupied slots, replay tsl's insertions (in first-index
//      order, with its displacement rule) to get the order of entries that share a home bucket.
//   5. output = entries by slot index. Pathological cases (cluster longer than kMaxCluster, table
//      nearly full) fall back to the CPU implementation.
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "DownsampleGpu.hpp"
#include "Internal.cuh"
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
using u64 = unsigned long long;
using i64 = long long;

constexpr int kBlockSize = 256;
constexpr int kMaxCluster = 128;

// counters_[] slots
constexpr int kNumHeads = 0;
constexpr int kNumClusters = 1;
constexpr int kNegatives = 2;
constexpr int kFallback = 3;
constexpr int kLastStart = 4;
constexpr int kNumCounters = 5;

struct P3 {
    double x, y, z;
};

struct MaxOp {
    __device__ i64 operator()(i64 a, i64 b) const { return a > b ? a : b; }
};

// std::hash<kiss_icp::Voxel> from VoxelUtils.hpp, bit for bit (uint32 arithmetic).
__device__ inline u64 CpuVoxelHash(int x, int y, int z) {
    const unsigned int h = static_cast<unsigned int>(x) * 73856093u ^
                           static_cast<unsigned int>(y) * 19349669u ^
                           static_cast<unsigned int>(z) * 83492791u;
    return static_cast<u64>(h);
}

__global__ void KeysKernel(const P3 *__restrict__ pts, int n, double voxel_size, Key *__restrict__ keys,
                           int *__restrict__ idx) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const P3 p = pts[i];
        keys[i] = detail::PackKey(detail::ToVoxel(p.x, voxel_size), detail::ToVoxel(p.y, voxel_size),
                                  detail::ToVoxel(p.z, voxel_size));
        idx[i] = i;
    }
}

// For each run head: (home bucket << 32) | first index.
__global__ void HeadsKernel(const P3 *__restrict__ pts,
                            const Key *__restrict__ sorted_keys,
                            const int *__restrict__ sorted_idx,
                            int n,
                            double voxel_size,
                            u64 mask,
                            u64 *__restrict__ ck,
                            unsigned char *__restrict__ head) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const bool h = (j == 0 || sorted_keys[j] != sorted_keys[j - 1]);
        head[j] = h ? 1 : 0;
        if (h) {
            const int i = sorted_idx[j];
            const P3 p = pts[i];
            const u64 bucket = CpuVoxelHash(detail::ToVoxel(p.x, voxel_size), detail::ToVoxel(p.y, voxel_size),
                                            detail::ToVoxel(p.z, voxel_size)) & mask;
            ck[j] = (bucket << 32) | static_cast<u64>(static_cast<unsigned int>(i));
        }
    }
}

// Sequence r = 0..m-1 is the (home, index)-sorted list rotated by c; entries moved from the end
// to the front get home - B so the sequence is non-decreasing.
__device__ inline i64 RotatedHome(const u64 *ck_sorted, int r, int m, int c, i64 B) {
    const int i = (c + r) % m;
    const i64 home = static_cast<i64>(ck_sorted[i] >> 32);
    return (c > 0 && r < m - c) ? home - B : home;
}

__global__ void ScanInputKernel(const u64 *__restrict__ ck_sorted, int m, int c, i64 B, i64 *__restrict__ v) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m; r += gridDim.x * blockDim.x) {
        v[r] = RotatedHome(ck_sorted, r, m, c, B) - r;
    }
}

// cluster starts; also the last start (needed for the wrap rotation)
__global__ void ClusterStartKernel(const i64 *__restrict__ scan,
                                   int m,
                                   unsigned char *__restrict__ start,
                                   int *__restrict__ last_start) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < m; r += gridDim.x * blockDim.x) {
        const bool s = (r == 0) || (r + scan[r] > (r - 1) + scan[r - 1] + 1);
        start[r] = s ? 1 : 0;
        if (s) atomicMax(last_start, r);
    }
}

// One thread per cluster: replay tsl::robin_map's insertions and write the entry of every slot.
__global__ void ClusterKernel(const u64 *__restrict__ ck_sorted,
                              const i64 *__restrict__ scan,
                              const int *__restrict__ starts,
                              int num_clusters,
                              int m,
                              int c,
                              i64 B,
                              int *__restrict__ entry_at,
                              int *__restrict__ negatives,
                              int *__restrict__ fallback) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < num_clusters; k += gridDim.x * blockDim.x) {
        const int s = starts[k];
        const int e = (k + 1 < num_clusters) ? starts[k + 1] : m;
        const int L = e - s;
        if (L > kMaxCluster) {
            atomicExch(fallback, 1);
            continue;
        }
        const i64 base = s + scan[s];  // slot of the first entry (= its home)
        i64 home[kMaxCluster];
        int fidx[kMaxCluster], order[kMaxCluster], slot[kMaxCluster];
        for (int t = 0; t < L; ++t) {
            home[t] = RotatedHome(ck_sorted, s + t, m, c, B);
            fidx[t] = static_cast<int>(ck_sorted[(c + s + t) % m] & 0xffffffffull);
            slot[t] = -1;
            int u = t;  // insertion sort by first index = CPU insertion order
            while (u > 0 && fidx[order[u - 1]] > fidx[t]) {
                order[u] = order[u - 1];
                --u;
            }
            order[u] = t;
        }
        bool ok = true;
        for (int q = 0; q < L && ok; ++q) {
            const int el = order[q];
            int j = static_cast<int>(home[el] - base);
            i64 d = 0;
            // tsl insert_impl: probe while our distance <= the occupant's
            while (j < L && slot[j] >= 0 && d <= (base + j) - home[slot[j]]) {
                ++j;
                ++d;
            }
            if (j >= L) {
                ok = false;
                break;
            }
            if (slot[j] < 0) {
                slot[j] = el;
                continue;
            }
            // tsl insert_value_impl: take the slot, carry the occupant on (robin hood)
            int carry = slot[j];
            i64 cd = (base + j) - home[carry];
            slot[j] = el;
            ++j;
            ++cd;
            while (true) {
                if (j >= L) {
                    ok = false;
                    break;
                }
                if (slot[j] < 0) {
                    slot[j] = carry;
                    break;
                }
                const i64 sd = (base + j) - home[slot[j]];
                if (cd > sd) {
                    const int tmp = slot[j];
                    slot[j] = carry;
                    carry = tmp;
                    cd = sd;
                }
                ++j;
                ++cd;
            }
        }
        int neg = 0;
        for (int t = 0; t < L && ok; ++t) {
            if (slot[t] < 0) {
                ok = false;
                break;
            }
            entry_at[s + t] = fidx[slot[t]];
            if (base + t < 0) ++neg;
        }
        if (!ok) {
            atomicExch(fallback, 1);
            continue;
        }
        if (neg > 0) atomicAdd(negatives, neg);
    }
}

// Slot order: slots with position >= 0 first, then the wrapped ones (position < 0 -> + B).
__global__ void GatherKernel(const P3 *__restrict__ in,
                             const int *__restrict__ entry_at,
                             int m,
                             int negatives,
                             P3 *__restrict__ out) {
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < m; q += gridDim.x * blockDim.x) {
        out[(q - negatives + m) % m] = in[entry_at[q]];
    }
}

// Bucket count of tsl::robin_map after reserve(n) with the default max load factor 0.5:
// rehash(ceil(float(n) / 0.5f)), rounded up to a power of two.
u64 RobinBucketCount(std::size_t n) {
    const auto count = static_cast<u64>(std::ceil(static_cast<float>(n) / 0.5f));
    u64 b = 1;
    while (b < count) b <<= 1;
    return b;
}

int Log2(u64 v) {
    int l = 0;
    while ((u64{1} << l) < v) ++l;
    return l;
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
        Reserve(n_points);
        KISS_DS_CHECK(cudaMemcpy(pts_, xyz, n_points * sizeof(P3), cudaMemcpyHostToDevice));
        const std::size_t m = DeviceLocked(pts_, n_points, voxel_size, pts_out_);
        double *out = alloc_out(m);
        if (m > 0) KISS_DS_CHECK(cudaMemcpy(out, pts_out_, m * sizeof(P3), cudaMemcpyDeviceToHost));
    }

    std::size_t Device(const P3 *d_in, std::size_t n_points, double voxel_size, P3 *d_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (num_sms_ == 0) num_sms_ = InitDeviceOrThrow();
        Reserve(n_points);
        return DeviceLocked(d_in, n_points, voxel_size, d_out);
    }

    std::size_t Fallbacks() const { return fallbacks_; }

private:
    std::size_t DeviceLocked(const P3 *d_in, std::size_t n_points, double voxel_size, P3 *d_out) {
        if (n_points == 0) return 0;
        const int n = static_cast<int>(n_points);
        const u64 B = RobinBucketCount(n_points);
        KISS_DS_CHECK(cudaMemset(counters_, 0, kNumCounters * sizeof(int)));

        // 1. voxel keys, stable sort, run heads
        KeysKernel<<<Grid(n_points), kBlockSize>>>(d_in, n, voxel_size, keys_, idx_);
        KISS_DS_CHECK(cudaGetLastError());
        std::size_t bytes = 0;
        KISS_DS_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_, keys_sorted_, idx_, idx_sorted_,
                                                      n, 0, detail::kKeyTotalBits));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceRadixSort::SortPairs(temp_, bytes, keys_, keys_sorted_, idx_, idx_sorted_,
                                                      n, 0, detail::kKeyTotalBits));
        HeadsKernel<<<Grid(n_points), kBlockSize>>>(d_in, keys_sorted_, idx_sorted_, n, voxel_size, B - 1, ck_,
                                                    flags_);
        KISS_DS_CHECK(cudaGetLastError());
        KISS_DS_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, ck_, flags_, ck_heads_, counters_ + kNumHeads, n));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceSelect::Flagged(temp_, bytes, ck_, flags_, ck_heads_, counters_ + kNumHeads, n));
        const int m = ReadCounter(kNumHeads);

        // 2./3. sort by (home bucket, first index)
        const int end_bit = 32 + Log2(B);
        KISS_DS_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, ck_heads_, ck_sorted_, m, 0, end_bit));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceRadixSort::SortKeys(temp_, bytes, ck_heads_, ck_sorted_, m, 0, end_bit));

        // slot positions; rotate if the last cluster runs past the end of the table
        int c = 0;
        Positions(m, c, B);
        i64 first = 0, last = 0;
        ReadPositions(m, first, last);
        if (last >= static_cast<i64>(B)) {
            c = ReadCounter(kLastStart);  // start of the cluster that wraps
            Positions(m, c, B);
            ReadPositions(m, first, last);
            if (last - first + 1 > static_cast<i64>(B) - 1) return Fallback(d_in, n_points, voxel_size, d_out);
        }
        const int num_clusters = CompactStarts(m);

        // 4. replay insertions per cluster
        ClusterKernel<<<Grid(static_cast<std::size_t>(num_clusters)), kBlockSize>>>(
            ck_sorted_, scan_, starts_, num_clusters, m, c, static_cast<i64>(B), entry_at_,
            counters_ + kNegatives, counters_ + kFallback);
        KISS_DS_CHECK(cudaGetLastError());
        if (ReadCounter(kFallback) != 0) return Fallback(d_in, n_points, voxel_size, d_out);

        // 5. output in slot order
        GatherKernel<<<Grid(static_cast<std::size_t>(m)), kBlockSize>>>(d_in, entry_at_, m,
                                                                         ReadCounter(kNegatives), d_out);
        KISS_DS_CHECK(cudaGetLastError());
        return static_cast<std::size_t>(m);
    }

    void Positions(int m, int c, u64 B) {
        ScanInputKernel<<<Grid(static_cast<std::size_t>(m)), kBlockSize>>>(ck_sorted_, m, c, static_cast<i64>(B),
                                                                           scan_in_);
        KISS_DS_CHECK(cudaGetLastError());
        std::size_t bytes = 0;
        KISS_DS_CHECK(cub::DeviceScan::InclusiveScan(nullptr, bytes, scan_in_, scan_, MaxOp{}, m));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceScan::InclusiveScan(temp_, bytes, scan_in_, scan_, MaxOp{}, m));
        // cluster starts (also gives the last start for a possible rotation)
        KISS_DS_CHECK(cudaMemset(counters_ + kLastStart, 0, sizeof(int)));
        ClusterStartKernel<<<Grid(static_cast<std::size_t>(m)), kBlockSize>>>(scan_, m, flags_,
                                                                               counters_ + kLastStart);
        KISS_DS_CHECK(cudaGetLastError());
    }

    void ReadPositions(int m, i64 &first, i64 &last) {
        i64 s0 = 0, sl = 0;
        KISS_DS_CHECK(cudaMemcpy(&s0, scan_, sizeof(i64), cudaMemcpyDeviceToHost));
        KISS_DS_CHECK(cudaMemcpy(&sl, scan_ + (m - 1), sizeof(i64), cudaMemcpyDeviceToHost));
        first = s0;
        last = (m - 1) + sl;
    }

    int CompactStarts(int m) {
        std::size_t bytes = 0;
        cub::CountingInputIterator<int> it(0);
        KISS_DS_CHECK(cub::DeviceSelect::Flagged(nullptr, bytes, it, flags_, starts_, counters_ + kNumClusters, m));
        Temp(bytes);
        KISS_DS_CHECK(cub::DeviceSelect::Flagged(temp_, bytes, it, flags_, starts_, counters_ + kNumClusters, m));
        return ReadCounter(kNumClusters);
    }

    // CPU implementation on the host copy (exactly the CPU result by definition).
    std::size_t Fallback(const P3 *d_in, std::size_t n, double voxel_size, P3 *d_out) {
        ++fallbacks_;
        std::vector<double> host(3 * n), out(3 * n);
        KISS_DS_CHECK(cudaMemcpy(host.data(), d_in, n * sizeof(P3), cudaMemcpyDeviceToHost));
        const std::size_t m = detail::VoxelDownsampleCpuInterleaved(host.data(), n, voxel_size, out.data());
        if (m > 0) KISS_DS_CHECK(cudaMemcpy(d_out, out.data(), m * sizeof(P3), cudaMemcpyHostToDevice));
        return m;
    }

    int ReadCounter(int slot) {
        int h = 0;
        KISS_DS_CHECK(cudaMemcpy(&h, counters_ + slot, sizeof(int), cudaMemcpyDeviceToHost));
        return h;
    }

    int Grid(std::size_t work) const {
        const std::size_t blocks = (work + kBlockSize - 1) / kBlockSize;
        const std::size_t cap = static_cast<std::size_t>(num_sms_) * 8;
        const std::size_t g = blocks < cap ? blocks : cap;
        return static_cast<int>(g > 0 ? g : 1);
    }

    template <typename T>
    static void Grow(T *&ptr, std::size_t count) {
        if (ptr != nullptr) KISS_DS_CHECK(cudaFree(ptr));
        ptr = nullptr;
        KISS_DS_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }

    void Reserve(std::size_t n) {
        if (counters_ == nullptr) KISS_DS_CHECK(cudaMalloc(&counters_, kNumCounters * sizeof(int)));
        if (n <= capacity_) return;
        const std::size_t cap = n + n / 4;  // headroom so slightly bigger scans don't reallocate
        Grow(pts_, cap);
        Grow(pts_out_, cap);
        Grow(keys_, cap);
        Grow(keys_sorted_, cap);
        Grow(idx_, cap);
        Grow(idx_sorted_, cap);
        Grow(flags_, cap);
        Grow(ck_, cap);
        Grow(ck_heads_, cap);
        Grow(ck_sorted_, cap);
        Grow(scan_in_, cap);
        Grow(scan_, cap);
        Grow(starts_, cap);
        Grow(entry_at_, cap);
        capacity_ = cap;
    }

    void Temp(std::size_t bytes) {
        if (bytes <= temp_bytes_) return;
        Grow(temp_, bytes);
        temp_bytes_ = bytes;
    }

    std::mutex mutex_;
    int num_sms_ = 0;
    std::size_t capacity_ = 0, temp_bytes_ = 0, fallbacks_ = 0;
    P3 *pts_ = nullptr, *pts_out_ = nullptr;
    Key *keys_ = nullptr, *keys_sorted_ = nullptr;
    int *idx_ = nullptr, *idx_sorted_ = nullptr, *starts_ = nullptr, *entry_at_ = nullptr, *counters_ = nullptr;
    u64 *ck_ = nullptr, *ck_heads_ = nullptr, *ck_sorted_ = nullptr;
    i64 *scan_in_ = nullptr, *scan_ = nullptr;
    unsigned char *flags_ = nullptr, *temp_ = nullptr;
};

Downsampler &Instance() {
    static Downsampler *instance = new Downsampler();  // intentionally leaked, see class comment
    return *instance;
}

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
    Instance().Run(xyz, n, voxel_size, alloc_out);
}

std::size_t DownsampleFallbackCount() { return Instance().Fallbacks(); }

namespace detail {
std::size_t VoxelDownsampleDevice(const double *d_in, std::size_t n, double voxel_size, double *d_out) {
    return Instance().Device(reinterpret_cast<const P3 *>(d_in), n, voxel_size, reinterpret_cast<P3 *>(d_out));
}
}  // namespace detail

}  // namespace kiss_icp::gpu
