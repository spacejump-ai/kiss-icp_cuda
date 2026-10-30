// MIT License
//
// CUDA voxel hash map for KISS-ICP, see VoxelMapGpu.hpp.
//
// Layout
// ------
//   cuco::static_map<uint64_t voxel_key, uint32_t block_index>   (open addressing, linear probing)
//   block pool, indexed by block_index:
//       block_key  [max_voxels]               key of the voxel owning the block, kEmptyKey if free
//       block_count[max_voxels]               number of valid points in the block
//       block_pts  [max_voxels * kMaxPoints]  fixed-size point slots (float xyz)
//   free_list / free_top                      stack of unused block indices
//
// Points never move once written: the hash map only stores indices into the pool. Erasing a voxel
// leaves a tombstone in the static_map; once the number of tombstones gets large, the map is
// rebuilt from the pool (cheap: only (key, index) pairs are re-inserted, no point data is copied).
//
// Insertion is lock-free and deterministic: incoming points are stable-sorted by voxel key and
// one thread owns one voxel, processing its points in input order, exactly like the sequential
// CPU loop in the original VoxelHashMap::AddPoints.
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <map>
#include <cuco/static_map.cuh>
#include <stdexcept>
#include <string>

#include "VoxelMapGpu.hpp"

#define KISS_CUDA_CHECK(expr)                                                                \
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
constexpr int kMaxPoints = kMaxPointsPerVoxel;
constexpr int kNumSums = 28;  // 21 (JTJ upper) + 6 (JTr) + 1 (count)

// 21 bits per axis, offset so that negative voxel coordinates map to positive ints.
// Covers +-1,048,576 voxels per axis (+-1048 km at 1 m voxels). Top bit is always 0, so the two
// all-ones sentinels below can never collide with a real key.
constexpr int kKeyBits = 21;
constexpr long long kKeyOffset = 1LL << (kKeyBits - 1);
constexpr unsigned long long kKeyMask = (1ULL << kKeyBits) - 1ULL;
constexpr unsigned long long kEmptyKey = ~0ULL;
constexpr unsigned long long kErasedKey = ~0ULL - 1ULL;
constexpr unsigned int kEmptyValue = ~0U;

using Key = unsigned long long;
using Value = unsigned int;
using MapType = cuco::static_map<Key,
                                 Value,
                                 cuco::extent<std::size_t>,
                                 cuda::thread_scope_device,
                                 thrust::equal_to<Key>,
                                 cuco::linear_probing<1, cuco::default_hash_function<Key>>>;

struct Pt {
    float x, y, z;
};

struct Mat34 {
    double m[12];
};

__host__ __device__ inline Key PackKey(int x, int y, int z) {
    return (static_cast<Key>(x + kKeyOffset) & kKeyMask) |
           ((static_cast<Key>(y + kKeyOffset) & kKeyMask) << kKeyBits) |
           ((static_cast<Key>(z + kKeyOffset) & kKeyMask) << (2 * kKeyBits));
}

__device__ inline int ToVoxel(double v, double voxel_size) {
    return static_cast<int>(floor(v / voxel_size));
}

// ---------------------------------------------------------------------------------------------
// Nearest neighbour inside the 27-voxel neighbourhood of q. Returns squared distance (FLT_MAX if
// no map point was found) and the neighbour in `best`.
template <typename Ref>
__device__ inline float NearestNeighbor(const Ref &ref,
                                        const Pt *__restrict__ block_pts,
                                        const int *__restrict__ block_count,
                                        double qx,
                                        double qy,
                                        double qz,
                                        double voxel_size,
                                        Pt &best) {
    const int vx = ToVoxel(qx, voxel_size);
    const int vy = ToVoxel(qy, voxel_size);
    const int vz = ToVoxel(qz, voxel_size);
    const float fx = static_cast<float>(qx);
    const float fy = static_cast<float>(qy);
    const float fz = static_cast<float>(qz);
    float best_d2 = FLT_MAX;
    best = Pt{fx, fy, fz};
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                const auto it = ref.find(PackKey(vx + dx, vy + dy, vz + dz));
                if (it == ref.end()) continue;
                const Value b = it->second;
                const int count = block_count[b];
                const Pt *pts = block_pts + static_cast<std::size_t>(b) * kMaxPoints;
                for (int k = 0; k < count; ++k) {
                    const Pt p = pts[k];
                    const float ex = p.x - fx;
                    const float ey = p.y - fy;
                    const float ez = p.z - fz;
                    const float d2 = ex * ex + ey * ey + ez * ez;
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best = p;
                    }
                }
            }
        }
    }
    return best_d2;
}

// ---------------------------------------------------------------------------------------------
__global__ void ComputeKeysKernel(const double *__restrict__ xyz,
                                  int n,
                                  double voxel_size,
                                  Key *__restrict__ keys,
                                  int *__restrict__ order) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        keys[i] = PackKey(ToVoxel(xyz[3 * i + 0], voxel_size), ToVoxel(xyz[3 * i + 1], voxel_size),
                          ToVoxel(xyz[3 * i + 2], voxel_size));
        order[i] = i;
    }
}

// One thread per distinct voxel of the incoming batch. Allocates the voxel if needed and appends
// its points in input order using the same rules as the CPU implementation.
template <typename Ref>
__global__ void InsertSegmentsKernel(Ref ref,
                                     const Key *__restrict__ unique_keys,
                                     const int *__restrict__ segment_start,
                                     int num_segments,
                                     int num_points,
                                     const int *__restrict__ order,
                                     const double *__restrict__ xyz,
                                     Key *__restrict__ block_key,
                                     int *__restrict__ block_count,
                                     Pt *__restrict__ block_pts,
                                     const Value *__restrict__ free_list,
                                     int *__restrict__ free_top,
                                     int *__restrict__ dropped,
                                     int max_points,
                                     double resolution2) {
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_segments;
         s += gridDim.x * blockDim.x) {
        const Key key = unique_keys[s];
        const int begin = segment_start[s];
        const int end = (s + 1 < num_segments) ? segment_start[s + 1] : num_points;

        Value b;
        const auto found = ref.find(key);
        if (found != ref.end()) {
            b = found->second;
        } else {
            // Pop a free block. No pushes happen concurrently (removal is a separate kernel).
            const int top = atomicSub(free_top, 1);
            if (top <= 0) {
                atomicAdd(free_top, 1);
                atomicAdd(dropped, end - begin);
                continue;
            }
            b = free_list[top - 1];
            block_key[b] = key;
            block_count[b] = 0;
            // This thread is the only one handling `key`, and find() just reported it absent,
            // so this cannot create a duplicate even if it lands on a tombstone.
            ref.insert(cuco::pair<Key, Value>{key, b});
        }

        int count = block_count[b];
        Pt *pts = block_pts + static_cast<std::size_t>(b) * kMaxPoints;
        for (int j = begin; j < end && count < max_points; ++j) {
            const int i = order[j];
            const Pt p{static_cast<float>(xyz[3 * i + 0]), static_cast<float>(xyz[3 * i + 1]),
                       static_cast<float>(xyz[3 * i + 2])};
            // Squared distance in double on the float-stored coordinates: differences and
            // squares are exact, so the decision is reproducible on the CPU (no FMA effects).
            bool too_close = false;
            for (int k = 0; k < count; ++k) {
                const double ex = static_cast<double>(pts[k].x) - static_cast<double>(p.x);
                const double ey = static_cast<double>(pts[k].y) - static_cast<double>(p.y);
                const double ez = static_cast<double>(pts[k].z) - static_cast<double>(p.z);
                if (ex * ex + ey * ey + ez * ez < resolution2) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) pts[count++] = p;
        }
        block_count[b] = count;
    }
}

__global__ void RemoveFarKernel(int num_blocks,
                                Key *__restrict__ block_key,
                                int *__restrict__ block_count,
                                const Pt *__restrict__ block_pts,
                                double ox,
                                double oy,
                                double oz,
                                double max_distance2,
                                Value *__restrict__ free_list,
                                int *__restrict__ free_top,
                                Key *__restrict__ erase_keys,
                                int *__restrict__ erase_count) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < num_blocks;
         b += gridDim.x * blockDim.x) {
        const Key key = block_key[b];
        if (key == kEmptyKey) continue;
        const Pt p = block_pts[static_cast<std::size_t>(b) * kMaxPoints];
        const double dx = static_cast<double>(p.x) - ox;
        const double dy = static_cast<double>(p.y) - oy;
        const double dz = static_cast<double>(p.z) - oz;
        if (dx * dx + dy * dy + dz * dz >= max_distance2) {
            erase_keys[atomicAdd(erase_count, 1)] = key;
            free_list[atomicAdd(free_top, 1)] = static_cast<Value>(b);
            block_key[b] = kEmptyKey;
            block_count[b] = 0;
        }
    }
}

__global__ void GatherPointsKernel(int num_blocks,
                                   const int *__restrict__ block_count,
                                   const Pt *__restrict__ block_pts,
                                   double *__restrict__ out,
                                   int *__restrict__ cursor) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < num_blocks;
         b += gridDim.x * blockDim.x) {
        const int count = block_count[b];
        if (count == 0) continue;
        const int base = atomicAdd(cursor, count);
        const Pt *pts = block_pts + static_cast<std::size_t>(b) * kMaxPoints;
        for (int k = 0; k < count; ++k) {
            out[3 * (base + k) + 0] = pts[k].x;
            out[3 * (base + k) + 1] = pts[k].y;
            out[3 * (base + k) + 2] = pts[k].z;
        }
    }
}

template <typename Ref>
__global__ void ClosestNeighborsKernel(Ref ref,
                                       const Pt *__restrict__ block_pts,
                                       const int *__restrict__ block_count,
                                       const double *__restrict__ query,
                                       int n,
                                       double voxel_size,
                                       double *__restrict__ nn,
                                       double *__restrict__ dist) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const double qx = query[3 * i + 0];
        const double qy = query[3 * i + 1];
        const double qz = query[3 * i + 2];
        Pt best;
        const float d2 =
            NearestNeighbor(ref, block_pts, block_count, qx, qy, qz, voxel_size, best);
        if (d2 == FLT_MAX) {
            nn[3 * i + 0] = 0.0;
            nn[3 * i + 1] = 0.0;
            nn[3 * i + 2] = 0.0;
            dist[i] = DBL_MAX;
        } else {
            nn[3 * i + 0] = best.x;
            nn[3 * i + 1] = best.y;
            nn[3 * i + 2] = best.z;
            const double ex = best.x - qx;
            const double ey = best.y - qy;
            const double ez = best.z - qz;
            dist[i] = sqrt(ex * ex + ey * ey + ez * ez);
        }
    }
}

// Fused data association + normal equations (Eq. 10/11 of the KISS-ICP paper).
// Each thread accumulates 28 doubles in registers, then a warp-shuffle + shared-memory block
// reduction, then one atomicAdd per sum and block.
template <typename Ref>
__global__ void __launch_bounds__(kBlockSize)
    LinearSystemKernel(Ref ref,
                       const Pt *__restrict__ block_pts,
                       const int *__restrict__ block_count,
                       const double *__restrict__ src,
                       int n,
                       double voxel_size,
                       Mat34 T,
                       double max_correspondence_distance2,
                       double kernel_scale,
                       double *__restrict__ sums) {
    double acc[kNumSums];
#pragma unroll
    for (int k = 0; k < kNumSums; ++k) acc[k] = 0.0;

    const double ks2 = kernel_scale * kernel_scale;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const double sx = src[3 * i + 0];
        const double sy = src[3 * i + 1];
        const double sz = src[3 * i + 2];
        const double px = T.m[0] * sx + T.m[1] * sy + T.m[2] * sz + T.m[3];
        const double py = T.m[4] * sx + T.m[5] * sy + T.m[6] * sz + T.m[7];
        const double pz = T.m[8] * sx + T.m[9] * sy + T.m[10] * sz + T.m[11];

        Pt q;
        if (NearestNeighbor(ref, block_pts, block_count, px, py, pz, voxel_size, q) == FLT_MAX)
            continue;
        const double rx = px - static_cast<double>(q.x);
        const double ry = py - static_cast<double>(q.y);
        const double rz = pz - static_cast<double>(q.z);
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (r2 >= max_correspondence_distance2) continue;

        const double w = ks2 / ((kernel_scale + r2) * (kernel_scale + r2));
        // J = [ I | -hat(p) ]
        const double J[3][6] = {{1.0, 0.0, 0.0, 0.0, pz, -py},
                                {0.0, 1.0, 0.0, -pz, 0.0, px},
                                {0.0, 0.0, 1.0, py, -px, 0.0}};
        int idx = 0;
#pragma unroll
        for (int a = 0; a < 6; ++a) {
#pragma unroll
            for (int b = a; b < 6; ++b) {
                acc[idx++] += w * (J[0][a] * J[0][b] + J[1][a] * J[1][b] + J[2][a] * J[2][b]);
            }
        }
#pragma unroll
        for (int a = 0; a < 6; ++a) {
            acc[21 + a] += w * (J[0][a] * rx + J[1][a] * ry + J[2][a] * rz);
        }
        acc[27] += 1.0;
    }

    // Block reduction.
    __shared__ double smem[kBlockSize / 32][kNumSums];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < kNumSums; ++k) {
        double v = acc[k];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) smem[warp][k] = v;
    }
    __syncthreads();
    if (warp == 0) {
        constexpr int kWarps = kBlockSize / 32;
#pragma unroll
        for (int k = 0; k < kNumSums; ++k) {
            double v = (lane < kWarps) ? smem[lane][k] : 0.0;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0 && v != 0.0) atomicAdd(&sums[k], v);
        }
    }
}

// thrust/CUB algorithms (sort, unique, reduce, copy_if) allocate temporary storage on every
// call. Plain cudaMalloc/cudaFree there costs far more than the work itself at our sizes and
// cudaFree synchronizes the device, so keep released blocks around and hand them out again.
class CachingAllocator {
public:
    using value_type = char;
    CachingAllocator() = default;
    CachingAllocator(const CachingAllocator &) = delete;
    CachingAllocator &operator=(const CachingAllocator &) = delete;
    ~CachingAllocator() {
        for (auto &kv : free_) cudaFree(kv.second);
        for (auto &kv : used_) cudaFree(kv.first);
    }
    char *allocate(std::ptrdiff_t n) {
        const auto bytes = static_cast<std::size_t>(n);
        auto it = free_.lower_bound(bytes);
        char *p = nullptr;
        std::size_t size = bytes;
        if (it != free_.end()) {
            size = it->first;
            p = it->second;
            free_.erase(it);
        } else {
            KISS_CUDA_CHECK(cudaMalloc(&p, bytes));
        }
        used_.emplace(p, size);
        return p;
    }
    void deallocate(char *p, std::size_t) {
        auto it = used_.find(p);
        if (it == used_.end()) return;
        free_.emplace(it->second, p);
        used_.erase(it);
    }

private:
    std::multimap<std::size_t, char *> free_;
    std::map<char *, std::size_t> used_;
};

struct IsLiveKey {
    __host__ __device__ bool operator()(Key k) const { return k != kEmptyKey; }
};

template <typename T>
inline T *Raw(thrust::device_vector<T> &v) {
    return thrust::raw_pointer_cast(v.data());
}
template <typename T>
inline const T *Raw(const thrust::device_vector<T> &v) {
    return thrust::raw_pointer_cast(v.data());
}

}  // namespace

// =============================================================================================
struct VoxelMapGpu::Impl {
    Impl(double voxel_size_, double max_distance_, unsigned int max_points_, std::size_t max_voxels_)
        : voxel_size(voxel_size_),
          max_distance(max_distance_),
          max_points(static_cast<int>(max_points_)),
          max_voxels(max_voxels_),
          resolution2(voxel_size_ * voxel_size_ / max_points_),
          map(cuco::extent<std::size_t>{2 * max_voxels_},
              cuco::empty_key<Key>{kEmptyKey},
              cuco::empty_value<Value>{kEmptyValue},
              cuco::erased_key<Key>{kErasedKey}) {
        if (max_points_ == 0 || max_points_ > static_cast<unsigned int>(kMaxPoints)) {
            throw std::invalid_argument(
                "[kiss_icp::gpu] max_points_per_voxel must be in [1, " +
                std::to_string(kMaxPoints) +
                "]; rebuild with -DKISS_ICP_CUDA_MAX_POINTS_PER_VOXEL=<n> for more");
        }
        if (max_voxels_ == 0 || max_voxels_ > (1u << 30)) {
            throw std::invalid_argument("[kiss_icp::gpu] invalid max_voxels");
        }
        int device = 0;
        KISS_CUDA_CHECK(cudaGetDevice(&device));
        KISS_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

        block_key.resize(max_voxels);
        block_count.resize(max_voxels);
        block_pts.resize(max_voxels * kMaxPoints);
        free_list.resize(max_voxels);
        counters.resize(kNumCounters);
        sums.resize(kNumSums);
        ResetPool();
    }

    // counters[] slots
    static constexpr int kFreeTop = 0;
    static constexpr int kDropped = 1;
    static constexpr int kEraseCount = 2;
    static constexpr int kCursor = 3;
    static constexpr int kNumCounters = 4;

    int Grid(std::size_t work) const {
        const std::size_t blocks = (work + kBlockSize - 1) / kBlockSize;
        const std::size_t cap = static_cast<std::size_t>(num_sms) * 8;
        return static_cast<int>(blocks < cap ? blocks : cap);
    }

    int ReadCounter(int slot) const {
        int h = 0;
        KISS_CUDA_CHECK(
            cudaMemcpy(&h, Raw(counters) + slot, sizeof(int), cudaMemcpyDeviceToHost));
        return h;
    }
    void WriteCounter(int slot, int value) {
        KISS_CUDA_CHECK(
            cudaMemcpy(Raw(counters) + slot, &value, sizeof(int), cudaMemcpyHostToDevice));
    }

    void ResetPool() {
        thrust::fill(block_key.begin(), block_key.end(), kEmptyKey);
        thrust::fill(block_count.begin(), block_count.end(), 0);
        // free_list = [max_voxels-1, ..., 1, 0] so that pops hand out low indices first
        thrust::sequence(free_list.rbegin(), free_list.rend(), Value{0});
        thrust::fill(counters.begin(), counters.end(), 0);
        WriteCounter(kFreeTop, static_cast<int>(max_voxels));
        num_voxels = 0;
        tombstones = 0;
    }

    void SyncNumVoxels() {
        num_voxels = max_voxels - static_cast<std::size_t>(ReadCounter(kFreeTop));
    }

    // Re-insert all live (key, block) pairs into an empty map to get rid of tombstones.
    void Rebuild() {
        map.clear();
        thrust::device_vector<Key> keys(num_voxels);
        thrust::device_vector<Value> vals(num_voxels);
        auto in_begin = thrust::make_zip_iterator(
            thrust::make_tuple(block_key.begin(), thrust::make_counting_iterator<Value>(0)));
        auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), vals.begin()));
        auto out_end = thrust::copy_if(thrust::cuda::par(alloc), in_begin, in_begin + max_voxels,
                                       block_key.begin(), out_begin, IsLiveKey{});
        const auto live = static_cast<std::size_t>(out_end - out_begin);
        auto pairs = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), vals.begin()));
        map.insert(pairs, pairs + live);
        tombstones = 0;
    }

    double voxel_size;
    double max_distance;
    int max_points;
    std::size_t max_voxels;
    double resolution2;
    int num_sms = 1;

    MapType map;
    thrust::device_vector<Key> block_key;
    thrust::device_vector<int> block_count;
    thrust::device_vector<Pt> block_pts;
    thrust::device_vector<Value> free_list;
    thrust::device_vector<int> counters;
    std::size_t num_voxels = 0;
    std::size_t tombstones = 0;

    // scratch (reused between calls to avoid reallocations)
    mutable CachingAllocator alloc;
    mutable thrust::device_vector<double> cloud_out;
    thrust::device_vector<double> in_xyz;
    thrust::device_vector<Key> keys;
    thrust::device_vector<int> order;
    thrust::device_vector<Key> unique_keys;
    thrust::device_vector<int> segment_start;
    thrust::device_vector<Key> erase_keys;
    mutable thrust::device_vector<double> src;
    mutable thrust::device_vector<double> sums;
    mutable thrust::device_vector<double> query, nn, dist;
};

// =============================================================================================
VoxelMapGpu::VoxelMapGpu(double voxel_size,
                         double max_distance,
                         unsigned int max_points_per_voxel,
                         std::size_t max_voxels)
    : impl_(std::make_unique<Impl>(voxel_size, max_distance, max_points_per_voxel, max_voxels)) {}

VoxelMapGpu::~VoxelMapGpu() = default;
VoxelMapGpu::VoxelMapGpu(VoxelMapGpu &&) noexcept = default;
VoxelMapGpu &VoxelMapGpu::operator=(VoxelMapGpu &&) noexcept = default;

void VoxelMapGpu::Clear() {
    impl_->map.clear();
    impl_->ResetPool();
}

std::size_t VoxelMapGpu::NumVoxels() const { return impl_->num_voxels; }
std::size_t VoxelMapGpu::Capacity() const { return impl_->max_voxels; }
std::size_t VoxelMapGpu::DroppedForCapacity() const {
    return static_cast<std::size_t>(impl_->ReadCounter(Impl::kDropped));
}

void VoxelMapGpu::AddPoints(const double *xyz, std::size_t n_points) {
    if (n_points == 0) return;
    auto &d = *impl_;
    const int n = static_cast<int>(n_points);
    d.in_xyz.resize(3 * n_points);
    d.keys.resize(n_points);
    d.order.resize(n_points);
    d.unique_keys.resize(n_points);
    d.segment_start.resize(n_points);
    KISS_CUDA_CHECK(cudaMemcpy(Raw(d.in_xyz), xyz, 3 * n_points * sizeof(double),
                               cudaMemcpyHostToDevice));

    ComputeKeysKernel<<<d.Grid(n_points), kBlockSize>>>(Raw(d.in_xyz), n, d.voxel_size,
                                                        Raw(d.keys), Raw(d.order));
    KISS_CUDA_CHECK(cudaGetLastError());

    // Group points by voxel, keeping input order inside each voxel.
    thrust::stable_sort_by_key(thrust::cuda::par(d.alloc), d.keys.begin(), d.keys.end(), d.order.begin());
    auto ends = thrust::unique_by_key_copy(thrust::cuda::par(d.alloc), d.keys.begin(), d.keys.end(),
                                           thrust::make_counting_iterator<int>(0),
                                           d.unique_keys.begin(), d.segment_start.begin());
    const int num_segments = static_cast<int>(ends.first - d.unique_keys.begin());

    InsertSegmentsKernel<<<d.Grid(static_cast<std::size_t>(num_segments)), kBlockSize>>>(
        d.map.ref(cuco::find, cuco::insert), Raw(d.unique_keys), Raw(d.segment_start),
        num_segments, n, Raw(d.order), Raw(d.in_xyz), Raw(d.block_key), Raw(d.block_count),
        Raw(d.block_pts), Raw(d.free_list), Raw(d.counters) + Impl::kFreeTop,
        Raw(d.counters) + Impl::kDropped, d.max_points, d.resolution2);
    KISS_CUDA_CHECK(cudaGetLastError());
    d.SyncNumVoxels();
}

void VoxelMapGpu::RemoveFarFromLocation(const double origin[3]) {
    auto &d = *impl_;
    if (d.num_voxels == 0) return;
    d.erase_keys.resize(d.num_voxels);
    d.WriteCounter(Impl::kEraseCount, 0);
    RemoveFarKernel<<<d.Grid(d.max_voxels), kBlockSize>>>(
        static_cast<int>(d.max_voxels), Raw(d.block_key), Raw(d.block_count), Raw(d.block_pts),
        origin[0], origin[1], origin[2], d.max_distance * d.max_distance, Raw(d.free_list),
        Raw(d.counters) + Impl::kFreeTop, Raw(d.erase_keys), Raw(d.counters) + Impl::kEraseCount);
    KISS_CUDA_CHECK(cudaGetLastError());
    const int erased = d.ReadCounter(Impl::kEraseCount);
    if (erased > 0) {
        d.map.erase(d.erase_keys.begin(), d.erase_keys.begin() + erased);
        d.tombstones += static_cast<std::size_t>(erased);
    }
    d.SyncNumVoxels();
    // Tombstones lengthen probe sequences for find(); rebuild once they reach 25% of the slots.
    if (d.tombstones > d.map.capacity() / 4) d.Rebuild();
}

std::vector<double> VoxelMapGpu::Pointcloud() const {
    auto &d = *impl_;
    const int total = thrust::reduce(thrust::cuda::par(d.alloc), d.block_count.begin(), d.block_count.end(), 0);
    std::vector<double> out(3 * static_cast<std::size_t>(total));
    if (total == 0) return out;
    d.cloud_out.resize(out.size());
    auto &d_out = d.cloud_out;
    d.WriteCounter(Impl::kCursor, 0);
    GatherPointsKernel<<<d.Grid(d.max_voxels), kBlockSize>>>(static_cast<int>(d.max_voxels),
                                                             Raw(d.block_count), Raw(d.block_pts),
                                                             Raw(d_out), Raw(d.counters) + Impl::kCursor);
    KISS_CUDA_CHECK(cudaGetLastError());
    KISS_CUDA_CHECK(cudaMemcpy(out.data(), Raw(d_out), out.size() * sizeof(double),
                               cudaMemcpyDeviceToHost));
    return out;
}

void VoxelMapGpu::ClosestNeighbors(const double *xyz,
                                   std::size_t n,
                                   double *nn_xyz,
                                   double *dist) const {
    if (n == 0) return;
    auto &d = *impl_;
    d.query.resize(3 * n);
    d.nn.resize(3 * n);
    d.dist.resize(n);
    KISS_CUDA_CHECK(
        cudaMemcpy(Raw(d.query), xyz, 3 * n * sizeof(double), cudaMemcpyHostToDevice));
    ClosestNeighborsKernel<<<d.Grid(n), kBlockSize>>>(
        d.map.ref(cuco::find), Raw(d.block_pts), Raw(d.block_count), Raw(d.query),
        static_cast<int>(n), d.voxel_size, Raw(d.nn), Raw(d.dist));
    KISS_CUDA_CHECK(cudaGetLastError());
    KISS_CUDA_CHECK(
        cudaMemcpy(nn_xyz, Raw(d.nn), 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
    KISS_CUDA_CHECK(cudaMemcpy(dist, Raw(d.dist), n * sizeof(double), cudaMemcpyDeviceToHost));
}

void VoxelMapGpu::SetSource(const double *xyz, std::size_t n) const {
    auto &d = *impl_;
    d.src.resize(3 * n);
    if (n == 0) return;
    KISS_CUDA_CHECK(cudaMemcpy(Raw(d.src), xyz, 3 * n * sizeof(double), cudaMemcpyHostToDevice));
}

LinearSystemSums VoxelMapGpu::LinearSystem(const double T[12],
                                           double max_correspondence_distance,
                                           double kernel_scale) const {
    auto &d = *impl_;
    LinearSystemSums result;
    const std::size_t n = d.src.size() / 3;
    if (n == 0 || d.num_voxels == 0) return result;

    Mat34 Tm;
    for (int k = 0; k < 12; ++k) Tm.m[k] = T[k];
    KISS_CUDA_CHECK(cudaMemset(Raw(d.sums), 0, kNumSums * sizeof(double)));
    LinearSystemKernel<<<d.Grid(n), kBlockSize>>>(
        d.map.ref(cuco::find), Raw(d.block_pts), Raw(d.block_count), Raw(d.src),
        static_cast<int>(n), d.voxel_size, Tm,
        max_correspondence_distance * max_correspondence_distance, kernel_scale, Raw(d.sums));
    KISS_CUDA_CHECK(cudaGetLastError());

    double h[kNumSums];
    KISS_CUDA_CHECK(cudaMemcpy(h, Raw(d.sums), sizeof(h), cudaMemcpyDeviceToHost));
    for (int k = 0; k < 21; ++k) result.JTJ_upper[static_cast<std::size_t>(k)] = h[k];
    for (int k = 0; k < 6; ++k) result.JTr[static_cast<std::size_t>(k)] = h[21 + k];
    result.num_correspondences = static_cast<std::size_t>(h[27]);
    return result;
}

}  // namespace kiss_icp::gpu
