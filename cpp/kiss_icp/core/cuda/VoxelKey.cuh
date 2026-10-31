// Voxel key packing shared by the CUDA voxel map and the CUDA downsampler.
#pragma once

#include <cmath>

namespace kiss_icp::gpu {
/// Creates the CUDA context; throws std::runtime_error with a readable hint if the driver is not
/// usable. Returns the multiprocessor count. Defined in VoxelMapGpu.cu.
int InitDeviceOrThrow();
}  // namespace kiss_icp::gpu

namespace kiss_icp::gpu::detail {

// 21 bits per axis, offset so that negative voxel coordinates map to positive ints.
// Covers +-1,048,576 voxels per axis (+-1048 km at 1 m voxels, +-524 km at 0.5 m). Top bit is
// always 0, so all-ones sentinels can never collide with a real key.
constexpr int kKeyBits = 21;
constexpr long long kKeyOffset = 1LL << (kKeyBits - 1);
constexpr unsigned long long kKeyMask = (1ULL << kKeyBits) - 1ULL;
constexpr int kKeyTotalBits = 3 * kKeyBits;  // 63

using Key = unsigned long long;

__host__ __device__ inline Key PackKey(int x, int y, int z) {
    return (static_cast<Key>(x + kKeyOffset) & kKeyMask) |
           ((static_cast<Key>(y + kKeyOffset) & kKeyMask) << kKeyBits) |
           ((static_cast<Key>(z + kKeyOffset) & kKeyMask) << (2 * kKeyBits));
}

// Same arithmetic as kiss_icp::PointToVoxel (IEEE double division + floor), so the voxel of a
// point is identical on CPU and GPU.
__device__ inline int ToVoxel(double v, double voxel_size) {
    return static_cast<int>(floor(v / voxel_size));
}

}  // namespace kiss_icp::gpu::detail
