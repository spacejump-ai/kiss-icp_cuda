// Self-check for the CUDA voxel map: compares it against a straight re-implementation of the
// original CPU VoxelHashMap logic, then runs a registration on a synthetic scene.
//
//   ./kiss_icp_cuda_check            -> prints PASS/FAIL per check and timings, exit code != 0 on failure
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>

#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/core/VoxelUtils.hpp"

namespace {
using kiss_icp::Voxel;
using Points = std::vector<Eigen::Vector3d>;
using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int failures = 0;
void Check(bool ok, const char *what, double value = 0.0) {
    std::printf("[%s] %-48s %g\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

// ---- reference: the original CPU VoxelHashMap (std::unordered_map instead of robin_map) ------
struct RefMap {
    double voxel_size, max_distance;
    unsigned int max_points;
    std::unordered_map<Voxel, Points> map;

    void Add(const Points &points) {
        const double res = std::sqrt(voxel_size * voxel_size / max_points);
        for (const auto &p : points) {
            const auto v = kiss_icp::PointToVoxel(p, voxel_size);
            auto it = map.find(v);
            if (it == map.end()) {
                Points vp;
                vp.push_back(p);
                map.emplace(v, std::move(vp));
                continue;
            }
            auto &vp = it->second;
            if (vp.size() == max_points ||
                std::any_of(vp.begin(), vp.end(), [&](const auto &q) { return (q - p).norm() < res; }))
                continue;
            vp.push_back(p);
        }
    }
    void Remove(const Eigen::Vector3d &origin) {
        for (auto it = map.begin(); it != map.end();) {
            if ((it->second.front() - origin).squaredNorm() >= max_distance * max_distance)
                it = map.erase(it);
            else
                ++it;
        }
    }
    std::size_t NumPoints() const {
        std::size_t n = 0;
        for (const auto &kv : map) n += kv.second.size();
        return n;
    }
    double Closest(const Eigen::Vector3d &q) const {
        const auto v = kiss_icp::PointToVoxel(q, voxel_size);
        double best = std::numeric_limits<double>::max();
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    auto it = map.find(v + Voxel(dx, dy, dz));
                    if (it == map.end()) continue;
                    for (const auto &p : it->second) best = std::min(best, (p - q).norm());
                }
        return best;
    }
};

// ---- synthetic "street": ground, two facades, poles, boxes ----------------------------------
Points MakeScene(std::mt19937 &rng, double x0, double x1) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.01);
    Points pts;
    const auto n_along = static_cast<int>((x1 - x0) * 400);
    for (int i = 0; i < n_along; ++i) {
        const double x = x0 + (x1 - x0) * u(rng);
        pts.emplace_back(x, -12.0 + 24.0 * u(rng), noise(rng));                    // ground
        pts.emplace_back(x, -12.0 + noise(rng), 10.0 * u(rng));                    // left facade
        pts.emplace_back(x, 12.0 + noise(rng), 10.0 * u(rng) + 0.3 * std::sin(x)); // right facade
    }
    for (double x = std::ceil(x0 / 15.0) * 15.0; x < x1; x += 15.0) {  // poles + boxes
        for (int k = 0; k < 400; ++k) {
            const double a = 2.0 * M_PI * u(rng);
            pts.emplace_back(x + 0.15 * std::cos(a), 6.0 + 0.15 * std::sin(a), 6.0 * u(rng));
            pts.emplace_back(x + 3.0 + 2.0 * u(rng), -4.0 + noise(rng), 1.5 * u(rng));
            pts.emplace_back(x + 3.0 + noise(rng), -4.0 - 2.0 * u(rng), 1.5 * u(rng));
        }
    }
    std::shuffle(pts.begin(), pts.end(), rng);
    return pts;
}
}  // namespace

int main() {
    const double voxel_size = 1.0, max_range = 60.0;
    const unsigned int max_points = 20;
    std::mt19937 rng(42);

    kiss_icp::VoxelHashMap gpu(voxel_size, max_range, max_points, /*max_voxels=*/1u << 17);
    RefMap ref{voxel_size, max_range, max_points, {}};

    // 1) Drive along x, insert + crop like KissICP does. Small capacity so the tombstone rebuild
    //    path gets exercised as voxels are dropped behind the vehicle.
    double t_update = 0.0;
    int frames = 0;
    for (double x = 0.0; x < 600.0; x += 2.0, ++frames) {
        Points scan = MakeScene(rng, x - 40.0, x + 60.0);
        scan.resize(std::min<std::size_t>(scan.size(), 40000));
        const Eigen::Vector3d origin(x, 0.0, 1.8);
        const auto t0 = Clock::now();
        gpu.Update(scan, origin);
        const auto t1 = Clock::now();
        t_update += Ms(t0, t1);
        ref.Add(scan);
        ref.Remove(origin);
    }
    const auto gpu_cloud = gpu.Pointcloud();
    Check(gpu.NumVoxels() == ref.map.size(), "voxel count matches reference",
          static_cast<double>(gpu.NumVoxels()));
    Check(gpu_cloud.size() == ref.NumPoints(), "point count matches reference",
          static_cast<double>(gpu_cloud.size()));
    Check(gpu.Gpu().DroppedForCapacity() == 0, "no points dropped for capacity");

    // Same points (float storage on the GPU -> compare after rounding to 1e-4 m).
    {
        auto key = [](const Eigen::Vector3d &p) {
            return std::array<long long, 3>{std::llround(p.x() * 1e4), std::llround(p.y() * 1e4),
                                            std::llround(p.z() * 1e4)};
        };
        std::vector<std::array<long long, 3>> a, b;
        for (const auto &p : gpu_cloud) a.push_back(key(p));
        for (const auto &kv : ref.map)
            for (const auto &p : kv.second) b.push_back(key(p));
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
            for (int k = 0; k < 3; ++k) mismatches += std::llabs(a[i][static_cast<std::size_t>(k)] - b[i][static_cast<std::size_t>(k)]) > 1;
        }
        Check(mismatches == 0, "map content identical (1e-4 m)", static_cast<double>(mismatches));
    }

    // 2) Batched nearest neighbour vs reference.
    {
        Points queries;
        std::uniform_real_distribution<double> ux(540.0, 600.0), uy(-13.0, 13.0), uz(-1.0, 11.0);
        for (int i = 0; i < 20000; ++i) queries.emplace_back(ux(rng), uy(rng), uz(rng));
        const auto t0 = Clock::now();
        const auto res = gpu.GetClosestNeighbors(queries);
        const auto t1 = Clock::now();
        double max_err = 0.0;
        std::size_t both_missing = 0, one_missing = 0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const double d_ref = ref.Closest(queries[i]);
            const double d_gpu = std::get<1>(res[i]);
            const bool m_ref = d_ref == std::numeric_limits<double>::max();
            const bool m_gpu = d_gpu == std::numeric_limits<double>::max();
            if (m_ref && m_gpu) { ++both_missing; continue; }
            if (m_ref != m_gpu) { ++one_missing; continue; }
            max_err = std::max(max_err, std::abs(d_ref - d_gpu));
        }
        Check(one_missing == 0, "NN found/not-found agrees", static_cast<double>(one_missing));
        Check(max_err < 1e-4, "NN distance max abs error [m]", max_err);
        std::printf("       %zu queries in %.3f ms (%zu without neighbour)\n", queries.size(), Ms(t0, t1), both_missing);
    }

    // 3) Registration: misalign a fresh scan of the current surroundings and align it back.
    {
        const double x = 598.0;
        Points scan = MakeScene(rng, x - 40.0, x + 60.0);
        Points frame;  // in sensor frame, sensor at (x, 0, 1.8)
        const Sophus::SE3d T_world_sensor(Sophus::SO3d(), Eigen::Vector3d(x, 0.0, 1.8));
        for (const auto &p : scan) frame.push_back(T_world_sensor.inverse() * p);
        frame = kiss_icp::VoxelDownsample(frame, 0.5 * voxel_size);

        const Sophus::SE3d delta = Sophus::SE3d::exp((Eigen::Matrix<double, 6, 1>() << 0.4, -0.25, 0.05, 0.004, -0.003, 0.03).finished());
        const Sophus::SE3d guess = T_world_sensor * delta;
        kiss_icp::Registration reg(500, 1e-4, 0);
        const auto t0 = Clock::now();
        const Sophus::SE3d est = reg.AlignPointsToMap(frame, gpu, guess, 3.0 * 1.0, 1.0);
        const auto t1 = Clock::now();
        const Sophus::SE3d err = T_world_sensor.inverse() * est;
        Check(err.translation().norm() < 0.02, "registration translation error [m]", err.translation().norm());
        Check(err.so3().log().norm() < 0.002, "registration rotation error [rad]", err.so3().log().norm());
        std::printf("       %zu source points, AlignPointsToMap %.3f ms\n", frame.size(), Ms(t0, t1));
    }

    std::printf("       %d map updates, avg %.3f ms, %zu voxels live\n", frames, t_update / frames, gpu.NumVoxels());
    std::printf("%s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
