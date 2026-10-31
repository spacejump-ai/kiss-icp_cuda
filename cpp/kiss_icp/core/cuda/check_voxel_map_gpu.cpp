// Self-check for the voxel map. Compares kiss_icp::VoxelHashMap (GPU in the KISS_ICP_USE_CUDA
// build, the original robin_map otherwise) against a reference re-implementation, then runs a
// registration and compares it with a reference ICP on the same map.
//
// In the CUDA build the reference mimics the GPU storage exactly (points rounded to float,
// too-close test as squared distance in double), so map content must match bit for bit.
//
//   ./kiss_icp_cuda_check     -> PASS/FAIL per check and timings, exit code != 0 on failure
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <random>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/core/VoxelUtils.hpp"

namespace {
using kiss_icp::Voxel;
using Points = std::vector<Eigen::Vector3d>;
using Clock = std::chrono::steady_clock;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

#ifdef KISS_ICP_CUDA
constexpr bool kGpu = true;
#else
constexpr bool kGpu = false;
#endif

double Ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int failures = 0;
void Check(bool ok, const char *what, double value = 0.0) {
    std::printf("[%s] %-48s %g\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

// ---- reference: the original CPU VoxelHashMap (std::unordered_map instead of robin_map) ------

Eigen::Vector3d ToFloat(const Eigen::Vector3d &p) { return p.cast<float>().cast<double>(); }

// ---- reference map ---------------------------------------------------------------------------
struct RefMap {
    double voxel_size, max_distance;
    unsigned int max_points;
    std::unordered_map<Voxel, Points> map;

    bool TooClose(const Eigen::Vector3d &q, const Eigen::Vector3d &p) const {
        if (kGpu) return (q - p).squaredNorm() < voxel_size * voxel_size / max_points;
        return (q - p).norm() < std::sqrt(voxel_size * voxel_size / max_points);  // original
    }
    void Add(const Points &points) {
        for (const auto &p_in : points) {
            const auto v = kiss_icp::PointToVoxel(p_in, voxel_size);  // voxel from the input
            const Eigen::Vector3d p = kGpu ? ToFloat(p_in) : p_in;     // stored value
            auto it = map.find(v);
            if (it == map.end()) {
                map.emplace(v, Points{p});
                continue;
            }
            auto &vp = it->second;
            if (vp.size() == max_points ||
                std::any_of(vp.begin(), vp.end(), [&](const auto &q) { return TooClose(q, p); }))
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
    std::pair<Eigen::Vector3d, double> Closest(const Eigen::Vector3d &q) const {
        const auto v = kiss_icp::PointToVoxel(q, voxel_size);
        double best = std::numeric_limits<double>::max();
        Eigen::Vector3d nn = Eigen::Vector3d::Zero();
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    auto it = map.find(v + Voxel(dx, dy, dz));
                    if (it == map.end()) continue;
                    for (const auto &p : it->second) {
                        const double d = (p - q).norm();
                        if (d < best) best = d, nn = p;
                    }
                }
        return {nn, best};
    }
    // KISS-ICP normal equations for the source transformed by T
    std::pair<Matrix6d, Vector6d> LinearSystem(const Points &src, const Sophus::SE3d &T,
                                               double max_corr, double ks) const {
        Matrix6d JTJ = Matrix6d::Zero();
        Vector6d JTr = Vector6d::Zero();
        for (const auto &s : src) {
            const Eigen::Vector3d p = T * s;
            const auto [nn, d] = Closest(p);
            if (d >= max_corr) continue;
            const Eigen::Vector3d r = p - nn;
            Eigen::Matrix<double, 3, 6> J;
            J.block<3, 3>(0, 0).setIdentity();
            J.block<3, 3>(0, 3) = -Sophus::SO3d::hat(p);
            const double w = ks * ks / ((ks + r.squaredNorm()) * (ks + r.squaredNorm()));
            JTJ += J.transpose() * w * J;
            JTr += J.transpose() * w * r;
        }
        return {JTJ, JTr};
    }
    Sophus::SE3d Align(const Points &src, const Sophus::SE3d &guess, double max_corr, double ks,
                       int max_iter, double conv) const {
        Sophus::SE3d T_icp;
        for (int j = 0; j < max_iter; ++j) {
            const auto [JTJ, JTr] = LinearSystem(src, T_icp * guess, max_corr, ks);
            const Vector6d dx = JTJ.ldlt().solve(-JTr);
            T_icp = Sophus::SE3d::exp(dx) * T_icp;
            if (dx.norm() < conv) break;
        }
        return T_icp * guess;
    }
};

// ---- synthetic street: ground, wavy facades, poles, boxes -------------------------------------
// The facades wiggle in y along x so that the travel direction is well constrained.
Points MakeScene(std::mt19937 &rng, double x0, double x1) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.01);
    Points pts;
    const auto n_along = static_cast<int>((x1 - x0) * 400);
    for (int i = 0; i < n_along; ++i) {
        const double x = x0 + (x1 - x0) * u(rng);
        pts.emplace_back(x, -12.0 + 24.0 * u(rng), noise(rng));  // ground
        pts.emplace_back(x, -12.0 + 1.0 * std::sin(0.7 * x) + noise(rng), 10.0 * u(rng));
        pts.emplace_back(x, 12.0 + 1.0 * std::cos(0.45 * x) + noise(rng), 10.0 * u(rng));
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

std::size_t NumVoxels(const kiss_icp::VoxelHashMap &m) { return m.NumVoxels(); }
}  // namespace

int main() {
    const double voxel_size = 1.0, max_range = 60.0;
    const unsigned int max_points = 20;
    std::mt19937 rng(42);
    std::printf("mode: %s\n", kGpu ? "CUDA voxel map" : "CPU voxel map (original)");

#ifdef KISS_ICP_CUDA
    kiss_icp::VoxelHashMap map(voxel_size, max_range, max_points, /*max_voxels=*/1u << 17);
#else
    kiss_icp::VoxelHashMap map(voxel_size, max_range, max_points);
#endif
    RefMap ref{voxel_size, max_range, max_points, {}};

    // 00) Preprocess (deskew + range crop) vs the CPU implementation.
    {
        std::uniform_real_distribution<double> ua(-M_PI / 3, M_PI / 3), ur(0.2, 140.0), uz(-0.3, 0.3);
        Points scan;
        std::vector<double> stamps;
        for (int i = 0; i < 650000; ++i) {   // 120 deg FoV, ranges across both crop limits
            const double a = ua(rng), r = ur(rng);
            scan.emplace_back(r * std::cos(a), r * std::sin(a), r * uz(rng));
            stamps.push_back(1.7e9 + 0.1 * (a + M_PI / 3) / (2 * M_PI / 3));  // 100 ms sweep
        }
        Vector6d xi;
        xi << 1.2, 0.05, 0.01, 0.002, -0.001, 0.05;  // ~1.2 m, ~3 deg per frame
        const Sophus::SE3d motion = Sophus::SE3d::exp(xi);
        for (bool deskew : {false, true}) {
            const kiss_icp::Preprocessor pre(100.0, 1.0, deskew, 0);
            const auto t0 = Clock::now();
            const Points gpu_out = pre.Preprocess(scan, stamps, motion);
            const auto t1 = Clock::now();
            const Points cpu_out = pre.PreprocessCpu(scan, stamps, motion);
            const auto t2 = Clock::now();
            double max_err = 0.0;
            if (gpu_out.size() == cpu_out.size())
                for (std::size_t i = 0; i < cpu_out.size(); ++i)
                    max_err = std::max(max_err, (gpu_out[i] - cpu_out[i]).norm());
            char what[96];
            std::snprintf(what, sizeof(what), "preprocess %s: same points/order as CPU", deskew ? "deskew+crop" : "crop only");
            Check(gpu_out.size() == cpu_out.size(), what, static_cast<double>(gpu_out.size()));
            std::snprintf(what, sizeof(what), "preprocess %s: max abs diff [m]", deskew ? "deskew+crop" : "crop only");
            Check(deskew ? max_err < 1e-9 : max_err == 0.0, what, max_err);
            std::printf("       %zu -> %zu points: Preprocess %.2f ms, CPU version %.2f ms\n", scan.size(),
                        gpu_out.size(), Ms(t0, t1), Ms(t1, t2));
        }
    }

    // 0) VoxelDownsample vs the CPU implementation (650k points, negative coordinates, points
    //    exactly on voxel borders, duplicates).
    {
        std::uniform_real_distribution<double> uc(-100.0, 100.0), uh(-3.0, 20.0);
        std::uniform_int_distribution<int> grid(-400, 400);
        Points cloud;
        cloud.reserve(650000);
        for (int i = 0; i < 650000; ++i) {
            if (i % 10 == 0) {         // on a 0.25 m lattice: exactly on 0.5 m borders
                cloud.emplace_back(0.25 * grid(rng), 0.25 * grid(rng), 0.25 * (grid(rng) % 40));
            } else if (i % 97 == 0) {  // exact duplicate of an earlier point
                cloud.push_back(cloud[static_cast<std::size_t>(i / 2)]);
            } else {
                cloud.emplace_back(uc(rng), uc(rng), uh(rng));
            }
        }
        for (double vs : {0.5, 1.5}) {
            // reference: first point of each voxel, in input order
            Points ref_ds;
            {
                std::unordered_map<Voxel, bool> seen;
                seen.reserve(cloud.size());
                for (const auto &p : cloud)
                    if (seen.emplace(kiss_icp::PointToVoxel(p, vs), true).second) ref_ds.push_back(p);
            }
            const auto t0 = Clock::now();
            const Points ds = kiss_icp::VoxelDownsample(cloud, vs);
            const auto t1 = Clock::now();
            const Points cpu = kiss_icp::VoxelDownsampleCpu(cloud, vs);
            const auto t2 = Clock::now();

            auto sorted = [](Points v) {
                std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
                    return std::lexicographical_compare(a.data(), a.data() + 3, b.data(), b.data() + 3);
                });
                return v;
            };
            char what[96];
            std::snprintf(what, sizeof(what), "downsample %.1f m: same points as CPU version", vs);
            Check(sorted(ds) == sorted(cpu), what, static_cast<double>(ds.size()));
            if (kGpu) {
                std::snprintf(what, sizeof(what), "downsample %.1f m: input order kept", vs);
                Check(ds == ref_ds, what);
            }
            std::printf("       %zu -> %zu points: VoxelDownsample %.2f ms, CPU version %.2f ms\n",
                        cloud.size(), ds.size(), Ms(t0, t1), Ms(t1, t2));
        }
        // below the GPU threshold the CPU path is used; result must still be right
        const Points small(cloud.begin(), cloud.begin() + 1000);
        const Points ds_small = kiss_icp::VoxelDownsample(small, 0.5);
        Check(ds_small.size() == kiss_icp::VoxelDownsampleCpu(small, 0.5).size(),
              "downsample small input (CPU path)", static_cast<double>(ds_small.size()));
    }

    // 1) Drive along x, insert + crop like KissICP does (exercises erase + tombstone rebuild).
    double t_update = 0.0;
    int frames = 0;
    for (double x = 0.0; x < 600.0; x += 2.0, ++frames) {
        Points scan = MakeScene(rng, x - 40.0, x + 60.0);
        scan.resize(std::min<std::size_t>(scan.size(), 40000));
        const Eigen::Vector3d origin(x, 0.0, 1.8);
        const auto t0 = Clock::now();
        map.Update(scan, origin);
        t_update += Ms(t0, Clock::now());
        ref.Add(scan);
        ref.Remove(origin);
    }
    const auto cloud = map.Pointcloud();
    std::printf("       map: %zu voxels / %zu points, reference: %zu voxels / %zu points\n",
                NumVoxels(map), cloud.size(), ref.map.size(), ref.NumPoints());
    Check(NumVoxels(map) == ref.map.size(), "voxel count matches reference",
          static_cast<double>(NumVoxels(map)));
    Check(cloud.size() == ref.NumPoints(), "point count matches reference",
          static_cast<double>(cloud.size()));
#ifdef KISS_ICP_CUDA
    Check(map.Gpu().DroppedForCapacity() == 0, "no points dropped for capacity");
#endif
    {
        using Key = std::array<double, 3>;
        std::vector<Key> a, b, diff;
        for (const auto &p : cloud) a.push_back({p.x(), p.y(), p.z()});
        for (const auto &kv : ref.map)
            for (const auto &p : kv.second) b.push_back({p.x(), p.y(), p.z()});
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                      std::back_inserter(diff));
        Check(diff.empty(), "map content identical (points not in both)",
              static_cast<double>(diff.size()));
    }

    // 2) Nearest neighbour vs reference.
    {
        Points queries;
        std::uniform_real_distribution<double> ux(540.0, 600.0), uy(-13.0, 13.0), uz(-1.0, 11.0);
        for (int i = 0; i < 20000; ++i) queries.emplace_back(ux(rng), uy(rng), uz(rng));
        const auto t0 = Clock::now();
#ifdef KISS_ICP_CUDA
        const auto res = map.GetClosestNeighbors(queries);
#else
        std::vector<std::tuple<Eigen::Vector3d, double>> res;
        for (const auto &q : queries) res.push_back(map.GetClosestNeighbor(q));
#endif
        const auto t1 = Clock::now();
        double max_err = 0.0;
        std::size_t missing = 0, disagree = 0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const double d_ref = ref.Closest(queries[i]).second;
            const double d_map = std::get<1>(res[i]);
            const bool m_ref = d_ref == std::numeric_limits<double>::max();
            const bool m_map = d_map == std::numeric_limits<double>::max();
            if (m_ref && m_map) { ++missing; continue; }
            if (m_ref != m_map) { ++disagree; continue; }
            max_err = std::max(max_err, std::abs(d_ref - d_map));
        }
        Check(disagree == 0, "NN found/not-found agrees", static_cast<double>(disagree));
        Check(max_err < 1e-4, "NN distance max abs error [m]", max_err);
        std::printf("       %zu queries in %.3f ms (%zu without neighbour)\n", queries.size(),
                    Ms(t0, t1), missing);
    }

    // 3) Registration of a fresh scan with a known offset, vs reference ICP on the same map.
    {
        const double x = 598.0;
        const Points scan = MakeScene(rng, x - 40.0, x + 60.0);
        const Sophus::SE3d T_world_sensor(Sophus::SO3d(), Eigen::Vector3d(x, 0.0, 1.8));
        Points frame;
        for (const auto &p : scan) frame.push_back(T_world_sensor.inverse() * p);
        frame = kiss_icp::VoxelDownsample(frame, 1.5 * voxel_size);  // like KissICP's source

        Vector6d xi;
        xi << 0.4, -0.25, 0.05, 0.004, -0.003, 0.03;
        const Sophus::SE3d guess = T_world_sensor * Sophus::SE3d::exp(xi);
        const double max_corr = 3.0, ks = 1.0, conv = 1e-4;
        const int max_iter = 500;

#ifdef KISS_ICP_CUDA
        {   // one linear system at the initial guess, GPU vs reference
            Points src(frame);
            const Sophus::SE3d T = guess;
            map.Gpu().SetSource(src.front().data(), src.size());
            const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> T34 = T.matrix3x4();
            const auto s = map.Gpu().LinearSystem(T34.data(), max_corr, ks);
            Matrix6d JTJ;
            int idx = 0;
            for (int a = 0; a < 6; ++a)
                for (int b = a; b < 6; ++b) JTJ(a, b) = JTJ(b, a) = s.JTJ_upper[std::size_t(idx++)];
            const Vector6d JTr = Eigen::Map<const Vector6d>(s.JTr.data());
            const auto [JTJ_ref, JTr_ref] = ref.LinearSystem(src, T, max_corr, ks);
            Check((JTJ - JTJ_ref).norm() / JTJ_ref.norm() < 1e-6, "JTJ rel. error vs reference",
                  (JTJ - JTJ_ref).norm() / JTJ_ref.norm());
            Check((JTr - JTr_ref).norm() / JTr_ref.norm() < 1e-6, "JTr rel. error vs reference",
                  (JTr - JTr_ref).norm() / JTr_ref.norm());
        }
#endif
        kiss_icp::Registration reg(max_iter, conv, 0);
        const auto t0 = Clock::now();
        const Sophus::SE3d est = reg.AlignPointsToMap(frame, map, guess, max_corr, ks);
        const auto t1 = Clock::now();
        const Sophus::SE3d est_ref = ref.Align(frame, guess, max_corr, ks, max_iter, conv);

        const Sophus::SE3d d_ref = est_ref.inverse() * est;
        Check(d_ref.translation().norm() < 1e-3, "pose vs reference ICP, translation [m]",
              d_ref.translation().norm());
        Check(d_ref.so3().log().norm() < 1e-4, "pose vs reference ICP, rotation [rad]",
              d_ref.so3().log().norm());
        const Sophus::SE3d err = T_world_sensor.inverse() * est;
        Check(err.translation().norm() < 0.02, "pose vs ground truth, translation [m]",
              err.translation().norm());
        Check(err.so3().log().norm() < 0.002, "pose vs ground truth, rotation [rad]",
              err.so3().log().norm());
        std::printf("       %zu source points, AlignPointsToMap %.3f ms\n", frame.size(), Ms(t0, t1));
    }

    std::printf("       %d map updates, avg %.3f ms\n", frames, t_update / frames);
    std::printf("%s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
