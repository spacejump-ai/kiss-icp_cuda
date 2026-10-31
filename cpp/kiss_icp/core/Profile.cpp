#include "Profile.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#ifdef KISS_ICP_CUDA
#include "cuda/VoxelMapGpu.hpp"
#endif

namespace kiss_icp::profile {
namespace {
using Clock = std::chrono::steady_clock;

int ReportEvery() {
    static const int every = [] {
        const char *env = std::getenv("KISS_ICP_PROFILE");
        if (env == nullptr) return 0;
        const int n = std::atoi(env);
        return n > 0 ? n : 50;  // set but not a number -> 50
    }();
    return every;
}

struct Frame {
    std::array<double, kNumStages> ms{};
    std::array<int, kNumStages> calls{};
    double period_ms = 0.0;
    int iterations = 0;
    std::size_t source_points = 0;
};

constexpr std::array<const char *, kNumStages> kNames = {
    "preprocess", "voxelize", "align", "map add", "map crop", "pointcloud", "nn single", "nn batch"};

struct State {
    std::mutex mutex;
    Frame current;
    std::vector<Frame> window;
    Clock::time_point last_begin;
    bool started = false;
    long frames_total = 0;

    ~State() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!window.empty()) Report();
    }

    static double Median(std::vector<double> v) {
        if (v.empty()) return 0.0;
        const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    }

    void Report() {
        auto median_of = [&](auto get) {
            std::vector<double> v;
            v.reserve(window.size());
            for (const auto &f : window) v.push_back(static_cast<double>(get(f)));
            return Median(std::move(v));
        };
#ifdef KISS_ICP_CUDA
        const char *backend = "cuda";
#else
        const char *backend = "cpu";
#endif
        std::string line = "[kiss_icp profile " + std::string(backend) + "] frames " +
                           std::to_string(frames_total - static_cast<long>(window.size()) + 1) +
                           "-" + std::to_string(frames_total) + ", median ms:";
        char buf[160];
        double stages_sum = 0.0;
        for (int s = 0; s < kNumStages; ++s) {
            const double calls = median_of([s](const Frame &f) { return f.calls[std::size_t(s)]; });
            double max_calls = 0.0;
            for (const auto &f : window) max_calls = std::max(max_calls, double(f.calls[std::size_t(s)]));
            if (max_calls == 0.0) continue;  // stage never used in this window
            const double ms = median_of([s](const Frame &f) { return f.ms[std::size_t(s)]; });
            stages_sum += ms;
            if (s == kAlign) {
                std::snprintf(buf, sizeof(buf), " %s %.1f (%.0f it, %.0f pts) |", kNames[std::size_t(s)], ms,
                              median_of([](const Frame &f) { return f.iterations; }),
                              median_of([](const Frame &f) { return f.source_points; }));
            } else if (calls != 1.0) {
                std::snprintf(buf, sizeof(buf), " %s %.1f (%.0f calls) |", kNames[std::size_t(s)], ms, calls);
            } else {
                std::snprintf(buf, sizeof(buf), " %s %.1f |", kNames[std::size_t(s)], ms);
            }
            line += buf;
        }
        const double outside = median_of([](const Frame &f) {
            double sum = 0.0;
            for (double v : f.ms) sum += v;
            return std::max(0.0, f.period_ms - sum);
        });
        const double period = median_of([](const Frame &f) { return f.period_ms; });
        std::snprintf(buf, sizeof(buf), " outside kiss %.1f | frame period %.1f (%.1f Hz)", outside,
                      period, period > 0.0 ? 1000.0 / period : 0.0);
        line += buf;
        std::fprintf(stderr, "%s\n", line.c_str());
        window.clear();
    }
};

State &GetState() {
    static State state;
    return state;
}
}  // namespace

bool Enabled() { return ReportEvery() > 0; }

namespace {
std::atomic<bool> g_in_align{false};
}
bool InAlign() { return g_in_align.load(std::memory_order_relaxed); }
void SetInAlign(bool in_align) { g_in_align.store(in_align, std::memory_order_relaxed); }

void BeginFrame() {
    if (!Enabled()) return;
    State &st = GetState();
    std::lock_guard<std::mutex> lock(st.mutex);
    const auto now = Clock::now();
    if (!st.started) {
        st.started = true;
#ifdef KISS_ICP_CUDA
        std::fprintf(stderr, "[kiss_icp profile] enabled, CUDA voxel map on %s, report every %d frames\n",
                     gpu::DeviceDescription().c_str(), ReportEvery());
#else
        std::fprintf(stderr, "[kiss_icp profile] enabled, CPU voxel map, report every %d frames\n",
                     ReportEvery());
#endif
    } else {
        st.current.period_ms = std::chrono::duration<double, std::milli>(now - st.last_begin).count();
        st.window.push_back(st.current);
        ++st.frames_total;
        if (static_cast<int>(st.window.size()) >= ReportEvery()) st.Report();
    }
    // A frame bucket runs from one AlignPointsToMap start to the next: align + map update of
    // frame k, whatever the node does in between, then preprocess + voxelize of frame k+1.
    st.current = Frame{};
    st.last_begin = now;
}

void SetAlignInfo(int icp_iterations, std::size_t source_points) {
    if (!Enabled()) return;
    State &st = GetState();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.current.iterations = icp_iterations;
    st.current.source_points = source_points;
}

void Add(Stage stage, double ms) {
    State &st = GetState();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.current.ms[std::size_t(stage)] += ms;
    st.current.calls[std::size_t(stage)] += 1;
}

}  // namespace kiss_icp::profile
