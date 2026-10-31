// Lightweight per-frame profiling of the KISS-ICP core functions.
//
// Enable with the environment variable KISS_ICP_PROFILE=<n>: a summary (medians over n frames) is
// printed to stderr every n frames and once more at normal program exit. A frame is the interval
// between two consecutive Registration::AlignPointsToMap calls, so this works with any pipeline
// built on the core (KissICP, or your own node). Disabled: one branch per instrumented call.
#pragma once

#include <chrono>
#include <cstddef>

namespace kiss_icp::profile {

enum Stage : int {
    kPreprocess = 0,  // Preprocessor::Preprocess
    kVoxelize,        // VoxelDownsample (all calls of the frame)
    kAlign,           // Registration::AlignPointsToMap
    kMapAdd,          // VoxelHashMap::AddPoints
    kMapCrop,         // VoxelHashMap::RemovePointsFarFromLocation
    kPointcloud,      // VoxelHashMap::Pointcloud (e.g. a viewer pulling the local map)
    kNnSingle,        // VoxelHashMap::GetClosestNeighbor, CUDA build only (one GPU round trip each)
    kNnBatch,         // VoxelHashMap::GetClosestNeighbors, CUDA build only
    kNumStages
};

/// True when KISS_ICP_PROFILE is set to a value > 0 (read once).
bool Enabled();

/// Called at the start of AlignPointsToMap; closes the previous frame.
void BeginFrame();
void SetAlignInfo(int icp_iterations, std::size_t source_points);
void Add(Stage stage, double ms);

class Scope {
public:
    explicit Scope(Stage stage) : stage_(stage), on_(Enabled()) {
        if (on_) t0_ = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (on_) {
            Add(stage_, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_)
                            .count());
        }
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    Stage stage_;
    bool on_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace kiss_icp::profile
