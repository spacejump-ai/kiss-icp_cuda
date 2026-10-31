# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
add_library(kiss_icp_core STATIC)
target_sources(kiss_icp_core PRIVATE Registration.cpp VoxelHashMap.cpp VoxelUtils.cpp Preprocessing.cpp Threshold.cpp Profile.cpp)
target_link_libraries(kiss_icp_core PUBLIC Eigen3::Eigen tsl::robin_map TBB::tbb Sophus::Sophus)
set_global_target_properties(kiss_icp_core)

if(KISS_ICP_USE_CUDA)
  target_sources(kiss_icp_core PRIVATE cuda/VoxelMapGpu.cu)
  target_link_libraries(kiss_icp_core PUBLIC CUDA::cudart PRIVATE cuco::cuco)
  target_compile_definitions(
    kiss_icp_core PUBLIC KISS_ICP_CUDA KISS_ICP_CUDA_MAX_POINTS_PER_VOXEL=${KISS_ICP_CUDA_MAX_POINTS_PER_VOXEL}
                         KISS_ICP_CUDA_MAX_VOXELS=${KISS_ICP_CUDA_MAX_VOXELS})
  target_compile_options(kiss_icp_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
  set_target_properties(kiss_icp_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

  option(KISS_ICP_CUDA_BUILD_CHECK "Build the CPU-vs-GPU voxel map check executable" ON)
  if(KISS_ICP_CUDA_BUILD_CHECK)
    add_executable(kiss_icp_cuda_check cuda/check_voxel_map_gpu.cpp)
    target_link_libraries(kiss_icp_cuda_check PRIVATE kiss_icp_core)
  endif()
endif()
