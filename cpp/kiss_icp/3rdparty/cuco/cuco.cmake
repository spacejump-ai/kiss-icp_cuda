# cuCollections (header-only). Pinned to the commit FAR-LIO uses.
# A parent project (or toolchain) may already provide cuco::cuco.
if(TARGET cuco::cuco)
  return()
endif()
include(FetchContent)
set(BUILD_TESTS OFF CACHE BOOL "Don't build cuCollections tests")
set(BUILD_BENCHMARKS OFF CACHE BOOL "Don't build cuCollections benchmarks")
set(BUILD_EXAMPLES OFF CACHE BOOL "Don't build cuCollections examples")
FetchContent_Declare(
  cuCollections
  GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
  GIT_TAG 67110a53135ffdd43814794c6c39751f8d74e080)
FetchContent_MakeAvailable(cuCollections)
