cmake_minimum_required(VERSION 3.9)
project(fem-reaction-diffusion-equation-plain-cuda)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(fem-reaction-diffusion-equation-plain-cuda fem-reaction-diffusion-equation-plain.cpp mesh.cpp)
target_link_libraries(fem-reaction-diffusion-equation-plain-cuda PUBLIC Ginkgo::ginkgo)
