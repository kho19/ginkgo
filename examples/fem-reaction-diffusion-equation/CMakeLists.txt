cmake_minimum_required(VERSION 3.9)
set(target_name "fem-reaction-diffusion-equation")
project(${target_name})

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

find_package(VTK COMPONENTS
        CommonColor
        CommonCore
        FiltersProgrammable
        FiltersSources
        IOGeometry
        InteractionStyle
        RenderingContextOpenGL2
        RenderingCore
        RenderingFreeType
        RenderingGL2PSOpenGL2
        RenderingOpenGL2
        QUIET)

if (VTK_FOUND)
    include(../../cmake/unified_kernels.cmake)
    add_executable(${target_name} ${target_name}.cpp mesh.cpp)
    ginkgo_add_unified_kernels(${target_name} kernels.cpp)
    target_link_libraries(${target_name} PUBLIC Ginkgo::ginkgo PRIVATE ${VTK_LIBRARIES})
    vtk_module_autoinit(TARGETS ${target_name} MODULES ${VTK_LIBRARIES})
    configure_file(./data/cube.obj data/cube.obj COPYONLY)
    configure_file(./data/cube16.obj data/cube16.obj COPYONLY)
    configure_file(./data/dragon.obj data/dragon.obj COPYONLY)
    configure_file(./data/sphere.obj data/sphere.obj COPYONLY)
    configure_file(./data/sphere6.obj data/sphere6.obj COPYONLY)
else()
    message(STATUS "No VTK found, disabling ${target_name} example")
endif()

