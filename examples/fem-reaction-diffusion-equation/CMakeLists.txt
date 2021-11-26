cmake_minimum_required(VERSION 3.9)
project(fem-reaction-diffusion-equation)

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
    add_executable(fem-reaction-diffusion-equation fem-reaction-diffusion-equation.cpp mesh.cpp)
    target_link_libraries(fem-reaction-diffusion-equation PUBLIC Ginkgo::ginkgo PRIVATE ${VTK_LIBRARIES})
    vtk_module_autoinit(TARGETS fem-reaction-diffusion-equation MODULES ${VTK_LIBRARIES})
    configure_file(./data/cube.obj data/cube.obj COPYONLY)
    configure_file(./data/cube16.obj data/cube16.obj COPYONLY)
    configure_file(./data/dragon.obj data/dragon.obj COPYONLY)
    configure_file(./data/sphere.obj data/sphere.obj COPYONLY)
    configure_file(./data/sphere6.obj data/sphere6.obj COPYONLY)

else()
    message(STATUS "No VTK found, disabling fem-reaction-diffusion-equation example")
endif()

