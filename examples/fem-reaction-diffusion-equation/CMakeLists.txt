cmake_minimum_required(VERSION 3.9)
project(fem-reaction-diffusion-equation)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

find_package(VTK COMPONENTS
        CommonColor
        CommonCore
        IOGeometry
        InteractionStyle
        RenderingContextOpenGL2
        RenderingCore
        RenderingFreeType
        RenderingGL2PSOpenGL2
        RenderingOpenGL2
        REQUIRED)
add_executable(fem-reaction-diffusion-equation fem-reaction-diffusion-equation.cpp mesh.cpp)
target_link_libraries(fem-reaction-diffusion-equation PUBLIC Ginkgo::ginkgo PRIVATE ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fem-reaction-diffusion-equation MODULES ${VTK_LIBRARIES})
