cmake_minimum_required(VERSION 2.8)

project(image-alignment)

if (WIN32)
    set(OpenCV_STATIC OFF)
	set(OpenCV_SHARED ON)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS")
endif()

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} "inc")

# Library
add_library(ialign
    inc/imagealign/imagealign.h
    inc/imagealign/config.h
    inc/imagealign/gradient.h
    inc/imagealign/sampling.h
    inc/imagealign/warp.h
    inc/imagealign/warp_image.h
    inc/imagealign/image_pyramid.h
    inc/imagealign/align_base.h
    inc/imagealign/forward_additive.h
    inc/imagealign/forward_compositional.h
    inc/imagealign/inverse_compositional.h
    src/unused.cpp
)
	
target_link_libraries(ialign ${OpenCV_LIBRARIES})
	
# Samples

add_executable(example_align examples/align.cpp)
target_link_libraries(example_align ialign ${OpenCV_LIBRARIES})

add_executable(example_optflow examples/optical_flow.cpp)
target_link_libraries(example_optflow ialign ${OpenCV_LIBRARIES})

# Tests

add_executable(tests
    tests/catch.hpp
    tests/warp.cpp
    tests/sampling.cpp
    tests/algorithms.cpp
)
target_link_libraries(tests ialign ${OpenCV_LIBRARIES})