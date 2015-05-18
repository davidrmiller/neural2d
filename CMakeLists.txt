cmake_minimum_required(VERSION 3.1)
project(neural2d CXX)

option(ENABLE_WEBSERVER "Enable webserver GUI." OFF)
option(BUILD_TESTS "Build tests (recommended when changing code in neural2d library)." OFF)

set(NEURAL2D_SOURCES
    src/neural2d-core.cpp
    src/parseTopologyConfig.cpp)

if(WIN32)
    set(ENABLE_WEBSERVER OFF)
endif()

if(ENABLE_WEBSERVER)
    list(APPEND NEURAL2D_SOURCES
         src/messagequeue.cpp
         src/webserver.cpp)
endif()

include_directories(includes)

# TODO: Create configuration file instead of adding defines from the command line.
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-DDISABLE_WEBSERVER -std=c++11")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "/DDISABLE_WEBSERVER")
endif()

add_library(neural2d ${NEURAL2D_SOURCES})

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
