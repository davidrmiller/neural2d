# CMakeLists.txt for neural2d
#
# Quick start guide:
#
# git clone https://github.com/davidrmiller/neural2d
# cd neural2d
# mkdir build
# cd build
# cmake ..        (or:  cmake -G generator-name .. )
# make test       (or open your favorite IDE or whatever)

cmake_minimum_required (VERSION 2.8.12)
#cmake_policy(SET CMP0037 OLD)  # This permits us to define a target named "test" without getting CMake warnings
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # This causes WIN32 to be 0 in Cygwin

# Build options:

# Set WEBSERVER to ON to include the integrated neural2d GUI:
option(WEBSERVER "Enable webserver GUI." OFF)

# For now, we don't compile the webserver for MSVC (needs to be fixed):

if(WIN32)
    set(WEBSERVER OFF)
endif()


project (neural2d)

set(CMAKE_BUILD_TYPE Debug)


# Directories:

# We'll set the install directory to the local build directory, even
# though at this time there is no install target:

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR} CACHE PATH "install path" FORCE)


# Sources for the two libraries:

set(NEURAL2D_CORE_LIB_SOURCES
    src/neural2d-core.cpp
    src/parseTopologyConfig.cpp
    src/imageReaderBMP.cpp
    src/imageReaderDat.cpp)

set(NEURAL2D_CORE_WEBSERVER_LIB_SOURCES
    src/messagequeue.cpp
    src/visualize.cpp
    src/webserver.cpp)

# We will build one or two libraries. They are the same, but with and without webserver support:
#     neural2d-core            - always built
#     neural2d-core-webserver  - build only if WEBSERVER is ON

add_library(neural2d-core ${NEURAL2D_CORE_LIB_SOURCES})

if(WEBSERVER)
    FIND_PACKAGE(Threads REQUIRED)
    add_library(neural2d-core-webserver ${NEURAL2D_CORE_LIB_SOURCES} ${NEURAL2D_CORE_WEBSERVER_LIB_SOURCES})
endif()


# Executables

add_executable(neural2d src/neural2d.cpp images/digits/test-1.bmp)
add_executable(unitTest EXCLUDE_FROM_ALL src/unitTest.cpp images/digits/test-1.bmp)


# Linking

if(WEBSERVER)
    target_link_libraries(neural2d neural2d-core-webserver ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(unitTest neural2d-core-webserver ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(neural2d neural2d-core)
    target_link_libraries(unitTest neural2d-core)
endif()


# For the "make test" target, we need to expand the digits.zip archive.

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/images/digits/test-1.bmp
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/images/digits/digits.zip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/images/digits/digits.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/images/digits/
    COMMENT "Preparing images/digits/..."
    VERBATIM
)


# Copy some text files into the build directory:

configure_file(src/topology.txt topology.txt)
configure_file(src/inputData.txt inputData.txt)
configure_file(src/topology-xor.txt topology-xor.txt)
configure_file(src/inputData-xor.txt inputData-xor.txt)

if(WEBSERVER)
    configure_file(src/http-response-template.txt http-response-template.txt)
endif()


# Build targets:

add_custom_target(
    test
    COMMAND neural2d topology.txt inputData.txt weights.txt
    DEPENDS neural2d)

add_custom_target(
    test-xor
    COMMAND neural2d topology-xor.txt inputData-xor.txt weights.txt
    DEPENDS neural2d)


# Compiler and compiler options:

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
else()
    set(warnings "-Wall -Wextra -pedantic")
    add_definitions("-std=c++11")
endif()


if(WEBSERVER)
    add_definitions("-pthread")
else()
    if(MSVC)
        add_definitions("/DDISABLE_WEBSERVER")
    else()
        add_definitions("-DDISABLE_WEBSERVER")
    endif()
endif()


# Show a summary of the more important build targets:
# Should we use a custom command and POST_BUILD for this? Or is it ok to
# just plop this here at the of the file?

message(STATUS "\nUseful build targets:\n"
     "    make all      -- makes only the main neural2d program\n"
     "    make neural2d -- makes only the main neural2d program\n"
     "    make unitTest -- makes the standalone test suite, for developing neural2d-core\n"
     "    make test     -- invokes neural2d to train a net to recognize digit images\n"
     "    make test-xor -- invokes neural2d to train a net to perform XOR\n"
     "    make clean    -- removes anything rebuildable\n")

