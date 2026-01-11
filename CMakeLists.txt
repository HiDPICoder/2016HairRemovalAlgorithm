cmake_minimum_required(VERSION 3.10)
project(HairRazor LANGUAGES CXX CUDA)

# 1. Set C++ Standard and MSVC specific definitions
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    # Disable warnings for "unsafe" functions like sprintf
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Define math constants like M_PI
    add_compile_definitions(_USE_MATH_DEFINES)
    
    # Enable fast floating point model ONLY for C++ files
    # (Prevents passing /fp:fast to nvcc, which causes the crash)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:fast>)
endif()

# 2. Configuration Options matching Makefile
# GUI is disabled as requested. MPI defaults to OFF for simplicity on Windows.
add_compile_definitions(GUI_SUPPORT=0)
add_compile_definitions(MPI_SUPPORT=0)

# 3. CUDA Configuration
set(CMAKE_CUDA_ARCHITECTURES native)
# Matches "--maxrregcount=20 -use_fast_math"
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -maxrregcount=20")

# 4. Include Directories
include_directories(
    ${CMAKE_SOURCE_DIR}/Code/src
    ${CMAKE_SOURCE_DIR}/Code/src/include
)

# 5. Define Sources for 'hairrazor'
# Note: 'gaps.cu' includes 'skelft.cu' directly, so we only compile gaps.cu.
set(SOURCES
    Code/src/main.cpp
    Code/src/color.cpp
    Code/src/flags.cpp
    Code/src/fmm.cpp
    Code/src/inpainting.cpp
    Code/src/io.cpp
    Code/src/mfmm.cpp
    Code/src/pgm.cpp
    Code/src/skelft_core.cpp
    Code/src/gaps.cu
)

# 6. Targets
add_executable(hairrazor ${SOURCES})

# 'batch' executable
add_executable(batch Code/src/batch.cpp)