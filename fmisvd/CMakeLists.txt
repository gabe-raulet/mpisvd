cmake_minimum_required(VERSION 3.10)

project(fmi_svd CXX)

set(CMAKE_CXX_STANDARD 17)

ADD_DEFINITIONS(-O2)

add_subdirectory(./fmi)

include_directories(./inc ${FMI_INCLUDE_DIRS} /opt/homebrew/Cellar/openblas/0.3.24/include /opt/homebrew/include)

add_library(lib_fmi_svd src/svd_fmi.cpp src/fmi_wrapper.cpp src/kiss.cpp src/mmio_dense.cpp 
            src/mmio.cpp src/svd_routines.cpp)

link_libraries(lib_fmi_svd)

add_executable(fmi_svd fmi_svd.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE FMI "-L/opt/homebrew/Cellar/lapack/3.12.0/lib -L/opt/homebrew/Cellar/openblas/0.3.24/lib -llapacke -lopenblas")
target_include_directories(${PROJECT_NAME} PRIVATE ${FMI_INCLUDE_DIRS})