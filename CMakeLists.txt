cmake_minimum_required(VERSION 3.0)
project(RecFLS)
#string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
#message(STATUS "lower case project name is "${PROJECT_NAME_LOWER})
message(STATUS ++++${CMAKE_BUILD_TOOL})

cmake_policy(SET CMP0074 NEW)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# Instruct CMake to create code from Qt designer ui files
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 14)

add_definitions(-DSTB_IMAGE_IMPLEMENTATION)
#add_definitions(-DEIGEN_DONT_ALIGN)
#add_definitions(-DPCL_ONLY_CORE_POINT_TYPES)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

include(FindCUDA)

# Find the QtWidgets library
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(Qt5 COMPONENTS Widgets Charts CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(flann REQUIRED)
find_package(VTK REQUIRED)
find_package(G2O REQUIRED)
find_package(CUDA)

message(STATUS G2O_INCLUDE_DIR=${G2O_INCLUDE_DIR})
message(STATUS G2O_FOUND=${G2O_FOUND})

set(__cuda_arch_bin "5.0 5.2 5.3 6.0 6.1")
IF(MSVC)
    set(CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4819\" --ftz=true;--prec-div=false;--prec-sqrt=false;--expt-relaxed-constexpr")
ELSE(MSVC)
    set(CUDA_NVCC_FLAGS "--ftz=true;--prec-div=false;--prec-sqrt=false;--expt-relaxed-constexpr")
ENDIF(MSVC)

set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ./src
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    ./third/stb_image
    ./third/uplinksimple
)

FILE(GLOB BASE_HEADERS src/*.h ./*.hpp)
FILE(GLOB BASE_SRCS src/*.cpp)

FILE(GLOB COMMON_HEADERS src/common/*.h)
FILE(GLOB COMMON_SRCS src/common/*.cpp)

FILE(GLOB CONTROLLER_HEADERS src/controller/*.h)
FILE(GLOB CONTROLLER_SRCS src/controller/*.cpp)

FILE(GLOB CUDA_HEADERS src/cuda/*.h)
FILE(GLOB CUDA_SRCS src/cuda/*.cu src/cuda/*.hpp)

FILE(GLOB DEVICE_HEADERS src/device/*.h)
FILE(GLOB DEVICE_SRCS src/device/*.cpp)

FILE(GLOB EXTRACTOR_HEADERS src/extractor/*.h)
FILE(GLOB EXTRACTOR_SRCS src/extractor/*.hpp src/extractor/*.cpp)

FILE(GLOB MATCHER_HEADERS src/matcher/*.h)
FILE(GLOB MATCHER_SRCS src/matcher/*hpp src/matcher/*.cpp)

FILE(GLOB ODOMETRY_HEADERS src/odometry/*.h)
FILE(GLOB ODOMETRY_SRCS src/odometry/*.cpp)

FILE(GLOB UI_HEADERS src/ui/*.h)
FILE(GLOB UI_SRCS src/ui/*.cpp)
FILE(GLOB UIS src/ui/*.ui)

FILE(GLOB UTIL_HEADERS src/util/*.h)
FILE(GLOB UTIL_SRCS src/util/*.cpp)

FILE(GLOB MOCS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_autogen/*.cpp)

set(OTHER_FILES
    config.ini
    config2.ini
    README
    .gitignore
)

source_group("ui" FILES ${UIS})
source_group("others" FILES ${OTHER_FILES})

set(HEADER_FILES
    ${HEADER_FILES}
    ${BASE_HEADERS}
    ${COMMON_HEADERS}
    ${CONTROLLER_HEADERS}
    ${CUDA_HEADERS}
    ${DEVICE_HEADERS}
    ${EXTRACTOR_HEADERS}
    ${MATCHER_HEADERS}
    ${ODOMETRY_HEADERS}
    ${UI_HEADERS}
    ${UTIL_HEADERS}
)

set(SRC_FILES
    ${SRC_FILES}
    ${BASE_SRCS}
    ${COMMON_SRCS}
    ${CONTROLLER_SRCS}
    ${CUDA_SRCS}
    ${DEVICE_SRCS}
    ${EXTRACTOR_SRCS}
    ${MATCHER_SRCS}
    ${ODOMETRY_SRCS}
    ${UI_SRCS}
    ${UIS}
    ${UTIL_SRCS}
    
)

IF(MSVC)
    source_group("source\\common" FILES ${COMMON_SRCS})
    source_group("header\\common" FILES ${COMMON_HEADERS})
    source_group("source\\controller" FILES ${CONTROLLER_SRCS})
    source_group("header\\controller" FILES ${CONTROLLER_HEADERS})
    source_group("source\\cuda" FILES ${CUDA_SRCS})
    source_group("header\\cuda" FILES ${CUDA_HEADERS})
    source_group("source\\device" FILES ${DEVICE_SRCS})
    source_group("header\\device" FILES ${DEVICE_HEADERS})
    source_group("source\\extractor" FILES ${EXTRACTOR_SRCS})
    source_group("header\\extractor" FILES ${EXTRACTOR_HEADERS})
    source_group("source\\matcher" FILES ${MATCHER_SRCS})
    source_group("header\\matcher" FILES ${MATCHER_HEADERS})
    source_group("source\\odometry" FILES ${ODOMETRY_SRCS})
    source_group("header\\odometry" FILES ${ODOMETRY_HEADERS})
    source_group("source\\ui" FILES ${UI_SRCS})
    source_group("header\\ui" FILES ${UI_HEADERS})
    source_group("source\\util" FILES ${UTIL_SRCS})
    source_group("header\\util" FILES ${UTIL_HEADERS})
    source_group("source\\mocs" FILES ${MOCS})
    source_group("source" FILES ${BASE_SRCS})
    set(SRC_FILES ${SRC_FILES} ${HEADER_FILES} ${OTHER_FILES})
ENDIF(MSVC)

include_directories(${INCLUDE_DIRS})

# Tell CMake to create the helloworld executable
cuda_add_executable(${PROJECT_NAME} ${SRC_FILES})

SET(G2O_LIBS_ITEM G2O_STUFF_LIBRARY G2O_CORE_LIBRARY G2O_CLI_LIBRARY G2O_SOLVER_CSPARSE G2O_SOLVER_CSPARSE_EXTENSION
             G2O_SOLVER_DENSE G2O_SOLVER_PCG G2O_SOLVER_SLAM2D_LINEAR G2O_SOLVER_STRUCTURE_ONLY G2O_SOLVER_EIGEN 
             G2O_TYPES_DATA G2O_TYPES_ICP G2O_TYPES_SBA G2O_TYPES_SCLAM2D G2O_TYPES_SIM3 G2O_TYPES_SLAM2D
             G2O_TYPES_SLAM3D G2O_OPENGL_HELPER)

SET(G2O_LIBS "")
FOREACH(_item ${G2O_LIBS_ITEM})
    # message(${${_item}_DEBUG})
    LIST(APPEND G2O_LIBS 
        debug ${${_item}_DEBUG}
        optimized ${${_item}})
ENDFOREACH()

# Use the Widgets module from Qt 5.
target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::Charts ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${FLANN_LIBRARIES} 
                                      ${VTK_LIBRARIES} ${CUDA_blas_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY}
                                      ${G2O_LIBS} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})

# Install the executable
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

