cmake_minimum_required(VERSION 3.0)
project(RecFLS)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# Instruct CMake to create code from Qt designer ui files
set(CMAKE_AUTOUIC ON)

add_definitions(-DSTB_IMAGE_IMPLEMENTATION)

# Find the QtWidgets library
#find_package(Boost REQUIRED QUIET)
find_package(Qt5 COMPONENTS Widgets Charts CONFIG REQUIRED)
find_package(OpenCV REQUIRED QUIET)
find_package(PCL REQUIRED QUIET COMPONENTS common features io kdtree search surface filters registration sample_consensus segmentation visualization)
find_package(OpenMP REQUIRED QUIET)
find_package(FLANN REQUIRED QUIET)
find_package(VTK REQUIRED QUIET)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED QUIET)

set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ./src
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ./third/stb_image
    ./third/uplinksimple
)

set(BASE_SRCS
    src/main.cpp
)

set(COMMON_HEADERS
    src/common/Parameters.h
    src/common/Frame.h
    src/common/ImuFrame.h
)

set(COMMON_SRCS
    src/common/Parameters.cpp
    src/common/Frame.cpp
    src/common/ImuFrame.cpp
)

set(CONTROLLER_HEADERS
    src/controller/Controller.h
    src/controller/FrameStepController.h
)

set(CONTROLLER_SRCS
    src/controller/Controller.cpp
    src/controller/FrameStepController.cpp
)

set(DEVICE_HEADERS
    src/device/Device.h
    src/device/SensorReaderDevice.h
)

set(DEVICE_SRCS
    src/device/Device.cpp
    src/device/SensorReaderDevice.cpp
)

set(EXTRACTOR_HEADERS
    src/extractor/EDLine3DExtractor.h
)

set(EXTRACTOR_SRCS
    src/extractor/EDLine3DExtractor.hpp
)

set(UI_HEADERS
    src/ui/MainWindow.h
    src/ui/ImageViewer.h
    src/ui/CloudViewer.h
    src/ui/CloudViewerCellPicker.h
    src/ui/vtkImageMatSource.h
)

set(UI_SRCS
    src/ui/MainWindow.ui
    src/ui/MainWindow.cpp
    src/ui/ImageViewer.cpp
    src/ui/CloudViewer.cpp
    src/ui/CloudViewerCellPicker.cpp
    src/ui/vtkImageMatSource.cpp
)

set(UTIL_HEADERS
    src/util/Utils.h
)

set(UTIL_SRCS
    src/util/Utils.cpp
)

set(HEADER_FILES
    ${HEADER_FILES}
    ${COMMON_HEADERS}
    ${CONTROLLER_HEADERS}
    ${DEVICE_HEADERS}
    ${EXTRACTOR_HEADERS}
    ${UI_HEADERS}
    ${UTIL_HEADERS}
)

set(SRC_FILES
    ${SRC_FILES}
    ${BASE_SRCS}
    ${COMMON_SRCS}
    ${CONTROLLER_SRCS}
    ${DEVICE_SRCS}
    ${EXTRACTOR_SRCS}
    ${UI_SRCS}
    ${UTIL_SRCS}
)

IF(MSVC)
    FILE(GLOB HEADERS ${HEADER_FILES})
    set(SRC_FILES ${SRC_FILES} ${HEADER_FILES})
ENDIF(MSVC)

include_directories(${INCLUDE_DIRS})

# Tell CMake to create the helloworld executable
add_executable(recfls ${SRC_FILES})

# Use the Widgets module from Qt 5.
target_link_libraries(recfls Qt5::Widgets Qt5::Charts ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${FLANN_LIBRARIES} ${VTK_LIBRARIES})

# Install the executable
install(TARGETS recfls DESTINATION bin)
