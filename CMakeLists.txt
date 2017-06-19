cmake_minimum_required(VERSION 2.8)

project(GenerateArucoMarkers)

find_package(OpenCV REQUIRED)

add_executable(generate_aruco_markers src/generate_aruco_markers.cpp)
add_executable(detect_aruco_markers src/detect_aruco_markers.cpp)
add_executable(reference_aruco_markers src/detect_aruco_markers_with_reference.cpp)

target_link_libraries(generate_aruco_markers ${OpenCV_LIBS})
target_link_libraries(detect_aruco_markers ${OpenCV_LIBS})
target_link_libraries(reference_aruco_markers ${OpenCV_LIBS})
