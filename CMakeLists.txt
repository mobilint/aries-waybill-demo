cmake_minimum_required(VERSION 3.16..3.22)

project(waybill_demo_kr_cpp LANGUAGES CXX)

# NOTE: MSVC expects the local codepage by default. Keep Korean strings and
# paths consistently interpreted as UTF-8 on Windows.
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

set(QBRUNTIME_PATH "" CACHE PATH "Path to qbruntime package root (contains include/, lib/, and on Windows bin/)")

if(NOT QBRUNTIME_PATH)
  set(QBRUNTIME_DEFAULT_CANDIDATES)
  if(DEFINED ENV{QBRUNTIME_PATH})
    list(APPEND QBRUNTIME_DEFAULT_CANDIDATES "$ENV{QBRUNTIME_PATH}")
  endif()
  if(DEFINED ENV{USERPROFILE})
    list(APPEND QBRUNTIME_DEFAULT_CANDIDATES "$ENV{USERPROFILE}/qbruntime")
  endif()
  if(DEFINED ENV{HOME})
    list(APPEND QBRUNTIME_DEFAULT_CANDIDATES "$ENV{HOME}/qbruntime")
  endif()

  foreach(QBRUNTIME_DEFAULT_CANDIDATE IN LISTS QBRUNTIME_DEFAULT_CANDIDATES)
    if(EXISTS "${QBRUNTIME_DEFAULT_CANDIDATE}")
      set(QBRUNTIME_PATH "${QBRUNTIME_DEFAULT_CANDIDATE}" CACHE PATH
        "Path to qbruntime package root (contains include/, lib/, and on Windows bin/)"
        FORCE
      )
      message(STATUS "Using default qbruntime path: ${QBRUNTIME_PATH}")
      break()
    endif()
  endforeach()
endif()

add_library(qbruntime SHARED IMPORTED GLOBAL)

if(WIN32)
  if(NOT QBRUNTIME_PATH)
    message(FATAL_ERROR
      "On Windows, qbruntime package root is required. "
      "Set -DQBRUNTIME_PATH=<path>, QBRUNTIME_PATH environment variable, "
      "or place qbruntime under your home folder as ~/qbruntime."
    )
  endif()

  get_filename_component(QBRUNTIME_ABS_PATH
    "${QBRUNTIME_PATH}"
    ABSOLUTE
    BASE_DIR "${CMAKE_BINARY_DIR}"
  )

  set_target_properties(qbruntime PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_IMPLIB_DEBUG "${QBRUNTIME_ABS_PATH}/lib/qbruntimed.lib"
    IMPORTED_LOCATION_DEBUG "${QBRUNTIME_ABS_PATH}/bin/qbruntimed.dll"
    IMPORTED_IMPLIB_RELEASE "${QBRUNTIME_ABS_PATH}/lib/qbruntime.lib"
    IMPORTED_LOCATION_RELEASE "${QBRUNTIME_ABS_PATH}/bin/qbruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${QBRUNTIME_ABS_PATH}/include"
  )
else()
  if(QBRUNTIME_PATH)
    get_filename_component(QBRUNTIME_ABS_PATH
      "${QBRUNTIME_PATH}"
      ABSOLUTE
      BASE_DIR "${CMAKE_BINARY_DIR}"
    )
    set(QBRUNTIME_INCLUDE_DIR "${QBRUNTIME_ABS_PATH}/include")
    set(QBRUNTIME_LIBRARY "${QBRUNTIME_ABS_PATH}/lib/libqbruntime.so")
  else()
    find_path(QBRUNTIME_INCLUDE_DIR
      NAMES qbruntime/qbruntime.h
    )
    find_library(QBRUNTIME_LIBRARY
      NAMES qbruntime
    )
  endif()

  if(NOT QBRUNTIME_INCLUDE_DIR OR NOT QBRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "qbruntime not found. Install mobilint-qb-runtime (e.g. sudo apt install mobilint-qb-runtime) "
      "or set -DQBRUNTIME_PATH=<path>."
    )
  endif()

  set_target_properties(qbruntime PROPERTIES
    IMPORTED_LOCATION "${QBRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${QBRUNTIME_INCLUDE_DIR}"
  )
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(OpenMP REQUIRED)

add_executable(waybill_demo_kr_cpp
  src/main.cc
  src/mxq_model.cc
  src/ocr_runtime.cc
  src/yolo_runtime.cc
)

target_include_directories(waybill_demo_kr_cpp PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
)

target_compile_features(waybill_demo_kr_cpp PRIVATE cxx_std_17)
set_target_properties(waybill_demo_kr_cpp PROPERTIES
  CXX_EXTENSIONS OFF
)


target_link_libraries(waybill_demo_kr_cpp PRIVATE
  qbruntime
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

if(MSVC)
  set_target_properties(waybill_demo_kr_cpp PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )

  # OpenCV's FFmpeg backend DLL is not always captured by TARGET_RUNTIME_DLLS.
  # Try common OpenCV Windows layouts and copy it explicitly when found.
  set(OPENCV_FFMPEG_HINT_DIRS
    "${OpenCV_DIR}"
    "${OpenCV_DIR}/bin"
    "${OpenCV_DIR}/x64/vc17/bin"
    "${OpenCV_DIR}/x64/vc16/bin"
    "${OpenCV_DIR}/x64/vc15/bin"
    "${OpenCV_DIR}/../.."
    "${OpenCV_DIR}/.."
  )

  find_file(OPENCV_FFMPEG_DLL
    NAMES opencv_videoio_ffmpeg*.dll
    HINTS ${OPENCV_FFMPEG_HINT_DIRS}
    PATH_SUFFIXES
      x64/vc17/bin
      x64/vc16/bin
      x64/vc15/bin
      bin
  )

  if(NOT OPENCV_FFMPEG_DLL)
    foreach(OPENCV_FFMPEG_HINT_DIR IN LISTS OPENCV_FFMPEG_HINT_DIRS)
      file(GLOB OPENCV_FFMPEG_DLL_CANDIDATES
        "${OPENCV_FFMPEG_HINT_DIR}/opencv_videoio_ffmpeg*.dll"
      )
      if(OPENCV_FFMPEG_DLL_CANDIDATES)
        list(GET OPENCV_FFMPEG_DLL_CANDIDATES 0 OPENCV_FFMPEG_DLL)
        break()
      endif()
    endforeach()
  endif()

  add_custom_command(TARGET waybill_demo_kr_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:waybill_demo_kr_cpp>
      $<TARGET_FILE_DIR:waybill_demo_kr_cpp>
    COMMAND_EXPAND_LISTS
  )

  if(OPENCV_FFMPEG_DLL)
    add_custom_command(TARGET waybill_demo_kr_cpp POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_FFMPEG_DLL}"
        "$<TARGET_FILE_DIR:waybill_demo_kr_cpp>"
    )
  else()
    message(WARNING
      "OpenCV FFmpeg backend DLL (opencv_videoio_ffmpeg*.dll) was not found. "
      "MP4 VideoCapture with CAP_FFMPEG may fail on Windows. "
      "Checked OpenCV_DIR='${OpenCV_DIR}'."
    )
  endif()
endif()
