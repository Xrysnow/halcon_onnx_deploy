cmake_minimum_required(VERSION 3.10)

set(lib_name onnx_deploy)
set(target_name ${lib_name})

project(${lib_name})
set(CMAKE_CXX_STANDARD 17)

set(halcon_root "D:/Programs/MVTec/HALCON-21.05-Progress")
set(halcon_lib_dir ${halcon_root}/x64-win64/lib)
set(onnx_include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(onnx_lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/lib)

if(NOT EXISTS ${halcon_root})
  message(FATAL_ERROR, "Please set correct halcon path")
endif()

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR})

set(src
  ${src_dir}/main.cpp
  ${src_dir}/HalconProcess.h
  ${src_dir}/HalconProcess.cpp
  ${src_dir}/OrtHelper.h
  ${src_dir}/OrtHelper.cpp
  ${src_dir}/OrtModel.h
  ${src_dir}/OrtModel.cpp
)

# add_library(${target_name} STATIC
#   ${src}
# )
add_executable(${target_name}
  ${src}
)

target_include_directories(${target_name} PUBLIC
  ${src_dir}
  ${onnx_include_dir}
  ${halcon_root}/include
)

target_link_libraries(${target_name}
  ${onnx_lib_dir}/onnxruntime.lib
  ${onnx_lib_dir}/onnxruntime_providers_shared.lib
  ${halcon_lib_dir}/halconc.lib
  ${halcon_lib_dir}/halconcpp.lib
)

target_compile_definitions(${target_name} PUBLIC
  ORT_DLL_IMPORT
)

if(MSVC)
    target_compile_definitions(${target_name} PUBLIC
        WIN32
        _WIN32
        _WINDOWS
        NOMINMAX
        UNICODE
        _UNICODE
        _CRT_SECURE_NO_WARNINGS
        _SCL_SECURE_NO_WARNINGS
        SE_ENABLE_INSPECTOR
    )
    target_compile_options(${target_name} PUBLIC /MP)
endif()
