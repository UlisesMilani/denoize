cmake_minimum_required(VERSION 3.22.1)
project(denoize_android_sdk LANGUAGES C)

set(DENOIZE_C_LIBRARY
    "${CMAKE_CURRENT_LIST_DIR}/src/main/prebuilt/${ANDROID_ABI}/libdenoize_c.so"
    CACHE FILEPATH "Path to the Rust denoize C ABI shared library")

if(NOT EXISTS "${DENOIZE_C_LIBRARY}")
  message(FATAL_ERROR
    "Missing denoize C ABI library for ${ANDROID_ABI}: ${DENOIZE_C_LIBRARY}. "
    "Build the pinned Rust android target before assembling the AAR.")
endif()

add_library(denoize_c SHARED IMPORTED)
set_target_properties(denoize_c PROPERTIES IMPORTED_LOCATION "${DENOIZE_C_LIBRARY}")

add_library(denoize_jni SHARED src/main/cpp/denoize_jni.c)
target_include_directories(denoize_jni PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../denoize-c/include")
target_compile_features(denoize_jni PRIVATE c_std_11)
target_compile_options(denoize_jni PRIVATE -Wall -Wextra -Werror -Wconversion -Wshadow)
target_link_libraries(denoize_jni PRIVATE denoize_c log)
