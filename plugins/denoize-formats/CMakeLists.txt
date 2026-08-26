cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED DENOIZE_PLUGIN_VERSION)
  message(FATAL_ERROR "DENOIZE_PLUGIN_VERSION is required")
endif()

project(denoize_formats VERSION "${DENOIZE_PLUGIN_VERSION}" LANGUAGES C CXX)

# clap-wrapper adds Objective-C and Objective-C++ sources on Apple platforms.
# Enable those languages before adding the pinned wrapper subdirectory so both
# Intel and Apple Silicon generators have complete compile rules.
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

# clap-wrapper uses C++17 library facilities such as std::filesystem.  Do not
# rely on generator defaults: Xcode 15 otherwise selects C++98, while current
# Visual Studio rejects the filesystem aliases compiled without /std:c++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Every wrapper/support archive is linked into a loadable plug-in bundle.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

foreach(required_path
    CLAP_WRAPPER_ROOT
    CLAP_SDK_ROOT
    VST3_SDK_ROOT
    DENOIZE_CLAP_STATICLIB)
  if(NOT DEFINED ${required_path} OR NOT EXISTS "${${required_path}}")
    message(FATAL_ERROR "${required_path} must name an existing pinned dependency")
  endif()
endforeach()
if(NOT DEFINED DENOIZE_RUST_NATIVE_LIBS OR DENOIZE_RUST_NATIVE_LIBS STREQUAL "")
  message(FATAL_ERROR "DENOIZE_RUST_NATIVE_LIBS is required")
endif()

set(CLAP_WRAPPER_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_AAX OFF CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_AUV2 OFF CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_AUV3 OFF CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_STANDALONE OFF CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_OUTPUT_NAME "denoize" CACHE STRING "" FORCE)
set(CLAP_WRAPPER_BUNDLE_IDENTIFIER "org.penguin425.denoize" CACHE STRING "" FORCE)
set(CLAP_WRAPPER_BUNDLE_VERSION "${DENOIZE_PLUGIN_VERSION}" CACHE STRING "" FORCE)

add_subdirectory("${CLAP_WRAPPER_ROOT}" clap-wrapper EXCLUDE_FROM_ALL)

add_library(denoize_clap_rust STATIC IMPORTED GLOBAL)
set_target_properties(denoize_clap_rust PROPERTIES
  IMPORTED_LOCATION "${DENOIZE_CLAP_STATICLIB}")

set(denoize_rust_native_libs ${DENOIZE_RUST_NATIVE_LIBS})
set(denoize_rust_native_link_options)
if(MSVC AND "/defaultlib:msvcrt" IN_LIST denoize_rust_native_libs)
  # rustc reports the MSVC CRT directive beside native .lib dependencies.
  # Passing it through INTERFACE_LINK_LIBRARIES makes CMake normalize the
  # leading slash into a path and MSBuild then searches for
  # `\defaultlib:msvcrt.obj`.  Preserve it as a linker option instead.
  list(REMOVE_ITEM denoize_rust_native_libs "/defaultlib:msvcrt")
  list(APPEND denoize_rust_native_link_options "/defaultlib:msvcrt")
endif()
if(UNIX AND NOT APPLE AND ("-lopus" IN_LIST denoize_rust_native_libs
    OR "opus" IN_LIST denoize_rust_native_libs))
  # Accept the runtime SONAME so developer machines do not need an otherwise
  # unused unversioned linker symlink.
  find_library(DENOIZE_OPUS_LIBRARY NAMES opus libopus.so.0 REQUIRED)
  list(REMOVE_ITEM denoize_rust_native_libs "-lopus" "opus")
  list(PREPEND denoize_rust_native_libs "${DENOIZE_OPUS_LIBRARY}")
endif()
set_property(TARGET denoize_clap_rust PROPERTY INTERFACE_LINK_LIBRARIES
  "${denoize_rust_native_libs}")
if(denoize_rust_native_link_options)
  set_property(TARGET denoize_clap_rust PROPERTY INTERFACE_LINK_OPTIONS
    "${denoize_rust_native_link_options}")
endif()

# The wrapper must resolve the exact Rust CLAP entry linked into the bundle. It
# must never scan user or system CLAP paths for a same-named binary.
target_compile_definitions(clap-wrapper-shared-detail
  PRIVATE STATICALLY_LINKED_CLAP_ENTRY=1)
# clap-wrapper-shared-detail owns the reference to `clap_entry`; expressing the
# imported archive as its public dependency preserves static link order.
target_link_libraries(clap-wrapper-shared-detail PUBLIC denoize_clap_rust)

add_library(denoize_vst3 MODULE)
target_add_vst3_wrapper(
  TARGET denoize_vst3
  OUTPUT_NAME "denoize"
  BUNDLE_IDENTIFIER "org.penguin425.denoize.vst3"
  BUNDLE_VERSION "${DENOIZE_PLUGIN_VERSION}")
target_compile_definitions(denoize_vst3-clap-wrapper-vst3-lib
  PRIVATE STATICALLY_LINKED_CLAP_ENTRY=1)
if(MSVC)
  # Windows headers define min/max macros unless explicitly suppressed.  They
  # corrupt std::numeric_limits::max() in the pinned VST3 SDK otherwise.
  target_compile_definitions(denoize_vst3-clap-wrapper-vst3-lib
    PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(UNIX AND NOT APPLE)
  target_link_options(denoize_vst3 PRIVATE
    "-Wl,-z,noexecstack" "-Wl,-z,relro" "-Wl,-z,now")
endif()

add_custom_target(denoize-formats ALL DEPENDS denoize_vst3)
