set(CMAKE_HIP_COMPILER
    "amdclang++"
    CACHE STRING "desired c++ compiler" FORCE)
if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Using ${CMAKE_HIP_COMPILER} to build for amdgpu backend")
else()
    message(
        FATAL_ERROR
            "'amdclang++' compiler required to compile test binaries for ROCm platform.")
endif()

set(VCOPY_SOURCES ${CMAKE_SOURCE_DIR}/sample/vcopy.cpp)
set_source_files_properties(${VCOPY_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vcopy ${VCOPY_SOURCES})
set_target_properties(vcopy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

set(KOKKOS_MAKEFILE_DIR ${CMAKE_SOURCE_DIR}/sample/kokkos-no-uvm/)
set(KOKKOS_EXEC ${CMAKE_SOURCE_DIR}/sample/kokkos-no-uvm/lulesh.kk)
add_custom_target(
    run_make ALL
    COMMAND ${CMAKE_COMMAND} -E env KOKKOS_USE_DEPRECATED_MAKEFILES=1
            ${CMAKE_MAKE_PROGRAM} -C ${KOKKOS_MAKEFILE_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${KOKKOS_EXEC} ${CMAKE_SOURCE_DIR}/tests
    COMMENT
        "Running Kokkos Makefile in ${KOKKOS_MAKEFILE_DIR} with KOKKOS_USE_DEPRECATED_MAKEFILES=1 set and copying lulesh.kk to ${CMAKE_SOURCE_DIR}/tests."
    )