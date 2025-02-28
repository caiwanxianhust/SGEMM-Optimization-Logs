set(sgemm_files
    sgemm.cu
)

set (SGEMM_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/SGEMM
    
)

include_directories(
    ${SGEMM_HEADER_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/SGEMM/lib)

message("-- Assign sgemm include directories (include_directories=${SGEMM_HEADER_DIR})")
message("-- in sgemm cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

add_library(sgemm SHARED ${sgemm_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(sgemm PUBLIC cxx_std_14)
target_compile_options(sgemm PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sgemm PUBLIC -lcudart)

