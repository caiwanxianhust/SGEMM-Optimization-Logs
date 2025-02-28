set (sample_sgemm_naive_files
    sgemm_naive.cu
)

set (sample_sgemm_sharedmem_tile_files
    sgemm_sharedmem_tile.cu
)

set (sample_sgemm_1D_tile_files
    sgemm_1D_tile.cu
)

set (sample_sgemm_2D_tile_files
    sgemm_2D_tile.cu
)

set (sample_sgemm_vector_load_files
    sgemm_vector_load.cu
)

set (sample_sgemm_warp_tile_files
    sgemm_warp_tile.cu
)

set (sample_sgemm_double_buffer_files
    sgemm_double_buffer.cu
)

set (sample_sgemm_tensor_core_files
    sgemm_tensorcore.cu
)

set (sample_sgemm_cublas_files
    sgemm_cublas.cu
)

set (SAMPLES_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/SGEMM

)

include_directories(
    ${SAMPLES_HEADER_DIR}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/SGEMM/lib
)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

add_executable(sgemm_naive ${sample_sgemm_naive_files})
target_compile_features(sgemm_naive PUBLIC cxx_std_14)
target_link_libraries(sgemm_naive PUBLIC sgemm)

add_executable(sgemm_sharedmem_tile ${sample_sgemm_sharedmem_tile_files})
target_compile_features(sgemm_sharedmem_tile PUBLIC cxx_std_14)
target_link_libraries(sgemm_sharedmem_tile PUBLIC sgemm)

add_executable(sgemm_1D_tile ${sample_sgemm_1D_tile_files})
target_compile_features(sgemm_1D_tile PUBLIC cxx_std_14)
target_link_libraries(sgemm_1D_tile PUBLIC sgemm)

add_executable(sgemm_2D_tile ${sample_sgemm_2D_tile_files})
target_compile_features(sgemm_2D_tile PUBLIC cxx_std_14)
target_link_libraries(sgemm_2D_tile PUBLIC sgemm)

add_executable(sgemm_vector_load ${sample_sgemm_vector_load_files})
target_compile_features(sgemm_vector_load PUBLIC cxx_std_14)
target_link_libraries(sgemm_vector_load PUBLIC sgemm)

add_executable(sgemm_warp_tile ${sample_sgemm_warp_tile_files})
target_compile_features(sgemm_warp_tile PUBLIC cxx_std_14)
target_link_libraries(sgemm_warp_tile PUBLIC sgemm)

add_executable(sgemm_double_buffer ${sample_sgemm_double_buffer_files})
target_compile_features(sgemm_double_buffer PUBLIC cxx_std_14)
target_link_libraries(sgemm_double_buffer PUBLIC sgemm)

add_executable(sgemm_tensor_core ${sample_sgemm_tensor_core_files})
target_compile_features(sgemm_tensor_core PUBLIC cxx_std_14)
target_link_libraries(sgemm_tensor_core PUBLIC sgemm)

add_executable(sgemm_cublas ${sample_sgemm_cublas_files})
target_compile_features(sgemm_cublas PUBLIC cxx_std_14)
target_link_libraries(sgemm_cublas PUBLIC sgemm)