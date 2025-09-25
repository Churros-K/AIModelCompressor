#include "compressor.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <nvcomp/zstd.hpp>

// CUDA 에러 체크 헬퍼 함수
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        throw std::runtime_error(cudaGetErrorString(e)); \
    } \
}

bool compress_safetensor(
    const std::string& json_header,
    const std::vector<char>& tensor_data,
    CompressionResult& result,
    int compression_level)
{

    try {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        { // 매니저 수명 관리를 위한 새 스코프
            const size_t internal_uncomp_chunk = 64 * 1024; // 64KB 권장
            nvcomp::ZstdManager manager(
                internal_uncomp_chunk,
                nvcompBatchedZstdCompressDefaultOpts,
                nvcompBatchedZstdDecompressDefaultOpts,
                stream);

            // 1) JSON 헤더 압축 (빈 헤더는 건너뜀)
            result.compressed_header.clear();
            if (!json_header.empty()) {
                void* d_uncompressed_header = nullptr;
                CUDA_CHECK(cudaMalloc(&d_uncompressed_header, json_header.size()));
                CUDA_CHECK(cudaMemcpyAsync(d_uncompressed_header, json_header.data(), json_header.size(), cudaMemcpyHostToDevice, stream));

                auto header_comp_config = manager.configure_compression(json_header.size());

                void* d_compressed_header = nullptr;
                CUDA_CHECK(cudaMalloc(&d_compressed_header, header_comp_config.max_compressed_buffer_size));

                manager.compress(
                    reinterpret_cast<const uint8_t*>(d_uncompressed_header),
                    reinterpret_cast<uint8_t*>(d_compressed_header),
                    header_comp_config);

                // 압축 완료 보장 후 크기 조회
                CUDA_CHECK(cudaStreamSynchronize(stream));

                size_t actual_header_comp_size =
                    manager.get_compressed_output_size(reinterpret_cast<const uint8_t*>(d_compressed_header));
                result.compressed_header.resize(actual_header_comp_size);

                // 동기 복사(추가 동기화 불필요)
                CUDA_CHECK(cudaMemcpy(result.compressed_header.data(),
                                      d_compressed_header,
                                      actual_header_comp_size,
                                      cudaMemcpyDeviceToHost));

                CUDA_CHECK(cudaFree(d_uncompressed_header));
                CUDA_CHECK(cudaFree(d_compressed_header));

                std::cout << "JSON header compressed (GPU): " << json_header.size()
                          << " -> " << actual_header_comp_size << " bytes" << std::endl;
            }

            // 2) 텐서 데이터 GPU 압축 (빈 입력은 건너뜀)
            result.compressed_tensors.clear();
            result.chunk_info.clear();

            if (!tensor_data.empty()) {
                const size_t chunk_size = 1024ULL * 1024ULL * 64ULL; // 64MB
                size_t num_chunks = (tensor_data.size() + chunk_size - 1) / chunk_size;
                std::cout << "Starting tensor compression with " << num_chunks
                          << " chunks on GPU using nvCOMP 5.0..." << std::endl;

                const size_t max_input_chunk = std::min(chunk_size, tensor_data.size());

                // 디바이스 버퍼를 반복 사용(과대할당 방지)
                void* d_uncompressed_chunk = nullptr;
                CUDA_CHECK(cudaMalloc(&d_uncompressed_chunk, max_input_chunk));

                auto comp_config_template = manager.configure_compression(max_input_chunk);
                void* d_compressed_chunk = nullptr;
                CUDA_CHECK(cudaMalloc(&d_compressed_chunk, comp_config_template.max_compressed_buffer_size));

                // 호스트 임시 버퍼(페이지드). 필요 시 cudaHostAlloc으로 변경 가능
                std::vector<char> host_comp_buf(comp_config_template.max_compressed_buffer_size);

                const char* current_tensor_ptr = tensor_data.data();
                size_t total_compressed_size = 0;

                for (size_t i = 0; i < num_chunks; ++i) {
                    const size_t current_chunk_size =
                        (i == num_chunks - 1)
                            ? (tensor_data.size() - i * chunk_size)
                            : chunk_size;

                    CUDA_CHECK(cudaMemcpyAsync(d_uncompressed_chunk,
                                               current_tensor_ptr,
                                               current_chunk_size,
                                               cudaMemcpyHostToDevice,
                                               stream));

                    // 청크별 압축 설정
                    auto comp_config = manager.configure_compression(current_chunk_size);

                    // 압축 실행
                    manager.compress(
                        reinterpret_cast<const uint8_t*>(d_uncompressed_chunk),
                        reinterpret_cast<uint8_t*>(d_compressed_chunk),
                        comp_config);

                    // 압축 완료 후 실제 크기 조회
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    const size_t actual_comp_size =
                        manager.get_compressed_output_size(reinterpret_cast<const uint8_t*>(d_compressed_chunk));

                    // 동기 복사로 호스트에 수신
                    CUDA_CHECK(cudaMemcpy(host_comp_buf.data(),
                                          d_compressed_chunk,
                                          actual_comp_size,
                                          cudaMemcpyDeviceToHost));

                    // 결과 누적
                    result.compressed_tensors.insert(result.compressed_tensors.end(),
                                                     host_comp_buf.begin(),
                                                     host_comp_buf.begin() + static_cast<std::ptrdiff_t>(actual_comp_size));
                    result.chunk_info.push_back({ current_chunk_size, actual_comp_size });
                    total_compressed_size += actual_comp_size;

                    current_tensor_ptr += current_chunk_size;
                }

                CUDA_CHECK(cudaFree(d_uncompressed_chunk));
                CUDA_CHECK(cudaFree(d_compressed_chunk));

                std::cout << "Tensor data compressed (GPU): " << tensor_data.size()
                          << " -> " << total_compressed_size << " bytes" << std::endl;
            }
        } // 스트림 파괴 전에 매니저가 먼저 소멸됨

        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred during GPU compression: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool decompress_kang(
    const std::vector<char>& compressed_header,
    const std::vector<char>& compressed_tensors,
    const std::vector<std::pair<size_t, size_t>>& chunk_info,
    std::string& json_header,
    std::vector<char>& tensor_data)
{
    try {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        { // 매니저 관리를 위한 새 스코프
            const size_t internal_uncomp_chunk = 64 * 1024;
            nvcomp::ZstdManager manager(
                internal_uncomp_chunk,
                nvcompBatchedZstdCompressDefaultOpts,
                nvcompBatchedZstdDecompressDefaultOpts,
                stream);

            // 1) JSON 헤더 해제 (빈 헤더는 건너뜀)
            json_header.clear();
            if (!compressed_header.empty()) {
                void* d_compressed_header = nullptr;
                CUDA_CHECK(cudaMalloc(&d_compressed_header, compressed_header.size()));
                CUDA_CHECK(cudaMemcpyAsync(d_compressed_header, compressed_header.data(), compressed_header.size(), cudaMemcpyHostToDevice, stream));

                auto header_decomp_config =
                    manager.configure_decompression(reinterpret_cast<const uint8_t*>(d_compressed_header));

                void* d_decompressed_header = nullptr;
                CUDA_CHECK(cudaMalloc(&d_decompressed_header, header_decomp_config.decomp_data_size));

                manager.decompress(
                    reinterpret_cast<uint8_t*>(d_decompressed_header),
                    reinterpret_cast<const uint8_t*>(d_compressed_header),
                    header_decomp_config);

                // 해제 완료 보장
                CUDA_CHECK(cudaStreamSynchronize(stream));

                json_header.resize(header_decomp_config.decomp_data_size);
                // 동기 복사(추가 동기화 불필요)
                CUDA_CHECK(cudaMemcpy(&json_header[0],
                                      d_decompressed_header,
                                      header_decomp_config.decomp_data_size,
                                      cudaMemcpyDeviceToHost));

                CUDA_CHECK(cudaFree(d_compressed_header));
                CUDA_CHECK(cudaFree(d_decompressed_header));
            }

            // 2) 텐서 데이터 해제 (청크가 없으면 건너뜀)
            if (chunk_info.empty()) {
                tensor_data.clear();
            } else {
                size_t total_decompressed_size = 0;
                size_t max_original_size = 0;
                size_t max_compressed_size = 0;
                for (const auto& info : chunk_info) {
                    total_decompressed_size += info.first;
                    if (info.first > max_original_size) max_original_size = info.first;
                    if (info.second > max_compressed_size) max_compressed_size = info.second;
                }
                tensor_data.resize(total_decompressed_size);

                std::cout << "Starting tensor decompression for " << chunk_info.size()
                          << " chunks on GPU using nvCOMP 5.0..." << std::endl;

                const char* current_compressed_ptr = compressed_tensors.data();
                char* current_decompressed_ptr = tensor_data.data();

                // 디바이스 버퍼 재사용(0 크기 방지)
                void* d_compressed_chunk = nullptr;
                CUDA_CHECK(cudaMalloc(&d_compressed_chunk, max_compressed_size));
                void* d_decompressed_chunk = nullptr;
                CUDA_CHECK(cudaMalloc(&d_decompressed_chunk, max_original_size));

                for (const auto& info : chunk_info) {
                    const size_t original_size = info.first;
                    const size_t compressed_size = info.second;

                    CUDA_CHECK(cudaMemcpyAsync(d_compressed_chunk,
                                               current_compressed_ptr,
                                               compressed_size,
                                               cudaMemcpyHostToDevice,
                                               stream));

                    // 해제 설정(디바이스에서 헤더 읽음)
                    auto decomp_config =
                        manager.configure_decompression(reinterpret_cast<const uint8_t*>(d_compressed_chunk));

                    // 검증: 예상 해제 크기 확인
                    if (decomp_config.decomp_data_size != original_size) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        CUDA_CHECK(cudaFree(d_compressed_chunk));
                        CUDA_CHECK(cudaFree(d_decompressed_chunk));
                        throw std::runtime_error("Decompressed size mismatch for chunk.");
                    }

                    manager.decompress(
                        reinterpret_cast<uint8_t*>(d_decompressed_chunk),
                        reinterpret_cast<const uint8_t*>(d_compressed_chunk),
                        decomp_config);

                    // 해제 완료 보장
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    // 동기 복사로 호스트에 수신
                    CUDA_CHECK(cudaMemcpy(current_decompressed_ptr,
                                          d_decompressed_chunk,
                                          original_size,
                                          cudaMemcpyDeviceToHost));

                    current_compressed_ptr += compressed_size;
                    current_decompressed_ptr += original_size;
                }

                CUDA_CHECK(cudaFree(d_compressed_chunk));
                CUDA_CHECK(cudaFree(d_decompressed_chunk));
            }
        } // 스트림 파괴 전에 매니저가 먼저 소멸

        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred during GPU decompression: " << e.what() << std::endl;
        return false;
    }
    return true;
}


