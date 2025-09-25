#ifndef COMPRESSOR_CUH
#define COMPRESSOR_CUH

#include <vector>
#include <string>

// 압축 결과를 담을 구조체
struct CompressionResult {
    std::vector<char> compressed_header;
    std::vector<char> compressed_tensors;
    std::vector<std::pair<size_t, size_t>> chunk_info; // <original_size, compressed_size>
};

// 압축 함수 인터페이스
bool compress_safetensor(
    const std::string& json_header,
    const std::vector<char>& tensor_data,
    CompressionResult& result,
    int compression_level = 10 // Zstd 압축 레벨 (높을수록 압축률 증가)
);

// 해제 함수 인터페이스
bool decompress_kang(
    const std::vector<char>& compressed_header,
    const std::vector<char>& compressed_tensors,
    const std::vector<std::pair<size_t, size_t>>& chunk_info,
    std::string& json_header,
    std::vector<char>& tensor_data
);


#endif //COMPRESSOR_CUH

