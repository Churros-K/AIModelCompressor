#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include "compressor.cuh"

namespace fs = std::filesystem;

void print_usage() {
    std::cout << "AI Model Compressor (.safetensors <-> .kang)" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  kang <command> [options] <input_path> <output_path>" << std::endl;
    std::cout << "\nCommands:" << std::endl;
    std::cout << "  compress      Compress a .safetensors file or a folder of them." << std::endl;
    std::cout << "  decompress    Decompress a .kang file or a folder of them." << std::endl;
    std::cout << "\nOptions for 'compress':" << std::endl;
    std::cout << "  -l, --level   Compression level (1-19, default: 10)." << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  kang compress model.safetensors model.kang" << std::endl;
    std::cout << "  kang compress -l 15 models_folder/ compressed_folder/" << std::endl;
}

const std::string KANG_SIGNATURE = "KANGCOMP";

static bool read_all(const fs::path& path, std::vector<char>& buffer)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    std::streamoff sz = in.tellg();
    if (sz < 0) return false;
    buffer.resize(static_cast<size_t>(sz));
    in.seekg(0, std::ios::beg);
    in.read(buffer.data(), buffer.size());
    return in.good();
}

// 단일 파일 압축 로직 (압축 레벨 인자 추가)
void handle_compression(const fs::path& input_path, const fs::path& output_path, int compression_level) {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Compressing " << input_path.string() << "\n-> to ->    " << output_path.string() << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. .safetensors 파일 읽기
    std::vector<char> file_buffer;
    if (!read_all(input_path, file_buffer)) {
        std::cerr << "Error: Cannot open file " << input_path.string() << std::endl;
        return;
    }

    if (file_buffer.size() < 8) {
        std::cerr << "Error: Invalid safetensors file (too small)." << std::endl;
        return;
    }

    // 2. 헤더와 텐서 데이터 분리 (unaligned 접근 피하기 위해 memcpy 사용)
    uint64_t header_len = 0;
    std::memcpy(&header_len, file_buffer.data(), sizeof(header_len));
    if (file_buffer.size() < 8 + header_len) {
        std::cerr << "Error: Invalid safetensors file (header size mismatch)." << std::endl;
        return;
    }
    std::string json_header(file_buffer.data() + 8, file_buffer.data() + 8 + header_len);
    std::vector<char> tensor_data(file_buffer.begin() + 8 + header_len, file_buffer.end());

    // 3. 압축 실행 (압축 레벨 전달)
    CompressionResult comp_result;
    if (!compress_safetensor(json_header, tensor_data, comp_result, compression_level)) {
        std::cerr << "Compression failed." << std::endl;
        return;
    }

    // 4. .kang 파일 쓰기
    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Cannot create output file " << output_path.string() << std::endl;
        return;
    }

    out_file.write(KANG_SIGNATURE.c_str(), KANG_SIGNATURE.size());
    uint64_t compressed_header_size = static_cast<uint64_t>(comp_result.compressed_header.size());
    out_file.write(reinterpret_cast<const char*>(&compressed_header_size), sizeof(compressed_header_size));
    out_file.write(comp_result.compressed_header.data(), comp_result.compressed_header.size());
    uint64_t num_chunks = static_cast<uint64_t>(comp_result.chunk_info.size());
    out_file.write(reinterpret_cast<const char*>(&num_chunks), sizeof(num_chunks));
    for (const auto& info : comp_result.chunk_info) {
        uint64_t orig = static_cast<uint64_t>(info.first);
        uint64_t comp = static_cast<uint64_t>(info.second);
        out_file.write(reinterpret_cast<const char*>(&orig), sizeof(orig));
        out_file.write(reinterpret_cast<const char*>(&comp), sizeof(comp));
    }
    if (!comp_result.compressed_tensors.empty())
        out_file.write(comp_result.compressed_tensors.data(), comp_result.compressed_tensors.size());
    out_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Compression successful! Took " << diff.count() << " seconds." << std::endl;
}

// 단일 파일 압축 해제 로직
void handle_decompression(const fs::path& input_path, const fs::path& output_path) {
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Decompressing " << input_path.string() << "\n-> to ->      " << output_path.string() << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::ifstream in_file(input_path, std::ios::binary);
    if (!in_file) {
        std::cerr << "Error: Cannot open input file " << input_path.string() << std::endl;
        return;
    }

    std::vector<char> signature_buf(KANG_SIGNATURE.size());
    in_file.read(signature_buf.data(), signature_buf.size());
    if (!in_file.good() || std::string(signature_buf.begin(), signature_buf.end()) != KANG_SIGNATURE) {
        std::cerr << "Error: Not a valid .kang file (invalid signature)." << std::endl;
        return;
    }

    auto read_u64 = [&](uint64_t& v) {
        in_file.read(reinterpret_cast<char*>(&v), sizeof(v));
        return in_file.good();
    };

    uint64_t compressed_header_size = 0;
    if (!read_u64(compressed_header_size)) { std::cerr << "Error reading header size." << std::endl; return; }
    std::vector<char> compressed_header(compressed_header_size);
    in_file.read(compressed_header.data(), compressed_header.size());
    if (!in_file.good()) { std::cerr << "Error reading compressed header." << std::endl; return; }

    uint64_t num_chunks_u64 = 0;
    if (!read_u64(num_chunks_u64)) { std::cerr << "Error reading num chunks." << std::endl; return; }

    std::vector<std::pair<size_t, size_t>> chunk_info;
    chunk_info.reserve(static_cast<size_t>(num_chunks_u64));
    for (uint64_t i = 0; i < num_chunks_u64; ++i) {
        uint64_t orig = 0, comp = 0;
        if (!read_u64(orig) || !read_u64(comp)) { std::cerr << "Error reading chunk info." << std::endl; return; }
        chunk_info.emplace_back(static_cast<size_t>(orig), static_cast<size_t>(comp));
    }

    // 남은 바이트 전체를 한 번에 읽기
    std::vector<char> compressed_tensors;
    {
        std::streampos cur = in_file.tellg();
        in_file.seekg(0, std::ios::end);
        std::streampos end = in_file.tellg();
        if (cur < 0 || end < cur) {
            std::cerr << "Error: Invalid file offsets." << std::endl; return;
        }
        size_t remain = static_cast<size_t>(end - cur);
        compressed_tensors.resize(remain);
        in_file.seekg(cur, std::ios::beg);
        in_file.read(compressed_tensors.data(), remain);
        if (!in_file.good()) { std::cerr << "Error reading tensor payload." << std::endl; return; }
    }
    in_file.close();

    std::string json_header;
    std::vector<char> tensor_data;
    if (!decompress_kang(compressed_header, compressed_tensors, chunk_info, json_header, tensor_data)) {
        std::cerr << "Decompression failed." << std::endl;
        return;
    }

    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) { std::cerr << "Error: Cannot create output file." << std::endl; return; }
    uint64_t header_len = static_cast<uint64_t>(json_header.size());
    out_file.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out_file.write(json_header.data(), json_header.size());
    if (!tensor_data.empty()) out_file.write(tensor_data.data(), tensor_data.size());
    out_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Decompression successful! Took " << diff.count() << " seconds." << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { // kang <command> <input> [<output>]
        print_usage();
        return 1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::string command;
    fs::path input_path;
    fs::path output_path;
    int compression_level = 10; // 기본 압축 레벨

    command = args[0];
    
    size_t path_arg_index = 1;
    if (command == "compress" && args.size() > 3) {
        if (args[1] == "-l" || args[1] == "--level") {
            if (args.size() < 5) { // kang compress -l 15 <input> <output>
                print_usage();
                return 1;
            }
            try {
                compression_level = std::stoi(args[2]);
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid compression level." << std::endl;
                print_usage();
                return 1;
            }
            path_arg_index = 3;
        }
    }

    if (args.size() < path_arg_index + 2) {
        print_usage();
        return 1;
    }
    input_path = args[path_arg_index];
    output_path = args[path_arg_index + 1];

    try {
        if (fs::is_directory(input_path)) {
            if (!fs::exists(output_path)) {
                std::cout << "Output directory does not exist. Creating: " << output_path.string() << std::endl;
                fs::create_directories(output_path);
            }

            int count = 0;
            if (command == "compress") {
                std::cout << "Starting batch compression from: " << input_path.string() << std::endl;
                for (const auto& entry : fs::directory_iterator(input_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                        fs::path out_file = output_path / entry.path().filename().replace_extension(".kang");
                        handle_compression(entry.path(), out_file, compression_level);
                        count++;
                    }
                }
            }
            else if (command == "decompress") {
                std::cout << "Starting batch decompression from: " << input_path.string() << std::endl;
                for (const auto& entry : fs::directory_iterator(input_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".kang") {
                        fs::path out_file = output_path / entry.path().filename().replace_extension(".safetensors");
                        handle_decompression(entry.path(), out_file);
                        count++;
                    }
                }
            }
            else {
                print_usage();
                return 1;
            }
            std::cout << "\nBatch processing finished. Total " << count << " files processed." << std::endl;

        }
        else if (fs::is_regular_file(input_path)) {
            if (command == "compress") {
                handle_compression(input_path, output_path, compression_level);
            }
            else if (command == "decompress") {
                handle_decompression(input_path, output_path);
            }
            else {
                print_usage();
                return 1;
            }
        }
        else {
            std::cerr << "Error: Input path is not a valid file or directory: " << input_path.string() << std::endl;
            return 1;
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

