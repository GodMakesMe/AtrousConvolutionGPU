#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int K = 3;
constexpr int R = K / 2;
__constant__ float d_kernel[K * K];
__constant__ float d_asppWeights[4];

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err__) << std::endl;        \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    } while (0)

struct RunResult {
    float cpuMs = 0.0f;
    float cpuAsppMs = 0.0f;
    float gpuBasicMs = 0.0f;
    float gpuTiledMs = 0.0f;
    float gpuAsppMs = 0.0f;
    float maxAbsDiffBasic = 0.0f;
    float maxAbsDiffTiled = 0.0f;
    float maxAbsDiffAspp = 0.0f;
};

void atrousConvolutionCPU(const std::vector<float> &input,
                          std::vector<float> &output,
                          int width,
                          int height,
                          int dilation,
                          const std::vector<float> &kernel) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = -R; ky <= R; ++ky) {
                for (int kx = -R; kx <= R; ++kx) {
                    const int ix = x + kx * dilation;
                    const int iy = y + ky * dilation;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        const float w = kernel[(ky + R) * K + (kx + R)];
                        sum += input[iy * width + ix] * w;
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

void atrousASPPCPU(const std::vector<float> &input,
                   std::vector<float> &output,
                   int width,
                   int height,
                   const std::vector<int> &dilations,
                   const std::vector<float> &weights,
                   const std::vector<float> &kernel) {
    std::fill(output.begin(), output.end(), 0.0f);
    std::vector<float> branch(output.size(), 0.0f);

    for (size_t b = 0; b < dilations.size(); ++b) {
        atrousConvolutionCPU(input, branch, width, height, dilations[b], kernel);
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += weights[b] * branch[i];
        }
    }
}

__global__ void atrousConvolutionBasicKernel(const float *input,
                                             float *output,
                                             int width,
                                             int height,
                                             int dilation) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -R; ky <= R; ++ky) {
        for (int kx = -R; kx <= R; ++kx) {
            const int ix = x + kx * dilation;
            const int iy = y + ky * dilation;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                const float w = d_kernel[(ky + R) * K + (kx + R)];
                sum += input[iy * width + ix] * w;
            }
        }
    }

    output[y * width + x] = sum;
}

__global__ void atrousConvolutionTiledKernel(const float *input,
                                             float *output,
                                             int width,
                                             int height,
                                             int dilation) {
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    const int halo = R * dilation;
    const int tileW = blockDim.x + 2 * halo;
    const int tileH = blockDim.y + 2 * halo;

    for (int sy = ty; sy < tileH; sy += blockDim.y) {
        for (int sx = tx; sx < tileW; sx += blockDim.x) {
            const int gx = blockIdx.x * blockDim.x + sx - halo;
            const int gy = blockIdx.y * blockDim.y + sy - halo;
            float value = 0.0f;
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                value = input[gy * width + gx];
            }
            tile[sy * tileW + sx] = value;
        }
    }

    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -R; ky <= R; ++ky) {
        for (int kx = -R; kx <= R; ++kx) {
            const int sx = tx + halo + kx * dilation;
            const int sy = ty + halo + ky * dilation;
            const float pixel = tile[sy * tileW + sx];
            const float w = d_kernel[(ky + R) * K + (kx + R)];
            sum += pixel * w;
        }
    }

    output[y * width + x] = sum;
}

__global__ void fuseASPPKernel(const float *branch0,
                               const float *branch1,
                               const float *branch2,
                               const float *branch3,
                               float *output,
                               int width,
                               int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;
    output[idx] = d_asppWeights[0] * branch0[idx] +
                  d_asppWeights[1] * branch1[idx] +
                  d_asppWeights[2] * branch2[idx] +
                  d_asppWeights[3] * branch3[idx];
}

float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float maxDiff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
    }
    return maxDiff;
}

float safeSpeedup(float baselineMs, float candidateMs) {
    return candidateMs > 1e-6f ? baselineMs / candidateMs : 0.0f;
}

bool readNextToken(std::istream &in, std::string &token) {
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string line;
            std::getline(in, line);
            continue;
        }
        return true;
    }
    return false;
}

bool readPGM(const std::string &path,
             std::vector<float> &image,
             int &width,
             int &height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    std::string magic;
    if (!readNextToken(in, magic)) {
        return false;
    }
    if (magic != "P5" && magic != "P2") {
        return false;
    }

    std::string token;
    if (!readNextToken(in, token)) return false;
    width = std::stoi(token);
    if (!readNextToken(in, token)) return false;
    height = std::stoi(token);
    if (!readNextToken(in, token)) return false;
    const int maxVal = std::stoi(token);

    if (width <= 0 || height <= 0 || maxVal <= 0) {
        return false;
    }

    image.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);

    if (magic == "P5") {
        in.get();
        std::vector<unsigned char> raw(static_cast<size_t>(width) * static_cast<size_t>(height));
        in.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (in.gcount() != static_cast<std::streamsize>(raw.size())) {
            return false;
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            image[i] = static_cast<float>(raw[i]) / static_cast<float>(maxVal);
        }
    } else {
        for (size_t i = 0; i < image.size(); ++i) {
            if (!readNextToken(in, token)) {
                return false;
            }
            image[i] = static_cast<float>(std::stoi(token)) / static_cast<float>(maxVal);
        }
    }

    return true;
}

void writePGM(const std::string &path,
              const std::vector<float> &image,
              int width,
              int height) {
    float minVal = image[0];
    float maxVal = image[0];
    for (float v : image) {
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }

    const float range = std::max(1e-6f, maxVal - minVal);

    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << width << " " << height << "\n255\n";
    for (float v : image) {
        const float norm = (v - minVal) / range;
        const unsigned char p = static_cast<unsigned char>(std::round(norm * 255.0f));
        out.write(reinterpret_cast<const char *>(&p), sizeof(unsigned char));
    }
}

RunResult run(const std::vector<float> &input,
              int width,
              int height,
              int dilation,
              int iterations,
              int blockX,
              int blockY,
              const std::string &outputPath) {
    const size_t numel = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t bytes = numel * sizeof(float);

    std::vector<float> kernel = {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
    };

    std::vector<float> outputCPU(numel, 0.0f);
    std::vector<float> outputCPUASPP(numel, 0.0f);
    std::vector<float> outputBasic(numel, 0.0f);
    std::vector<float> outputTiled(numel, 0.0f);
    std::vector<float> outputASPP(numel, 0.0f);

    const std::vector<int> asppDilations = {1, 2, 4, 8};
    const std::vector<float> asppWeights = {0.4f, 0.3f, 0.2f, 0.1f};

    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        atrousConvolutionCPU(input, outputCPU, width, height, dilation, kernel);
    }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    const float cpuMs = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count() /
                        static_cast<float>(iterations);

    auto cpuAsppStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        atrousASPPCPU(input, outputCPUASPP, width, height, asppDilations, asppWeights, kernel);
    }
    auto cpuAsppEnd = std::chrono::high_resolution_clock::now();
    const float cpuAsppMs =
        std::chrono::duration<float, std::milli>(cpuAsppEnd - cpuAsppStart).count() /
        static_cast<float>(iterations);

    float *dInput = nullptr;
    float *dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInput, bytes));
    CUDA_CHECK(cudaMalloc(&dOutput, bytes));
    CUDA_CHECK(cudaMemcpy(dInput, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel.data(), K * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_asppWeights, asppWeights.data(), asppWeights.size() * sizeof(float)));

    dim3 block(blockX, blockY);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        atrousConvolutionBasicKernel<<<grid, block>>>(dInput, dOutput, width, height, dilation);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsedBasicMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedBasicMs, start, stop));
    elapsedBasicMs /= static_cast<float>(iterations);
    CUDA_CHECK(cudaMemcpy(outputBasic.data(), dOutput, bytes, cudaMemcpyDeviceToHost));

    const int halo = R * dilation;
    const int tileW = static_cast<int>(block.x) + 2 * halo;
    const int tileH = static_cast<int>(block.y) + 2 * halo;
    const size_t sharedBytes = static_cast<size_t>(tileW) * tileH * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        atrousConvolutionTiledKernel<<<grid, block, sharedBytes>>>(dInput, dOutput, width, height, dilation);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsedTiledMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTiledMs, start, stop));
    elapsedTiledMs /= static_cast<float>(iterations);
    CUDA_CHECK(cudaMemcpy(outputTiled.data(), dOutput, bytes, cudaMemcpyDeviceToHost));

    std::array<float *, 4> dBranches = {nullptr, nullptr, nullptr, nullptr};
    std::array<cudaStream_t, 4> streams = {nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaMalloc(&dBranches[i], bytes));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    const int asppMaxDilation = *std::max_element(asppDilations.begin(), asppDilations.end());
    const int asppHalo = R * asppMaxDilation;
    const int asppTileW = static_cast<int>(block.x) + 2 * asppHalo;
    const int asppTileH = static_cast<int>(block.y) + 2 * asppHalo;
    const size_t asppSharedBytes = static_cast<size_t>(asppTileW) * asppTileH * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iterations; ++it) {
        for (int b = 0; b < 4; ++b) {
            atrousConvolutionTiledKernel<<<grid, block, asppSharedBytes, streams[b]>>>(
                dInput,
                dBranches[b],
                width,
                height,
                asppDilations[b]);
        }
        for (int b = 0; b < 4; ++b) {
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        }
        fuseASPPKernel<<<grid, block>>>(
            dBranches[0],
            dBranches[1],
            dBranches[2],
            dBranches[3],
            dOutput,
            width,
            height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsedAsppMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedAsppMs, start, stop));
    elapsedAsppMs /= static_cast<float>(iterations);
    CUDA_CHECK(cudaMemcpy(outputASPP.data(), dOutput, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(dBranches[i]));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    RunResult result;
    result.cpuMs = cpuMs;
    result.cpuAsppMs = cpuAsppMs;
    result.gpuBasicMs = elapsedBasicMs;
    result.gpuTiledMs = elapsedTiledMs;
    result.gpuAsppMs = elapsedAsppMs;
    result.maxAbsDiffBasic = maxAbsDiff(outputCPU, outputBasic);
    result.maxAbsDiffTiled = maxAbsDiff(outputCPU, outputTiled);
    result.maxAbsDiffAspp = maxAbsDiff(outputCPUASPP, outputASPP);

    writePGM(outputPath, outputASPP, width, height);

    return result;
}

}  // namespace

int main(int argc, char **argv) {
    int width = 2048;
    int height = 2048;
    int dilation = 2;
    int iterations = 20;
    int blockX = 16;
    int blockY = 16;
    std::string inputPath;
    std::string outputPath = "results/output_aspp.pgm";

    if (argc > 1) width = std::max(16, std::atoi(argv[1]));
    if (argc > 2) height = std::max(16, std::atoi(argv[2]));
    if (argc > 3) dilation = std::max(1, std::atoi(argv[3]));
    if (argc > 4) iterations = std::max(1, std::atoi(argv[4]));
    if (argc > 5) blockX = std::clamp(std::atoi(argv[5]), 4, 32);
    if (argc > 6) blockY = std::clamp(std::atoi(argv[6]), 4, 32);
    if (argc > 7) inputPath = argv[7];
    if (argc > 8) outputPath = argv[8];

    std::vector<float> input;
    if (!inputPath.empty()) {
        int loadedW = 0;
        int loadedH = 0;
        if (!readPGM(inputPath, input, loadedW, loadedH)) {
            std::cerr << "Failed to read input PGM image: " << inputPath << "\n";
            return 1;
        }
        width = loadedW;
        height = loadedH;
    } else {
        const size_t numel = static_cast<size_t>(width) * static_cast<size_t>(height);
        input.resize(numel);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (float &v : input) {
            v = dist(rng);
        }
    }

    const std::filesystem::path outPath(outputPath);
    if (!outPath.parent_path().empty()) {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::cout << "Atrous convolution benchmark\n";
    std::cout << "  width=" << width
              << " height=" << height
              << " dilation=" << dilation
              << " iterations=" << iterations
              << " block=" << blockX << "x" << blockY << "\n\n";

    const RunResult result = run(input, width, height, dilation, iterations, blockX, blockY, outputPath);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU avg ms:        " << result.cpuMs << "\n";
    std::cout << "CPU ASPP avg ms:   " << result.cpuAsppMs << "\n";
    std::cout << "GPU basic avg ms:  " << result.gpuBasicMs << "\n";
    std::cout << "GPU tiled avg ms:  " << result.gpuTiledMs << "\n";
    std::cout << "GPU ASPP avg ms:   " << result.gpuAsppMs << "\n";

    std::cout << "\nSpeedup basic: " << safeSpeedup(result.cpuMs, result.gpuBasicMs) << "x\n";
    std::cout << "Speedup tiled: " << safeSpeedup(result.cpuMs, result.gpuTiledMs) << "x\n";
    std::cout << "Speedup ASPP:  " << safeSpeedup(result.cpuAsppMs, result.gpuAsppMs) << "x\n";

    std::cout << "\nValidation max abs diff (CPU vs basic): " << result.maxAbsDiffBasic << "\n";
    std::cout << "Validation max abs diff (CPU vs tiled): " << result.maxAbsDiffTiled << "\n";
    std::cout << "Validation max abs diff (CPU ASPP vs GPU ASPP): " << result.maxAbsDiffAspp << "\n";

    if (inputPath.empty()) {
        std::cout << "\nInput source: random tensor\n";
    } else {
        std::cout << "\nInput source: " << inputPath << "\n";
    }
    std::cout << "Wrote output image: " << outputPath << "\n";
    return 0;
}
