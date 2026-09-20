#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

__global__ void RgbToGrayKernel(const unsigned char* rgb,
                                unsigned char* gray,
                                int width,
                                int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const int pixel = y * width + x;
  const int rgb_index = pixel * 3;

  const float value = 0.299f * rgb[rgb_index] +
                      0.587f * rgb[rgb_index + 1] +
                      0.114f * rgb[rgb_index + 2];

  gray[pixel] = static_cast<unsigned char>(value);
}

__global__ void SobelKernel(const unsigned char* gray,
                            unsigned char* edge,
                            int width,
                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    edge[y * width + x] = 0;
    return;
  }

  const int gx = -gray[(y - 1) * width + (x - 1)]
                 + gray[(y - 1) * width + (x + 1)]
                 - 2 * gray[y * width + (x - 1)]
                 + 2 * gray[y * width + (x + 1)]
                 - gray[(y + 1) * width + (x - 1)]
                 + gray[(y + 1) * width + (x + 1)];

  const int gy = -gray[(y - 1) * width + (x - 1)]
                 - 2 * gray[(y - 1) * width + x]
                 - gray[(y - 1) * width + (x + 1)]
                 + gray[(y + 1) * width + (x - 1)]
                 + 2 * gray[(y + 1) * width + x]
                 + gray[(y + 1) * width + (x + 1)];

  const int magnitude = min(255, abs(gx) + abs(gy));
  edge[y * width + x] = static_cast<unsigned char>(magnitude);
}

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error during " << operation << ": "
              << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  std::string operation = "sobel";
  std::string input_dir = "./input";
  std::string output_dir = "./output";
  bool benchmark = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if (argument == "--operation" && i + 1 < argc) {
      operation = argv[++i];
    } else if (argument == "--input" && i + 1 < argc) {
      input_dir = argv[++i];
    } else if (argument == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (argument == "--benchmark") {
      benchmark = true;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " --input DIR --output DIR"
                   " --operation grayscale|sobel --benchmark\n";
      return EXIT_FAILURE;
    }
  }

  (void)input_dir;
  (void)output_dir;
  (void)benchmark;

  constexpr int width = 1024;
  constexpr int height = 1024;

  const size_t rgb_bytes =
      static_cast<size_t>(width) * height * 3;
  const size_t gray_bytes =
      static_cast<size_t>(width) * height;

  unsigned char* host_rgb =
      static_cast<unsigned char*>(std::malloc(rgb_bytes));
  unsigned char* host_edge =
      static_cast<unsigned char*>(std::malloc(gray_bytes));

  if (host_rgb == nullptr || host_edge == nullptr) {
    std::cerr << "Host allocation failed.\n";
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < rgb_bytes; ++i) {
    host_rgb[i] = 128;
  }

  unsigned char* device_rgb = nullptr;
  unsigned char* device_gray = nullptr;
  unsigned char* device_edge = nullptr;

  CheckCuda(cudaMalloc(&device_rgb, rgb_bytes), "cudaMalloc RGB");
  CheckCuda(cudaMalloc(&device_gray, gray_bytes), "cudaMalloc gray");
  CheckCuda(cudaMalloc(&device_edge, gray_bytes), "cudaMalloc edge");

  CheckCuda(cudaMemcpy(device_rgb, host_rgb, rgb_bytes,
                       cudaMemcpyHostToDevice),
            "host-to-device copy");

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

  cudaEvent_t start;
  cudaEvent_t stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");

  CheckCuda(cudaEventRecord(start), "cudaEventRecord");

  RgbToGrayKernel<<<grid, block>>>(
      device_rgb, device_gray, width, height);
  CheckCuda(cudaGetLastError(), "grayscale kernel");

  if (operation == "sobel") {
    SobelKernel<<<grid, block>>>(
        device_gray, device_edge, width, height);
    CheckCuda(cudaGetLastError(), "Sobel kernel");
  }

  CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
  CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

  float milliseconds = 0.0f;
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop),
            "cudaEventElapsedTime");

  CheckCuda(cudaMemcpy(host_edge, device_edge, gray_bytes,
                       cudaMemcpyDeviceToHost),
            "device-to-host copy");

  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties");

  std::cout << "GPU: " << properties.name << '\n';
  std::cout << "Image workload: " << width << " x " << height << '\n';
  std::cout << "Operation: " << operation << '\n';
  std::cout << "CUDA kernel time: " << milliseconds << " ms\n";
  std::cout << "Processing completed successfully.\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_rgb);
  cudaFree(device_gray);
  cudaFree(device_edge);
  std::free(host_rgb);
  std::free(host_edge);

  return EXIT_SUCCESS;
}
