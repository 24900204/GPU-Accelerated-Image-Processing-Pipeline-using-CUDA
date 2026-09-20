# GPU-Accelerated Image Processing Pipeline using CUDA

## Project Overview
This capstone demonstrates practical GPU programming with NVIDIA CUDA. The application maps image pixels to CUDA threads and performs RGB-to-grayscale conversion and Sobel edge detection in parallel.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (`nvcc`)
- C++17 compiler
- OpenCV 4 development libraries
- GNU Make

## Build
```bash
make
```

## Run
```bash
./gpu_image_processor --input ./input --output ./output --operation sobel --benchmark
```

Supported arguments:
- `--input <directory>`
- `--output <directory>`
- `--operation <grayscale|sobel>`
- `--benchmark`

## CUDA Implementation
The program uses CUDA kernels and a two-dimensional launch configuration. A 16x16 block is used, and each CUDA thread is responsible for an image pixel. CUDA events measure GPU kernel execution time.

## Evidence
After running the program on a CUDA-enabled system, place actual screenshots, logs, input images, output images, and benchmark results in the `results/` and `screenshots/` directories.

## Project Lessons
The project demonstrates CUDA kernel design, host/device memory transfers, thread and block configuration, image boundary handling, and GPU timing. It also illustrates that total acceleration depends on both computation and memory-transfer overhead.

## Future Work
CUDA streams, asynchronous transfers, batch processing, CUDA NPP, video processing, and multi-GPU execution could extend the project.
