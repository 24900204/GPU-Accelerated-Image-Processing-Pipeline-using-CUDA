#!/bin/bash
set -e
make
./gpu_image_processor --input ./input --output ./output --operation sobel --benchmark
