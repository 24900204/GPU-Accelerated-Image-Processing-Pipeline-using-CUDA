NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2

TARGET := gpu_image_processor

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(CXXFLAGS) src/main.cu -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) --input ./input --output ./output --operation sobel --benchmark
