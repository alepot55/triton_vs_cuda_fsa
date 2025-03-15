CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
NVCCFLAGS = -g -G -arch=sm_70 -I../common/include -I./src
LDFLAGS =

COMMON_DIR = ../common
SRC_DIR = ./src
BENCH_DIR = ./benchmarks
OBJ_DIR = ./obj

# Create obj directory if it doesn't exist
$(shell mkdir -p $(OBJ_DIR))

# Define source files
COMMON_SRC = $(COMMON_DIR)/src/regex_conversion.cpp
SRC_FILES = $(SRC_DIR)/fsa_engine.cu $(BENCH_DIR)/benchmark_fsa.cu

# Define object files
COMMON_OBJS = $(OBJ_DIR)/regex_conversion.o
CUDA_OBJS = $(OBJ_DIR)/fsa_engine.o $(OBJ_DIR)/benchmark_fsa.o

# Define the target executable
TARGET = fsa_engine_cuda

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(COMMON_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^

# Compile common C++ code
$(OBJ_DIR)/regex_conversion.o: $(COMMON_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA code
$(OBJ_DIR)/fsa_engine.o: $(SRC_DIR)/fsa_engine.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark_fsa.o: $(BENCH_DIR)/benchmark_fsa.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
