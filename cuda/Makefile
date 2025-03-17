# Directories
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = obj
BENCHMARKS_DIR = benchmarks
COMMON_DIR = ../common
COMMON_INCLUDE_DIR = $(COMMON_DIR)/include
COMMON_BENCHMARK_DIR = $(COMMON_DIR)/benchmark
COMMON_TEST_DIR = $(COMMON_DIR)/test
COMMON_SRC_DIR = $(COMMON_DIR)/src

# Compiler flags
NVCC = nvcc
CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_INCLUDE_DIR) -I$(COMMON_DIR) -std=c++17
NVCCFLAGS = $(CFLAGS) -arch=sm_70 -Xcompiler -Wall,-Wextra

# List all source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)
BENCHMARK_CU_FILES = $(wildcard $(BENCHMARKS_DIR)/*.cu)
BENCHMARK_CPP_FILES = $(wildcard $(BENCHMARKS_DIR)/*.cpp)
COMMON_BENCHMARK_FILES = $(wildcard $(COMMON_BENCHMARK_DIR)/*.cpp)
COMMON_TEST_FILES = $(wildcard $(COMMON_TEST_DIR)/*.cpp)
COMMON_SRC_FILES = $(wildcard $(COMMON_SRC_DIR)/*.cpp)

# Generate object file names
SRC_OBJ = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(filter %.cu,$(SRC_FILES))) $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(filter %.cpp,$(SRC_FILES)))
BENCHMARK_CU_OBJ = $(patsubst $(BENCHMARKS_DIR)/%.cu,$(OBJ_DIR)/%.o,$(BENCHMARK_CU_FILES))
BENCHMARK_CPP_OBJ = $(patsubst $(BENCHMARKS_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(BENCHMARK_CPP_FILES))
COMMON_BENCHMARK_OBJ = $(patsubst $(COMMON_BENCHMARK_DIR)/%.cpp,$(OBJ_DIR)/common_%.o,$(COMMON_BENCHMARK_FILES))
COMMON_TEST_OBJ = $(patsubst $(COMMON_TEST_DIR)/%.cpp,$(OBJ_DIR)/common_test_%.o,$(COMMON_TEST_FILES))
COMMON_SRC_OBJ = $(patsubst $(COMMON_SRC_DIR)/%.cpp,$(OBJ_DIR)/common_src_%.o,$(COMMON_SRC_FILES))

# Executable
TARGET = fsa_engine_cuda

# Default target
.PHONY: all clean debug rebuild
all: $(OBJ_DIR) compile

compile: $(TARGET)

rebuild: clean all

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Print debugging information
debug:
	@echo "Source files:"
	@echo $(SRC_FILES)
	@echo "Benchmark CU files:"
	@echo $(BENCHMARK_CU_FILES)
	@echo "Benchmark CPP files:"
	@echo $(BENCHMARK_CPP_FILES)
	@echo "Common Benchmark files:"
	@echo $(COMMON_BENCHMARK_FILES)
	@echo "Common Test files:"
	@echo $(COMMON_TEST_FILES)
	@echo "Common Source files:"
	@echo $(COMMON_SRC_FILES)

# Link target
$(TARGET): $(SRC_OBJ) $(BENCHMARK_CU_OBJ) $(BENCHMARK_CPP_OBJ) $(COMMON_BENCHMARK_OBJ) $(COMMON_TEST_OBJ) $(COMMON_SRC_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lnvidia-ml

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA benchmark files
$(OBJ_DIR)/%.o: $(BENCHMARKS_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ benchmark files
$(OBJ_DIR)/%.o: $(BENCHMARKS_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile Common benchmark files
$(OBJ_DIR)/common_%.o: $(COMMON_BENCHMARK_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile Common test files
$(OBJ_DIR)/common_test_%.o: $(COMMON_TEST_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile Common source files
$(OBJ_DIR)/common_src_%.o: $(COMMON_SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)
