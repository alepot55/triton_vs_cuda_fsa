NVCC := nvcc  # Changed to just 'nvcc' - assuming it's in PATH

# Flags di compilazione
NVCCFLAGS := -g -G -arch=sm_70 -I../common/include -I./src # Adatta l'architettura GPU (es: sm_70 per RTX 20xx, sm_80 per A100, ecc.)

# File sorgente e eseguibile
SRC_FILES := src/fsa_engine.cu src/utils.cu benchmarks/benchmark_fsa.cu  # ADDED benchmarks/benchmark_fsa.cu HERE
EXECUTABLE := fsa_engine_cuda

OBJECTS := $(SRC_FILES:.cu=.o) # Create object files from source files

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) # Link object files to create executable
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu # Rule to compile .cu files to .o object files
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(EXECUTABLE) $(OBJECTS) # Clean object files as well
