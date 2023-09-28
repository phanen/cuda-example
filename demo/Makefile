NVCC=nvcc

NVCC_FLAGS=-MMD
CUDA_SRCS = $(shell find $(abspath .) -name "*.cu")

OBJS := ${CUDA_SRCS:.cu=.o}
BIN=a.out

$(BIN): $(OBJS)
	$(NVCC) $^ -o $@

%.o: %.cu
	nvcc -dc $(NVCC_FLAGS) $< -o $@

clean:
	rm *.o *.out -rf

all: $(CU_SRCS)
	nvcc $^
