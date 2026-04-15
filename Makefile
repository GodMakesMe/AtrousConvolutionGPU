NVCC ?= nvcc
TARGET := bin/atrous
SRC := src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p bin
	$(NVCC) -O3 -std=c++17 $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) 2048 2048 2 20

clean:
	rm -rf bin results/*.pgm results/*.txt results/*.csv

.PHONY: all run clean
