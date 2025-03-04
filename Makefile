# Makefile for building the L-BFGS optimizer in C

CC = gcc
CFLAGS = -O2 -std=c99 -Wall

# Uncomment the following lines if using clang with OpenMP support
# For clang, you may need: 
# CFLAGS += -Xpreprocessor -fopenmp
# LDFLAGS += -lomp

# For GCC with OpenMP, add:
CFLAGS += -fopenmp
LDFLAGS += -fopenmp

SRC = src/main.c src/optimizer.c src/objective.c
OBJ = $(SRC:.c=.o)

TARGET = lbfgs_optimizer

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
