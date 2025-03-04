CC = clang
CFLAGS = -O2 -std=c99 -Wall

# Explicitly specify libomp location (manual override)
CFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp

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
