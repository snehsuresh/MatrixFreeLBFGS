CC = clang
CFLAGS = -O3 -std=c99 -Wall -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

all: lbfgs_optimizer

lbfgs_optimizer: src/main.o src/optimizer.o src/objective.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o lbfgs_optimizer
