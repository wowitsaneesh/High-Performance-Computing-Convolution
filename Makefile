CC=gcc-15
CFLAGS=-O3 -march=native -fopenmp -Wall -Wextra -std=c11
LDFLAGS=

all: conv_test

conv_test: conv_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f conv_test *.o *.out *.txt

