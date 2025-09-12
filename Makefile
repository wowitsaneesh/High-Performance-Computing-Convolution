# Use whatever GCC is available on the system or loaded via modules
CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c11
# Put -fopenmp in both compile and link flags (some toolchains need it twice)
OMPFLAGS := -fopenmp
LDFLAGS ?=
LDLIBS ?=

all: conv_test

conv_test: conv_test.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f conv_test *.o *.out

