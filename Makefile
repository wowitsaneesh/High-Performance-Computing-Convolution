# Use whatever GCC is available on the system or loaded via modules
CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c11
OMPFLAGS := -fopenmp
LDFLAGS ?=
LDLIBS ?=

all: conv_test_new

conv_test_new: conv_test_new.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f conv_test_new *.o *.out
