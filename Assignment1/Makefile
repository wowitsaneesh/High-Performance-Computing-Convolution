# STUDENT ID : 22751096 , STUDENT NAME : JALIL INAYAT-HUSSAIN
# STUDENT ID : 24553634 , STUDENT NAME : ANEESH KUMAR BANDARI

CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c11
OMPFLAGS := -fopenmp
LDFLAGS ?=
LDLIBS ?=

all: conv_test

conv_test: conv_test.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f conv_test *.o *.out
