# 2D Convolution (Serial + OpenMP)

This program performs 2D convolution with same (zero) padding using both a serial and a parallel (OpenMP) implementation.  

## Build
```bash
make            # builds ./conv_test
make clean      # remove executable + extras
```

> On macOS, use Homebrew GCC: `make CC=gcc-14`  
> On Kaya, load GCC: `module load gcc/12.2.0`

## Usage
You can either **load matrices from files** or **generate random ones**.

### Generate random matrices
```bash
./conv_test -H 1024 -W 1024 -kH 5 -kW 5 -o out.txt
```

### Use input files
```bash
./conv_test -f input.txt -g kernel.txt -o out.txt
```

### Options
- `-H, -W` → input matrix size (if generating)  
- `-kH, -kW` → kernel size (if generating)  
- `-f <file>` → input matrix file  
- `-g <file>` → kernel matrix file  
- `-o <file>` → write output  

## Controlling Threads
```bash
export OMP_NUM_THREADS=8
./conv_test -H 2048 -W 2048 -kH 7 -kW 7
```

## Example SLURM (Kaya)
```bash
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
module load gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./conv_test -H 4096 -W 4096 -kH 7 -kW 7 -o out.txt
```
