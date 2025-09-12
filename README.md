# 2D Convolution (Serial + OpenMP)

```bash
STUDENT ID : 22751096 , STUDENT NAME : JALIL INAYAT-HUSSAIN
STUDENT ID : 24553634 , STUDENT NAME : ANEESH KUMAR BANDARI
```

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
./conv_test -H 1024 -W 1024 -kH 5 -kW 5 
```

### Generate random matrices and save outputs
```bash
./conv_test -H 1024 -W 1024 -kH 5 -kW 5 -o out.txt
```

### Generate random matrices and save inputs and outputs
```bash
./conv_test -H 1024 -W 1024 -kH 5 -kW 5 -f f.txt -g g.txt -o o.txt
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
#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --mem=1024G
#SBATCH --job-name=convd
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=22751096@student.uwa.edu.au
#SBATCH --partition=cits3402
#SBATCH --output=conv_test_1250000x200000_3x3_dynamic.out
#SBATCH --error=conv_test.err
 
 
gcc -fopenmp conv_test.c -o conv_test
 
./conv_test -H 1250000 -W 200000 -kH 3 -kW 3
```
