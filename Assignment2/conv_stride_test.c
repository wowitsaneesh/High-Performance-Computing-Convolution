// STUDENT NAME: JALIL INAYAT-HUSSAIN , STUDENT ID : 22751096
// STUDENT NAME: ANEESH KUMAR BANDARI , STUDENT ID : 24553634

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>

float **allocate_matrix(int H, int W);
void free_matrix(float **matrix, int H);

// Calculate output dimensions with stride
int calc_output_height(int H, int kH, int sH) {
    return (int)ceil((double)H / sH);
}

int calc_output_width(int W, int kW, int sW) {
    return (int)ceil((double)W / sW);
}

// Serial convolution (stride supported)
static void conv2d_serial_stride(
    float **f, int H, int W,
    float **g, int kH, int kW,
    int sH, int sW,
    float **output) {

    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);
    int padH = kH / 2, padW = kW / 2;

    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++) {
            float sum = 0.0;
            int input_i = i * sH;
            int input_j = j * sW;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int fi = input_i + ki - padH;
                    int fj = input_j + kj - padW;
                    if (fi >= 0 && fi < H && fj >= 0 && fj < W) {
                        sum += f[fi][fj] * g[ki][kj];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// OpenMP parallel convolution
static void conv2d_omp_stride(
    float **f, int H, int W,
    float **g, int kH, int kW,
    int sH, int sW,
    float **output) {

    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);
    int padH = kH / 2, padW = kW / 2;

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++) {
            float sum = 0.0;
            int input_i = i * sH;
            int input_j = j * sW;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int fi = input_i + ki - padH;
                    int fj = input_j + kj - padW;
                    if (fi >= 0 && fi < H && fj >= 0 && fj < W) {
                        sum += f[fi][fj] * g[ki][kj];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

// MPI + OpenMP convolution (hybrid)
void conv2d_stride(
    float **f, int H, int W,
    float **g, int kH, int kW,
    int sH, int sW,
    float **output, MPI_Comm comm) {

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);

    if (size == 1) {
        conv2d_omp_stride(f, H, W, g, kH, kW, sH, sW, output);
        return;
    }

    int rows_per_proc = outH / size;
    int extra_rows = outH % size;

    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

    int padH = kH / 2;
    int input_start = start_row * sH - padH;
    int input_end = (start_row + local_rows - 1) * sH + padH;
    if (input_start < 0) input_start = 0;
    if (input_end >= H) input_end = H - 1;
    int input_rows_needed = input_end - input_start + 1;

    float **local_f = allocate_matrix(input_rows_needed, W);
    for (int i = 0; i < input_rows_needed; i++)
        for (int j = 0; j < W; j++)
            local_f[i][j] = (input_start + i < H) ? f[input_start + i][j] : 0.0;

    float **local_output = allocate_matrix(local_rows, outW);

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < outW; j++) {
            float sum = 0.0;
            int global_out_i = start_row + i;
            int input_i = global_out_i * sH;
            int input_j = j * sW;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int fi = input_i + ki - padH - input_start;
                    int fj = input_j + kj - (kW / 2);
                    if (fi >= 0 && fi < input_rows_needed && fj >= 0 && fj < W) {
                        sum += local_f[fi][fj] * g[ki][kj];
                    }
                }
            }
            local_output[i][j] = sum;
        }
    }

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int proc_rows = rows_per_proc + (i < extra_rows ? 1 : 0);
        recvcounts[i] = proc_rows * outW;
        displs[i] = i == 0 ? 0 : displs[i-1] + recvcounts[i-1];
    }

    float *local_flat = malloc(local_rows * outW * sizeof(float));
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < outW; j++)
            local_flat[i * outW + j] = local_output[i][j];

    float *output_flat = NULL;
    if (rank == 0) output_flat = malloc(outH * outW * sizeof(float));

    MPI_Gatherv(local_flat, local_rows * outW, MPI_FLOAT,
                output_flat, recvcounts, displs, MPI_FLOAT,
                0, comm);

    if (rank == 0) {
        for (int i = 0; i < outH; i++)
            for (int j = 0; j < outW; j++)
                output[i][j] = output_flat[i * outW + j];
        free(output_flat);
    }

    free(local_flat);
    free_matrix(local_output, local_rows);
    free_matrix(local_f, input_rows_needed);
    free(recvcounts);
    free(displs);
}

// Utility functions
float **allocate_matrix(int H, int W) {
    if (H <= 0 || W <= 0) return NULL;
    float **matrix = malloc(H * sizeof(float*));
    for (int i = 0; i < H; i++) {
        matrix[i] = malloc(W * sizeof(float));
    }
    return matrix;
}

void free_matrix(float **matrix, int H) {
    for (int i = 0; i < H; i++) free(matrix[i]);
    free(matrix);
}

float **create_matrix(int H, int W) {
    float **matrix = allocate_matrix(H, W);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            matrix[i][j] = (float)rand() / RAND_MAX;
    return matrix;
}

// Main driver with argument parsing
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand(42);

    int H = 64, W = 64, kH = 3, kW = 3, sH = 1, sW = 1;

    int opt;
    while ((opt = getopt(argc, argv, "H:W:kH:kW:sH:sW:")) != -1) {
        switch (opt) {
            case 'H':
                H = atoi(optarg);
                break;

            case 'W':
                W = atoi(optarg);
                break;

            case 'kH':
                kH = atoi(optarg);
                break;

            case 'kW':
                kW = atoi(optarg);
                break;

            case 'sH':
                sH = atoi(optarg);
                break;

            case 'sW':
                sW = atoi(optarg);
                break;

            default:
                break;
        }
    }

    if (rank == 0) {
        printf("Matrix H=%d, W=%d, Kernel H=%d, W=%d, Stride H=%d, W=%d\n",
               H, W, kH, kW, sH, sW);
    }

    float **f = create_matrix(H, W);
    float **g = create_matrix(kH, kW);
    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);
    float **out = allocate_matrix(outH, outW);

    if (rank == 0) {
        float **serial_out = allocate_matrix(outH, outW);
        double t1 = MPI_Wtime();
        conv2d_serial_stride(f, H, W, g, kH, kW, sH, sW, serial_out);
        double t2 = MPI_Wtime();
        printf("Serial execution time: %f seconds\n", t2 - t1);
        free_matrix(serial_out, outH);
    }

    if (rank == 0) {
        float **omp_out = allocate_matrix(outH, outW);
        double t1 = MPI_Wtime();
        conv2d_omp_stride(f, H, W, g, kH, kW, sH, sW, omp_out);
        double t2 = MPI_Wtime();
        printf("OpenMP execution time: %f seconds\n", t2 - t1);
        free_matrix(omp_out, outH);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    conv2d_stride(f, H, W, g, kH, kW, sH, sW, out, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    if (rank == 0) {
        printf("MPI+OpenMP execution time: %f seconds\n", t2 - t1);
    }

    free_matrix(f, H);
    free_matrix(g, kH);
    free_matrix(out, outH);
    MPI_Finalize();
    return 0;
}
