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

// Calculate output dimensions with stride
int calc_output_height(int H, int kH, int sH) {
    return (int)ceil((double)H / sH);
}

int calc_output_width(int W, int kW, int sW) {
    return (int)ceil((double)W / sW);
}

static void conv2d_serial_stride(
    float **f, 
    int H, 
    int W,
    float **g, 
    int kH, 
    int kW,
    int sH,
    int sW,
    float **output) {
    
    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);
    
    int padH = kH / 2;
    int padW = kW / 2;
    
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++) {
            float sum = 0.0;
            
            // Calculate input position with stride
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

static void conv2d_omp_stride(
    float **f, 
    int H, 
    int W,
    float **g, 
    int kH, 
    int kW,
    int sH,
    int sW,
    float **output) {
    
    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);
    
    int padH = kH / 2;
    int padW = kW / 2;
    
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++) {
            float sum = 0.0;
            
            int input_i = i * sH;
            int input_j = j * sW;
            
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int fi = input_i + ki - padH;
                    int fj = input_j + kj - padW;   // <-- FIXED: was kW instead of kj
                    
                    if (fi >= 0 && fi < H && fj >= 0 && fj < W) {
                        sum += f[fi][fj] * g[ki][kj];
                    }
                }
            }
            
            output[i][j] = sum;
        }
    }
}

void conv2d_stride(
    float **f,
    int H, int W,
    float **g,
    int kH, int kW,
    int sH, int sW,
    float **output,
    MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int outH = calc_output_height(H, kH, sH);
    int outW = calc_output_width(W, kW, sW);
    
    if (size == 1) {
        // Single process - use OpenMP only
        conv2d_omp_stride(f, H, W, g, kH, kW, sH, sW, output);
        return;
    }
    
    // Row-based decomposition
    int rows_per_proc = outH / size;
    int extra_rows = outH % size;
    
    int start_row = rank * rows_per_proc + (rank < extra_rows ? rank : extra_rows);
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
    
    // Calculate input rows needed for this process
    int padH = kH / 2;
    int input_start = start_row * sH - padH;
    int input_end = (start_row + local_rows - 1) * sH + padH;
    
    // Ensure bounds
    if (input_start < 0) input_start = 0;
    if (input_end >= H) input_end = H - 1;
    
    int input_rows_needed = input_end - input_start + 1;
    
    // Allocate local input buffer
    float **local_f = allocate_matrix(input_rows_needed, W);
    if (!local_f && input_rows_needed > 0) {
        fprintf(stderr, "Error: Failed to allocate local input buffer\n");
        MPI_Abort(comm, 1);
    }
    
    // Copy relevant input rows
    for (int i = 0; i < input_rows_needed; i++) {
        for (int j = 0; j < W; j++) {
            int global_row = input_start + i;
            if (global_row >= 0 && global_row < H) {
                local_f[i][j] = f[global_row][j];
            } else {
                local_f[i][j] = 0.0; // Padding
            }
        }
    }
    
    // Allocate local output buffer
    float **local_output = allocate_matrix(local_rows, outW);
    if (!local_output && local_rows > 0) {
        fprintf(stderr, "Error: Failed to allocate local output buffer\n");
        free_matrix(local_f, input_rows_needed);
        MPI_Abort(comm, 1);
    }
    
    // Perform local convolution with OpenMP
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
    
    // Gather results
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        int proc_rows = rows_per_proc + (i < extra_rows ? 1 : 0);
        recvcounts[i] = proc_rows * outW;
        displs[i] = i == 0 ? 0 : displs[i-1] + recvcounts[i-1];
    }
    
    // Flatten local output for MPI_Gatherv
    float *local_flat = NULL;
    if (local_rows > 0) {
        local_flat = malloc(local_rows * outW * sizeof(float));
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < outW; j++) {
                local_flat[i * outW + j] = local_output[i][j];
            }
        }
    }
    
    float *output_flat = NULL;
    if (rank == 0) {
        output_flat = malloc(outH * outW * sizeof(float));
    }
    
    MPI_Gatherv(local_flat, local_rows * outW, MPI_FLOAT,
                output_flat, recvcounts, displs, MPI_FLOAT,
                0, comm);
    
    // Copy back to output matrix on rank 0
    if (rank == 0) {
        for (int i = 0; i < outH; i++) {
            for (int j = 0; j < outW; j++) {
                output[i][j] = output_flat[i * outW + j];
            }
        }
        free(output_flat);
    }
    
    // Cleanup
    if (local_flat) free(local_flat);
    if (local_rows > 0) free_matrix(local_output, local_rows);
    if (input_rows_needed > 0) free_matrix(local_f, input_rows_needed);
    free(recvcounts);
    free(displs);
}

float **allocate_matrix(int H, int W) {
    if (H <= 0 || W <= 0) return NULL;
    
    float **matrix = malloc(H * sizeof(float*));
    if (matrix == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < H; i++) {
        matrix[i] = malloc(W * sizeof(float));
        if (matrix[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    
    return matrix;
}

void free_matrix(float **matrix, int H) {
    if (!matrix) return;
    for (int i = 0; i < H; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

static int read_matrix_txt(const char *path, float ***A, int *H, int *W) {
    FILE *fp = fopen(path, "r");    
    if (!fp) {
        return -1;
    }

    if ((fscanf(fp, "%d %d", H, W) != 2) || *H <= 0 || *W <= 0) {
        fclose(fp);
        return -1;
    }

    float **M = allocate_matrix(*H, *W);
    if (!M) {
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < *H; i++) {
        for (int j = 0; j < *W; j++) {
            if (fscanf(fp, "%f", &M[i][j]) != 1) {
                free_matrix(M, *H);
                fclose(fp);
                return -1;
            }
        }
    }
    
    fclose(fp);
    *A = M;
    return 0;
}

float **create_matrix(int H, int W) {
    float **matrix = allocate_matrix(H, W);
    if (!matrix) {
        return NULL;
    }
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX;
        }
    }
    
    return matrix;
}

static int write_matrix_txt(const char *path, float **matrix, int H, int W) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        return -1;
    }
    
    if (fprintf(fp, "%d %d\n", H, W) < 0) {
        fclose(fp);
        return -1;
    }
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (fprintf(fp, "%.6f", matrix[i][j]) < 0) {
                fclose(fp);
                return -1;
            }
            
            if (j < W - 1) {
                if (fprintf(fp, " ") < 0) {
                    fclose(fp);
                    return -1;
                }
            } else {
                if (fprintf(fp, "\n") < 0) {
                    fclose(fp);
                    return -1;
                }
            }
        }
    }
    
    fclose(fp);
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(42);  

    const char *fpath = NULL;
    const char *gpath = NULL;
    const char *opath = NULL;

    int H = 0, W = 0;
    int kH = 0, kW = 0;
    int sH = 1, sW = 1; // Default stride = 1

    // Parse stride parameters first
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-sH") == 0) {
            sH = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-sW") == 0) {
            sW = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    // Parse kernel dimensions
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-kH") == 0) {
            kH = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-kW") == 0) {
            kW = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    // Parse remaining options
    int opt;
    opterr = 0;
    while((opt = getopt(argc, argv, "f:g:o:H:W:")) != -1) {
        switch(opt) {
            case 'f': 
                fpath = optarg;
                break;
            case 'g': 
                gpath = optarg;
                break;
            case 'o': 
                opath = optarg;
                break;
            case 'H': 
                H = atoi(optarg);
                break;
            case 'W': 
                W = atoi(optarg);
                break;
            default:
                break;
        }
    }

    // Input validation
    if (sH <= 0 || sW <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid stride values sH=%d, sW=%d\n", sH, sW);
        }
        MPI_Finalize();
        return 1;
    }

    float **f = NULL, **g = NULL, **out = NULL;
    int iH = 0, iW = 0, ikH = 0, ikW = 0; 
    bool inputGenerated = false;
    bool kernelGenerated = false;

    // Only rank 0 handles I/O
    if (rank == 0) {
        // Read or generate input matrix
        if (fpath) {
            if (read_matrix_txt(fpath, &f, &iH, &iW) != 0) {
                if (H <= 0 || W <= 0) {
                    fprintf(stderr, "Error: Failed to read input matrix and no -H/-W provided\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                f = create_matrix(H, W);
                if (!f) {
                    fprintf(stderr, "Error: Failed to create input matrix\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                inputGenerated = true;
                iH = H;
                iW = W;
            }
        } else {
            if (H <= 0 || W <= 0) {
                fprintf(stderr, "Error: Missing -f or both -H and -W\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            f = create_matrix(H, W);
            if (!f) {
                fprintf(stderr, "Error: Failed to create input matrix\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            inputGenerated = true;
            iH = H;
            iW = W;
        }

        // Read or generate kernel matrix
        if (gpath) {
            if (read_matrix_txt(gpath, &g, &ikH, &ikW) != 0) {
                if (kH <= 0 || kW <= 0) {
                    fprintf(stderr, "Error: Failed to read kernel matrix and no -kH/-kW provided\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                g = create_matrix(kH, kW);
                if (!g) {
                    fprintf(stderr, "Error: Failed to create kernel matrix\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                kernelGenerated = true;
                ikH = kH;
                ikW = kW;
            }
        } else {
            if (kH <= 0 || kW <= 0) {
                fprintf(stderr, "Error: Missing -g or both -kH and -kW\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            g = create_matrix(kH, kW);
            if (!g) {
                fprintf(stderr, "Error: Failed to create kernel matrix\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            kernelGenerated = true;
            ikH = kH;
            ikW = kW;
        }
        
        printf("Input height: %i\n", iH);
        printf("Input width: %i\n", iW);
        printf("Kernel height: %i\n", ikH);
        printf("Kernel width: %i\n", ikW);
        printf("Stride height: %i\n", sH);
        printf("Stride width: %i\n", sW);
        printf("Output height: %i\n", calc_output_height(iH, ikH, sH));
        printf("Output width: %i\n", calc_output_width(iW, ikW, sW));
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&iH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iW, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ikH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ikW, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sW, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate matrices on all processes
    if (rank != 0) {
        f = allocate_matrix(iH, iW);
        g = allocate_matrix(ikH, ikW);
    }
    
    // Broadcast input and kernel data
    if (f && g) {
        for (int i = 0; i < iH; i++) {
            MPI_Bcast(f[i], iW, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < ikH; i++) {
            MPI_Bcast(g[i], ikW, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }
    
    int outH = calc_output_height(iH, ikH, sH);
    int outW = calc_output_width(iW, ikW, sW);
    
    if (rank == 0) {
        out = allocate_matrix(outH, outW);
        if (!out) {
            fprintf(stderr, "Error: Allocation out failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Time the parallel convolution
    double start = MPI_Wtime();
    conv2d_stride(f, iH, iW, g, ikH, ikW, sH, sW, out, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    
    if (rank == 0) {
        printf("MPI+OpenMP Execution time: %f seconds\n", end - start);
        
        // Optional output file
        if (opath) {
            if (write_matrix_txt(opath, out, outH, outW) != 0) {
                fprintf(stderr, "Error: failed to write output to %s\n", opath);
            }
        }
        
        // Save generated matrices if requested
        if (fpath && inputGenerated) {
            write_matrix_txt(fpath, f, iH, iW);
        }
        if (gpath && kernelGenerated) {
            write_matrix_txt(gpath, g, ikH, ikW);
        }
    }

    // Cleanup
    if (f) free_matrix(f, iH);
    if (g) free_matrix(g, ikH);
    if (out && rank == 0) free_matrix(out, outH);
    
    MPI_Finalize();
    return 0;

}
