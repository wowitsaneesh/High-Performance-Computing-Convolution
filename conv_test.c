// STUDENT NAME: JALIL INAYAT-HUSSAIN , STUDENT ID : 22751096
// STUDENT NAME: ANEESH KUMAR BANDARI , STUDENT ID : 24553634

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>     
#include <string.h>
#include <omp.h>

static void conv2d_serial(
		float **f, 
		int H, 
		int W,
		float **g, 
		int kH, 
		int kW,
		float **output) {
    
    // With same padding, output size = input size
    int outH = H;
    int outW = W;
    
    // Calculate padding needed
    int padH = kH / 2;  // Padding for height
    int padW = kW / 2;  // Padding for width
    
    // For each position in the output
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++) {
            
            float sum = 0.0;
            
            // Apply kernel at position (i,j)
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    
                    // Calculate input coordinates with padding
                    int fi = i + ki - padH;
                    int fj = j + kj - padW;
                    
                    // Check bounds - use 0 for out-of-bounds (zero padding)
                    if (fi >= 0 && fi < H && fj >= 0 && fj < W) {
                        sum += f[fi][fj] * g[ki][kj];
                    }
                    // Out-of-bounds pixels are treated as 0 (implicit)
                }
            }
            
            output[i][j] = sum;
        }
    }
}

static void conv2d_omp(
		float **f, 
		int H, 
		int W,
		float **g, 
		int kH, 
		int kW,
		float **output) {
    
    int padH = kH / 2;
    int padW = kW / 2;
    
    // Collapse both outer loops for better load balancing
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0;
            
            // Keep kernel loops together for cache efficiency
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int fi = i + ki - padH;
                    int fj = j + kj - padW;
                    
                    if (fi >= 0 && fi < H && fj >= 0 && fj < W) {
                        sum += f[fi][fj] * g[ki][kj];
                    }
                }
            }
            
            output[i][j] = sum;
        }
    }
}

float **allocate_matrix(int H, int W) {
    // Allocate array of row pointers
    float **matrix = malloc(H * sizeof(float*));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Failed to allocate row pointers\n");
        return NULL;
    }
    
    // Allocate each row
    for (int i = 0; i < H; i++) {
        matrix[i] = malloc(W * sizeof(float));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Error: Failed to allocate row %d\n", i);
            // Clean up previously allocated rows
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
    for (int i = 0; i < H; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


static int read_matrix_txt(const char *path, float ***A, int *H, int *W) {
	FILE *fp = fopen(path, "r");	

	if (!fp) {
		fprintf(stderr, "Error: Could not open file '%s'\n", path);
		return -1;
	}

	if ((fscanf(fp, "%d %d", H, W) != 2) || *H <= 0 || *W <= 0) {
		fprintf(stderr, "Error: Invalid header in file '%s'\n", path);
		fclose(fp);
		return -1;
	}

	float **M = allocate_matrix(*H, *W);

	if (!M) {
		fprintf(stderr, "Error: Allocation failed for '%s'\n", path);
		fclose(fp);
		return -1;
	}

	// Fill matrix
	for (int i = 0; i < *H; i++) {
		for (int j = 0; j < *W; j++) {
			if (fscanf(fp, "%f", &M[i][j]) != 1) {
				fprintf(stderr, "Error: Invalid data in %s at pos(%i, %i)\n", path, i, j);
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
    
    // Fill with random values between 0 and 1
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
        fprintf(stderr, "Error: Could not open file '%s' for writing\n", path);
        return -1;
    }
    
    // Write header (dimensions)
    if (fprintf(fp, "%d %d\n", H, W) < 0) {
        fprintf(stderr, "Error: Failed to write header to '%s'\n", path);
        fclose(fp);
        return -1;
    }
    
    // Write matrix data
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (fprintf(fp, "%.6f", matrix[i][j]) < 0) {
                fprintf(stderr, "Error: Failed to write data to '%s' at pos(%d,%d)\n", 
                        path, i, j);
                fclose(fp);
                return -1;
            }
            
            // Add space between elements, newline at end of row
            if (j < W - 1) {
                if (fprintf(fp, " ") < 0) {
                    fprintf(stderr, "Error: Failed to write separator to '%s'\n", path);
                    fclose(fp);
                    return -1;
                }
            } else {
                if (fprintf(fp, "\n") < 0) {
                    fprintf(stderr, "Error: Failed to write newline to '%s'\n", path);
                    fclose(fp);
                    return -1;
                }
            }
        }
    }
    
    fclose(fp);
    return 0;
}

void print_matrix(float **M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", M[i][j]);  // Use %f for floats, M[i][j] for access
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
	// Set seed for random number generation
	srand(42);  

	// Create pointers to store file paths
	const char *fpath = NULL;
	const char *gpath = NULL;
	const char *opath = NULL;

	// Create variables to store input and kernel dimensions
	int H = 0, W = 0;
	int kH = 0, kW = 0;

	// 1) Parse -kH and -kW FIRST (before getopt)
	// after parsing kH/kW manually
	for (int i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-kH") == 0) {
			kH = atoi(argv[i + 1]);
			// remove them from argv
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

	// 2) Now use getopt for the rest
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

	// 2) Create or read in input and kernel from files

	// Create storage for the matrices
	float **f = NULL, **g = NULL, **out = NULL;

	// Create storage for the input kernel width and heights
	int iH = 0, iW = 0, ikH = 0, ikW = 0; 


	// Read input matrix from file or create input matrix
	if (fpath) {
		if (read_matrix_txt(fpath, &f, &iH, &iW) != 0) {
			fprintf(stderr, "Error: Failed to read input matrix\n");
			return 1;
		}
	} else {
		if (H <= 0 || W <= 0) {
			fprintf(stderr, "Error: Missing -f or both -H and -W\n");
			return 1;
		}
		// Create input matrix with H x W dimensions
		f = create_matrix(H, W);
		if (!f) {
			fprintf(stderr, "Error: Failed to create input matrix\n");
			return 1;
		}
		iH = H;    // Input height = H
		iW = W;    // Input width = W
	}

	// Read kernel matrix from file or create kernel matrix  
	if (gpath) {
		if (read_matrix_txt(gpath, &g, &ikH, &ikW) != 0) {
			fprintf(stderr, "Error: Failed to read kernel matrix\n");
			return 1;
		}
	} else {
		if (kH <= 0 || kW <= 0) {  // Check kH, kW (not H, W)
			fprintf(stderr, "Error: Missing -g or both -kH and -kW\n");
			return 1;
		}
		// Create kernel matrix with kH x kW dimensions
		g = create_matrix(kH, kW);  // Use kH, kW here!
		if (!g) {
			fprintf(stderr, "Error: Failed to create kernel matrix\n");
			return 1;
		}
		ikH = kH;  // Kernel height = kH
		ikW = kW;  // Kernel width = kW
	}
	
	// Let have a look at the matrix
	printf("Input height: %i\n", H);
	printf("Input width: %i\n", W);
	printf("kernel height: %i\n", kH);
	printf("kernel width: %i\n", kW);

    out = create_matrix(iH, iW);

    if (!out) { 
		fprintf(stderr, "Error: Allocation out failed\n"); 
		free_matrix(f, H); 
		free_matrix(g, W); 
		return 1; 
	}

	// Serial convolution
	double start, end;
	double cpu_time_used;

	start = omp_get_wtime();

 conv2d_serial(f, iH, iW, g, ikH, ikW, out);

	end = omp_get_wtime();
	cpu_time_used = (double) (end - start) ;

	printf("Serial Execution:\n");
	printf("CPU time used: %f\n", cpu_time_used);

	// Parallel convolution
	start = omp_get_wtime();

    conv2d_omp(f, iH, iW, g, ikH, ikW, out);

	end = omp_get_wtime();
	cpu_time_used = (double) (end - start) ;

	printf("Parallel Execution:\n");
	printf("CPU time used: %f\n", cpu_time_used);

    // Optional output file
    if (opath) {
        if (write_matrix_txt(opath, out, iH, iW) != 0) {
            fprintf(stderr, "Error: failed to write %s\n", opath);
        }
    }

	free_matrix(f, H);
	free_matrix(g, kH);
	free_matrix(out, H);
	return 0;

}
