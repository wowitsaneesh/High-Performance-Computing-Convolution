// conv_test.c — CITS3402/CITS5507 A1: Fast parallel 2D convolution with OpenMP
// Author: <Your Name> (<Student ID>)

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>     // getopt
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------
// Utility: timing (monotonic)
// -----------------------------
static inline double now_seconds(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ----------------------------------------------
// Matrix helpers: contiguous + row pointer view
// ----------------------------------------------
static float **alloc_matrix(int H, int W) {
    if (H <= 0 || W <= 0) return NULL;
    size_t rowsz = (size_t)W * sizeof(float);
    float **rowptrs = (float**)malloc((size_t)H * sizeof(float*));
    float  *block   = (float*) malloc((size_t)H * rowsz); // portable on macOS
    if (!rowptrs || !block) { free(rowptrs); free(block); return NULL; }
    for (int i = 0; i < H; ++i) rowptrs[i] = block + (size_t)i * W;
    return rowptrs;
}

static void free_matrix(float **A) {
    if (!A) return;
    free(A[0]); // contiguous block
    free(A);
}

static void fill_random(float **A, int H, int W) {
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            A[i][j] = (float)drand48(); // [0,1)
}

// ---------------------------------
// File I/O per assignment spec
// ---------------------------------
static int read_matrix_txt(const char *path, float ***A, int *H, int *W) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno)); return -1; }
    if (fscanf(fp, "%d %d", H, W) != 2 || *H <= 0 || *W <= 0) {
        fprintf(stderr, "Invalid header in %s\n", path); fclose(fp); return -1;
    }
    float **M = alloc_matrix(*H, *W);
    if (!M) { fprintf(stderr, "alloc failed %s\n", path); fclose(fp); return -1; }
    for (int i = 0; i < *H; ++i) {
        for (int j = 0; j < *W; ++j) {
            if (fscanf(fp, "%f", &M[i][j]) != 1) {
                fprintf(stderr, "Invalid data in %s at (%d,%d)\n", path, i, j);
                free_matrix(M); fclose(fp); return -1;
            }
        }
    }
    fclose(fp);
    *A = M; return 0;
}

static int write_matrix_txt(const char *path, float **A, int H, int W) {
    FILE *fp = fopen(path, "w"); if (!fp) { fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno)); return -1; }
    fprintf(fp, "%d %d\n", H, W);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            fprintf(fp, (j+1==W)? "%g\n" : "%g ", A[i][j]);
        }
    }
    fclose(fp); return 0;
}

// -------------------------------------------------
// SAME-padding 2D convolution — serial reference
// -------------------------------------------------
static void conv2d_serial(float **f, int H, int W,
                          float **g, int kH, int kW,
                          float **out) {
    int padY = kH / 2;
    int padX = kW / 2;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int ky = 0; ky < kH; ++ky) {
                int iy = y + ky - padY;
                if ((unsigned)iy >= (unsigned)H) continue; // out-of-bounds => zero
                for (int kx = 0; kx < kW; ++kx) {
                    int ix = x + kx - padX;
                    if ((unsigned)ix >= (unsigned)W) continue;
                    acc += f[iy][ix] * g[ky][kx];
                }
            }
            out[y][x] = acc;
        }
    }
}

// -------------------------------------------------
// SAME-padding 2D convolution — OpenMP parallel
// -------------------------------------------------
static void conv2d_omp(float **f, int H, int W,
                       float **g, int kH, int kW,
                       float **out) {
    int padY = kH / 2;
    int padX = kW / 2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int ky = 0; ky < kH; ++ky) {
                int iy = y + ky - padY;
                if ((unsigned)iy >= (unsigned)H) continue;
                for (int kx = 0; kx < kW; ++kx) {
                    int ix = x + kx - padX;
                    if ((unsigned)ix >= (unsigned)W) continue;
                    acc += f[iy][ix] * g[ky][kx];
                }
            }
            out[y][x] = acc;
        }
    }
}

// -------------------------------
// CLI / main (with -f/-g/-o and -kH/-kW)
// -------------------------------
static void usage(const char *prog) {
    fprintf(stderr,
    "Usage: %s [options]\n"
    "  -f <f.txt>        Input matrix f (file)\n"
    "  -g <g.txt>        Kernel g (file)\n"
    "  -o <o.txt>        Output file (optional)\n"
    "  -H <int>          Generate input height (if no -f)\n"
    "  -W <int>          Generate input width  (if no -f)\n"
    "  -kH <int>         Generate kernel height (if no -g)\n"
    "  -kW <int>         Generate kernel width  (if no -g)\n"
    "  -s <int>          RNG seed for generation\n"
    "  -t <int>          Threads (OMP_NUM_THREADS)\n"
    "  -p <mode>         Mode: serial | omp   (default: omp)\n"
    "  -C                Compare serial vs omp and report speedup\n"
    "  -v                Verbose\n",
    prog);
}

static int parse_int_arg(const char *flag, int argc, char **argv, int *out) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], flag) == 0) {
            *out = atoi(argv[i+1]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *fpath = NULL, *gpath = NULL, *opath = NULL;
    int H = 0, W = 0, kH = 0, kW = 0, threads = 0, compare = 0, verbose = 0;
    unsigned int seed = 12345;
    enum { MODE_OMP, MODE_SERIAL } mode = MODE_OMP;

    // Seed the RNG for drand48
    srand48((long)seed);

    // Accept -kH and -kW as long options (manual scan before getopt):
    (void)parse_int_arg("-kH", argc, argv, &kH);
    (void)parse_int_arg("-kW", argc, argv, &kW);

    // Parse the short options with getopt
    int opt;
    opterr = 0; // ignore unknowns so -kH/-kW don't produce errors
    while ((opt = getopt(argc, argv, "f:g:o:H:W:s:t:p:Cvh")) != -1) {
        if (opt == 'h') { usage(argv[0]); return 0; }
        switch (opt) {
            case 'f': fpath = optarg; break;
            case 'g': gpath = optarg; break;
            case 'o': opath = optarg; break;
            case 'H': H = atoi(optarg); break;
            case 'W': W = atoi(optarg); break;
            case 's': seed = (unsigned int)strtoul(optarg, NULL, 10); srand48((long)seed); break;
            case 't': threads = atoi(optarg); break;
            case 'p':
                if (strcmp(optarg, "serial") == 0) mode = MODE_SERIAL;
                else if (strcmp(optarg, "omp") == 0) mode = MODE_OMP;
                else { fprintf(stderr, "-p must be serial|omp\n"); return 1; }
                break;
            case 'C': compare = 1; break;
            case 'v': verbose = 1; break;
            default: /* ignore unknowns */ break;
        }
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    if (verbose) {
        int nt = omp_get_max_threads();
        fprintf(stderr, "[info] OpenMP max threads = %d\n", nt);
    }
#else
    if (threads > 1 && verbose) fprintf(stderr, "[warn] OpenMP not enabled.\n");
#endif

    // Load or generate f and g as per spec
    float **f = NULL, **g = NULL, **out = NULL, **tmp = NULL;
    int iH=0, iW=0, ikH=0, ikW=0;

    // Input matrix f
    if (fpath) {
        if (read_matrix_txt(fpath, &f, &iH, &iW) != 0) return 1;
    } else {
        if (H <= 0 || W <= 0) {
            fprintf(stderr, "Missing -f or both -H and -W\n");
            usage(argv[0]); return 1;
        }
        f = alloc_matrix(H, W); if (!f) { fprintf(stderr, "alloc f failed\n"); return 1; }
        fill_random(f, H, W); iH = H; iW = W;
    }

    // Kernel g
    if (gpath) {
        if (read_matrix_txt(gpath, &g, &ikH, &ikW) != 0) { free_matrix(f); return 1; }
    } else {
        if (kH <= 0 || kW <= 0) {
            fprintf(stderr, "Missing -g or both -kH and -kW\n");
            usage(argv[0]); free_matrix(f); return 1;
        }
        g = alloc_matrix(kH, kW); if (!g) { fprintf(stderr, "alloc g failed\n"); free_matrix(f); return 1; }
        fill_random(g, kH, kW); ikH = kH; ikW = kW;
    }

    if (verbose) fprintf(stderr, "[info] f: %dx%d, g: %dx%d\n", iH, iW, ikH, ikW);

    out = alloc_matrix(iH, iW);
    if (!out) { fprintf(stderr, "alloc out failed\n"); free_matrix(f); free_matrix(g); return 1; }

    // Compute (exclude I/O from timing)
    double t0, t1;

    if (compare) {
        // Serial baseline
        tmp = alloc_matrix(iH, iW); if (!tmp) { fprintf(stderr, "alloc tmp failed\n"); goto cleanup; }
        t0 = now_seconds();
        conv2d_serial(f, iH, iW, g, ikH, ikW, tmp);
        t1 = now_seconds();
        double t_serial = t1 - t0;

        // OpenMP (or serial if not compiled with OMP)
        t0 = now_seconds();
#ifdef _OPENMP
        conv2d_omp(f, iH, iW, g, ikH, ikW, out);
#else
        conv2d_serial(f, iH, iW, g, ikH, ikW, out);
#endif
        t1 = now_seconds();
        double t_omp = t1 - t0;

        float mad = 0.0f;
        for (int i = 0; i < iH; ++i)
            for (int j = 0; j < iW; ++j) {
                float d = fabsf(tmp[i][j] - out[i][j]);
                if (d > mad) mad = d;
            }

        printf("serial: %.6f s\n", t_serial);
        printf("parallel: %.6f s\n", t_omp);
        if (t_omp > 0) printf("speedup: %.2fx\n", t_serial / t_omp);
        printf("max_abs_diff: %.6g\n", mad);
    } else {
        t0 = now_seconds();
        if (mode == MODE_SERIAL) {
            conv2d_serial(f, iH, iW, g, ikH, ikW, out);
        } else {
#ifdef _OPENMP
            conv2d_omp(f, iH, iW, g, ikH, ikW, out);
#else
            conv2d_serial(f, iH, iW, g, ikH, ikW, out);
#endif
        }
        t1 = now_seconds();
        printf("compute_time: %.6f s\n", t1 - t0);
    }

    // Optional output file
    if (opath) {
        if (write_matrix_txt(opath, out, iH, iW) != 0) {
            fprintf(stderr, "failed to write %s\n", opath);
        }
    }

cleanup:
    free_matrix(f);
    free_matrix(g);
    free_matrix(out);
    free_matrix(tmp);
    return 0;
}

