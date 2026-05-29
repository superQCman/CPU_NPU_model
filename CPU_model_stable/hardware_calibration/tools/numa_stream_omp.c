#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint64_t parse_u64(const char *text) {
    char *end = NULL;
    errno = 0;
    {
        unsigned long long value = strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0') {
            fprintf(stderr, "invalid integer: %s\n", text);
            exit(2);
        }
        return (uint64_t)value;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <elements_per_array> <repetitions> <scalar>\n", argv[0]);
        return 2;
    }

    {
        const uint64_t n = parse_u64(argv[1]);
        const uint64_t reps = parse_u64(argv[2]);
        const double scalar = atof(argv[3]);
        double *a = NULL;
        double *b = NULL;
        double *c = NULL;
        double best_gb_s = 0.0;
        double bytes_moved_per_rep = 3.0 * (double)n * (double)sizeof(double);

        if (posix_memalign((void **)&a, 64, n * sizeof(double)) != 0 ||
            posix_memalign((void **)&b, 64, n * sizeof(double)) != 0 ||
            posix_memalign((void **)&c, 64, n * sizeof(double)) != 0) {
            perror("posix_memalign");
            free(a);
            free(b);
            free(c);
            return 2;
        }

        #pragma omp parallel for simd schedule(static) aligned(a, b, c:64)
        for (uint64_t i = 0; i < n; ++i) {
            a[i] = 1.0;
            b[i] = 2.0;
            c[i] = 0.5;
        }

        for (uint64_t rep = 0; rep < reps; ++rep) {
            double t0 = monotonic_ns();
            double *restrict ar = a;
            double *restrict br = b;
            double *restrict cr = c;
            #pragma omp parallel for simd schedule(static) aligned(ar, br, cr:64)
            for (uint64_t i = 0; i < n; ++i) {
                ar[i] = br[i] + scalar * cr[i];
            }
            {
                double t1 = monotonic_ns();
                double gb_s = bytes_moved_per_rep / (t1 - t0);
                if (gb_s > best_gb_s) {
                    best_gb_s = gb_s;
                }
                printf(
                    "mode=stream_triad rep=%" PRIu64 " elements=%" PRIu64
                    " threads=%d total_ns=%.0f gb_s=%.4f\n",
                    rep,
                    n,
                    omp_get_max_threads(),
                    t1 - t0,
                    gb_s
                );
            }
        }

        printf(
            "mode=stream_triad_summary elements=%" PRIu64 " reps=%" PRIu64
            " threads=%d best_gb_s=%.4f\n",
            n,
            reps,
            omp_get_max_threads(),
            best_gb_s
        );

        free(a);
        free(b);
        free(c);
    }
    return 0;
}
