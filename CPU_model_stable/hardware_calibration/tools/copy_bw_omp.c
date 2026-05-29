#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double run_copy(float* dst, const float* src, size_t elements, int iterations) {
  double best = 0.0;
  for (int it = 0; it < iterations; ++it) {
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < elements; ++i) {
      dst[i] = src[i];
    }
    double elapsed = omp_get_wtime() - t0;
    if (elapsed > 0.0) {
      double gb_s = (2.0 * (double)elements * (double)sizeof(float)) / elapsed / 1e9;
      if (gb_s > best) {
        best = gb_s;
      }
    }
  }
  return best;
}

int main(int argc, char** argv) {
  size_t elements = (argc > 1) ? strtoull(argv[1], NULL, 10) : 50000000ULL;
  int iterations = (argc > 2) ? atoi(argv[2]) : 5;
  float* src = NULL;
  float* dst = NULL;
  if (posix_memalign((void**)&src, 64, elements * sizeof(float)) != 0) {
    return 2;
  }
  if (posix_memalign((void**)&dst, 64, elements * sizeof(float)) != 0) {
    free(src);
    return 3;
  }
  for (size_t i = 0; i < elements; ++i) {
    src[i] = (float)(i % 1024);
    dst[i] = 0.0f;
  }
  double best = run_copy(dst, src, elements, iterations);
  printf("elements=%zu iterations=%d threads=%d best_gb_s=%.6f\n", elements, iterations, omp_get_max_threads(), best);
  free(src);
  free(dst);
  return 0;
}
