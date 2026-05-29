#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t lcg32(uint32_t x) {
  return x * 1664525u + 1013904223u;
}

int main(int argc, char** argv) {
  size_t elements = (argc > 1) ? strtoull(argv[1], NULL, 10) : 25000000ULL;
  int iterations = (argc > 2) ? atoi(argv[2]) : 5;
  float* values = NULL;
  uint32_t* index = NULL;
  if (posix_memalign((void**)&values, 64, elements * sizeof(float)) != 0) {
    return 2;
  }
  if (posix_memalign((void**)&index, 64, elements * sizeof(uint32_t)) != 0) {
    free(values);
    return 3;
  }
  uint32_t seed = 1u;
  for (size_t i = 0; i < elements; ++i) {
    values[i] = (float)(i % 1024);
    seed = lcg32(seed);
    index[i] = seed % (uint32_t)elements;
  }

  volatile float sink = 0.0f;
  double best = 0.0;
  for (int it = 0; it < iterations; ++it) {
    double t0 = omp_get_wtime();
#pragma omp parallel
    {
      float local = 0.0f;
#pragma omp for schedule(static)
      for (size_t i = 0; i < elements; ++i) {
        local += values[index[i]];
      }
#pragma omp atomic
      sink += local;
    }
    double elapsed = omp_get_wtime() - t0;
    if (elapsed > 0.0) {
      double touched = (double)elements * (double)(sizeof(float) + sizeof(uint32_t));
      double gb_s = touched / elapsed / 1e9;
      if (gb_s > best) {
        best = gb_s;
      }
    }
  }
  printf("elements=%zu iterations=%d threads=%d best_gb_s=%.6f sink=%f\n", elements, iterations, omp_get_max_threads(), best, sink);
  free(values);
  free(index);
  return 0;
}
