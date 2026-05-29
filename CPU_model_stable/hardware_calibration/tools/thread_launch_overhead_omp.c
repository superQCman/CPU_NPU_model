#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double parallel_for_empty(int iterations) {
  double t0 = omp_get_wtime();
  for (int it = 0; it < iterations; ++it) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < omp_get_max_threads(); ++i) {
    }
  }
  return (omp_get_wtime() - t0) * 1e6 / (double)iterations;
}

static double task_spawn_empty(int iterations) {
  double total = 0.0;
  for (int it = 0; it < iterations; ++it) {
    double t0 = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
      {
        for (int i = 0; i < omp_get_max_threads(); ++i) {
#pragma omp task
          {
          }
        }
#pragma omp taskwait
      }
    }
    total += (omp_get_wtime() - t0) * 1e6;
  }
  return total / (double)iterations;
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "parallel_for_empty";
  int iterations = (argc > 2) ? atoi(argv[2]) : 10000;
  double avg_launch_us = 0.0;
  if (strcmp(mode, "parallel_for_empty") == 0) {
    avg_launch_us = parallel_for_empty(iterations);
  } else if (strcmp(mode, "task_spawn_empty") == 0) {
    avg_launch_us = task_spawn_empty(iterations);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    return 2;
  }
  printf("mode=%s iterations=%d threads=%d avg_launch_us=%.6f\n", mode, iterations, omp_get_max_threads(), avg_launch_us);
  return 0;
}
