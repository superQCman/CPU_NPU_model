#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t lcg_next(uint64_t x) {
  return x * 6364136223846793005ULL + 1442695040888963407ULL;
}

static double run_slice_copy(uint8_t* dst,
                             const uint8_t* src,
                             const size_t* indices,
                             size_t block_size,
                             size_t slice_count,
                             size_t source_slice_count,
                             int iterations) {
  double best_write_gb_s = 0.0;
  for (int it = 0; it < iterations; ++it) {
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < slice_count; ++i) {
      const size_t src_index = indices[i] % source_slice_count;
      memcpy(dst + i * block_size, src + src_index * block_size, block_size);
    }
    double elapsed = omp_get_wtime() - t0;
    if (elapsed > 0.0) {
      const double write_gb_s = ((double)slice_count * (double)block_size) / elapsed / 1e9;
      if (write_gb_s > best_write_gb_s) {
        best_write_gb_s = write_gb_s;
      }
    }
  }
  return best_write_gb_s;
}

int main(int argc, char** argv) {
  const size_t block_size = (argc > 1) ? strtoull(argv[1], NULL, 10) : 800ULL;
  const size_t slice_count = (argc > 2) ? strtoull(argv[2], NULL, 10) : 1000000ULL;
  const size_t source_slice_count = (argc > 3) ? strtoull(argv[3], NULL, 10) : slice_count;
  const int iterations = (argc > 4) ? atoi(argv[4]) : 5;
  if (block_size == 0 || slice_count == 0 || source_slice_count == 0 || iterations <= 0) {
    return 2;
  }

  uint8_t* src = NULL;
  uint8_t* dst = NULL;
  size_t* indices = NULL;
  if (posix_memalign((void**)&src, 64, source_slice_count * block_size) != 0) {
    return 3;
  }
  if (posix_memalign((void**)&dst, 64, slice_count * block_size) != 0) {
    free(src);
    return 4;
  }
  if (posix_memalign((void**)&indices, 64, slice_count * sizeof(size_t)) != 0) {
    free(src);
    free(dst);
    return 5;
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < source_slice_count * block_size; ++i) {
    src[i] = (uint8_t)(i & 0xffU);
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < slice_count * block_size; ++i) {
    dst[i] = 0;
  }

  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (size_t i = 0; i < slice_count; ++i) {
    state = lcg_next(state);
    indices[i] = (size_t)(state % source_slice_count);
  }

  const double write_gb_s = run_slice_copy(dst, src, indices, block_size, slice_count, source_slice_count, iterations);
  printf("block_size=%zu slices=%zu source_slices=%zu iterations=%d threads=%d write_gb_s=%.6f rw_gb_s=%.6f\n",
         block_size,
         slice_count,
         source_slice_count,
         iterations,
         omp_get_max_threads(),
         write_gb_s,
         2.0 * write_gb_s);
  free(src);
  free(dst);
  free(indices);
  return 0;
}
