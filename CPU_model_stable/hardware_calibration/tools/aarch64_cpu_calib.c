#define _GNU_SOURCE

#include <arm_neon.h>
#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile uint64_t g_sink_u64;
static volatile double g_sink_f64;
static volatile float g_sink_f32;

static uint64_t parse_u64(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "invalid integer: %s\n", text);
        exit(2);
    }
    return (uint64_t)value;
}

static void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        exit(2);
    }
}

static double monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void shuffle_u64(uint64_t *values, uint64_t n, uint64_t seed) {
    if (n == 0) {
        return;
    }
    for (uint64_t i = n - 1; i > 0; --i) {
        seed = seed * 6364136223846793005ULL + 1ULL;
        {
            uint64_t j = seed % (i + 1ULL);
            uint64_t tmp = values[i];
            values[i] = values[j];
            values[j] = tmp;
        }
    }
}

static void run_ptrchase(uint64_t size_bytes, uint64_t rounds, int cpu) {
    const uint64_t line_bytes = 64;
    const uint64_t node_count = size_bytes / line_bytes;
    if (size_bytes < 4096 || (size_bytes % line_bytes) != 0 || node_count < 64) {
        fprintf(stderr, "size_bytes must be >= 4096 and aligned to 64 bytes\n");
        exit(2);
    }

    pin_cpu(cpu);

    /* Use a real 64-byte stride per node so the working set matches size_bytes. */
    uint8_t *buffer = NULL;
    if (posix_memalign((void **)&buffer, line_bytes, size_bytes) != 0) {
        perror("posix_memalign");
        exit(2);
    }
    memset(buffer, 0, size_bytes);

    uint64_t *order = malloc(node_count * sizeof(uint64_t));
    if (!order) {
        perror("malloc");
        free(buffer);
        exit(2);
    }
    for (uint64_t i = 0; i < node_count; ++i) {
        order[i] = i;
    }
    shuffle_u64(order, node_count, 1ULL);
    for (uint64_t i = 0; i + 1 < node_count; ++i) {
        *(uint64_t *)(buffer + order[i] * line_bytes) = order[i + 1];
    }
    *(uint64_t *)(buffer + order[node_count - 1] * line_bytes) = order[0];

    volatile uint64_t idx = order[0];
    const uint64_t total_loads = node_count * rounds;
    uint64_t i = 0;
    for (i = 0; i + 8 <= node_count; i += 8) {
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
    }
    for (; i < node_count; ++i) {
        idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
    }

    {
        double t0 = monotonic_ns();
        for (uint64_t r = 0; r < rounds; ++r) {
            uint64_t j = 0;
            for (j = 0; j + 8 <= node_count; j += 8) {
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
            }
            for (; j < node_count; ++j) {
                idx = *(volatile uint64_t *)(buffer + idx * line_bytes);
            }
        }
        {
            double t1 = monotonic_ns();
            g_sink_u64 = idx;
            printf(
                "mode=ptrchase cpu=%d size_bytes=%" PRIu64 " node_count=%" PRIu64
                " total_loads=%" PRIu64 " total_ns=%.0f ns_per_load=%.4f\n",
                cpu,
                size_bytes,
                node_count,
                total_loads,
                t1 - t0,
                (t1 - t0) / (double)total_loads
            );
        }
    }

    free(order);
    free(buffer);
}

static void run_issue_loop(uint64_t iters, int cpu) {
    pin_cpu(cpu);
    {
        uint64_t original_iters = iters;
        register uint64_t a0 = 1, a1 = 2, a2 = 3, a3 = 4;
        register uint64_t a4 = 5, a5 = 6, a6 = 7, a7 = 8;
        register uint64_t a8 = 9, a9 = 10, a10 = 11, a11 = 12;
        asm volatile(
            "1:\n"
            "add %[a0], %[a0], #1\n"
            "add %[a1], %[a1], #1\n"
            "add %[a2], %[a2], #1\n"
            "add %[a3], %[a3], #1\n"
            "add %[a4], %[a4], #1\n"
            "add %[a5], %[a5], #1\n"
            "add %[a6], %[a6], #1\n"
            "add %[a7], %[a7], #1\n"
            "add %[a8], %[a8], #1\n"
            "add %[a9], %[a9], #1\n"
            "add %[a10], %[a10], #1\n"
            "add %[a11], %[a11], #1\n"
            "add %[a0], %[a0], #1\n"
            "add %[a1], %[a1], #1\n"
            "add %[a2], %[a2], #1\n"
            "add %[a3], %[a3], #1\n"
            "add %[a4], %[a4], #1\n"
            "add %[a5], %[a5], #1\n"
            "add %[a6], %[a6], #1\n"
            "add %[a7], %[a7], #1\n"
            "add %[a8], %[a8], #1\n"
            "add %[a9], %[a9], #1\n"
            "add %[a10], %[a10], #1\n"
            "add %[a11], %[a11], #1\n"
            "add %[a0], %[a0], #1\n"
            "add %[a1], %[a1], #1\n"
            "add %[a2], %[a2], #1\n"
            "add %[a3], %[a3], #1\n"
            "add %[a4], %[a4], #1\n"
            "add %[a5], %[a5], #1\n"
            "add %[a6], %[a6], #1\n"
            "add %[a7], %[a7], #1\n"
            "add %[a8], %[a8], #1\n"
            "add %[a9], %[a9], #1\n"
            "add %[a10], %[a10], #1\n"
            "add %[a11], %[a11], #1\n"
            "add %[a0], %[a0], #1\n"
            "add %[a1], %[a1], #1\n"
            "add %[a2], %[a2], #1\n"
            "add %[a3], %[a3], #1\n"
            "add %[a4], %[a4], #1\n"
            "add %[a5], %[a5], #1\n"
            "add %[a6], %[a6], #1\n"
            "add %[a7], %[a7], #1\n"
            "add %[a8], %[a8], #1\n"
            "add %[a9], %[a9], #1\n"
            "add %[a10], %[a10], #1\n"
            "add %[a11], %[a11], #1\n"
            "subs %[iters], %[iters], #1\n"
            "b.ne 1b\n"
            : [iters] "+r"(iters),
              [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3),
              [a4] "+r"(a4), [a5] "+r"(a5), [a6] "+r"(a6), [a7] "+r"(a7),
              [a8] "+r"(a8), [a9] "+r"(a9), [a10] "+r"(a10), [a11] "+r"(a11)
            :
            : "cc"
        );
        g_sink_u64 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;
        printf("mode=issue cpu=%d adds_per_iter=48 iterations=%" PRIu64 "\n", cpu, original_iters);
    }
}

static void run_scalar_chain(const char *op, uint64_t iters, int cpu) {
    pin_cpu(cpu);
    {
        uint64_t original_iters = iters;
        register double a = 1.0;
        register double b = 2.0;
        register double c = 3.0;
        if (strcmp(op, "fadd64") == 0) {
            asm volatile(
                "1:\n"
                ".rept 32\n"
                "fadd %d[a], %d[a], %d[b]\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters), [a] "+w"(a)
                : [b] "w"(b)
                : "cc");
        } else if (strcmp(op, "fmul64") == 0) {
            asm volatile(
                "1:\n"
                ".rept 32\n"
                "fmul %d[a], %d[a], %d[b]\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters), [a] "+w"(a)
                : [b] "w"(b)
                : "cc");
        } else if (strcmp(op, "fma64") == 0) {
            asm volatile(
                "1:\n"
                ".rept 32\n"
                "fmadd %d[a], %d[a], %d[b], %d[c]\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters), [a] "+w"(a)
                : [b] "w"(b), [c] "w"(c)
                : "cc");
        } else {
            fprintf(stderr, "unknown scalar op: %s\n", op);
            exit(2);
        }
        g_sink_f64 = a;
        printf("mode=scalar_chain cpu=%d op=%s ops_per_iter=32 iterations=%" PRIu64 "\n", cpu, op, original_iters);
    }
}

static void run_neon_chain(const char *op, uint64_t iters, int cpu) {
    pin_cpu(cpu);
    {
        uint64_t original_iters = iters;
        register float32x4_t a asm("v0") = vdupq_n_f32(1.0f);
        register float32x4_t b asm("v1") = vdupq_n_f32(2.0f);
        register float32x4_t c asm("v2") = vdupq_n_f32(3.0f);
        if (strcmp(op, "fadd4s") == 0) {
            asm volatile(
                "1:\n"
                ".rept 32\n"
                "fadd v0.4s, v0.4s, v1.4s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters), "+w"(a)
                : "w"(b)
                : "cc");
        } else if (strcmp(op, "fmul4s") == 0) {
            asm volatile(
                "1:\n"
                ".rept 32\n"
                "fmul v0.4s, v0.4s, v1.4s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters), "+w"(a)
                : "w"(b)
                : "cc");
        } else if (strcmp(op, "fma4s") == 0) {
            asm volatile(
                "1:\n"
                ".rept 32\n"
                "fmla v0.4s, v1.4s, v2.4s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters), "+w"(a)
                : "w"(b), "w"(c)
                : "cc");
        } else {
            fprintf(stderr, "unknown NEON op: %s\n", op);
            exit(2);
        }
        {
            float lane0 = vgetq_lane_f32(a, 0);
            g_sink_f32 = lane0;
        }
        printf("mode=neon_chain cpu=%d op=%s ops_per_iter=32 iterations=%" PRIu64 "\n", cpu, op, original_iters);
    }
}

static void usage(const char *argv0) {
    fprintf(
        stderr,
        "usage:\n"
        "  %s ptrchase <cpu> <size_bytes> <rounds>\n"
        "  %s issue <cpu> <iterations>\n"
        "  %s scalar_chain <cpu> <fadd64|fmul64|fma64> <iterations>\n"
        "  %s neon_chain <cpu> <fadd4s|fmul4s|fma4s> <iterations>\n",
        argv0, argv0, argv0, argv0
    );
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "ptrchase") == 0) {
        if (argc != 5) {
            usage(argv[0]);
            return 2;
        }
        run_ptrchase(parse_u64(argv[3]), parse_u64(argv[4]), (int)parse_u64(argv[2]));
        return 0;
    }
    if (strcmp(argv[1], "issue") == 0) {
        if (argc != 4) {
            usage(argv[0]);
            return 2;
        }
        run_issue_loop(parse_u64(argv[3]), (int)parse_u64(argv[2]));
        return 0;
    }
    if (strcmp(argv[1], "scalar_chain") == 0) {
        if (argc != 5) {
            usage(argv[0]);
            return 2;
        }
        run_scalar_chain(argv[3], parse_u64(argv[4]), (int)parse_u64(argv[2]));
        return 0;
    }
    if (strcmp(argv[1], "neon_chain") == 0) {
        if (argc != 5) {
            usage(argv[0]);
            return 2;
        }
        run_neon_chain(argv[3], parse_u64(argv[4]), (int)parse_u64(argv[2]));
        return 0;
    }
    usage(argv[0]);
    return 2;
}
