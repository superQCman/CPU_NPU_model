#define _GNU_SOURCE

#include <arm_neon.h>
#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile float g_sink_f32;

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

static void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        exit(2);
    }
}

static void run_neon_throughput(const char *op, uint64_t iters, int cpu) {
    const uint64_t vector_insts_per_iter = 64;
    pin_cpu(cpu);

    {
        uint64_t original_iters = iters;
        register float32x4_t a0 asm("v0") = vdupq_n_f32(1.0f);
        register float32x4_t a1 asm("v1") = vdupq_n_f32(1.0f);
        register float32x4_t a2 asm("v2") = vdupq_n_f32(1.0f);
        register float32x4_t a3 asm("v3") = vdupq_n_f32(1.0f);
        register float32x4_t a4 asm("v4") = vdupq_n_f32(1.0f);
        register float32x4_t a5 asm("v5") = vdupq_n_f32(1.0f);
        register float32x4_t a6 asm("v6") = vdupq_n_f32(1.0f);
        register float32x4_t a7 asm("v7") = vdupq_n_f32(1.0f);
        register float32x4_t b asm("v16") = vdupq_n_f32(2.0f);
        register float32x4_t c asm("v17") = vdupq_n_f32(3.0f);

        if (strcmp(op, "fadd4s") == 0) {
            asm volatile(
                "1:\n"
                ".rept 8\n"
                "fadd v0.4s, v0.4s, v16.4s\n"
                "fadd v1.4s, v1.4s, v16.4s\n"
                "fadd v2.4s, v2.4s, v16.4s\n"
                "fadd v3.4s, v3.4s, v16.4s\n"
                "fadd v4.4s, v4.4s, v16.4s\n"
                "fadd v5.4s, v5.4s, v16.4s\n"
                "fadd v6.4s, v6.4s, v16.4s\n"
                "fadd v7.4s, v7.4s, v16.4s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters),
                  "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                  "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7)
                : "w"(b)
                : "cc"
            );
        } else if (strcmp(op, "fmul4s") == 0) {
            asm volatile(
                "1:\n"
                ".rept 8\n"
                "fmul v0.4s, v0.4s, v16.4s\n"
                "fmul v1.4s, v1.4s, v16.4s\n"
                "fmul v2.4s, v2.4s, v16.4s\n"
                "fmul v3.4s, v3.4s, v16.4s\n"
                "fmul v4.4s, v4.4s, v16.4s\n"
                "fmul v5.4s, v5.4s, v16.4s\n"
                "fmul v6.4s, v6.4s, v16.4s\n"
                "fmul v7.4s, v7.4s, v16.4s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters),
                  "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                  "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7)
                : "w"(b)
                : "cc"
            );
        } else if (strcmp(op, "fma4s") == 0) {
            asm volatile(
                "1:\n"
                ".rept 8\n"
                "fmla v0.4s, v16.4s, v17.4s\n"
                "fmla v1.4s, v16.4s, v17.4s\n"
                "fmla v2.4s, v16.4s, v17.4s\n"
                "fmla v3.4s, v16.4s, v17.4s\n"
                "fmla v4.4s, v16.4s, v17.4s\n"
                "fmla v5.4s, v16.4s, v17.4s\n"
                "fmla v6.4s, v16.4s, v17.4s\n"
                "fmla v7.4s, v16.4s, v17.4s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters),
                  "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                  "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7)
                : "w"(b), "w"(c)
                : "cc"
            );
        } else {
            fprintf(stderr, "unknown NEON throughput op: %s\n", op);
            exit(2);
        }

        {
            float lane0 = vgetq_lane_f32(a0, 0) + vgetq_lane_f32(a1, 0) + vgetq_lane_f32(a2, 0) + vgetq_lane_f32(a3, 0);
            lane0 += vgetq_lane_f32(a4, 0) + vgetq_lane_f32(a5, 0) + vgetq_lane_f32(a6, 0) + vgetq_lane_f32(a7, 0);
            g_sink_f32 = lane0;
        }

        printf(
            "mode=neon_throughput cpu=%d op=%s vector_insts_per_iter=%" PRIu64 " iterations=%" PRIu64 "\n",
            cpu,
            op,
            vector_insts_per_iter,
            original_iters
        );
    }
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s <cpu> <fadd4s|fmul4s|fma4s> <iterations>\n", argv0);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        usage(argv[0]);
        return 2;
    }
    run_neon_throughput(argv[2], parse_u64(argv[3]), (int)parse_u64(argv[1]));
    return 0;
}
