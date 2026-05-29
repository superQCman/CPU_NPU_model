#define _GNU_SOURCE

#include <arm_sve.h>
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

static void run_sve_throughput(const char *op, uint64_t iters, int cpu) {
    const uint64_t vector_insts_per_iter = 64;
    pin_cpu(cpu);

    {
        uint64_t original_iters = iters;
        register svfloat32_t a0 asm("z0") = svdup_f32(1.0f);
        register svfloat32_t a1 asm("z1") = svdup_f32(1.0f);
        register svfloat32_t a2 asm("z2") = svdup_f32(1.0f);
        register svfloat32_t a3 asm("z3") = svdup_f32(1.0f);
        register svfloat32_t a4 asm("z4") = svdup_f32(1.0f);
        register svfloat32_t a5 asm("z5") = svdup_f32(1.0f);
        register svfloat32_t a6 asm("z6") = svdup_f32(1.0f);
        register svfloat32_t a7 asm("z7") = svdup_f32(1.0f);
        register svfloat32_t b asm("z16") = svdup_f32(2.0f);
        register svfloat32_t c asm("z17") = svdup_f32(3.0f);
        register svbool_t pg asm("p0") = svptrue_b32();

        if (strcmp(op, "fadds") == 0) {
            asm volatile(
                "1:\n"
                ".rept 8\n"
                "fadd z0.s, p0/m, z0.s, z16.s\n"
                "fadd z1.s, p0/m, z1.s, z16.s\n"
                "fadd z2.s, p0/m, z2.s, z16.s\n"
                "fadd z3.s, p0/m, z3.s, z16.s\n"
                "fadd z4.s, p0/m, z4.s, z16.s\n"
                "fadd z5.s, p0/m, z5.s, z16.s\n"
                "fadd z6.s, p0/m, z6.s, z16.s\n"
                "fadd z7.s, p0/m, z7.s, z16.s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters),
                  "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                  "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7),
                  "+Upl"(pg)
                : "w"(b)
                : "cc"
            );
        } else if (strcmp(op, "fmuls") == 0) {
            asm volatile(
                "1:\n"
                ".rept 8\n"
                "fmul z0.s, p0/m, z0.s, z16.s\n"
                "fmul z1.s, p0/m, z1.s, z16.s\n"
                "fmul z2.s, p0/m, z2.s, z16.s\n"
                "fmul z3.s, p0/m, z3.s, z16.s\n"
                "fmul z4.s, p0/m, z4.s, z16.s\n"
                "fmul z5.s, p0/m, z5.s, z16.s\n"
                "fmul z6.s, p0/m, z6.s, z16.s\n"
                "fmul z7.s, p0/m, z7.s, z16.s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters),
                  "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                  "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7),
                  "+Upl"(pg)
                : "w"(b)
                : "cc"
            );
        } else if (strcmp(op, "fmas") == 0) {
            asm volatile(
                "1:\n"
                ".rept 8\n"
                "fmla z0.s, p0/m, z16.s, z17.s\n"
                "fmla z1.s, p0/m, z16.s, z17.s\n"
                "fmla z2.s, p0/m, z16.s, z17.s\n"
                "fmla z3.s, p0/m, z16.s, z17.s\n"
                "fmla z4.s, p0/m, z16.s, z17.s\n"
                "fmla z5.s, p0/m, z16.s, z17.s\n"
                "fmla z6.s, p0/m, z16.s, z17.s\n"
                "fmla z7.s, p0/m, z16.s, z17.s\n"
                ".endr\n"
                "subs %[iters], %[iters], #1\n"
                "b.ne 1b\n"
                : [iters] "+r"(iters),
                  "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                  "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7),
                  "+Upl"(pg)
                : "w"(b), "w"(c)
                : "cc"
            );
        } else {
            fprintf(stderr, "unknown SVE throughput op: %s\n", op);
            exit(2);
        }

        {
            float lane0 = svlastb_f32(pg, a0) + svlastb_f32(pg, a1) + svlastb_f32(pg, a2) + svlastb_f32(pg, a3);
            lane0 += svlastb_f32(pg, a4) + svlastb_f32(pg, a5) + svlastb_f32(pg, a6) + svlastb_f32(pg, a7);
            g_sink_f32 = lane0;
        }

        printf(
            "mode=sve_throughput cpu=%d op=%s vector_insts_per_iter=%" PRIu64 " iterations=%" PRIu64 "\n",
            cpu,
            op,
            vector_insts_per_iter,
            original_iters
        );
    }
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s <cpu> <fadds|fmuls|fmas> <iterations>\n", argv0);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        usage(argv[0]);
        return 2;
    }
    run_sve_throughput(argv[2], parse_u64(argv[3]), (int)parse_u64(argv[1]));
    return 0;
}
