#!/usr/bin/env python3
import argparse
import math


def generate_sizes(lo, hi, points, phase):
    base = []
    for i in range(points):
        x = math.log2(lo) + (i + phase) * (math.log2(hi) - math.log2(lo)) / points
        s = int(round((2 ** x) / 4096.0) * 4096)
        s = max(lo, min(hi, s))
        base.append(s)

    sizes = set()
    for s in base:
        a = (s // 131072) * 131072
        if a < 131072:
            a = 131072

        g1 = a
        g2 = a + 32768
        g3 = a + 4096
        for v in (g1, g2, g3):
            if lo <= v <= hi:
                sizes.add(int(v))

    return sorted(sizes)


def main():
    parser = argparse.ArgumentParser(description="Generate message sizes covering Group1/2/3.")
    parser.add_argument("--lo", type=int, default=4 * 1024)
    parser.add_argument("--hi", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--points", type=int, default=84)
    parser.add_argument("--phase", type=float, default=0.0)
    args = parser.parse_args()

    sizes = generate_sizes(args.lo, args.hi, args.points, args.phase)
    for s in sizes:
        print(s)


if __name__ == "__main__":
    main()
