#!/usr/bin/env python3
"""List the scalars a Neuronte bundle was trained on: its range table.

    scalars.py FILE.nrt [--net N] > scalars.txt

One U+XXXX per line, ascending, for neuronte_gpu_bench --scalars. A drawing
file's bundle is found inside it. --net keeps the scalars of one member. A
file with no table (one net) lists nothing.
"""

import argparse
import struct


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def bundle(b):
    ver = u32(b, 8)
    if ver == 2:
        return b
    if ver == 5:
        for i in range(u32(b, 12)):
            off, n = u32(b, 16 + 8 * i), u32(b, 20 + 8 * i)
            if u32(b, off + 8) == 2:
                return b[off:off + n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--net", type=int)
    args = ap.parse_args()
    b = bundle(open(args.file, "rb").read())
    if b is None:
        return
    k, r = u32(b, 12), u32(b, 16)
    for i in range(r):
        first, last, net = struct.unpack_from("<III", b, 28 + 4 * k + 12 * i)
        if args.net is None or args.net == net:
            for s in range(first, last + 1):
                print(f"U+{s:04X}")


if __name__ == "__main__":
    main()
