#!/usr/bin/env python3
"""Describe Neuronte weight files: kind, members, hidden widths, tables.

    nrtinfo.py FILE...

One line per image, indented by nesting. Reads the format only; needs no
library. Useful for seeing what a file asks of a GPU backend.
"""

import struct
import sys

HEADER = 8 + 5 * 4 + 4 + 43 * 4
COLOUR_FIXED = 420


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def f32(b, o):
    return struct.unpack_from("<f", b, o)[0]


def describe(b, depth, out):
    pad = "  " * depth
    if len(b) < 12 or b[:8] != b"NEURONTE":
        out.append(f"{pad}not a Neuronte image ({len(b)} bytes)")
        return
    ver = u32(b, 8)
    if ver == 1:
        out.append(f"{pad}v1 distance net H={u32(b, 20)} tau={f32(b, 28):.6g} bytes={len(b)}")
    elif ver == 2:
        k, r, fb, g = u32(b, 12), u32(b, 16), u32(b, 20), u32(b, 24)
        off = 28 + 4 * k
        ranges = [struct.unpack_from("<III", b, off + 12 * i) for i in range(r)]
        scalars = sum(last - first + 1 for first, last, _ in ranges)
        per = [0] * k
        for first, last, net in ranges:
            per[net] += last - first + 1
        out.append(f"{pad}v2 bundle nets={k} ranges={r} scalars={scalars} fallback={fb} gate_h={g} "
                   f"bytes={len(b)} scalars/net={per}")
        off += 12 * r
        if g:
            off += 4 * (43 + g * 64 + g + k * g + k)
        for i in range(k):
            n = u32(b, 28 + 4 * i)
            describe(b[off:off + n], depth + 1, out)
            off += n
    elif ver == 3:
        h, k, r, e, snap = u32(b, 20), u32(b, 32), u32(b, 44), u32(b, 48), u32(b, 52)
        out.append(f"{pad}v3 colour net H={h} K={k} R={r} edge-records={e} snap={snap} "
                   f"tau={f32(b, 28):.6g} bytes={len(b)}")
    elif ver == 4:
        out.append(f"{pad}v4 advance table records={u32(b, 24)} fallback=U+{u32(b, 28):04X} bytes={len(b)}")
    elif ver == 5:
        m = u32(b, 12)
        out.append(f"{pad}v5 drawing file members={m} bytes={len(b)}")
        for i in range(m):
            off, n = u32(b, 16 + 8 * i), u32(b, 16 + 8 * i + 4)
            describe(b[off:off + n], depth + 1, out)
    elif ver == 6:
        c, r, m = u32(b, 12), u32(b, 16), u32(b, 20)
        prefix = 32 + 4 * (c + 1 + r + 1) + 16 * c * r + 4 * m
        xe = [f32(b, 32 + 4 * i) for i in range(c + 1)]
        ye = [f32(b, 32 + 4 * (c + 1) + 4 * i) for i in range(r + 1)]
        out.append(f"{pad}v6 tiled net {c}x{r} nets={m} U+{u32(b, 24):04X}..U+{u32(b, 28):04X} "
                   f"xe={xe} ye={ye} bytes={len(b)}")
        off = prefix
        for i in range(m):
            n = u32(b, prefix - 4 * m + 4 * i)
            describe(b[off:off + n], depth + 1, out)
            off += n
    elif ver == 7:
        out.append(f"{pad}v7 letter edge table snap={u32(b, 12)} records={u32(b, 16)} bytes={len(b)}")
    else:
        out.append(f"{pad}unknown version {ver}")


def main():
    for path in sys.argv[1:]:
        out = []
        describe(open(path, "rb").read(), 1, out)
        print(path)
        print("\n".join(out))


if __name__ == "__main__":
    main()
