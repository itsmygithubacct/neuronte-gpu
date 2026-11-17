#!/usr/bin/env python3
"""Compare a GPU backend's coverage with the reference CPU library's, byte for byte.

    exact.py --bench build/neuronte_gpu_bench --backend cuda [--device NAME]
             [--cells 8x16,12x24,16x32] [--untrained 0,1] [--fast] [--out FILE.jsonl] FILE.nrt ...

For each weight file, cell size and untrained mode, the scalars are the
file's own table of trained characters (tools/scalars.py; printable ASCII
when it has none) and a few it was not trained on, so the fallback net, the
replacement character and the untrained mark are compared too. The bench's
"cpu" backend is the reference library itself. The GPU draws the set twice;
both calls must give the same bytes. One line per comparison: identical, or
how many glyphs differ and the largest difference.
"""

import argparse
import array
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# not trained by any file, a private-use scalar, a surrogate and a value past the last scalar (both
# become the replacement character), the replacement character itself, and common ones
EXTRA = [0x378, 0xE000, 0x10FFFD, 0xD800, 0x110000, 0xFFFD, 0x20, 0x41, 0x2800, 0x28FF, 0x2588, 0x1F311]


def scalars_of(path):
    out = subprocess.run([sys.executable, os.path.join(HERE, "scalars.py"), path], capture_output=True, text=True,
                         check=True).stdout.split()
    table = [int(s[2:], 16) for s in out] or list(range(0x20, 0x7F))
    seen = set(table)
    return table + [s for s in EXTRA if s not in seen]


def run(bench, nrt, cell, backend, device, untrained, snap, fast, listing, raster):
    cmd = [bench, "--nrt", nrt, "--cell", cell, "--backend", backend, "--scalars", listing, "--repeats", "1",
           "--warmup", "0", "--passes", "1", "--coverage-out", raster, "--max-batch", "4096"]
    if backend != "cpu":
        cmd += ["--no-fallback"]
        if device:
            cmd += ["--device", device]
        if fast:
            cmd += ["--fast"]
    if untrained is not None:
        cmd += ["--untrained", str(untrained)]
    if snap is not None:
        cmd += ["--snap", str(snap)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except ValueError:
        d = {"error": (p.stderr or "no output").strip()[-300:]}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="build/neuronte_gpu_bench")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--device")
    ap.add_argument("--cells", default="8x16,12x24,16x32")
    ap.add_argument("--untrained", default="0,1")
    ap.add_argument("--snap", type=int)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    log = open(args.out, "a") if args.out else None
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        listing, ref, got = (os.path.join(tmp, n) for n in ("scalars.txt", "cpu.f32", "gpu.f32"))
        for nrt in args.files:
            scalars = scalars_of(nrt)
            with open(listing, "w") as f:
                f.write("".join(f"0x{s:X}\n" for s in scalars))
            for cell in args.cells.split(","):
                w, h = (int(v) for v in cell.split("x"))
                for mode in args.untrained.split(","):
                    untrained = int(mode)
                    c = run(args.bench, nrt, cell, "cpu", None, untrained, args.snap, False, listing, ref)
                    g = run(args.bench, nrt, cell, args.backend, args.device, untrained, args.snap, args.fast,
                            listing, got)
                    rec = {"file": os.path.basename(nrt), "cell": cell, "untrained": untrained,
                           "scalars": len(scalars), "backend": g.get("backend"), "device": g.get("device"),
                           "samples": g.get("samples")}
                    err = c.get("error") or g.get("error")
                    if err or not c.get("coverage_written") or not g.get("coverage_written"):
                        rec["result"] = "error: " + (err or "coverage not written")
                        bad += 1
                    else:
                        a, b = open(ref, "rb").read(), open(got, "rb").read()
                        st = g["rounds"][0]["stages"]
                        rec.update(cpu_glyphs=st["cpu_glyphs"], marks=st["marks"], warped=st["warped"])
                        if not g.get("first_call_equals_last", False):
                            # the same scalars twice must give the same bytes: the first call counts too
                            rec["result"] = "differs between the first call and the second"
                            bad += 1
                        elif a == b and len(a) == len(scalars) * w * h * 4:
                            rec["result"] = "identical"
                        else:
                            fa, fb = array.array("f"), array.array("f")
                            fa.frombytes(a)
                            fb.frombytes(b[:len(a)])
                            plane = w * h
                            worst, glyphs = 0.0, 0
                            for i in range(0, len(fa), plane):
                                if a[i * 4:(i + plane) * 4] != b[i * 4:(i + plane) * 4]:
                                    glyphs += 1
                                    worst = max(worst, max(abs(x - y) for x, y in zip(fa[i:i + plane],
                                                                                    fb[i:i + plane])))
                            rec.update(result="differs", glyphs_differing=glyphs, largest_difference=worst)
                            if not args.fast:
                                bad += 1
                    print(f"{rec['file']:36s} {cell:7s} untrained={untrained} n={len(scalars):5d} "
                          f"{str(rec.get('backend')):11s} {rec['result']}"
                          + (f" ({rec['glyphs_differing']} glyphs, largest {rec['largest_difference']:.3g})"
                             if "glyphs_differing" in rec else "")
                          + (f" [library drew {rec.get('cpu_glyphs', 0)} glyphs, {rec.get('marks', 0)} marks; "
                             f"{rec.get('warped', 0)} warped]" if "cpu_glyphs" in rec else ""), flush=True)
                    if log:
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
