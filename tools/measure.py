#!/usr/bin/env python3
"""Time a weight file on a GPU backend against the reference CPU library on the same machine.

    measure.py --bench build/neuronte_gpu_bench --backend cuda [--device NAME] --cells 8x16,12x24,16x32
               [--threads N] [--repeats 10] [--out FILE.jsonl] [--skip-cpu] FILE.nrt ...

For each file and cell size, over the file's whole table of trained
characters (printable ASCII when it has none), end to end:

  one core      the library, one process pinned to a CPU
  all threads   one pinned process per hardware thread at once, rates added
  GPU batch     the whole set in one call
  GPU single    one glyph per call

and what the bench reports of device memory. Where the NVIDIA driver reports
power, it is sampled while the GPU batch runs. Appends one JSON line per
file and cell to --out and prints a table row.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def table(path):
    out = subprocess.run([sys.executable, os.path.join(HERE, "scalars.py"), path], capture_output=True, text=True,
                         check=True).stdout
    return out if out.strip() else "".join(f"U+{s:04X}\n" for s in range(0x20, 0x7F))


def bench(args, extra):
    p = subprocess.run(args + extra, capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"error": (p.stderr or "no output").strip()[-300:]}


def per_glyph_ns(d):
    r = [x["wall"] for x in d["rounds"] if not x["warmup"]]
    return statistics.median(r) / d["glyphs_per_round"]


class Power(threading.Thread):
    """Samples the NVIDIA driver's power reading, in watts, about ten times a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split()
                self.samples.append((time.time(), float(out[0])))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                return
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="build/neuronte_gpu_bench")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--device")
    ap.add_argument("--cells", default="8x16,12x24,16x32")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--cpu-repeats", type=int, default=3)
    ap.add_argument("--single", type=int, default=200)
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    log = open(a.out, "a") if a.out else None
    with tempfile.TemporaryDirectory() as tmp:
        listing = os.path.join(tmp, "scalars.txt")
        for nrt in a.files:
            text = table(nrt)
            n = len(text.split())
            open(listing, "w").write(text)
            for cell in a.cells.split(","):
                base = [a.bench, "--nrt", nrt, "--cell", cell, "--scalars", listing, "--max-batch", str(max(n, 1))]
                rec = {"file": os.path.basename(nrt), "cell": cell, "glyphs": n, "when": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                if not a.skip_cpu:
                    cpu = ["--backend", "cpu", "--warmup", "1", "--passes", "1", "--repeats", str(a.cpu_repeats)]
                    one = bench(base, cpu + ["--cpu", "2"])
                    if "error" in one:
                        print(rec["file"], cell, "cpu:", one["error"])
                        continue
                    rec["cpu_one_ns"] = per_glyph_ns(one)
                    rec["cpu_load"] = one["hardware_start"]["loadavg"]
                    procs = [subprocess.Popen(base + cpu + ["--cpu", str(c)], stdout=subprocess.PIPE, text=True)
                             for c in range(a.threads)]
                    rates = []
                    for p in procs:
                        try:
                            rates.append(1e9 / per_glyph_ns(json.loads(p.communicate()[0])))
                        except (ValueError, KeyError):
                            pass
                    rec["cpu_all_threads"] = len(rates)
                    rec["cpu_all_ns"] = 1e9 / sum(rates) if rates else None
                gpu = ["--backend", a.backend, "--no-fallback", "--warmup", "2", "--passes", "1", "--repeats",
                       str(a.repeats), "--single", str(a.single), "--warmup-ms", "300"]
                if a.device:
                    gpu += ["--device", a.device]
                if a.fast:
                    gpu += ["--fast"]
                power = Power()
                power.start()
                t0 = time.time()
                d = bench(base, gpu)
                t1 = time.time()
                power.stop = True
                power.join(timeout=6)
                if "error" in d:
                    print(rec["file"], cell, a.backend + ":", d["error"])
                    continue
                rounds = [x for x in d["rounds"] if not x["warmup"]]
                rec.update(
                    backend=d["backend"], device=d["device"], versions=d["versions"], samples=d["samples"],
                    exact=d["exact"], open_ms=d["open_ns"] / 1e6, first_call_ms=d["first_call"]["wall"] / 1e6,
                    gpu_batch_ns=per_glyph_ns(d),
                    gpu_kernel_ns=statistics.median(x["stages"]["run"] for x in rounds) / d["glyphs_per_round"],
                    gpu_plan_ns=statistics.median(x["stages"]["plan"] for x in rounds) / d["glyphs_per_round"],
                    gpu_batch_ms=statistics.median(x["wall"] for x in rounds) / 1e6,
                    gpu_single_ns=statistics.median(d["single"]) if d.get("single") else None,
                    library_glyphs=rounds[0]["stages"]["cpu_glyphs"], launches=rounds[0]["stages"]["launches"],
                    memory=d.get("memory_first_call"), battery=d["hardware_start"].get("battery_percent"),
                    ac=d["hardware_start"].get("ac_online"), load=d["hardware_start"]["loadavg"],
                    nvml_start=d["hardware_start"].get("nvml"), nvml_end=d["hardware_end"].get("nvml"))
                # power: samples in the last 60% of the run are the timed rounds and single calls
                late = [w for t, w in power.samples if t > t0 + 0.4 * (t1 - t0)]
                rec["gpu_watts_running"] = statistics.median(late) if late else None
                rec["gpu_watts_samples"] = len(late)
                if log:
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                one, allc = rec.get("cpu_one_ns"), rec.get("cpu_all_ns")
                mem = rec["memory"] or {}
                print(f"{rec['file']:34s} {cell:6s} n={n:5d} {rec['backend']:10s} batch {rec['gpu_batch_ns'] / 1e3:8.2f} us/glyph"
                      f" ({rec['gpu_batch_ms']:7.2f} ms the set) single {(rec['gpu_single_ns'] or 0) / 1e3:8.1f} us"
                      + (f" | cpu one {one / 1e3:8.1f} us, all {allc / 1e3:7.1f} us"
                         f" | x{one / rec['gpu_batch_ns']:.1f} one, x{allc / rec['gpu_batch_ns']:.2f} all" if one and allc else "")
                      + f" | dev mem {mem.get('process_device_bytes', 0) / 2**20:.0f} MiB, buffers {mem.get('buffers', 0) / 2**20:.1f} MiB"
                      + (f" | {rec['gpu_watts_running']:.0f} W" if rec["gpu_watts_running"] else ""), flush=True)


if __name__ == "__main__":
    main()
