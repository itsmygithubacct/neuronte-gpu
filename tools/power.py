#!/usr/bin/env python3
"""Power of an NVIDIA GPU while neuronte-gpu draws, sampled through the driver.

    power.py --lib build/libneuronte_gpu.so --nrt FILE.nrt --scalars LIST --out FILE.json
             [--cells 8x16,16x32] [--sustain 60] [--bursts 30] [--burst-glyphs 3] [--burst-gap 0.5]

Phases, each sampled every 0.2 s (watts, utilisation, temperature) and
bracketed by the driver's energy counter where it has one:
  idle           before anything is opened
  open-idle      handles open, nothing drawn
  fill WxH       one call that draws the whole list, then five seconds after
  bursts         a terminal's pattern: a few new glyphs, a gap, again
  sustain        whole-list calls back to back
  after          ten seconds after the last call
and then the same fills, bursts and a shorter sustained run on the CPU
library (through the same handle type, backend "cpu"), timed. Energy above
idle is the counter's difference less idle watts times the time.
"""
import argparse, ctypes, json, os, threading, time


class Options(ctypes.Structure):
    _fields_ = [('layout_path', ctypes.c_char_p), ('model_dir', ctypes.c_char_p), ('nrt_path', ctypes.c_char_p),
                ('cell_w', ctypes.c_int), ('cell_h', ctypes.c_int), ('samples', ctypes.c_int), ('backend', ctypes.c_int),
                ('device', ctypes.c_char_p), ('flags', ctypes.c_int), ('max_batch', ctypes.c_int), ('block', ctypes.c_int)]


class Nvml:
    def __init__(self):
        self.n = ctypes.CDLL('libnvidia-ml.so.1')
        self.n.nvmlInit_v2()
        self.h = ctypes.c_void_p()
        self.n.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h))

    def watts(self):
        v = ctypes.c_uint()
        return v.value / 1000.0 if self.n.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(v)) == 0 else None

    def joules(self):
        v = ctypes.c_ulonglong()
        return v.value / 1000.0 if self.n.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(v)) == 0 else None

    def util(self):
        v = (ctypes.c_uint * 2)()
        return v[0] if self.n.nvmlDeviceGetUtilizationRates(self.h, v) == 0 else None

    def temp(self):
        v = ctypes.c_uint()
        return v.value if self.n.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(v)) == 0 else None


def battery():
    def read(name):
        try:
            return open(name).read().strip()
        except OSError:
            return None
    ac = [read(os.path.join('/sys/class/power_supply', d, 'online')) for d in sorted(os.listdir('/sys/class/power_supply')) if d.startswith('A')]
    return {'percent': read('/sys/class/power_supply/BAT0/capacity'), 'status': read('/sys/class/power_supply/BAT0/status'), 'ac': ac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True); ap.add_argument('--nrt', required=True); ap.add_argument('--scalars', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--cells', default='8x16,16x32')
    ap.add_argument('--sustain', type=float, default=60); ap.add_argument('--bursts', type=float, default=30)
    ap.add_argument('--burst-glyphs', type=int, default=3); ap.add_argument('--burst-gap', type=float, default=0.5)
    ap.add_argument('--backend', type=int, default=1)  # CUDA
    ap.add_argument('--tail', type=float, default=0, help='instead: one small burst, then this many idle seconds, twice, then close')
    a = ap.parse_args()
    scalars = [int(l.strip()[2:], 16) if l.strip().upper().startswith('U+') else int(l, 0) for l in open(a.scalars) if l.strip()]
    lib = ctypes.CDLL(a.lib)
    lib.neuronte_gpu_open.argtypes = [ctypes.POINTER(Options), ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
    lib.neuronte_gpu_coverage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]
    lib.neuronte_gpu_close.argtypes = [ctypes.c_void_p]
    lib.neuronte_gpu_device.restype = ctypes.c_char_p; lib.neuronte_gpu_device.argtypes = [ctypes.c_void_p]
    nv = Nvml()
    samples = []; phase = ['idle']; stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append((time.monotonic(), phase[0], nv.watts(), nv.util(), nv.temp()))
            stop.wait(0.2)
    th = threading.Thread(target=sampler); th.start()
    marks = []  # (phase, t0, t1, joules0, joules1, glyphs, calls)

    def span(name, fn=None, seconds=0.0):
        phase[0] = name
        j0, t0 = nv.joules(), time.monotonic()
        glyphs, calls = fn() if fn else (0, 0)
        if seconds:
            time.sleep(seconds)
        marks.append((name, t0, time.monotonic(), j0, nv.joules(), glyphs, calls))

    def opened(backend, cell):
        w, h = map(int, cell.split('x'))
        o = Options(); o.nrt_path = os.fsencode(a.nrt); o.cell_w, o.cell_h = w, h; o.backend = backend; o.flags = 16; o.max_batch = 2048
        g = ctypes.c_void_p(); why = ctypes.create_string_buffer(512)
        if lib.neuronte_gpu_open(ctypes.byref(o), ctypes.byref(g), why, 512):
            raise SystemExit('open failed: ' + why.value.decode())
        return g, (ctypes.c_float * (len(scalars) * w * h))()

    def draw(g, buf, which):
        arr = (ctypes.c_uint32 * len(which))(*which)
        if lib.neuronte_gpu_coverage(g, arr, len(which), buf, None):
            raise SystemExit('draw failed')

    def fill(g, buf):
        return lambda: (draw(g, buf, scalars), (len(scalars), 1))[1]

    def bursts(g, buf, seconds):
        def run():
            end = time.monotonic() + seconds; at = glyphs = calls = 0
            while time.monotonic() < end:
                draw(g, buf, [scalars[(at + k) % len(scalars)] for k in range(a.burst_glyphs)])
                at += a.burst_glyphs; glyphs += a.burst_glyphs; calls += 1
                time.sleep(a.burst_gap)
            return glyphs, calls
        return run

    def sustain(g, buf, seconds):
        def run():
            end = time.monotonic() + seconds; glyphs = calls = 0
            while time.monotonic() < end:
                draw(g, buf, scalars); glyphs += len(scalars); calls += 1
            return glyphs, calls
        return run

    cells = a.cells.split(',')
    res = {'tool': 'power.py', 'file': os.path.basename(a.nrt), 'glyphs_in_list': len(scalars), 'when': time.strftime('%FT%TZ', time.gmtime()),
           'battery_start': battery(), 'load': open('/proc/loadavg').read().split()[:3],
           'burst': {'glyphs': a.burst_glyphs, 'gap_s': a.burst_gap}}
    span('idle', seconds=5)
    if a.tail:
        c = cells[0]
        g, buf = opened(a.backend, c)
        res['device'] = lib.neuronte_gpu_device(g).decode()
        draw(g, buf, scalars[:4])
        one = lambda: (draw(g, buf, scalars[10:10 + a.burst_glyphs]), (a.burst_glyphs, 1))[1]
        span('tail after open', seconds=a.tail)
        span('burst 1', one)
        span('tail 1', seconds=a.tail)
        span('burst 2', one)
        span('tail 2', seconds=a.tail)
        lib.neuronte_gpu_close(g)
        span('closed', seconds=20)
    else:
        gpu = {c: opened(a.backend, c) for c in cells}
        res['device'] = lib.neuronte_gpu_device(gpu[cells[0]][0]).decode()
        for c in cells:  # a first call each, so nothing below pays a first-use cost
            draw(*gpu[c], scalars[:4])
        span('open-idle', seconds=5)
        for c in cells:
            span('fill ' + c, fill(*gpu[c]))
            span('after fill ' + c, seconds=5)
        span('bursts ' + cells[0], bursts(*gpu[cells[0]], a.bursts))
        span('after bursts', seconds=10)
        span('sustain ' + cells[0], sustain(*gpu[cells[0]], a.sustain))
        span('after', seconds=10)
        # the same work on the CPU library, one thread
        cpu = {c: opened(4, c) for c in cells}
        for c in cells:
            span('cpu fill ' + c, fill(*cpu[c]))
        span('cpu bursts ' + cells[0], bursts(*cpu[cells[0]], min(a.bursts, 10)))
        span('cpu sustain ' + cells[0], sustain(*cpu[cells[0]], min(a.sustain, 10)))
    stop.set(); th.join()
    idle = [s[2] for s in samples if s[1] == 'idle' and s[2] is not None]
    idle_w = sum(idle) / len(idle)
    res['idle_watts'] = idle_w
    res['phases'] = []
    for name, t0, t1, j0, j1, glyphs, calls in marks:
        ws = [s[2] for s in samples if t0 <= s[0] <= t1 and s[2] is not None]
        us = [s[3] for s in samples if t0 <= s[0] <= t1 and s[3] is not None]
        ts = [s[4] for s in samples if t0 <= s[0] <= t1 and s[4] is not None]
        row = {'phase': name, 'seconds': t1 - t0, 'glyphs': glyphs, 'calls': calls,
               'watts_mean': sum(ws) / len(ws) if ws else None, 'watts_peak': max(ws) if ws else None,
               'utilisation_mean': sum(us) / len(us) if us else None, 'utilisation_peak': max(us) if us else None,
               'temperature_max': max(ts) if ts else None, 'gpu_joules': None if j0 is None else j1 - j0}
        if j0 is not None:
            row['gpu_joules_above_idle'] = (j1 - j0) - idle_w * (t1 - t0)
            if glyphs:
                row['gpu_joules_per_1000_glyphs'] = (j1 - j0) / glyphs * 1000
                row['gpu_joules_above_idle_per_1000_glyphs'] = row['gpu_joules_above_idle'] / glyphs * 1000
        if glyphs:
            row['us_per_glyph_busy'] = None if name.find('bursts') >= 0 else (t1 - t0) / glyphs * 1e6
        res['phases'].append(row)
    res['samples'] = [(round(s[0] - samples[0][0], 2), s[1], s[2], s[3], s[4]) for s in samples]
    res['battery_end'] = battery()
    json.dump(res, open(a.out, 'w'), indent=1)
    if a.tail:
        for t, ph, w, u, tc in res['samples']:
            if int(t * 5) % 10 == 0: print('%6.1f s %-16s %5.1f W util %s temp %s' % (t, ph, w, u, tc))
    print(res['device'], 'idle %.1f W' % idle_w, res['battery_start'], '->', res['battery_end'])
    for r in res['phases']:
        print('%-18s %6.2f s glyphs %7d  mean %5.1f W peak %5.1f W  util mean %s peak %s  temp %s  gpu J %s  above idle %s  J/1000 glyphs %s (above idle %s)' % (
            r['phase'], r['seconds'], r['glyphs'], r['watts_mean'] or 0, r['watts_peak'] or 0,
            None if r['utilisation_mean'] is None else round(r['utilisation_mean']), r['utilisation_peak'], r['temperature_max'],
            None if r['gpu_joules'] is None else round(r['gpu_joules'], 2), round(r.get('gpu_joules_above_idle', 0), 2),
            None if 'gpu_joules_per_1000_glyphs' not in r else round(r['gpu_joules_per_1000_glyphs'], 2),
            None if 'gpu_joules_above_idle_per_1000_glyphs' not in r else round(r['gpu_joules_above_idle_per_1000_glyphs'], 2)))


if __name__ == '__main__':
    main()
