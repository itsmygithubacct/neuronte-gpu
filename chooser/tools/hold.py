#!/usr/bin/env python3
"""What an NVIDIA GPU's power does after calls through the loader, for one idle_close_ms.

    hold.py --lib libneuronte-backend.so.0 --nrt FILE.nrt --scalars LIST --idle-ms N --out FILE.json
            [--wake-ms M] [--cell 8x16] [--window 30]

The environment must already choose a GPU (NEURONTE_BACKEND_CONF or the
settings, and a current warm stamp); the script reads that backend.conf,
writes a copy with idle_close_ms=N and wake_ms=M and points the loader at the
copy.

One handle, and one scenario after another, each sampled for --window
seconds from its first call (watts, performance state and clock through the
driver, ten times a second):
  3 glyphs, 16 glyphs, the whole list   one call, then nothing
  3 glyphs twice, 5 s apart             what the second call finds
  typing                                3 glyphs every 0.5 s for 10 s
  the whole list twice, 5 s apart
For every call: how long it took and how many glyphs the CPU and the GPU
answered. Energy is the watts summed over the window; above idle is that
less the idle watts measured before the first scenario times the window.
"""
import argparse, ctypes, json, os, tempfile, threading, time


class Info(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_int), ('min_glyphs', ctypes.c_int), ('backend', ctypes.c_char * 32),
                ('device', ctypes.c_char * 128), ('reason', ctypes.c_char * 256), ('from_cells', ctypes.c_uint64),
                ('from_cpu', ctypes.c_uint64), ('from_gpu', ctypes.c_uint64), ('ready', ctypes.c_int)]


class Nvml:
    def __init__(self):
        self.n = ctypes.CDLL('libnvidia-ml.so.1')
        self.n.nvmlInit_v2()
        self.h = ctypes.c_void_p()
        self.n.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h))

    def read(self):
        w, c, p = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_int()
        watts = w.value / 1000.0 if self.n.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(w)) == 0 else None
        clock = c.value if self.n.nvmlDeviceGetClockInfo(self.h, 0, ctypes.byref(c)) == 0 else None
        state = p.value if self.n.nvmlDeviceGetPerformanceState(self.h, ctypes.byref(p)) == 0 else None
        return watts, clock, state


def battery():
    out = {}
    base = '/sys/class/power_supply'
    for name in sorted(os.listdir(base)):
        try:
            kind = open(f'{base}/{name}/type').read().strip()
            if kind == 'Battery':
                out['percent'] = int(open(f'{base}/{name}/capacity').read())
                out['status'] = open(f'{base}/{name}/status').read().strip()
            elif kind == 'Mains':
                out['ac'] = int(open(f'{base}/{name}/online').read())
        except (OSError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True); ap.add_argument('--nrt', required=True); ap.add_argument('--scalars', required=True)
    ap.add_argument('--idle-ms', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cell', default='8x16'); ap.add_argument('--window', type=float, default=30)
    ap.add_argument('--wake-ms', type=int, default=0)
    a = ap.parse_args()
    scalars = [int(l.strip()[2:], 16) if l.strip().upper().startswith('U+') else int(l, 0) for l in open(a.scalars) if l.strip()]
    w, h = (int(v) for v in a.cell.split('x'))

    lib = ctypes.CDLL(a.lib)
    lib.neuronte_backend_choice.argtypes = [ctypes.POINTER(Info)]
    lib.neuronte_backend_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.neuronte_backend_coverage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.neuronte_backend_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
    lib.neuronte_backend_close.argtypes = [ctypes.c_void_p]

    # the machine's backend.conf, with this run's idle_close_ms
    conf = os.environ.get('NEURONTE_BACKEND_CONF')
    if not conf:
        state = os.environ.get('NEURONTE_SYSTEM_STATE') or '/var/lib/neuronte'
        settings = os.environ.get('NEURONTE_SETUP_CONF') or '/etc/neuronte/setup.conf'
        if not os.environ.get('NEURONTE_SYSTEM_STATE') and os.path.exists(settings):
            for line in open(settings):
                if line.startswith('state='):
                    state = line.split('=', 1)[1].strip()
        conf = os.path.join(state, 'backend.conf')
    tmp = tempfile.NamedTemporaryFile('w', suffix='.conf', delete=False)
    tmp.write(open(conf).read() + f'idle_close_ms={a.idle_ms}\nwake_ms={a.wake_ms}\n')
    tmp.close()
    os.environ['NEURONTE_BACKEND_CONF'] = tmp.name

    info = Info()
    if not lib.neuronte_backend_choice(ctypes.byref(info)):
        raise SystemExit(f'hold.py: no GPU is chosen here: {info.reason.decode()}')
    nvml = Nvml()
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append((time.monotonic(),) + nvml.read())
            time.sleep(0.1)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    handle = ctypes.c_void_p()
    if lib.neuronte_backend_open(os.fsencode(a.nrt), w, h, ctypes.byref(handle)) != 0:
        raise SystemExit('hold.py: the file did not open')
    buf = (ctypes.c_float * (len(scalars) * w * h))()
    at = [0]

    def call(count):
        """Draw `count` scalars not asked for before (the list goes round)."""
        picked = scalars if count >= len(scalars) else [scalars[(at[0] + k) % len(scalars)] for k in range(count)]
        at[0] += count
        arr = (ctypes.c_uint32 * len(picked))(*picked)
        lib.neuronte_backend_info(handle, ctypes.byref(info))
        cpu0, gpu0, ready = info.from_cpu, info.from_gpu, info.ready
        t0 = time.monotonic()
        code = lib.neuronte_backend_coverage(handle, arr, len(picked), buf)
        took = time.monotonic() - t0
        lib.neuronte_backend_info(handle, ctypes.byref(info))
        return {'at': t0, 'glyphs': len(picked), 'ms': round(took * 1000, 3), 'gpu_open_before': ready, 'cpu': info.from_cpu - cpu0,
                'gpu': info.from_gpu - gpu0, 'code': code}

    def idle_watts(seconds):
        t0 = time.monotonic()
        time.sleep(seconds)
        got = [s[1] for s in samples if s[0] >= t0 and s[1] is not None]
        return sum(got) / len(got) if got else None

    def settle(limit=40):
        """Wait for the lowest performance state the driver showed at the start."""
        end = time.monotonic() + limit
        while time.monotonic() < end:
            time.sleep(0.5)
            recent = [s for s in samples if s[0] >= time.monotonic() - 1.5]
            if recent and all(s[3] == base_state for s in recent):
                return True
        return False

    time.sleep(1.0)
    base_state = samples[-1][3]
    idle = idle_watts(4)
    out = {'tool': 'hold.py', 'when': time.strftime('%FT%TZ', time.gmtime()), 'file': os.path.basename(a.nrt), 'cell': [w, h],
           'glyphs_in_list': len(scalars), 'idle_close_ms': a.idle_ms, 'wake_ms': a.wake_ms, 'window_s': a.window, 'idle_watts': idle,
           'idle_state': base_state, 'backend': info.backend.decode(), 'device': info.device.decode(), 'battery_before': battery(),
           'load': os.getloadavg(), 'scenarios': []}

    def scenario(name, plan):
        """plan: (seconds from the start, glyphs) pairs."""
        settled = settle()
        t0 = time.monotonic()
        calls = []
        for when, count in plan:
            wait = t0 + when - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            c = call(count)
            c['at'] = round(c['at'] - t0, 3)
            calls.append(c)
        wait = t0 + a.window - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        t1 = t0 + a.window
        win = [s for s in samples if t0 <= s[0] <= t1 and s[1] is not None]
        joules = sum((win[k + 1][0] - win[k][0]) * (win[k][1] + win[k + 1][1]) / 2 for k in range(len(win) - 1))
        raised = sum(win[k + 1][0] - win[k][0] for k in range(len(win) - 1) if win[k][3] != base_state)
        res = {'name': name, 'settled_before': settled, 'joules': round(joules, 1),
               'joules_above_idle': round(joules - idle * (win[-1][0] - win[0][0]), 1) if idle else None,
               'seconds_above_idle_state': round(raised, 1), 'peak_watts': max(s[1] for s in win), 'calls': calls}
        out['scenarios'].append(res)
        ms = sorted(c['ms'] for c in calls)
        print(f"idle_close_ms={a.idle_ms:5d} wake_ms={a.wake_ms:3d} {name:28s} {res['joules']:7.1f} J, {res['joules_above_idle']:7.1f} above idle, "
              f"{raised:5.1f} s raised, peak {res['peak_watts']:5.1f} W; calls {len(calls)}: median {ms[len(ms) // 2]:.2f} ms, "
              f"max {ms[-1]:.2f} ms, cpu {sum(c['cpu'] for c in calls)}, gpu {sum(c['gpu'] for c in calls)}", flush=True)

    scenario('3 glyphs', [(0, 3)])
    scenario('16 glyphs', [(0, 16)])
    scenario('whole list', [(0, len(scalars))])
    scenario('3 glyphs, again after 5 s', [(0, 3), (5, 3)])
    scenario('typing: 3 every 0.5 s, 10 s', [(k * 0.5, 3) for k in range(20)])
    scenario('whole list twice, 5 s apart', [(0, len(scalars)), (5, len(scalars))])
    lib.neuronte_backend_close(handle)
    stop.set()
    thread.join()
    out['battery_after'] = battery()
    os.unlink(tmp.name)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
