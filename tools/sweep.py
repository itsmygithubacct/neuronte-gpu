#!/usr/bin/env python3
"""How parallel a GPU run is: batch-size sweep, host-thread share, device utilisation, two feeders at once.
   sweep.py BENCH NRT CELL SCALARS OUT.json [--backend cuda]"""
import json, os, resource, statistics as st, subprocess, sys, threading, time
bench, nrt, cell, scal, out = sys.argv[1:6]
backend = sys.argv[7] if len(sys.argv) > 7 else 'cuda'
n_all = sum(1 for l in open(scal) if l.strip())
res = {'file': os.path.basename(nrt), 'cell': cell, 'glyphs': n_all, 'backend': backend, 'when': time.strftime('%FT%TZ', time.gmtime()),
       'load': open('/proc/loadavg').read().split()[:3]}
def run(per_call, repeats, extra=()):
    cmd = [bench, '--nrt', nrt, '--cell', cell, '--backend', backend, '--no-fallback', '--scalars', scal, '--per-call', str(per_call),
           '--repeats', str(repeats), '--warmup', '2', '--passes', '1', '--max-batch', '2048', *extra]
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    d['_cpu_s'] = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime); d['_wall_s'] = time.time() - t0
    return d
sweep = []
for pc in (1, 4, 16, 64, 256, 1024, n_all):
    if pc > n_all: continue
    d = run(pc, 5)
    if 'error' in d: res['error'] = d['error']; break
    rounds = [r for r in d['rounds'] if not r.get('warmup')]; g = d['glyphs_per_round']
    row = {'per_call': pc, 'us_per_glyph': st.median(r['wall'] for r in rounds) / g / 1e3,
           'host_cpu_share': st.median(r['cpu'] / r['wall'] for r in rounds),
           'us_per_glyph_by_stage': {k: round(st.median(r['stages'].get(k, 0) for r in rounds) / g / 1e3, 3) for k in ('plan', 'upload', 'run', 'wait', 'download', 'cpu_eval')},
           'launches_per_call': st.median(r['stages'].get('launches', 0) for r in rounds)}
    sweep.append(row); res['device'] = d.get('device'); res['versions'] = d.get('versions'); res['samples'] = d.get('samples')
res['sweep'] = sweep
if 'error' not in res:
    # a steady run of whole-set calls: device utilisation and power, and the host's CPU share
    samples = []; stop = threading.Event()
    def watch():
        while not stop.is_set():
            o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip().split('\n')[0]
            samples.append(o); stop.wait(0.25)
    th = threading.Thread(target=watch); th.start()
    reps = max(20, int(4.0 / (sweep[-1]['us_per_glyph'] * n_all / 1e6)))
    d = run(n_all, min(reps, 400)); stop.set(); th.join()
    res['steady'] = {'repeats': min(reps, 400), 'us_per_glyph': st.median(r['wall'] for r in d['rounds'] if not r.get('warmup')) / d['glyphs_per_round'] / 1e3,
                     'host_cpu_share': st.median(r['cpu'] / r['wall'] for r in d['rounds'] if not r.get('warmup')),
                     'process_cpu_s': d['_cpu_s'], 'process_wall_s': d['_wall_s'], 'nvidia_smi_util_power_mem': samples}
    # two feeders at once, each a process with its own context
    ps = [subprocess.Popen([bench, '--nrt', nrt, '--cell', cell, '--backend', backend, '--no-fallback', '--scalars', scal, '--repeats', str(min(reps, 400)),
                            '--warmup', '2', '--passes', '1', '--max-batch', '2048'], stdout=subprocess.PIPE, text=True) for _ in range(2)]
    outs = [json.loads(p.communicate()[0]) for p in ps]
    res['two_feeders'] = {'us_per_glyph_each': [st.median(r['wall'] for r in o['rounds'] if not r.get('warmup')) / o['glyphs_per_round'] / 1e3 for o in outs]}
json.dump(res, open(out, 'w'), indent=1)
print(res.get('device'), res.get('versions', '')[:160])
for r in res.get('sweep', []): print(r)
print('two feeders', res.get('two_feeders'), res.get('error'))
if 'steady' in res: print('steady', res['steady']['us_per_glyph'], 'host share', res['steady']['host_cpu_share'], 'cpu', res['steady']['process_cpu_s'], 'wall', res['steady']['process_wall_s'], res['steady']['nvidia_smi_util_power_mem'][2:14])
