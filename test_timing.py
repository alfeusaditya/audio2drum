"""Timing self-check for audio2drum.py: synthesise a humanised groove with known hit times,
run the converter, and report how far each MIDI note lands from the true attack.

python test_timing.py [extra audio2drum args, e.g. --no-quantize]
"""
import os, subprocess, sys, tempfile, numpy as np, soundfile as sf

SR = 44100
rng = np.random.default_rng(1)

def kick(n=int(0.35 * SR)):
    t = np.arange(n) / SR
    return 0.9 * np.sin(2 * np.pi * (45 * t + 40 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 9)

def snare(n=int(0.25 * SR)):
    t = np.arange(n) / SR
    return (0.5 * rng.standard_normal(n) * np.exp(-t * 18) + 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 25))

def hat(n=int(0.06 * SR)):
    x = np.diff(rng.standard_normal(n + 1)) * 0.6
    return x * np.exp(-np.arange(n) / SR * 60)

bpm, bars = 100, 16
beat = 60 / bpm
truth, y = [], np.zeros(int((bars * 4 * beat + 2) * SR))
for i in range(bars * 8):  # 8ths
    t0 = 0.5 + i * beat / 2 + rng.uniform(-0.012, 0.012)  # human jitter +-12 ms
    parts = [(49, hat)]
    if i % 4 == 0: parts.append((36, kick))
    if i % 4 == 2: parts.append((38, snare))
    if i % 8 == 7: parts.append((38, lambda: 0.25 * snare()))  # ghost snare
    for note, fn in parts:
        s = fn(); a = int(t0 * SR); y[a:a + len(s)] += s; truth.append((t0, note))
y += 0.002 * rng.standard_normal(len(y))

d = os.environ.get('TT_DIR') or tempfile.mkdtemp()
wav, txt = os.path.join(d, 'groove.wav'), os.path.join(d, 'groove.txt')
sf.write(wav, y / np.abs(y).max() * 0.9, SR)
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, 'audio2drum.py'), wav, os.path.join(d, 'g.mid'),
                '--no-stems', '--notes', txt, *sys.argv[1:]], check=True, capture_output=True)
notes = [(float(l.split()[1]), int(l.split()[2])) for l in open(txt) if l.startswith('NOTE')]

err, miss = [], 0
for t0 in sorted({t for t, _ in truth}):  # timing only: any note near the hit, whatever drum it was called
    c = [t for t, p in notes if abs(t - t0) < 0.04]
    if c: err.append(min(c, key=lambda t: abs(t - t0)) - t0)
    else: miss += 1
err = np.abs(np.array(err)) * 1000
print(f'{len({t for t, _ in truth})} true hit times, {len(err)} found, {miss} missed; '
      f'timing error ms: median {np.median(err):.1f}, 95% {np.percentile(err, 95):.1f}, max {err.max():.1f}')
