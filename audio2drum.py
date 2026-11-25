"""Drum audio -> quantized MIDI in the Addictive Drums 2 map.

Two ways to find the drums:
  * stems mode (best): separated stems <name>_kick/_snare/_toms/_hihat/_cymbals.wav
    (drum-separation tool naming), found next to the input or in a sub-folder,
    or given with --stems DIR. Each stem = one instrument, so no guessing.
  * mix mode (fallback): band-energy heuristics on the drum audio itself.
Then: toms split by pitch, hats closed/half/open by decay, cymbals crash/ride,
flams as a grace note 30 ms early, snapped to a 1/16 grid (--no-quantize: exact attack times).

Grid: 'fixed' = one tempo (loops, click tracks); 'map' = follow the drummer beat
by beat (live playing, no click). 'auto' = fixed if the hits fit one tempo all song, else map.

python audio2drum.py in.wav out.mid [--meter 4/4] [--anchor SEC] [--bpm N]
                     [--grid auto|fixed|map] [--stems DIR] [--notes out.txt]
--notes writes plain text for the REAPER script (times in seconds of the audio file).
"""
import argparse, glob, os, numpy as np, librosa, mido

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--meter', default='4/4')
ap.add_argument('--anchor', type=float, help='time (s) of a beat 1; default = first kick')
ap.add_argument('--bpm', type=float, help='force one fixed tempo')
ap.add_argument('--grid', default='auto', choices=['auto', 'fixed', 'map'])
ap.add_argument('--stems', help='folder with separated stems')
ap.add_argument('--no-stems', action='store_true', help='ignore stems, use mix heuristics')
ap.add_argument('--notes', help='also write notes/beats text file for REAPER')
ap.add_argument('--no-quantize', action='store_true', help='notes at the exact attack, not snapped to 1/16')
A = ap.parse_args()
num, den = map(int, A.meter.split('/'))

# Addictive Drums 2 keymap (official AD2 Keymap.pdf). NB: 42 = snare sidestick in AD2, not hat.
KICK, SNARE, HAT, HAT_HALF, HAT_OPEN, CRASH, RIDE = 36, 38, 49, 54, 56, 77, 60
UNKNOWN = KICK - 2  # a transient we heard but could not name (mix mode); sort it by hand
TOMS = [71, 69, 67, 65]  # Tom 1 (high) .. Tom 4 (floor), open hit
LABEL = {KICK: 'K', SNARE: 'S', HAT: 'h', HAT_HALF: 'o', HAT_OPEN: 'O', CRASH: 'C', RIDE: 'R',
         71: '1', 69: '2', 67: '3', 65: '4', UNKNOWN: '?'}
FLAM_GAP, FLAM_MS = 0.075, 0.030  # same-drum onsets closer than 75ms = flam; grace 30ms early
SR, HOP = 44100, 256
STEMS = ['kick', 'snare', 'toms', 'hihat', 'cymbals']

def find_stems(src):
    if A.no_stems: return None
    here = os.path.dirname(os.path.abspath(src))
    base = os.path.splitext(os.path.basename(src))[0]
    dirs = [A.stems] if A.stems else [here] + sorted(glob.glob(os.path.join(here, '*', '')))
    for d in dirs:
        for pat in (f'{base}_{{}}.wav', '{}.wav'):  # ISO: name_kick.wav, LarsNet: kick.wav
            paths = {s: os.path.join(d, pat.format(s)) for s in STEMS}
            if all(os.path.exists(p) for p in paths.values()): return paths
    if A.stems: raise SystemExit(f'stems not found in {A.stems}')
    return None

def hat_note(dec):
    return HAT if dec < 0.1 else HAT_HALF if dec < 0.3 else HAT_OPEN if dec < 0.8 else CRASH

def decay(env, a, stop):  # seconds until env falls to 25% of its attack peak (or until next hit)
    seg = env[a:min(a + 200, stop)] / (env[a:a + 4].max() + 1e-9)
    return np.argmax(seg < 0.25) * HOP / SR if (seg < 0.25).any() else len(seg) * HOP / SR

# ---------- hits: [time, family, note (tom: set later by pitch), velocity, tom_hz]
hits = []
stems = find_stems(A.src)
yst, _ = librosa.load(A.src, sr=SR, mono=False)  # stereo kept for the pan rule (mix mode)
y = librosa.to_mono(yst)
dur = len(y) / SR
if stems:
    BAND = {'kick': (30, 250), 'snare': (150, 10000), 'toms': (60, 700),
            'hihat': (5000, 16000), 'cymbals': (3000, 16000)}
    DELTA = {'kick': 0.12, 'snare': 0.10, 'toms': 0.15, 'hihat': 0.08, 'cymbals': 0.15}
    found, beat_env = {}, 0
    for s in STEMS:
        ys, _ = librosa.load(stems[s], sr=SR, mono=True)
        lo, hi = BAND[s]
        mel = librosa.feature.melspectrogram(y=ys, sr=SR, n_fft=4096, hop_length=HOP, n_mels=24, fmin=lo, fmax=hi)
        env = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max), sr=SR, hop_length=HOP)
        env = env / (np.percentile(env, 99.5) + 1e-9)  # stem-relative, so quiet stems still count
        fr = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, delta=DELTA[s],
                                        wait=5, pre_max=3, post_max=3, pre_avg=40, post_avg=10)  # wait 29ms
        rms = librosa.feature.rms(y=ys, frame_length=1024, hop_length=HOP)[0]
        bleed = np.percentile(rms, 10) / (np.percentile(rms, 99.5) + 1e-9)  # never silent = other instruments
        lvl = 20 * np.log10(np.array([rms[a:a + 6].max() for a in fr]) + 1e-9)
        rise = lvl - 20 * np.log10(np.array([rms[max(a - 6, 0):a + 1].min() for a in fr]) + 1e-9)
        ok = (env[fr] > (0.45 if bleed > 0.05 else 0.25)) & (rise > (6 if bleed > 0.05 else 3)) \
             & (lvl > 20 * np.log10(np.percentile(rms, 99.5) + 1e-9) - 30)  # real hit: jumps up, not too quiet
        fr = fr[ok]
        if bleed > 0.05:
            print(f'warning: {s} stem is never silent ({bleed:.0%} floor) - leakage from other instruments; '
                  'run drum separation first for full songs')
        found[s] = (fr, env[fr], ys)
        if s in ('kick', 'snare'): beat_env = beat_env + env
    # bleed between neighbouring stems: drop the weaker hit if the neighbour is 2.5x stronger
    for a, b in (('toms', 'kick'), ('toms', 'snare'), ('kick', 'toms'), ('snare', 'toms'),
                 ('hihat', 'cymbals'), ('cymbals', 'hihat')):
        fa, sa, ya = found[a]; fb, sb, _ = found[b]
        keep = [not any(abs(int(x) - int(z)) * HOP / SR < 0.015 and w >= 2.5 * v for z, w in zip(fb, sb))
                for x, v in zip(fa, sa)]
        found[a] = (fa[keep], sa[keep], ya)
    for s in STEMS:
        fr, st, ys = found[s]
        t = fr * HOP / SR
        vel = np.clip(30 + 97 * np.sqrt(np.clip(st, 0, 1)), 1, 127).astype(int)
        if s == 'kick': hits += [[x, 'kick', KICK, v, 0] for x, v in zip(t, vel)]
        elif s == 'snare': hits += [[x, 'snare', SNARE, v, 0] for x, v in zip(t, vel)]
        elif s == 'toms':
            S = np.abs(librosa.stft(ys, n_fft=4096, hop_length=HOP)); f = librosa.fft_frequencies(sr=SR, n_fft=4096)
            lm = (f >= 60) & (f < 400)
            hits += [[x, 'tom', None, v, f[lm][np.argmax(S[lm, a:a + 10].sum(1))]] for x, v, a in zip(t, vel, fr)]
        else:
            env = np.abs(librosa.stft(ys, n_fft=1024, hop_length=HOP))[64:].sum(0)  # > ~2.7 kHz
            nxt = list(fr[1:]) + [len(env)]
            for x, v, a, b in zip(t, vel, fr, nxt):
                if s == 'hihat': hits.append([x, 'hat', hat_note(decay(env, a, b)), v, 0])
                else: hits.append([x, 'cym', CRASH, v, 0])
    cym = [h for h in hits if h[1] == 'cym']
    for i, h in enumerate(cym):  # a cymbal played as a steady pattern (< 0.6 s both sides) = ride
        if 0 < i < len(cym) - 1 and h[0] - cym[i - 1][0] < 0.6 and cym[i + 1][0] - h[0] < 0.6:
            h[2] = RIDE
else:
    # Features per hit, each band normalised by its 98th percentile over the whole file so the
    # rules follow this kit/mix. A produced snare can have MORE body (90-300) than a tom, so
    # snare vs tom = mid (300-1500) relative to body + how long the body rings.
    ob = librosa.onset.onset_detect(y=y, sr=SR, units='frames', hop_length=HOP, delta=0.07)
    # every fainter transient too: never guessed into a drum, always written as UNKNOWN
    weak = librosa.onset.onset_detect(y=y, sr=SR, units='frames', hop_length=HOP, delta=0.03)
    weak = [a for a in weak if not len(ob) or np.abs(ob - a).min() > 2]  # > ~12 ms from a main onset
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP)); f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    bands = [(30, 90), (90, 300), (300, 1500), (2500, 8000), (8000, 20000)]  # low, body, mid, noise, air
    E = np.array([S[(f >= lo) & (f < hi)].sum(0) for lo, hi in bands])
    pm = (f >= 70) & (f < 400)
    # 50-300 Hz per channel for the pan rule; a mono file gives pan 0 (rule never fires)
    bm = (f >= 50) & (f < 300)
    EL, ER = (np.abs(librosa.stft(c, n_fft=4096, hop_length=HOP))[bm].sum(0) for c in yst[:2]) \
        if yst.ndim == 2 else (E[1], E[1])
    rows = []  # t, low, body, mid, noise, air, body ring, pitch, noise decay, air decay, weak?
    for a in sorted([*ob, *weak]):
        i = np.searchsorted(ob, a, 'right')
        nxt = ob[i] if i < len(ob) else len(E[1])  # ring stops at the next MAIN onset, as before
        e = E[:, a:a + 8].sum(1)
        # body still sounding ~90-230 ms later, stopping before the next hit so it can't leak in
        w = E[1, a + 15:min(a + 40, nxt - 1)]
        ring = w.mean() / (E[1, a:a + 6].mean() + 1e-9) if len(w) >= 4 else \
            E[1, a + 8:max(nxt - 1, a + 9)].mean() / (E[1, a:a + 6].mean() + 1e-9) if a + 9 < E.shape[1] else 0  # fast hits: earlier window; 0 at file end
        rows.append((a * HOP / SR, *e, ring, f[pm][np.argmax(S[pm, a:a + 8].sum(1))],
                     decay(E[3], a, a + 200), decay(E[4], a, a + 200), a not in ob))
    R = np.array(rows)
    P = np.percentile(R[R[:, 10] == 0][:, 1:6], 98, axis=0) + 1e-9  # main onsets only: rules unchanged
    N = R[:, 1:6] / P
    snare_mid, tom_info = [], []
    for r, n in zip(R, N):
        t, low, body, mid, noise, air, ring, pitch, ndec, adec, is_weak = r
        vel = lambda x: int(np.clip(35 + 92 * min(x, 1) ** 0.7, 1, 127))
        n0 = len(hits)
        if is_weak or n.max() <= 0.12:  # faint / bleed-level: record it, don't guess what it is
            hits.append([t, 'unk', UNKNOWN, vel(n.max()), 0]); continue
        # Low ring = 30-90 Hz still sounding 90-230 ms later vs the attack: kicks 0.2-0.6, toms > 1.
        a_ = int(round(t * SR / HOP))
        low_ring = E[0, a_ + 15:a_ + 40].mean() / (E[0, a_:a_ + 6].mean() + 1e-9) \
            if a_ + 15 < E.shape[1] else 0  # file ends before the ring can be heard
        dry = ring < 0.15 and low_ring < 0.7  # dies at once: a kick's thud, never a ringing tom (user)
        # kick: low-dominant, or strong low that is dry (punchy kicks carry a fat 90-300 Hz body)
        kick = n[0] > 0.35 and (low >= 0.9 * body or (dry and low >= 0.3 * body))
        # User rule: kick/snare sit in the centre, toms almost never. Off-centre low that rings = tom.
        # Fast rolls cut the ring short: there a tom is told from a panned snare by its top (user):
        # > 5 kHz vs 50-300 Hz is >= 0.61 on centred snares, ~0.2-0.4 on toms. Kicks keep the ring test.
        # Kick/snare measure |pan| <= 0.13 (even under a tom's tail, which low_ring >= 1.3 excludes).
        # rise: the hit brings its OWN 50-300 Hz, not a hat/snare landing on a panned tom's tail.
        l_, r_ = EL[a_:a_ + 8].sum(), ER[a_:a_ + 8].sum()
        pre = EL[max(a_ - 14, 0):max(a_ - 8, 1)].mean() + ER[max(a_ - 14, 0):max(a_ - 8, 1)].mean()
        top = S[f >= 5000, a_:a_ + 8].sum() / (S[bm, a_:a_ + 8].sum() + 1e-9)
        if abs(r_ - l_) >= 0.18 * (r_ + l_ + 1e-9) and max(n[0], n[1]) >= 0.12 \
                and (low_ring >= 1.3 or (not kick and top < 0.6)) \
                and (EL[a_:a_ + 6].mean() + ER[a_:a_ + 6].mean()) >= 1.15 * pre:
            hits.append([t, 'tom', None, vel(max(n[0], n[1])), pitch]); continue
        # snare: wires + short body ring + mid not far below body; tom: rings longer or is darker.
        # Calibrated on 3 kits: snares mid/body >= 0.63 with ring < 0.3; short toms mid/body <= 0.60.
        snare = not kick and max(n[1], n[2]) > 0.4 and n[3] > 0.3 and ring < 0.3 and n[2] >= 0.62 * n[1]
        tom = not snare and not (kick and dry) and pitch >= 75 and low < 1.2 * body and \
            (n[1] > 0.55 or (n[1] > 0.3 and ring > 0.8))
        if tom and n[3] >= n[1] and n[2] >= 0.62 * n[1] and n[3] > 0.3:
            snare, tom = True, False  # wire buzz >= body + crack: a snare whose shell just rings long
        if kick: hits.append([t, 'kick', KICK, vel(n[0]), 0])
        if snare: hits.append([t, 'snare', SNARE, vel(max(n[3], n[1])), pitch]); snare_mid.append(n[2])
        if tom:
            hits.append([t, 'tom', None, vel(n[1]), pitch])
            tom_info.append((hits[-1], (n[3] + n[4]) / (2 * n[1] + 1e-9), n[2]))  # wire buzz vs body, crack
        hi_e = max(n[3], n[4])
        thin = hi_e >= 0.1 and max(n[0], n[1], n[2]) <= 0.3 * hi_e  # energy ONLY up high: a light hat
        hatlike = not snare and not tom and ((n[3] > 0.2 and noise >= 0.6 * mid) or thin)
        if hatlike and not kick:
            # User rule: a hit that brings its own 200-300 Hz body is a snare, never a hat, however
            # quiet. "Own" = rise over the 50-30 ms before (room/ringing drums excluded), as a
            # resonance standing above 100-180 and 400-800 Hz.
            d = np.clip(S[:, a_:a_ + 6].mean(1) - S[:, max(a_ - 14, 0):max(a_ - 8, 1)].mean(1), 0, None) \
                if (a_ := int(round(t * SR / HOP))) else S[:, :6].mean(1)
            bd = lambda lo, hi: d[(f >= lo) & (f < hi)].mean()
            body23 = bd(190, 310) / (0.5 * (bd(100, 180) + bd(400, 800)) + 1e-9)
            if (n[1] >= 0.3 and n[2] >= 0.6) or (n[1] >= 0.1 and body23 >= 1.5):
                hits.append([t, 'snare', SNARE, vel(max(n[3], n[1])), pitch]); snare_mid.append(n[2])
                hatlike = False
        if hatlike:  # hats/cymbals: noise without snare body
            # crash = WIDE (strong 2.5-8 kHz wash) and ringing, or still ringing after the whole
            # 1.16 s window. A sizzle living mostly above 8 kHz that dies within ~1 s is an open hat
            # (user: crash/open hat were swapped by an earlier 'more air = crash' rule).
            note = min(hat_note(ndec), HAT_OPEN)
            if ndec >= 1.1 or (n[3] >= 0.5 and ndec >= 0.5) or (n[4] > 0.5 and adec > 0.8 and not thin):
                note = CRASH
            hits.append([t, 'cym' if note == CRASH else 'hat', note, vel(hi_e), 0])
        if len(hits) == n0:
            # User rule: tom = strong low, rings a bit longer than a kick, no snare 'crack'.
            lo_e = max(n[0], n[1])
            if lo_e >= 0.12 and low_ring >= 0.9 and n[2] <= lo_e and n[4] <= lo_e:
                hits.append([t, 'tom', None, vel(lo_e), pitch])
            else:
                hits.append([t, 'unk', UNKNOWN, vel(n.max()), 0])  # heard, but no rule fits
    # a 'tom' at the snare's own body pitch is a snare whose ring we couldn't measure (fast hits)
    # With fewer than 3 snares (fast chops: ring is never measurable) the pitch alone is not trusted;
    # the hit also needs snare-wire buzz. Ghosts lack the crack, so velocity = mid vs the real snares.
    # shortcut: wire >= 0.72 sits between toms (<=0.66) and ghosts (>=0.76) of one clip; retune on misses.
    sp = [h[4] for h in hits if h[1] == 'snare']
    if sp:
        med, sm = np.median(sp), np.median(snare_mid)
        for h, wire, m in tom_info:
            if abs(h[4] / med - 1) < 0.05:
                if len(sp) >= 3: h[1], h[2] = 'snare', SNARE
                elif wire >= 0.72: h[1], h[2], h[3] = 'snare', SNARE, vel(m / sm)
    beat_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
assert hits, 'no hits detected'
hits.sort(key=lambda h: h[0])

# ---------- exact attack: onset frames are ~6 ms coarse and often late; move each hit to where its
# transient starts climbing, at 0.25 ms resolution. Envelope = |audio| (what the waveform shows)
# with a TRAILING 6 ms max: it never rises before the sound does, and it has no dips at the
# zero crossings of a low kick/tom wave (those dips used to stop the search inside the hit).
BLK = 11
dy = np.abs(y)
env1 = dy[:len(dy) // BLK * BLK].reshape(-1, BLK).max(1)
env1 = np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(23), env1], 24).max(1)

def attack(t, lo_t, hi_t):  # search only between the neighbouring hits' midpoints
    ms = SR / BLK / 1000
    lo = max(int(max(t - 0.035, lo_t) * SR / BLK), 0)  # onset frames can be up to ~30 ms late
    hi = min(int(min(t + 0.020, hi_t) * SR / BLK), len(env1) - 1)
    if hi - lo < 4: return t
    # The detector's STFT window can fire ~20-25 ms BEFORE a hit; then nothing loud is near t yet.
    # If the next 30 ms hold a hit at least twice as loud, that hit is the one this onset meant.
    # shortcut: a flam grace under half the main stroke's level merges into it; lower 0.5 if flams vanish.
    c, ahead = int(t * SR / BLK), min(int((t + 0.030) * SR / BLK), len(env1) - 1)
    if c <= ahead and env1[lo:hi + 1].max() < 0.5 * env1[c:ahead + 1].max(): hi = ahead  # c > ahead: file end
    pk = lo + int(np.argmax(env1[lo:hi + 1]))
    a0 = max(lo, pk - int(25 * ms))  # a drum climbs to its peak in < 25 ms; earlier = another sound
    fl = a0 + int(np.argmin(env1[a0:pk + 1]))  # quietest point before the peak = before this hit
    th = env1[fl] + 0.2 * (env1[pk] - env1[fl])
    k = fl + int(np.argmax(env1[fl:pk + 1] >= th))  # first crossing going forward from the quiet
    return max(k - int(0.3 * ms), 0) * BLK / SR  # ~0.3 ms before the crossing = the very start

ut = sorted({h[0] for h in hits})  # hits at the same frame share one attack
mids = [-1.0] + [(a + b) / 2 for a, b in zip(ut, ut[1:])] + [dur + 1]
exact = {t: attack(t, mids[i], mids[i + 1]) for i, t in enumerate(ut)}
# estimates < 10 ms apart are one hit seen from two onsets: all take the latest (the earlier one
# came from a wider search that also caught the quiet just before the hit)
groups = []
for v in sorted(set(exact.values())):
    if groups and v - groups[-1][-1] < 0.010: groups[-1].append(v)
    else: groups.append([v])
same = {v: g[-1] for g in groups for v in g}
for h in hits: h[0] = same[exact[h[0]]]
# two onsets that turned out to be one hit: keep one per drum family (the louder), else it reads as a flam
best = {}
for h in hits:
    k = (round(h[0], 4), h[1])
    if k not in best or h[3] > best[k][3]: best[k] = h
hits = sorted(best.values(), key=lambda h: h[0])

# ---------- toms: pitch clusters (gap > ~2.4 semitones = different drum), highest -> Tom 1
tp = sorted({h[4] for h in hits if h[1] == 'tom'})
clusters = []
for p in tp:
    if clusters and p / clusters[-1][-1] < 1.15: clusters[-1].append(p)
    else: clusters.append([p])
k = len(clusters)
tom_note = {p: TOMS[round((k - 1 - i) * 3 / max(k - 1, 1))] if k > 1 else TOMS[1]
            for i, c in enumerate(clusters) for p in c}
for h in hits:
    if h[1] == 'tom': h[2] = tom_note[h[4]]

# ---------- beat grid (quarter-note beat times in seconds)
if True:  # tempo from the body hits only (hats/ghosts sit between grid lines more often)
    strong = np.array([h[0] for h in hits if h[1] in ('kick', 'snare', 'tom') and h[3] > 70]
                      or [h[0] for h in hits])
if A.anchor is None:
    kicks = [h[0] for h in hits if h[1] == 'kick']
    A.anchor = kicks[0] if kicks else hits[0][0]
mode = 'fixed' if A.bpm else A.grid
def fit_fixed():
    def grid_err(st):
        return min(np.mean(np.abs(((strong - o0) / st + .5) % 1 - .5)) for o0 in strong[:8])
    cand = [(round(grid_err(60 / b / 4), 3), b) for b in np.arange(70, 180.01, 0.25)]
    st = 60 / min(cand)[1] / 4
    for n in (8, 24, len(strong)):  # least-squares refine; growing window avoids rounding slips
        w = strong[:n]
        st = np.polyfit(np.round((w - w[0]) / st), w, 1)[0]
    return st, np.mean(np.abs(((strong - strong[0]) / st + .5) % 1 - .5))
if mode == 'auto':
    mode = 'fixed'
    if dur > 30 and den == 4 and not A.bpm:
        step_fit, err = fit_fixed()  # recorded to a click: hits stay within ~10% of a 1/16 all song
        if err > 0.1: mode = 'map'
if mode == 'fixed':
    step = 60 / A.bpm / 4 if A.bpm else fit_fixed()[0]
    bl = 4 * step
    beats = A.anchor + bl * np.arange(-int(A.anchor / bl) - 4 * num - 4, int((dur - A.anchor) / bl) + 4 * num + 4)
else:
    _, beats = librosa.beat.beat_track(onset_envelope=beat_env, sr=SR, hop_length=HOP, units='time', tightness=400)
    ib = np.median(np.diff(beats))
    pre = beats[0] - ib * np.arange(int(beats[0] / ib) + 4 * num + 4, 0, -1)
    post = beats[-1] + ib * np.arange(1, int((dur - beats[-1]) / ib) + 4 * num + 4)
    beats = np.concatenate([pre, beats, post])
bidx = np.arange(len(beats), dtype=float)
to_beat = lambda t: np.interp(t, beats, bidx)
to_time = lambda b: np.interp(b, bidx, beats)
bpb = num * 4 / den  # quarter beats per bar

# ---------- quantize, flams, bar 1
for h in hits: h.append(round(to_beat(h[0]) * 4) / 4)  # h[5] = quantized beat position (1/16)
for i, h in enumerate(hits):  # flam: same-drum hit just before another -> soft grace of that one
    nxt = None if h[1] in ('kick', 'unk') else \
        next((g for g in hits[i + 1:] if g[0] - h[0] < FLAM_GAP and g[1] == h[1]), None)  # feet don't flam
    h.append(nxt is not None)  # h[6] = grace?
    if nxt is not None: h[5] = nxt[5]; h[3] = min(h[3], 70)
D = round(to_beat(A.anchor))
first = min(h[5] for h in hits)
B0 = D - bpb * np.ceil((D - first) / bpb - 1e-9)  # bar 1 = last downbeat at/before first hit
if any(h[6] and h[5] <= B0 for h in hits): B0 -= bpb  # grace before bar 1 -> prepend a bar

notes = {}  # (seconds in audio: exact attack, or on the quantized grid, pitch) -> velocity
for h in hits:
    t = h[0] if A.no_quantize else float(to_time(h[5])) - (FLAM_MS if h[6] else 0)
    key = (round(t, 5), h[2])
    notes[key] = max(notes.get(key, 0), h[3])
t_bar1 = float(to_time(B0))

# ---------- MIDI file (tick 0 = bar 1); tempo map follows the beat grid
tpb = 480
mid = mido.MidiFile(ticks_per_beat=tpb); tr = mido.MidiTrack(); mid.tracks.append(tr)
last_b = max(h[5] for h in hits)
ev = [(0, 0, mido.MetaMessage('time_signature', numerator=num, denominator=den))]
prev = None
for b in np.arange(B0, last_b + 1):
    tempo = mido.bpm2tempo(60 / (to_time(b + 1) - to_time(b)))
    if prev is None or abs(tempo - prev) > 500:  # skip near-identical tempo events
        ev.append((int(round((b - B0) * tpb)), 0, mido.MetaMessage('set_tempo', tempo=tempo)))
        prev = tempo
for (t, p), v in notes.items():
    a = int(round((to_beat(t) - B0) * tpb))
    ev += [(a, 2, mido.Message('note_on', channel=9, note=p, velocity=int(v))),
           (a + tpb // 8, 1, mido.Message('note_off', channel=9, note=p, velocity=0))]
last = 0
for t, _, m in sorted(ev, key=lambda e: (e[0], e[1])):
    m.time = t - last; tr.append(m); last = t
mid.save(A.dst)

if A.notes:  # plain text for the REAPER script
    with open(A.notes, 'w') as fh:
        fh.write(f'METER {num} {den}\nBAR1 {t_bar1:.6f}\n')
        for b in np.arange(B0, last_b + 2):
            fh.write(f'BEAT {to_time(b):.6f} {60 / (to_time(b + 1) - to_time(b)):.4f}\n')
        for (t, p), v in sorted(notes.items()):
            fh.write(f'NOTE {t:.6f} {p} {int(v)}\n')

# ---------- report
tempi = 60 / np.diff(to_time(np.arange(B0, last_b + 1)))
print('stems: ' + os.path.dirname(stems['kick']) if stems else 'mix mode (no stems found)')
print(f"grid {mode}, tempo {np.median(tempi):.2f} BPM (range {tempi.min():.1f}-{tempi.max():.1f}), "
      f"meter {num}/{den}, bar 1 at {t_bar1:.3f}s in audio")
print(f"{k} tom(s) at {[round(np.mean(c)) for c in clusters]} Hz, "
      f"{sum(h[6] for h in hits)} flam grace note(s), {len(notes)} notes")
cnt = {}
for (_, p) in notes: cnt[LABEL[p]] = cnt.get(LABEL[p], 0) + 1
print('counts:', ' '.join(f'{c}={n}' for c, n in sorted(cnt.items())))
print("K kick  S snare  1-4 tom  h closed hat  o half-open  O open  C crash  R ride  ? unknown (note 34)  ~ flam  (column = 16th)")
grid = {}
for h in hits:
    q = int(round((h[5] - B0) * 4))
    c = ('~' if h[6] else '') + LABEL[h[2]]
    if c not in grid.get(q, ''): grid[q] = grid.get(q, '') + c
spb = int(bpb * 4)
for q in range(0, min(max(grid) + 1, int(os.environ.get('SHOW_BARS', 8)) * spb)):
    if q % spb == 0: print(f"\nbar {q // spb + 1:2}: ", end='')
    print(grid.get(q, '.').ljust(4), end='')
print()
