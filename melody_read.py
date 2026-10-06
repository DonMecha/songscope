#!/usr/bin/env python3
"""**分離したボーカルのステム**から旋律を読む。

    python3 melody_read.py <ボーカルのステム> [--gap 1.2]

2mix では伴奏に引っぱられて旋律を追えない。**Lead Vocal のステムだけ**に
すればHPSで基音が取れる。この道具はそれを前提に、次を数字で出す。

- **歌っている区間**(無音で切る)とその尺
- 区間ごとの **中央値の音高 / 最高音**(Hzと音名)
- 全体の **音域**、**最高音がどこか**

サビが上がるかどうかは、**区間ごとの最高音を並べて**見る。Aメロの区間より
サビの区間が上でなければ、そのテイクは捨てる。
"""
import subprocess, os, argparse, tempfile
import numpy as np
from scipy import signal

NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note(hz):
    if not np.isfinite(hz) or hz <= 0: return '-'
    n = int(round(69 + 12 * np.log2(hz / 440.0)))
    return f'{NAMES[n % 12]}{n // 12 - 1}'


def load(path, sr=16000):
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f: tmp = f.name
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-ac', '1',
                    '-ar', str(sr), '-f', 'f32le', tmp], check=True)
    x = np.fromfile(tmp, dtype=np.float32); os.unlink(tmp)
    return x, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('--gap', type=float, default=1.2)
    ap.add_argument('--sr', type=int, default=16000)
    a = ap.parse_args()
    x, sr = load(a.src, a.sr)

    f, t, Z = signal.stft(x, fs=sr, nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    rms = 20 * np.log10(np.sqrt(P.sum(axis=0)) + 1e-12)
    live = rms > (np.percentile(rms, 92) - 26)      # 歌っているか

    Pv = P.copy(); Pv[(f < 140) | (f > 3000)] = 0
    lo, hi = np.searchsorted(f, 150), np.searchsorted(f, 700)
    pit = np.full(P.shape[1], np.nan)
    for i in range(P.shape[1]):
        if not live[i]: continue
        c = Pv[:, i]
        if c.sum() < 1e-12: continue
        acc = c.copy()
        for h in (2, 3, 4):
            d = c[::h]; acc[:len(d)] = acc[:len(d)] * d
        k = lo + int(np.argmax(acc[lo:hi]))
        pit[i] = f[k]

    # 歌っている区間へまとめる
    dt = t[1] - t[0]
    segs, s = [], None
    silence = 0
    for i in range(len(live)):
        if live[i]:
            if s is None: s = i
            silence = 0
        else:
            if s is not None:
                silence += dt
                if silence >= a.gap:
                    if (t[i] - t[s]) > 3.0: segs.append((s, i))
                    s = None; silence = 0
    if s is not None and (t[-1] - t[s]) > 3.0: segs.append((s, len(live) - 1))

    print(f'{os.path.basename(a.src)}  尺 {len(x)/sr:.1f}s  歌の区間 {len(segs)}本')
    print(f'{"#":>2} {"入り":>7} {"尺":>6} {"中央":>10} {"最高":>10}')
    tops = []
    for n, (i0, i1) in enumerate(segs, 1):
        p = pit[i0:i1]; p = p[np.isfinite(p)]
        if len(p) < 5: continue
        med, top = np.median(p), np.percentile(p, 97)
        tops.append(top)
        print(f'{n:>2} {t[i0]:>6.1f}s {t[i1]-t[i0]:>5.1f}s '
              f'{med:>6.0f}Hz {note(med):>4} {top:>6.0f}Hz {note(top):>4}')
    allp = pit[np.isfinite(pit)]
    if len(allp):
        k = int(np.nanargmax(np.where(np.isfinite(pit), pit, -1)))
        print(f'音域 {np.percentile(allp,3):.0f}Hz({note(np.percentile(allp,3))}) '
              f'〜 {np.percentile(allp,99):.0f}Hz({note(np.percentile(allp,99))}) / '
              f'最高音 {pit[k]:.0f}Hz({note(pit[k])}) @ {t[k]:.1f}s')
        if len(tops) >= 2:
            print(f'区間の最高音のばらつき {min(tops):.0f}〜{max(tops):.0f}Hz '
                  f'(差 {12*np.log2(max(tops)/min(tops)):.1f}半音)')


if __name__ == '__main__':
    main()
