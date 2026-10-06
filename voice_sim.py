#!/usr/bin/env python3
"""声質が参照音声にどれだけ寄ったかを機械で見る(耳の代替ではなく、効いているかの確認)。

    python3 voice_sim.py <参照音声> <候補> [<候補>...]

やること2つ:
- 声の帯(200〜4000Hz)の長時間平均スペクトル(LTAS)を正規化して、参照とのコサイン類似度を出す
- 自己相関で F0(基本周波数)を拾い、周期性の強いフレームだけの中央値を出す

伴奏が違うと LTAS は伴奏に引っぱられるので、**同じ編曲・同じ条件で参照ありとなしを
並べたときの差**だけを根拠にする。
"""
import argparse
import os

import numpy as np
import soundfile as sf

SR = 24000


def load(path):
    a, sr = sf.read(path, always_2d=True)
    x = a.mean(1)
    if sr != SR:  # 素朴な線形リサンプル(比較用なので十分)
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x.astype(np.float32)


def ltas(x):
    n = 2048
    hop = 1024
    w = np.hanning(n)
    frames = [x[i:i + n] * w for i in range(0, len(x) - n, hop)]
    p = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], 0)
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f >= 200) & (f <= 4000)
    v = np.log10(p[band] + 1e-12)
    return (v - v.mean()) / (v.std() + 1e-12)


def f0_median(x):
    n = 1024
    hop = 512
    lo, hi = int(SR / 500), int(SR / 80)   # 80〜500Hz を探す
    vals = []
    for i in range(0, len(x) - n, hop):
        f = x[i:i + n] - x[i:i + n].mean()
        e = np.sqrt(np.mean(f ** 2))
        if e < 0.02:
            continue
        r = np.correlate(f, f, "full")[n - 1:]
        r = r / (r[0] + 1e-12)
        seg = r[lo:hi]
        k = int(np.argmax(seg))
        if seg[k] > 0.45:          # 周期性が強いフレームだけ
            vals.append(SR / (lo + k))
    return float(np.median(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="参照音声との声質の近さを測る")
    ap.add_argument("ref", help="参照音声")
    ap.add_argument("candidates", nargs="+", help="比べる音声")
    a = ap.parse_args()

    ref = load(a.ref)
    rl, rf = ltas(ref), f0_median(ref)
    print(f"{'file':38s} {'声帯域の類似度':>10s} {'F0中央値':>10s}")
    print(f"{os.path.basename(a.ref):38s} {1.0:10.3f} {rf:9.1f}Hz  ←参照")
    for p in a.candidates:
        x = load(p)
        sim = float(np.dot(rl, ltas(x)) / len(rl))
        print(f"{os.path.basename(p):38s} {sim:10.3f} {f0_median(x):9.1f}Hz")


if __name__ == "__main__":
    main()
