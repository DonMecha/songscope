#!/usr/bin/env python3
"""曲を「見える形」に変える。音を聴けない側が品質を判定するための道具。

    python3 songscope.py <音源> [-o out_dir] [--fmin E3 --fmax D6]

**なぜ要るか。** AIエージェントは音を聴けない。だが**絵は見られる**ので、音を絵と数字へ
翻訳すれば判定に手が届く。翻訳するのは次の3つで、どれも「良い曲かどうか」ではなく
**「狙った形になっているか」**を測る。

1. **旋律(F0)** — librosa.pyin でボーカルの高さを追う。サビがAメロより上がっているか、
   最高音がどこで出ているか、音域はどれだけあるかを数字にする。
2. **勢い(RMS)と明るさ(spectral centroid)** — サビで音が厚く・明るくなっているか。
   「サビで開く」形を狙った曲で、ここが平らなテイクは形になっていない。
3. **メルスペクトログラム** — 上2つを重ねてPNGにする。**これを目で見る。**

**サビは音量では見つからない。** マスタリングで潰れた曲は音量が真っ平らになる。
代わりに**旋律の高止まり**で切る——中央音より4半音以上高い所に0.4秒以上留まる区間を
「張り」とみなす。サビで高い音を伸ばす形の曲では、これがサビと山場の代理になる。
出力は `<out>/<名前>.png` と `<名前>.json`。
"""
import argparse, json, os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
        break

SR = 22050
# 既定は**女声の歌だけを追う帯**。下限を低く取りすぎると pyin がベースを掴み、
# 中央音がベースの高さになる。男声なら --fmin C2 --fmax G5 などに広げる。
FMIN_NOTE, FMAX_NOTE = "E3", "D6"


def analyze(path, sr=SR, fmin=FMIN_NOTE, fmax=FMAX_NOTE):
    y, sr = librosa.load(path, sr=sr, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    # ハーモニック成分だけ残し、さらに歌の帯へ絞ってから高さを追う
    yh = librosa.effects.harmonic(y, margin=3.0)
    yv = librosa.effects.preemphasis(yh, coef=0.97)
    f0, voiced, _ = librosa.pyin(yv, fmin=librosa.note_to_hz(fmin), fmax=librosa.note_to_hz(fmax), sr=sr, hop_length=hop,
                                 fill_na=np.nan)
    tf0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    # 高域比 = 2kHz以上のエネルギー / 全体。サビで音が開くと上がる
    Sm = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    hf = Sm[freqs >= 2000].sum(0) / (Sm.sum(0) + 1e-9)
    return dict(y=y, sr=sr, dur=len(y) / sr, t=t, rms=rms, cen=cen, hf=hf,
                f0=f0, tf0=tf0, tempo=tempo)


def plateaus(a, up=4.0, min_sec=0.4):
    """旋律の高止まり(=張っている所)を拾う。サビと山場の代理。"""
    f0, t = a["f0"], a["tf0"]
    ok = np.isfinite(f0)
    if ok.sum() < 10:
        return [], None
    midi = np.full(len(f0), np.nan)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    med = np.nanmedian(midi)
    high = np.nan_to_num(midi, nan=-99) >= med + up
    dt = t[1] - t[0]
    segs, i = [], 0
    while i < len(high):
        if not high[i]:
            i += 1
            continue
        j = i
        while j < len(high) and high[j]:
            j += 1
        if (j - i) * dt >= min_sec:
            segs.append(dict(開始=round(float(t[i]), 1), 長さ=round((j - i) * dt, 2),
                             音高=round(float(np.nanmean(midi[i:j])), 1)))
        i = j
    return segs, med


def summarize(a):
    f0 = a["f0"]
    ok = np.isfinite(f0)
    midi = librosa.hz_to_midi(f0[ok]) if ok.any() else np.array([])
    db = librosa.amplitude_to_db(a["rms"], ref=np.max)
    hi = db >= np.percentile(db, 90)
    lo = db <= np.percentile(db, 40)
    out = {
        "尺_秒": round(a["dur"], 2),
        "BPM推定": round(a["tempo"], 1),
        "旋律が取れた割合": round(float(ok.mean()), 3),
        "音域_半音": round(float(midi.max() - midi.min()), 1) if midi.size else None,
        "最高音": librosa.midi_to_note(float(midi.max())) if midi.size else None,
        "最高音の時刻_秒": round(float(a["tf0"][ok][int(np.argmax(midi))]), 1) if midi.size else None,
        "中央音": librosa.midi_to_note(float(np.median(midi))) if midi.size else None,
        "強い所と弱い所の差_dB": round(float(db[hi].mean() - db[lo].mean()), 1),
        "強い所の明るさ_Hz": round(float(a["cen"][hi].mean())),
        "弱い所の明るさ_Hz": round(float(a["cen"][lo].mean())),
        "強い所の高域比": round(float(a["hf"][hi].mean()), 3),
        "弱い所の高域比": round(float(a["hf"][lo].mean()), 3),
    }
    segs, med = plateaus(a)
    if segs:
        top = max(segs, key=lambda s: s["音高"])
        longest = max(segs, key=lambda s: s["長さ"])
        out["張りの回数"] = len(segs)
        out["張りの合計_秒"] = round(sum(s["長さ"] for s in segs), 1)
        out["いちばん高い張り"] = librosa.midi_to_note(top["音高"])
        out["その時刻_秒"] = top["開始"]
        out["最高音は後半3割か"] = top["開始"] / a["dur"] >= 0.7
        out["いちばん長い張り_秒"] = longest["長さ"]
        out["張りと地の差_半音"] = round(float(np.mean([s["音高"] for s in segs]) - med), 1)
    else:
        out["張りの回数"] = 0
    out["_張り"] = segs
    return out


def draw(a, path, title, segs=None):
    y, sr = a["y"], a["sr"]
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.4, 1.4]})
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", fmax=8000,
                             ax=ax[0], cmap="magma")
    ok = np.isfinite(a["f0"])
    ax[0].plot(a["tf0"][ok], a["f0"][ok], color="#00e5ff", lw=1.6)
    ax[0].set_title(title, fontsize=13)
    ax[0].set_ylim(0, 8000)
    db = librosa.amplitude_to_db(a["rms"], ref=np.max)
    ax[1].plot(a["t"], db, color="#ff5252")
    ax[1].axhline(np.percentile(db, 90), color="#888", ls="--", lw=.8)
    ax[1].set_ylabel("loudness dB"); ax[1].grid(alpha=.3)
    if ok.any():
        ax[2].plot(a["tf0"][ok], librosa.hz_to_midi(a["f0"][ok]), ".", ms=2, color="#40c057")
    for s in (segs or []):
        ax[2].axvspan(s["開始"], s["開始"] + s["長さ"], color="#ffd43b", alpha=.35)
    ax[2].set_ylabel("pitch MIDI"); ax[2].set_xlabel("sec"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default="songscope-out")
    ap.add_argument("--fmin", default=FMIN_NOTE, help="旋律を追う下限の音名(既定 E3)")
    ap.add_argument("--fmax", default=FMAX_NOTE, help="旋律を追う上限の音名(既定 D6)")
    g = ap.parse_args()
    os.makedirs(g.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(g.src))[0]
    a = analyze(g.src, fmin=g.fmin, fmax=g.fmax)
    s = summarize(a)
    png = os.path.join(g.out, name + ".png")
    draw(a, png, name, s.pop("_張り", None))
    with open(os.path.join(g.out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    print(json.dumps({"曲": name, **s}, ensure_ascii=False, indent=1))
    print("絵 →", png)


if __name__ == "__main__":
    main()
