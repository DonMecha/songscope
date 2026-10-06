#!/usr/bin/env python3
"""生成された歌が歌詞どおりに歌っているかを機械で採点する。

    python3 lyrics_adherence.py <歌詞ファイル> <音源> [<音源>...] [--model small] [--threshold 0.45]

出す指標2つ:
- 網羅率: 歌詞の各行が書き起こしのどこかに出ているか(文字bigramのDice係数 >= しきい値 で「出た」)
- 逆行数: 出てきた順が歌詞の順とずれた回数。同じ行の歌い直しや節の入れ替わりで増える

書き起こしは faster-whisper。日本語の歌は誤字が多いので、しきい値は緩めに取っている。
歌詞ファイルは1行1フレーズ。漢字の読み違いを避けたいなら、ひらがな表記にしておくと当たりやすい。
**絶対値でなく、同じ歌詞・同じしきい値で版どうしを比べるために使う。**
"""
import argparse
import os
import re

KANA = re.compile(r"[^ぁ-んァ-ヶー一-龥]")


def norm(s):
    return KANA.sub("", s)


def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)}


def dice(a, b):
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return 2 * len(A & B) / (len(A) + len(B))


def score(lines, hyp, threshold):
    """歌詞の行と書き起こし(時刻, 本文)から、網羅率・逆行数・落ちた行を出す。"""
    hit, missed = [], []
    for i, ln in enumerate(lines):
        best = max(((dice(ln, h), t) for t, h in hyp), default=(0, 0))
        if best[0] >= threshold:
            hit.append((i, best[1]))
        else:
            missed.append(ln[:12])
    # 順序は「歌詞に1回しか出てこない行」だけで見る。
    # サビの繰り返し行は同じ時刻に当たって偽の逆行を作るので除く。
    once = {i for i, l in enumerate(lines) if lines.count(l) == 1}
    seq = [t for i, t in hit if i in once]
    inv = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
    return len(hit) / len(lines), inv, missed


def main():
    ap = argparse.ArgumentParser(description="歌詞の追従を採点する")
    ap.add_argument("lyrics", help="歌詞ファイル(1行1フレーズ)")
    ap.add_argument("audio", nargs="+", help="採点する音源")
    ap.add_argument("--model", default="small", help="faster-whisper のモデル名(既定 small)")
    ap.add_argument("--threshold", type=float, default=0.45, help="行が出たとみなすDice係数")
    ap.add_argument("--language", default="ja")
    a = ap.parse_args()

    with open(a.lyrics, encoding="utf-8") as f:
        lines = [norm(l) for l in f.read().splitlines()]
    lines = [l for l in lines if len(l) >= 5]
    if not lines:
        raise SystemExit("5文字以上の歌詞行がない")

    from faster_whisper import WhisperModel
    m = WhisperModel(a.model, device="cpu", compute_type="int8")
    print(f"{'file':26s} {'網羅率':>7s} {'逆行':>5s}  {'落ちた行'}")
    for path in a.audio:
        segs = list(m.transcribe(path, language=a.language, vad_filter=False)[0])
        hyp = [(s.start, norm(s.text)) for s in segs if norm(s.text)]
        cov, inv, missed = score(lines, hyp, a.threshold)
        print(f"{os.path.basename(path):26s} {cov:6.0%} {inv:5d}  {' / '.join(missed[:6])}")


if __name__ == "__main__":
    main()
