"""合成した「歌っぽい音」で各道具を通しで動かす。

Aメロ相当(低め)→無音→サビ相当(4半音以上高く伸ばす)の倍音つき音を作り、
サビが上がったことを各道具が数字で拾えるかを見る。
lyrics_adherence は書き起こしモデルを使わず、採点部分だけを検査する。
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parents[1]
SR = 22050


def tone(freqs, sec_each, sr=SR, seed=0):
    """倍音つきの音を音高の列どおりに並べる(軽いビブラートとノイズ入り)。"""
    rng = np.random.default_rng(seed)
    out = []
    for f0 in freqs:
        t = np.arange(int(sec_each * sr)) / sr
        f = f0 * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
        ph = 2 * np.pi * np.cumsum(f) / sr
        x = sum(np.sin(k * ph) / k for k in range(1, 7))
        env = np.minimum(1, np.minimum(t, t[::-1]) / 0.03)
        out.append(0.25 * x * env)
    y = np.concatenate(out)
    return (y + 0.003 * rng.standard_normal(len(y))).astype(np.float32)


def song(sr=SR):
    hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
    verse = tone([hz(m) for m in (64, 65, 67, 65, 64, 62, 64, 65, 67, 65, 64, 62)], 0.6, sr)  # E4〜G4
    gap = np.zeros(int(1.5 * sr), dtype=np.float32)
    chorus = tone([hz(m) for m in (72, 74, 76, 76)], 0.9, sr, seed=1)                  # C5〜E5
    return np.concatenate([verse, gap, chorus, gap[: sr // 2]])


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp()
        cls.wav = os.path.join(cls.td, "fixture.wav")
        sf.write(cls.wav, song(), SR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def run_tool(self, *args):
        r = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return r.stdout

    def test_songscope_finds_high_chorus(self):
        out = os.path.join(self.td, "out")
        self.run_tool(str(HERE / "songscope.py"), self.wav, "-o", out)
        self.assertTrue(os.path.getsize(os.path.join(out, "fixture.png")) > 10_000)
        with open(os.path.join(out, "fixture.json"), encoding="utf-8") as f:
            s = json.load(f)
        self.assertGreater(s["旋律が取れた割合"], 0.5)
        self.assertIn(s["最高音"], ("D#5", "E5", "F5"))
        self.assertGreaterEqual(s["張りの回数"], 1)
        self.assertGreater(s["その時刻_秒"], 8.0)   # Aメロ7.2秒+無音1.5秒の後
        self.assertGreaterEqual(s["張りと地の差_半音"], 4)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg が無い")
    def test_melody_read_sections_rise(self):
        out = self.run_tool(str(HERE / "melody_read.py"), self.wav)
        self.assertIn("歌の区間 2本", out)
        rows = [l for l in out.splitlines() if l.strip()[:1].isdigit() and "Hz" in l]
        self.assertEqual(len(rows), 2, out)
        top = [float(r.split()[-2].rstrip("Hz")) for r in rows]
        self.assertGreater(top[1], top[0] * 2 ** (3 / 12), out)

    def test_voice_sim_ranks_same_voice_higher(self):
        same = os.path.join(self.td, "same.wav")
        other = os.path.join(self.td, "other.wav")
        sf.write(same, song(), SR)
        noise = np.random.default_rng(3).standard_normal(len(song())).astype(np.float32) * 0.1
        sf.write(other, noise, SR)
        out = self.run_tool(str(HERE / "voice_sim.py"), self.wav, same, other)
        sims = {l.split()[0]: float(l.split()[1]) for l in out.splitlines()[2:]}
        self.assertGreater(sims["same.wav"], 0.99)
        self.assertLess(sims["other.wav"], sims["same.wav"])

    def test_lyrics_score(self):
        la = load_module("lyrics_adherence")
        lines = [la.norm(x) for x in ("あさのひかりがまぶしい", "きみのなまえをよんだ", "あさのひかりがまぶしい")]
        hyp = [(1.0, la.norm("朝の光がまぶしい")), (3.0, "きみのなまえをよんだ"), (5.0, "あさのひかりがまぶしい")]
        cov, inv, missed = la.score(lines, hyp, 0.45)
        self.assertEqual(cov, 1.0)
        self.assertEqual(inv, 0)
        cov, inv, missed = la.score(lines + ["まったくちがうことばです"], hyp, 0.45)
        self.assertLess(cov, 1.0)
        self.assertEqual(missed, ["まったくちがうことばです"])
        swapped = [(3.0, "あさのひかりがまぶしい"), (1.0, "きみのなまえをよんだ")]
        _, inv, _ = la.score(["きみのなまえをよんだ", "あさのひかりがまぶしい", "ほかのぎょうがここにある"],
                             swapped + [(0.5, "ほかのぎょうがここにある")], 0.45)
        self.assertEqual(inv, 1)   # 時刻の並びが 1.0 → 3.0 → 0.5 で、戻りは1回


if __name__ == "__main__":
    unittest.main()
