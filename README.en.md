# songscope

[日本語](README.md) | [English](README.en.md)

AI agents cannot hear. They still need to check whether a generated song came out the way it was meant to. songscope is a small set of tools that turns a song into **pictures and numbers**, so takes can be checked without listening. It is meant for comparing and picking takes from music-generation AI.

None of the tools judge whether a song is good. They measure whether it has the **intended shape**: does the chorus rise, does the singer follow the lyrics, did the voice move toward the reference.

| Tool | Input | Output |
|---|---|---|
| `songscope.py` | A full mix | A PNG overlaying mel spectrogram, melody and loudness, plus JSON with range, peak position, chorus "lift" and brightness |
| `melody_read.py` | An isolated vocal stem | Median and top pitch per sung section, overall range |
| `lyrics_adherence.py` | A lyrics file and one or more takes | Lyric coverage, order inversions, missed lines |
| `voice_sim.py` | A reference voice and candidates | Spectral similarity in the voice band and median F0 |

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` (numpy, scipy, soundfile, librosa, matplotlib, faster-whisper)
- `melody_read.py` uses ffmpeg
- `lyrics_adherence.py` downloads a faster-whisper model on first run (default `small`, runs on CPU)

## Usage

### songscope.py: turn a song into a picture and numbers

```sh
python3 songscope.py song.mp3 -o songscope-out
```

This writes `songscope-out/song.png` and `songscope-out/song.json`. The PNG shows, from top to bottom: the mel spectrogram with the melody overlaid in cyan, loudness, and pitch over time. Chorus "lift" sections are shaded yellow in the pitch panel.

Main JSON fields (keys are in Japanese):

- `音域_半音` range in semitones, `最高音` top note, `最高音の時刻_秒` time of the top note, `中央音` median note
- `張りの回数` number of lifts, `いちばん高い張り` highest lift, `最高音は後半3割か` whether the top lift is in the last 30%, `張りと地の差_半音` lift height above the median in semitones
- `強い所と弱い所の差_dB` loud/quiet difference, `強い所の明るさ_Hz` / `弱い所の明るさ_Hz` spectral centroid, high-frequency ratios

**You cannot find the chorus by loudness.** Heavily mastered songs are flat in loudness. songscope uses the melody instead: a section that stays at least 4 semitones above the median note for at least 0.4 seconds counts as a "lift" and stands in for the chorus and climaxes.

The melody band defaults to E3–D6, tuned for female vocals. If the lower bound is too low, pyin locks onto the bass and the median becomes the bass pitch. For male vocals, widen it, e.g. `--fmin C2 --fmax G5`.

### melody_read.py: read the melody from a vocal stem

```sh
python3 melody_read.py vocals.wav [--gap 1.2]
```

The melody cannot be tracked reliably in a full mix because the backing pulls it off, so pass only the lead vocal stem. The tool splits sung sections at silences (1.2 s or longer by default) and lists the median and top pitch of each section. Use it to check that the chorus sections peak higher than the verse sections.

### lyrics_adherence.py: is it singing the lyrics?

```sh
python3 lyrics_adherence.py lyrics.txt take1.mp3 take2.mp3 [--model small] [--threshold 0.45] [--language ja]
```

It transcribes each take with faster-whisper and checks every lyric line using the Dice coefficient of character bigrams. The lyrics file has one phrase per line. Lines with only markup such as `[Verse]` and lines shorter than 5 characters are ignored.

- **Coverage**: the share of lyric lines that appear somewhere in the transcript
- **Inversions**: how many times the order of appearance goes backwards (counted only on lines that occur once in the lyrics)

Transcripts of sung Japanese contain many errors, so the threshold is lenient. Do not read the numbers as absolute; use them to **compare takes against the same lyrics with the same threshold**. Scoring currently looks at Japanese characters only (hiragana, katakana, kanji).

### voice_sim.py: did the voice move toward the reference?

```sh
python3 voice_sim.py reference.wav candidate1.wav candidate2.wav
```

It prints the cosine similarity of the long-term average spectrum in the 200–4000 Hz band and the median F0. A different backing track pulls the spectrum, so only rely on the difference between versions with and without the reference under the same arrangement and settings.

## Limitations

- It measures shape only. It does not judge whether a song is good or pleasant.
- Melody estimation gets less reliable with dense harmonic backing or heavy effects.
- Output keys and messages are in Japanese.

## Tests

```sh
python3 -m unittest discover -s tests
```

The tests synthesize a "verse, silence, higher chorus" signal and run `songscope.py`, `melody_read.py` and `voice_sim.py` end to end, checking that the rising chorus shows up in the numbers. `lyrics_adherence.py` is tested on its scoring logic without loading a model.

## License

MIT
