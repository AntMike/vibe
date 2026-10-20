# /// script
# requires-python = ">=3.10"
# ///
"""Cut [start, end) seconds from a PCM wav: cut_clip.py <in.wav> <start_s> <end_s> <out.wav>"""
import sys
import wave
from pathlib import Path

src, t0, t1, dst = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
with wave.open(src, "rb") as r:
    sr = r.getframerate()
    r.setpos(int(t0 * sr))
    frames = r.readframes(int((t1 - t0) * sr))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(dst, "wb") as w:
        w.setparams(r.getparams())
        w.writeframes(frames)
print(f"{dst}: {len(frames) // r.getsampwidth() // r.getnchannels() / sr:.3f}s @ {sr} Hz")
