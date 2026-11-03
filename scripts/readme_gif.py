"""The README's GIF: rollouts from one start scene under different commands, side by side, from saved eval videos.

    .venv/bin/python scripts/readme_gif.py outputs/min_smolvla/eval/eval_colours --seed 100001 \
        --variants S1:B T1:B:left T1:B:right --out docs/media/same_scene_three_commands.gif \
        --footer "SmolVLA, same start scene (held-out cube colour), 2x speed"

Uses the overview videos `sim.evaluate --videos N` saves (`<variant>_<seed>_ok|fail.mp4`) and each rollout's command
from `episodes.jsonl` in the same directory. Every 8th frame of 50 fps played at 12.5 fps (2x speed); shorter clips
hold their last frame. Needs ffmpeg for the GIF palette.
"""
import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, CAP, FOOT = 320, 240, 82, 30  # panel size of the overview camera; CAP + H + FOOT a multiple of 16 for H.264
STEP, FPS, HOLD = 8, 12.5, 18  # HOLD: extra frames on the last one, ~1.5 s
FONTS = "/System/Library/Fonts/Supplemental"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="an evaluation condition directory with videos and episodes.jsonl")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variants", nargs="+", required=True, help="e.g. S1:B T1:B:left")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--footer", default="SmolVLA, same start scene, 2x speed")
    a = p.parse_args()

    episodes = {}
    for line in (a.run / "episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["seed"] == a.seed:
            episodes[":".join(x for x in e["variant"] if x)] = e
    videos = []
    for v in a.variants:
        (path,) = a.run.glob(f"{v.replace(':', '_')}_{a.seed}_*.mp4")
        videos.append(iio.imread(path)[::STEP])
        print(f"{v}: {path.name}, success {episodes[v]['success']}, cube {episodes[v]['cube_colour']}")

    k = len(a.variants)
    bold = ImageFont.truetype(f"{FONTS}/Arial Bold.ttf", 14)
    reg = ImageFont.truetype(f"{FONTS}/Arial.ttf", 12)
    header = Image.new("RGB", (k * W, CAP), "white")
    d = ImageDraw.Draw(header)
    for i, v in enumerate(a.variants):
        d.text((i * W + 8, 6), v, font=bold, fill=(20, 20, 20))
        for j, line in enumerate(textwrap.wrap(f'"{episodes[v]["command"]}"', 50)):
            d.text((i * W + 8, 26 + 15 * j), line, font=reg, fill=(70, 70, 70))
    footer = Image.new("RGB", (k * W, FOOT), "white")
    ImageDraw.Draw(footer).text((8, 8), a.footer, font=reg, fill=(110, 110, 110))

    frames = []
    for t in range(max(len(v) for v in videos) + HOLD):
        canvas = Image.new("RGB", (k * W, CAP + H + FOOT), "white")
        canvas.paste(header, (0, 0))
        for i, v in enumerate(videos):
            canvas.paste(Image.fromarray(v[min(t, len(v) - 1)]), (i * W, CAP))
        canvas.paste(footer, (0, CAP + H))
        frames.append(np.asarray(canvas))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "panels.mp4"
        iio.imwrite(mp4, np.stack(frames), fps=FPS, codec="libx264", quality=9)
        palette = f"fps={FPS},split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf", palette, "-loop", "0", a.out], check=True)
    print(f"{a.out}: {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
