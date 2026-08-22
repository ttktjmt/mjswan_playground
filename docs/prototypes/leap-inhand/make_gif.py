"""Turn the recorded webm into a GIF of a target playback length.

Playwright's bundled ffmpeg can mux webm and encode PNG, nothing else — so ffmpeg
decodes to frames and Pillow does the palette and the GIF.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FFMPEG = "/opt/pw-browsers/ffmpeg-1011/ffmpeg-linux"
VFRAMES = HERE / "vframes"

SOURCE_FPS = 25  # Playwright records at 25


def extract(video: Path, crop: str, width: int) -> list[Path]:
    if VFRAMES.exists():
        shutil.rmtree(VFRAMES)
    VFRAMES.mkdir()
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-r", str(SOURCE_FPS), "-i", str(video),
            "-vf", f"crop={crop},scale={width}:-1",
            "-c:v", "png", "-f", "image2", str(VFRAMES / "f_%05d.png"),
        ],
        check=True,
    )
    return sorted(VFRAMES.glob("*.png"))


def first_scene_frame(paths: list[Path]) -> int:
    """First frame the cube is actually on screen.

    Brightness will not do: the app's "Loading scene…" page is dark too, so a
    dark-frame test starts the clip on the spinner. The cube is the one saturated
    orange thing in a blue-grey scene (`rgba 0.85 0.3 0.15`), so look for its pixels.
    """
    for i in range(0, len(paths), 5):
        rgb = np.asarray(Image.open(paths[i]).convert("RGB"), dtype=np.int16)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cube = (red > 90) & (red - green > 40) & (red - blue > 40)
        if cube.mean() > 0.01:  # ~1% of the frame; the cube is far bigger once shown
            return min(i + SOURCE_FPS, len(paths) - 1)
    return 0


def build(paths: list[Path], seconds: float, fps: int, colors: int, out: Path) -> None:
    start = first_scene_frame(paths)
    usable = paths[start:]
    want = max(2, int(seconds * fps))
    stride = max(1, len(usable) // want)
    sel = usable[::stride][:want]

    frames = [
        Image.open(p).convert("RGB").quantize(
            colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG
        )
        for p in sel
    ]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
    )
    covered = len(sel) * stride / SOURCE_FPS
    print(
        f"{out.name}: {out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, "
        f"{frames[0].size[0]}x{frames[0].size[1]}, {len(frames) / fps:.1f}s playback, "
        f"{covered:.0f}s of browser time at {stride * fps / SOURCE_FPS:.1f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--crop", default="800:620:110:100")
    ap.add_argument("--out", type=Path, default=HERE / "leap_inhand_rotation.gif")
    args = ap.parse_args()

    video = args.video or max((HERE / "video").glob("*.webm"), key=lambda p: p.stat().st_size)
    print(f"source: {video.name} ({video.stat().st_size / 1e6:.1f} MB)")
    paths = extract(video, args.crop, args.width)
    print(f"decoded {len(paths)} frames")
    build(paths, args.seconds, args.fps, args.colors, args.out)


if __name__ == "__main__":
    main()
