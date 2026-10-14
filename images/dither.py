"""
4x4 Bayer dither → 16 colours.

  python images/dither.py photo.jpg images/messages/2026-10-name.png
  python images/dither.py photo.jpg images/projects/foo.png --preset square
  python images/dither.py photo.jpg out.png --size 400x400
  python images/dither.py photo.jpg out.png --keep-size

Presets: messages 396x704, square 320x320, wide max 800 wide, album max 512 long side.
Skip: pfp, favicons, album.gif, diary.gif.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

COLORS = 16
STRENGTH = 32       # dither grain (0–64)
SATURATION = 1.06   # 1 = natural
BLACK_LIFT = 0      # raise shadows; 0 keeps real black
WARMTH = 0          # amber in shadows; 0 = none

BAYER = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float32,
)

PRESETS = {
    "messages": ("fill", (396, 704)),
    "square": ("fill", (320, 320)),
    "wide": ("fit", 800),
    "album": ("fit", 512),
}

SKIP = {"pfp.jpg", "pfp.png", "favicon.ico", "album.gif", "diary.gif"}


def crop_fill(im, tw, th):
    """Crop to aspect, then resize."""
    w, h = im.size
    if w / h > tw / th:
        nw = round(h * tw / th)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = round(w * th / tw)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize((tw, th), Image.Resampling.LANCZOS)


def fit_long_side(im, limit):
    """Shrink longest side to limit; keep aspect."""
    scale = limit / max(im.size)
    if scale >= 1:
        return im
    return im.resize(
        (round(im.width * scale), round(im.height * scale)),
        Image.Resampling.LANCZOS,
    )


def grade(im, saturation=SATURATION, black=BLACK_LIFT, warmth=WARMTH):
    """Light grade before picking the palette."""
    im = ImageEnhance.Color(im).enhance(saturation)
    im = ImageEnhance.Contrast(im).enhance(1.04)
    arr = np.asarray(im, dtype=np.float32)

    if black > 0:
        arr = black + arr * (255 - black) / 255

    if warmth > 0:
        shadow = 1 - arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32) / 255
        tint = np.array([1.0, -0.1, -1.1], dtype=np.float32) * warmth
        arr += shadow[:, :, None] * tint

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_palette(im, colors=COLORS):
    """16 colours from the photo, plus a couple bright accents if needed."""
    pal_img = im.quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    pal = np.array(pal_img.getpalette()[: colors * 3], dtype=np.float32).reshape(
        colors, 3
    )

    arr = np.asarray(im, dtype=np.float32).reshape(-1, 3)
    chroma = arr.max(axis=1) - arr.min(axis=1)
    injected = 0
    for i in np.argsort(chroma)[::-1][: min(4000, len(arr))]:
        if chroma[i] < 40:
            break
        c = arr[i]
        if float(np.linalg.norm(pal - c, axis=1).min()) < 28:
            continue
        # swap out the dullest slot for this accent colour
        pal_chroma = pal.max(axis=1) - pal.min(axis=1)
        pal[int(pal_chroma.argmin())] = c
        injected += 1
        if injected >= 2:
            break
    return pal


def dither(im, colors=COLORS, strength=STRENGTH, palette=None):
    """Map pixels to palette with a 4x4 Bayer pattern."""
    if palette is None:
        palette = make_palette(im, colors)
    else:
        palette = np.asarray(palette, dtype=np.float32).reshape(-1, 3)
        colors = len(palette)

    arr = np.asarray(im, dtype=np.float32)
    h, w = arr.shape[:2]
    tile = np.tile(BAYER / 16 - 0.5, (h // 4 + 1, w // 4 + 1))[:h, :w]
    nudged = arr + tile[:, :, None] * strength

    diff = nudged.reshape(-1, 1, 3) - palette[None]
    index = (diff * diff).sum(2).argmin(1).astype(np.uint8)

    out = Image.fromarray(index.reshape(h, w), mode="P")
    out.putpalette([int(v) for v in palette.ravel()] + [0] * (768 - colors * 3))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("src", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--preset", choices=sorted(PRESETS), default="messages")
    p.add_argument("--size", help="WxH, e.g. 400x400")
    p.add_argument("--keep-size", action="store_true")
    p.add_argument("--strength", type=float, default=STRENGTH)
    p.add_argument("--saturation", type=float, default=SATURATION)
    p.add_argument("--black", type=float, default=BLACK_LIFT)
    p.add_argument("--warmth", type=float, default=WARMTH)
    p.add_argument(
        "--palette-from-full",
        action="store_true",
        help="pick colours from the full photo so crop + full match",
    )
    args = p.parse_args()

    if not args.src.is_file():
        sys.exit(f"not found: {args.src}")
    if args.src.name.lower() in SKIP:
        sys.exit(f"refusing to dither protected file: {args.src.name}")

    graded = grade(
        Image.open(args.src).convert("RGB"),
        args.saturation,
        args.black,
        args.warmth,
    )
    palette = make_palette(graded) if args.palette_from_full else None

    if args.keep_size:
        framed = graded
    elif args.size:
        w, h = (int(n) for n in args.size.lower().split("x"))
        framed = crop_fill(graded, w, h)
    else:
        mode, value = PRESETS[args.preset]
        framed = (
            crop_fill(graded, *value) if mode == "fill" else fit_long_side(graded, value)
        )

    if palette is None:
        palette = make_palette(framed)

    out = dither(framed, strength=args.strength, palette=palette)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.dest, format="PNG", optimize=True)
    print(f"wrote {args.dest} ({out.width}x{out.height}, {COLORS} colours, 4x4 Bayer)")


if __name__ == "__main__":
    main()
