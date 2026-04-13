#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pdf2image import convert_from_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a sorted PDF/PNG sequence into an animated GIF."
    )
    parser.add_argument("input_dir", help="Directory containing PDF/PNG frames")
    parser.add_argument(
        "output_gif",
        nargs="?",
        help="Output GIF path. Defaults to <input_dir>.gif",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Filename glob used to find frames. Default: %(default)s",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=120,
        help="Frame duration in milliseconds. Default: %(default)s",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count. Default: %(default)s (infinite)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_gif = Path(args.output_gif) if args.output_gif else input_dir.with_suffix(".gif")
    frames = sorted(input_dir.glob(args.glob))
    if not frames:
        raise SystemExit(f"No frames matched {args.glob} in {input_dir}")

    images = []
    for frame in frames:
        if frame.suffix.lower() == ".png":
            img = Image.open(frame)
            images.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
        elif frame.suffix.lower() == ".pdf":
            image = convert_from_path(frame)
            img = image[0]
            images.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=args.loop,
        optimize=False,
    )
    print(output_gif)


if __name__ == "__main__":
    main()
