"""Downscale the image features of a LeRobot dataset (image dtype, not video).

Decoding 640x480 PNGs on the CPU is the training bottleneck. This writes a copy of the
dataset where every camera frame is resized (default 320x240) and re-encoded as JPEG,
which decodes several times faster. Everything else (actions, state, stats, episodes
metadata) is copied unchanged; only the image shapes in meta/info.json are updated.

Usage:
    python downscale_dataset.py SRC_ROOT DST_ROOT [--height 240] [--width 320] [--quality 95]

Only needed for datasets recorded at 640x480 (the first dataset version). Current
recordings are already 320x240.

Example:
    python scripts/downscale_dataset.py \
        ~/.cache/huggingface/lerobot/MattoTT/ur5e_pick_place \
        ~/.cache/huggingface/lerobot/local/ur5e_pick_place_small
"""

import argparse
import io
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_SIZE: tuple[int, int] = (320, 240)  # (W, H) as PIL expects
_QUALITY: int = 95


def _init_worker(size: tuple[int, int], quality: int) -> None:
    global _SIZE, _QUALITY
    _SIZE, _QUALITY = size, quality


def _resize(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # PIL's bilinear resize is antialiased when downscaling.
    img = img.resize(_SIZE, Image.Resampling.BILINEAR)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=_QUALITY)
    return out.getvalue()


def _resize_column(col: pa.StructArray, pool: ProcessPoolExecutor) -> pa.StructArray:
    # Images are stored by HF `datasets` as struct<bytes: binary, path: string>.
    field_names = [col.type.field(i).name for i in range(col.type.num_fields)]
    children = dict(zip(field_names, col.flatten()))
    raw = children["bytes"].to_pylist()
    if any(b is None for b in raw):
        raise SystemExit("Found an image without embedded bytes; is this dataset's data/ complete?")

    new_bytes = list(pool.map(_resize, raw, chunksize=32))
    children["bytes"] = pa.array(new_bytes, type=col.type.field("bytes").type)
    return pa.StructArray.from_arrays(
        [children[n] for n in field_names],
        fields=[col.type.field(i) for i in range(col.type.num_fields)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Source dataset root (contains meta/ and data/)")
    parser.add_argument("dst", type=Path, help="Output dataset root (must not exist)")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"{dst} already exists; delete it or choose another path.")

    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    img_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "image"]
    if not img_keys:
        raise SystemExit("No features with dtype 'image' found; nothing to do.")
    print(f"Image features: {img_keys} -> {args.height}x{args.width} JPEG q{args.quality}")

    # Copy everything except the frame data (and any leftover temp images/ dir).
    shutil.copytree(src, dst, ignore=lambda d, names: ["data", "images"] if Path(d) == src else [])

    files = sorted((src / "data").rglob("*.parquet"))
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=((args.width, args.height), args.quality),
    ) as pool:
        for i, f in enumerate(files, 1):
            table = pq.read_table(f)
            schema_meta = table.schema.metadata  # keeps HF `datasets` feature info
            for key in img_keys:
                idx = table.schema.get_field_index(key)
                new_col = _resize_column(table.column(idx).combine_chunks(), pool)
                table = table.set_column(idx, table.schema.field(idx), new_col)
            table = table.replace_schema_metadata(schema_meta)

            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out)
            print(f"[{i}/{len(files)}] {f.relative_to(src)}  {table.num_rows} rows  ({time.time() - t0:.0f}s)")

    for key in img_keys:
        info["features"][key]["shape"] = [args.height, args.width, 3]
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    def size_gb(p: Path) -> float:
        return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1e9

    print(f"Done in {time.time() - t0:.0f}s. Size: {size_gb(src):.2f} GB -> {size_gb(dst):.2f} GB")
    print(f"Train with: --dataset.root={dst}")


if __name__ == "__main__":
    main()
