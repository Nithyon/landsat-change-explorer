"""Discover Landsat C2 L2 bands and plot real, QA-masked natural-color images.

Usage: python visualize_landsat.py --data-dir "path/to/existing/data"
No input files are modified. B5 is loaded but RGB uses B4, B3, B2.
"""
from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

DATES = ("2015-12-13", "2024-07-30")
BANDS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "QA_PIXEL")
# USGS QA_PIXEL: fill, dilated cloud, cirrus, cloud, cloud shadow, snow.
BAD_BITS = sum(1 << bit for bit in range(6))


def discover(root):
    paths = sorted(p.resolve() for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    print(f"Found {len(paths)} GeoTIFF(s) under {root}")
    for path in paths:
        print(f"  {path}")
    found = {}
    problems = []
    for date in DATES:
        found[date] = {}
        for band in BANDS:
            matches = []
            for path in paths:
                relative = str(path.relative_to(root))
                # Standard Landsat names contain acquisition and processing dates.
                standard = re.search(r"LC08_L2\w{2}_\d{6}_(\d{8})_", path.name, re.I)
                date_ok = (standard.group(1) == date.replace("-", "") if standard
                           else any(d in relative for d in
                                    (date, date.replace("-", ""), date.replace("-", "_"))))
                if date_ok and re.search(rf"(?:^|_){band}(?:_|\.|$)", path.name, re.I):
                    matches.append(path)
            if len(matches) != 1:
                problems.append(f"{date} {band}: expected exactly one file, found {len(matches)}")
            else:
                found[date][band] = matches[0]
    if problems:
        raise ValueError("\n".join(problems) +
                         "\nSupply --data-dir with the existing imagery folder. "
                         "Names or parent folders must identify the acquisition date; "
                         "ambiguous matches are never selected automatically.")
    return found


def grid(src):
    if src.crs is None:
        raise ValueError(f"Missing CRS: {src.name}")
    if src.count != 1:
        raise ValueError(f"Expected a single-band file: {src.name}")
    return dict(crs=src.crs, transform=src.transform,
                width=src.width, height=src.height)


def same_grid(a, b):
    return (a['crs'] == b['crs'] and a['width'] == b['width']
            and a['height'] == b['height'] and a['transform'] == b['transform'])


def warp(array, source, target):
    if same_grid(source, target):
        return array
    output = np.full((target['height'], target['width']), np.nan, dtype=np.float32)
    # Nearest neighbor preserves discrete QA and avoids mixing cloud-edge pixels.
    reproject(array, output, src_transform=source['transform'], src_crs=source['crs'],
              dst_transform=target['transform'], dst_crs=target['crs'],
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    return output


def load_date(files, target, report):
    # QA is first aligned to each band's native grid and applied before warping.
    with rasterio.open(files['QA_PIXEL']) as src:
        qa_grid = grid(src)
        qa = src.read(1, masked=True).astype(np.float32).filled(np.nan)
    result = []
    for band in BANDS[:-1]:
        with rasterio.open(files[band]) as src:
            native = grid(src)
            dn = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        band_qa = warp(qa, qa_grid, native)
        flags = np.nan_to_num(band_qa, nan=1).astype(np.uint16)
        valid = (np.isfinite(dn) & (dn != 0) & np.isfinite(band_qa)
                 & ((flags & BAD_BITS) == 0))
        reflectance = dn * np.float32(0.0000275) - np.float32(0.2)
        reflectance[~valid] = np.nan
        result.append(warp(reflectance, native, target))
        report[band]['aligned_to_reference'] = not same_grid(native, target)
    return np.stack(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--no-show', action='store_true', help='Save without opening a plot window')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Data directory does not exist: {root}")
    files = discover(root)
    report = {}
    grids = {}
    for date in DATES:
        report[date] = {}
        grids[date] = {}
        for band, path in files[date].items():
            with rasterio.open(path) as src:
                grids[date][band] = grid(src)
                report[date][band] = dict(path=str(path), crs=str(src.crs),
                    transform=list(src.transform), shape=[src.height, src.width],
                    bounds=list(src.bounds), nodata=src.nodata)
    target = grids[DATES[0]]['SR_B4']
    for date in DATES:
        for band in BANDS:
            match = same_grid(grids[date][band], target)
            report[date][band]['matches_reference_grid'] = match
            print(f"{date} {band}: {'matches' if match else 'will align to'} 2015 B4 grid")
    arrays = [load_date(files[d], target, report[d]) for d in DATES]
    # Retain the full 2015 reference extent; non-overlap is transparent.
    # Use joint four-band validity for consistent support between dates.
    common = np.isfinite(arrays[0]).all(axis=0) & np.isfinite(arrays[1]).all(axis=0)
    if not common.any():
        raise ValueError('No common valid pixels after QA masking and alignment; no comparison produced.')
    print(f"Common valid pixels: {common.sum():,} / {common.size:,}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    step = max(1, int(np.ceil(max(common.shape) / 1600)))
    for ax, date, stack in zip(axes, DATES, arrays):
        rgb = np.moveaxis(stack[:, ::step, ::step][[2, 1, 0]], 0, -1)
        # Identical fixed display scale on both dates; scientific values stay unchanged.
        rgba = np.dstack((np.clip(np.nan_to_num(rgb) / 0.3, 0, 1), common[::step, ::step].astype(np.float32)))
        ax.set_facecolor('#d9d9d9')
        ax.imshow(rgba, interpolation='nearest', extent=(0, target['width'], target['height'], 0))
        ax.set_title(f"{date} | natural color (B4/B3/B2)")
        ax.set_xlabel('Reference-grid column')
        ax.set_ylabel('Reference-grid row')
    fig.suptitle('Landsat 8 imagery supplied for Noida\n'
                 'Shared reflectance display range: 0–0.30; gray = excluded in either date\n'
                 'Different seasons: visual differences alone do not establish land-cover change', fontsize=11)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'landsat_natural_color_comparison.png'
    fig.savefig(destination, dpi=180)
    report['processing'] = dict(scale=0.0000275, offset=-0.2,
        excluded_qa_bits=list(range(6)), alignment='nearest; 2015 B4 grid and full extent',
        common_valid_pixels=int(common.sum()), total_pixels=int(common.size),
        display_reflectance_range=[0, 0.3], display_stride=step, rgb_bands=['SR_B4', 'SR_B3', 'SR_B2'])
    (args.output_dir / 'alignment_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Saved {destination.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, rasterio.errors.RasterioError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
