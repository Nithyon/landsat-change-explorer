"""Build cached, real-data results used by the terminal presentation."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "colab_test_data"
OUT = ROOT / "outputs"
RECOVERED = ROOT / "recovered_inputs"
DATES = {"2015": "20151213", "2024": "20240730"}
BANDS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5")
BAD_QA_BITS = sum(1 << bit for bit in range(6))
SCALE_FACTOR = 5  # 30 m source pixels -> 150 m analysis/display grid
RNG = np.random.default_rng(42)


def locate(date_token: str, band: str) -> Path:
    recovered = RECOVERED / f"LC08_L2SP_146040_{date_token}_{'20200908' if date_token == '20151213' else '20240801'}_02_T1_{band}.TIF"
    if recovered.exists():
        return recovered
    matches = sorted(DATA.glob(f"LC08_L2SP_*_{date_token}_*_{band}.TIF"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {date_token} {band}, found {len(matches)}")
    return matches[0]


def reference_grid():
    path = locate(DATES["2015"], "SR_B4")
    with rasterio.open(path) as src:
        height = (src.height + SCALE_FACTOR - 1) // SCALE_FACTOR
        width = (src.width + SCALE_FACTOR - 1) // SCALE_FACTOR
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        return {
            "crs": src.crs,
            "transform": transform,
            "height": height,
            "width": width,
            "source_height": src.height,
            "source_width": src.width,
            "source_transform": src.transform,
            "bounds": list(src.bounds),
        }


def read_to_grid(path: Path, target, resampling=Resampling.nearest) -> np.ndarray:
    output = np.zeros((target["height"], target["width"]), dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=output,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=target["transform"], dst_crs=target["crs"], dst_nodata=0,
            resampling=resampling,
        )
    return output


def load_year(year: str, target):
    token = DATES[year]
    qa = read_to_grid(locate(token, "QA_PIXEL"), target, Resampling.nearest).astype(np.uint16)
    valid_qa = (qa & BAD_QA_BITS) == 0
    arrays = []
    for band in BANDS:
        dn = read_to_grid(locate(token, band), target, Resampling.nearest)
        valid = valid_qa & (dn > 0)
        reflectance = dn * np.float32(0.0000275) - np.float32(0.2)
        reflectance[~valid] = np.nan
        arrays.append(reflectance)
    stack = np.stack(arrays)
    return stack, np.isfinite(stack).all(axis=0)


def seed_masks(stack, valid):
    blue, green, red, nir = stack
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    visible = stack[:3]
    finite = np.isfinite(visible)
    brightness = np.divide(np.nansum(visible, axis=0), finite.sum(axis=0),
                           out=np.full(valid.shape, np.nan, dtype=np.float32),
                           where=finite.sum(axis=0) > 0)
    values = lambda a: a[valid & np.isfinite(a)]
    nq = np.quantile(values(ndvi), [0.30, 0.55, 0.80])
    wq = np.quantile(values(ndwi), 0.90)
    bq = np.quantile(values(brightness), [0.45, 0.75])
    masks = {
        "Water-like": valid & (ndwi >= wq) & (ndvi <= nq[0]),
        "Vegetation-like": valid & (ndvi >= nq[2]),
        "Bright low-vegetation": valid & (brightness >= bq[1]) & (ndvi <= nq[1]),
        "Bare/other-like": valid & (brightness >= bq[0]) & (ndvi > nq[0]) & (ndvi < nq[2]),
    }
    return masks


def fit_mle(stack, masks):
    pixels = np.moveaxis(stack, 0, -1)
    models, seed_counts = [], {}
    for class_id, (name, mask) in enumerate(masks.items()):
        samples = pixels[mask]
        seed_counts[name] = int(len(samples))
        if len(samples) > 5000:
            samples = samples[RNG.choice(len(samples), 5000, replace=False)]
        if len(samples) < 20:
            raise RuntimeError(f"Not enough provisional samples for {name}: {len(samples)}")
        mean = samples.mean(axis=0)
        covariance = np.cov(samples, rowvar=False) + np.eye(4) * 1e-6
        models.append((class_id, name, mean, np.linalg.pinv(covariance),
                       max(float(np.linalg.slogdet(covariance)[1]), -100.0)))
    return models, seed_counts


def classify_mle(stack, valid, models):
    pixels = np.moveaxis(stack, 0, -1)
    positions = np.flatnonzero(valid)
    x = pixels.reshape(-1, 4)[positions]
    best = np.full(len(x), -np.inf)
    labels = np.full(len(x), -1, dtype=np.int8)
    for class_id, _name, mean, inv_cov, logdet in models:
        delta = x - mean
        score = -0.5 * (logdet + np.einsum("ij,jk,ik->i", delta, inv_cov, delta))
        select = score > best
        labels[select] = class_id
        best[select] = score[select]
    result = np.full(valid.shape, -1, dtype=np.int8)
    result.flat[positions] = labels
    return result


def regression_and_change(old, new, common):
    old_pixels = np.moveaxis(old, 0, -1)[common]
    new_pixels = np.moveaxis(new, 0, -1)[common]
    sample_n = min(100_000, len(old_pixels))
    sample_idx = RNG.choice(len(old_pixels), sample_n, replace=False)
    residuals = np.empty_like(new_pixels, dtype=np.float32)
    details = {}
    for index, band in enumerate(BANDS):
        model = LinearRegression().fit(old_pixels[sample_idx, index, None],
                                       new_pixels[sample_idx, index])
        predicted = model.predict(old_pixels[:, index, None])
        residuals[:, index] = new_pixels[:, index] - predicted
        details[band] = {
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "r2_sample": float(model.score(old_pixels[sample_idx, index, None],
                                                   new_pixels[sample_idx, index])),
            "rmse_all": float(np.sqrt(np.mean(residuals[:, index] ** 2))),
        }
    scale = np.median(np.abs(residuals), axis=0) + 1e-6
    score_values = np.sqrt(np.mean((residuals / scale) ** 2, axis=1))
    median = float(np.median(score_values))
    mad = float(np.median(np.abs(score_values - median)))
    threshold = median + 3.0 * 1.4826 * mad
    changed = score_values > threshold
    score_map = np.full(common.shape, np.nan, dtype=np.float32)
    change_map = np.full(common.shape, -1, dtype=np.int8)
    score_map[common] = score_values
    change_map[common] = changed.astype(np.int8)
    return details, sample_n, score_map, change_map, threshold


def display_rgb(stack, valid):
    rgb = np.moveaxis(stack[[2, 1, 0]], 0, -1)
    rgb = np.clip(np.nan_to_num(rgb) / 0.30, 0, 1)
    return np.dstack([rgb, valid.astype(np.float32)])


def main():
    global DATA
    parser = argparse.ArgumentParser(description="Build Landsat terminal-demo results")
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Folder containing the ten Landsat GeoTIFF files")
    args = parser.parse_args()
    DATA = args.data_dir.expanduser().resolve()
    if not DATA.is_dir():
        raise RuntimeError(f"Dataset folder missing: {DATA}")
    OUT.mkdir(exist_ok=True)
    target = reference_grid()
    old, valid_old = load_year("2015", target)
    new, valid_new = load_year("2024", target)
    common = valid_old & valid_new

    masks = seed_masks(old, valid_old)
    models, seed_counts = fit_mle(old, masks)
    mle_old = classify_mle(old, valid_old, models)
    mle_new = classify_mle(new, valid_new, models)
    regression, sample_n, score_map, change_map, threshold = regression_and_change(old, new, common)

    class_names = [m[1] for m in models]
    mle_counts = {}
    for label, classification, valid in (("2015", mle_old, valid_old), ("2024", mle_new, valid_new)):
        mle_counts[label] = {name: int(np.sum(classification[valid] == i))
                             for i, name in enumerate(class_names)}

    changed = int(np.sum(change_map == 1))
    unchanged = int(np.sum(change_map == 0))
    file_sizes = [p.stat().st_size for token in DATES.values()
                  for band in (*BANDS, "QA_PIXEL") for p in [locate(token, band)]]

    report = {
        "dataset": {
            "folder": str(DATA), "files": 10, "bytes": int(sum(file_sizes)),
            "source_shape": [target["source_height"], target["source_width"]],
            "pixels_per_date": int(target["source_height"] * target["source_width"]),
            "crs": str(target["crs"]), "source_resolution_m": 30,
            "analysis_resolution_m": 150, "analysis_shape": [target["height"], target["width"]],
            "dates": ["2015-12-13", "2024-07-30"],
            "recovery_note": "2015 SR_B4 was unreadable in the extracted download; the intact same-named file from the original ZIP archive was used",
        },
        "preprocessing": {
            "scale": 0.0000275, "offset": -0.2,
            "qa_exclusions": ["fill", "dilated cloud", "cirrus", "cloud", "cloud shadow", "snow"],
            "reference_grid": "2015 SR_B4", "2024_shift_west_m": 2100,
            "valid_2015_analysis_pixels": int(valid_old.sum()),
            "valid_2024_analysis_pixels": int(valid_new.sum()),
            "common_valid_analysis_pixels": int(common.sum()),
        },
        "mle": {
            "status": "Demonstration only: provisional spectral-rule seeds, not ground truth",
            "features": list(BANDS), "seed_counts": seed_counts, "class_counts": mle_counts,
        },
        "regression": {"training_sample_pixels": sample_n, "bands": regression},
        "change_classification": {
            "status": "Possible spectral change, not verified land-cover change",
            "method": "robust regression-residual threshold: median + 3 x 1.4826 x MAD",
            "threshold": float(threshold), "possible_change_pixels": changed,
            "unchanged_pixels": unchanged,
            "possible_change_percent": float(100 * changed / (changed + unchanged)),
        },
        "limitations": [
            "2015 is December and 2024 is July; seasonal effects can resemble change",
            "No independent ground-truth labels were supplied, so true accuracy is not reported",
            "MLE labels are provisional classes created only to demonstrate the required concept",
        ],
    }
    (OUT / "terminal_demo_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    colors = plt.matplotlib.colors.ListedColormap(["#2b83ba", "#1a9850", "#d8b365", "#a6611a"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    axes[0, 0].imshow(display_rgb(old, common)); axes[0, 0].set_title("2015 natural color")
    axes[0, 1].imshow(display_rgb(new, common)); axes[0, 1].set_title("2024 natural color")
    im0 = axes[0, 2].imshow(np.ma.masked_less(mle_old, 0), cmap=colors, vmin=0, vmax=3)
    axes[0, 2].set_title("2015 provisional MLE")
    axes[1, 0].imshow(np.ma.masked_less(mle_new, 0), cmap=colors, vmin=0, vmax=3)
    axes[1, 0].set_title("2024 provisional MLE")
    im1 = axes[1, 1].imshow(score_map, cmap="magma", vmin=0,
                           vmax=np.nanpercentile(score_map, 98))
    axes[1, 1].set_title("Regression residual score"); fig.colorbar(im1, ax=axes[1, 1], shrink=.75)
    change_cmap = plt.matplotlib.colors.ListedColormap(["#202020", "#e31a1c"])
    change_cmap.set_bad("#d9d9d9")
    axes[1, 2].imshow(np.ma.masked_less(change_map, 0), cmap=change_cmap, vmin=0, vmax=1)
    axes[1, 2].set_title("Possible change (red); invalid (gray)")
    for ax in axes.flat:
        ax.axis("off")
    cbar = fig.colorbar(im0, ax=[axes[0, 2], axes[1, 0]], ticks=range(4), shrink=.75)
    cbar.ax.set_yticklabels(class_names)
    fig.suptitle("Real Landsat data | 150 m analysis grid | demonstration outputs", fontsize=15)
    fig.savefig(OUT / "terminal_demo_results.png", dpi=160)
    plt.close(fig)
    print(f"Built {OUT / 'terminal_demo_results.json'}")
    print(f"Built {OUT / 'terminal_demo_results.png'}")


if __name__ == "__main__":
    main()
