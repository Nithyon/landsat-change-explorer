"""Reusable, local Landsat Collection 2 change-analysis pipeline.

No inputs are modified. Results are held in memory for display and download.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from sklearn.linear_model import LinearRegression

BANDS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5")
REQUIRED = (*BANDS, "QA_PIXEL")
BAD_QA_BITS = sum(1 << bit for bit in range(6))
CLASS_NAMES = ("Water-like", "Vegetation-like", "Bright low-vegetation", "Bare/other-like")
CLASS_COLORS = ("#2b83ba", "#1a9850", "#d8b365", "#a6611a")
PATTERN = re.compile(r"(?P<id>LC0[89]_L2S[PR]_\d{6}_(?P<date>\d{8})_\d{8}_02_T[12])_(?P<band>SR_B[2345]|QA_PIXEL)\.tif$", re.I)

class AnalysisError(ValueError):
    """An input or analysis problem that can be shown to the user."""

@dataclass
class Acquisition:
    identifier: str
    date: str
    files: dict[str, Path]
    replacement: str | None = None


def discover(folder: str) -> dict[str, Acquisition]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise AnalysisError(f"Folder not found: {root}")
    groups = {}
    for path in sorted(root.iterdir()):
        match = PATTERN.fullmatch(path.name) if path.is_file() else None
        if not match:
            continue
        identifier = match['id'].upper()
        try:
            date = datetime.strptime(match['date'], '%Y%m%d').date().isoformat()
        except ValueError as exc:
            raise AnalysisError(f"Invalid acquisition date in {path.name}") from exc
        acquisition = groups.setdefault(identifier, Acquisition(identifier, date, {}))
        band = match['band'].upper()
        if band in acquisition.files:
            raise AnalysisError(f"Duplicate {band} for {identifier}.")
        acquisition.files[band] = path.resolve()
    if not groups:
        raise AnalysisError("No supported Landsat 8/9 Collection 2 Level-2 SR files found. Use original Landsat filenames and GeoTIFFs.")
    return groups


def replace_red_band(acquisition: Acquisition, replacement: str) -> Acquisition:
    if not replacement.strip():
        return acquisition
    path = Path(replacement).expanduser()
    match = PATTERN.fullmatch(path.name)
    if not path.is_file() or not match or match['id'].upper() != acquisition.identifier or match['band'].upper() != 'SR_B4':
        raise AnalysisError("The replacement must be a readable SR_B4 GeoTIFF with the same Landsat acquisition identifier.")
    return Acquisition(acquisition.identifier, acquisition.date, {**acquisition.files, 'SR_B4': path.resolve()}, str(path.resolve()))


def validate_pair(before: Acquisition, after: Acquisition):
    if before.date >= after.date:
        raise AnalysisError("Before must have an earlier acquisition date than After.")
    references = []
    for acquisition in (before, after):
        missing = set(REQUIRED) - set(acquisition.files)
        if missing:
            raise AnalysisError(f"{acquisition.date}: missing {', '.join(sorted(missing))}.")
        metadata = []
        for band in REQUIRED:
            path = acquisition.files[band]
            try:
                with rasterio.open(path) as src:
                    if src.count != 1 or src.crs is None or src.width < 1 or src.height < 1:
                        raise AnalysisError(f"{path.name}: requires one band and valid geographic metadata.")
                    if not np.isfinite(tuple(src.transform)).all() or abs(src.transform.determinant) < 1e-12:
                        raise AnalysisError(f"{path.name}: invalid geographic transform.")
                    if np.dtype(src.dtypes[0]).kind != 'u' or np.dtype(src.dtypes[0]).itemsize != 2:
                        raise AnalysisError(f"{path.name}: expected unsigned 16-bit Landsat data.")
                    src.read(1, window=((0, min(16, src.height)), (0, min(16, src.width))))
                    metadata.append((src.crs, src.transform, src.width, src.height, src.bounds))
            except (rasterio.errors.RasterioError, OSError) as exc:
                raise AnalysisError(f"Cannot read {path.name}. Check the GeoTIFF or select an intact replacement. {exc}") from exc
        if any(m[:4] != metadata[0][:4] for m in metadata[1:]):
            raise AnalysisError(f"{acquisition.date}: bands and QA must share the same source grid.")
        references.append(metadata[2])
    old, new = references
    try:
        b = transform_bounds(new[0], old[0], *new[4], densify_pts=21)
    except Exception as exc:
        raise AnalysisError("Cannot transform acquisition bounds into a shared CRS.") from exc
    a = old[4]
    if max(a.left, b[0]) >= min(a.right, b[2]) or max(a.bottom, b[1]) >= min(a.top, b[3]):
        raise AnalysisError("The acquisitions have no geographic overlap.")


def reference_grid(acquisition):
    with rasterio.open(acquisition.files['SR_B4']) as src:
        height, width = (src.height + 4) // 5, (src.width + 4) // 5
        if height * width > 12_000_000:
            raise AnalysisError("This scene is too large for the local prototype. Use a smaller, consistently cropped Landsat scene.")
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        return dict(crs=src.crs, transform=transform, height=height, width=width,
                    source_shape=[src.height, src.width], source_resolution=list(src.res))


def load_acquisition(acquisition, target):
    qa = read_to_grid(acquisition.files['QA_PIXEL'], target).astype(np.uint16)
    qa_valid = (qa & BAD_QA_BITS) == 0
    layers = []
    for band in BANDS:
        dn = read_to_grid(acquisition.files[band], target)
        reflectance = dn * np.float32(0.0000275) - np.float32(0.2)
        reflectance[~(qa_valid & (dn > 0))] = np.nan
        layers.append(reflectance)
    stack = np.stack(layers)
    return stack, np.isfinite(stack).all(axis=0)

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

def fit_mle(stack, masks, rng):
    pixels = np.moveaxis(stack, 0, -1)
    models, seed_counts = [], {}
    for class_id, (name, mask) in enumerate(masks.items()):
        samples = pixels[mask]
        seed_counts[name] = int(len(samples))
        if len(samples) > 5000:
            samples = samples[rng.choice(len(samples), 5000, replace=False)]
        if len(samples) < 20:
            raise AnalysisError(f"Not enough provisional samples for {name}: {len(samples)}")
        mean = samples.mean(axis=0)
        covariance = np.cov(samples, rowvar=False) + np.eye(4) * 1e-6
        models.append((class_id, name, mean, np.linalg.pinv(covariance),
                       float(np.linalg.slogdet(covariance)[1])))
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

def regression_and_change(old, new, common, rng):
    old_pixels = np.moveaxis(old, 0, -1)[common]
    new_pixels = np.moveaxis(new, 0, -1)[common]
    sample_n = min(100_000, len(old_pixels))
    sample_idx = rng.choice(len(old_pixels), sample_n, replace=False)
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
    return details, sample_n, score_map, change_map, threshold, old_pixels[sample_idx[:3000]], new_pixels[sample_idx[:3000]]

def display_rgb(stack, valid):
    rgb = np.moveaxis(stack[[2, 1, 0]], 0, -1)
    rgb = np.clip(np.nan_to_num(rgb) / 0.30, 0, 1)
    return np.dstack([rgb, valid.astype(np.float32)])


def analyze(before: Acquisition, after: Acquisition, progress=None):
    """Return arrays, a JSON-safe report, and regression scatter samples."""
    emit = progress or (lambda fraction, message: None)
    emit(0.05, "Checking input files and geographic overlap")
    validate_pair(before, after)
    target = reference_grid(before)
    try:
        emit(0.15, "Loading and masking the Before acquisition")
        old, valid_old = load_acquisition(before, target)
        emit(0.35, "Aligning and masking the After acquisition")
        new, valid_new = load_acquisition(after, target)
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise AnalysisError(f"A GeoTIFF could not be read during processing: {exc}") from exc
    common = valid_old & valid_new
    total = int(common.sum())
    if total < 2:
        raise AnalysisError("No usable common valid pixels (at least two are needed). Check overlap and cloud coverage.")
    rng = np.random.default_rng(42)
    emit(0.55, "Fitting provisional MLE classes")
    models, seeds = fit_mle(old, seed_masks(old, valid_old), rng)
    mle_old = classify_mle(old, valid_old, models)
    mle_new = classify_mle(new, valid_new, models)
    emit(0.75, "Fitting regression and calculating possible spectral change")
    regression, sample_n, score, change, threshold, scatter_old, scatter_new = regression_and_change(old, new, common, rng)
    class_counts = {}
    individual_counts = {}
    for acquisition, labels, valid in ((before, mle_old, valid_old), (after, mle_new, valid_new)):
        class_counts[acquisition.date] = dict(zip(CLASS_NAMES, np.bincount(labels[common], minlength=4).astype(int).tolist()))
        individual_counts[acquisition.date] = dict(zip(CLASS_NAMES, np.bincount(labels[valid], minlength=4).astype(int).tolist()))
    changed, unchanged = int((change == 1).sum()), int((change == 0).sum())
    report = {
        'dataset': {'dates': [before.date, after.date], 'acquisitions': [
            {'identifier': a.identifier, 'date': a.date,
             'files': {k: str(v) for k, v in a.files.items()}, 'red_band_replacement': a.replacement}
            for a in (before, after)],
            'source_shape': target['source_shape'], 'crs': str(target['crs']),
            'source_resolution_crs_units': target['source_resolution'],
            'analysis_resolution_crs_units': [abs(target['transform'].a), abs(target['transform'].e)],
            'analysis_shape': [target['height'], target['width']]},
        'settings': {'grid_factor': 5, 'random_seed': 42, 'scale': 0.0000275, 'offset': -0.2,
                     'resampling': 'nearest', 'reference_grid': before.identifier + '_SR_B4',
                     'qa_excluded_bits': list(range(6)), 'maximum_seed_samples_per_class': 5000},
        'preprocessing': {'valid_before_pixels': int(valid_old.sum()), 'valid_after_pixels': int(valid_new.sum()),
                          'common_valid_analysis_pixels': total},
        'mle': {'status': 'Provisional spectral-rule classes; not ground truth',
                'seed_counts': seeds, 'class_counts': class_counts,
                'class_counts_individual_footprints': individual_counts},
        'regression': {'training_sample_pixels': sample_n, 'bands': regression},
        'change_classification': {'status': 'Possible spectral change; not verified land-cover change',
            'method': 'median + 3 x 1.4826 x MAD of standardized residual scores',
            'threshold': float(threshold), 'possible_change_pixels': changed,
            'unchanged_pixels': unchanged, 'possible_change_percent': 100 * changed / total},
        'limitations': ['Seasonal differences may contribute to spectral change.',
                       'MLE classes use provisional spectral-rule samples.',
                       'No independent ground truth: measured accuracy is unavailable.',
                       'The scene extent is shown; no administrative boundary is assumed.']}
    # Use an identical footprint and crop for every displayed map.
    rows, cols = np.where(common)
    crop = (slice(max(0, rows.min()-20), min(common.shape[0], rows.max()+21)),
            slice(max(0, cols.min()-20), min(common.shape[1], cols.max()+21)))
    result = dict(report=report, rgb_before=display_rgb(old, common)[crop],
                  rgb_after=display_rgb(new, common)[crop],
                  mle_before=np.where(common, mle_old, -1)[crop],
                  mle_after=np.where(common, mle_new, -1)[crop],
                  score=score[crop], change=change[crop],
                  scatter_before=scatter_old, scatter_after=scatter_new)
    emit(1.0, "Analysis complete")
    return result
