"""Menu-driven terminal presentation of cached Landsat results."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "outputs" / "terminal_demo_results.json"
FIGURE = ROOT / "outputs" / "terminal_demo_results.png"


def line(char="=", width=72):
    print(char * width)


def heading(title):
    print()
    line()
    print(title)
    line()


def dataset(r):
    d = r["dataset"]
    heading("1. DATASET - REAL LANDSAT 8 GEOTIFF FILES")
    print("Source              : 10 locally processed Landsat GeoTIFF files")
    print(f"Files               : {d['files']} GeoTIFFs")
    print(f"Disk size           : {d['bytes'] / 1024**2:,.2f} MB")
    print(f"Dates               : {d['dates'][0]} and {d['dates'][1]}")
    print(f"Raster per date     : {d['source_shape'][0]:,} x {d['source_shape'][1]:,}")
    print(f"Pixels per date     : {d['pixels_per_date']:,}")
    print(f"CRS                 : {d['crs']}")
    print(f"Native resolution   : {d['source_resolution_m']} m")
    print("Bands/features      : SR_B2, SR_B3, SR_B4, SR_B5")
    print("Quality mask        : QA_PIXEL (not a model feature)")


def preprocessing(r):
    p, d = r["preprocessing"], r["dataset"]
    heading("2. PREPROCESSING AND ALIGNMENT")
    print(f"Reflectance formula : DN x {p['scale']} {p['offset']:+g}")
    print(f"QA exclusions       : {', '.join(p['qa_exclusions'])}")
    print(f"Coordinate system   : {d['crs']} for both dates")
    print(f"Grid result         : 2024 origin is {p['2024_shift_west_m']:,} m west")
    print(f"Alignment reference : {p['reference_grid']}")
    print(f"Demo analysis grid  : {d['analysis_shape'][0]:,} x {d['analysis_shape'][1]:,} at {d['analysis_resolution_m']} m")
    print(f"Common valid pixels : {p['common_valid_analysis_pixels']:,}")
    print("Impact              : masking avoids cloud artifacts; alignment makes")
    print("                      corresponding pixels represent the same location.")


def mle(r):
    m = r["mle"]
    heading("3. MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
    print(f"Features             : {', '.join(m['features'])}")
    print(f"Training status      : {m['status']}")
    print("Provisional seeds:")
    for name, count in m["seed_counts"].items():
        print(f"  {name:<24} {count:>10,}")
    print("Classified pixels:")
    print(f"  {'Class':<24} {'2015':>10} {'2024':>10}")
    for name in m["seed_counts"]:
        print(f"  {name:<24} {m['class_counts']['2015'][name]:>10,} {m['class_counts']['2024'][name]:>10,}")
    print("Interpretation       : MLE assigns the most statistically likely class.")
    print("Claim limit          : these are demonstration classes, not validated labels.")


def regression(r):
    reg = r["regression"]
    heading("4. LINEAR REGRESSION - 2015 TO 2024")
    print(f"Fitting sample       : {reg['training_sample_pixels']:,} common valid pixels")
    print(f"  {'Band':<10} {'Slope':>10} {'Intercept':>12} {'R2':>10} {'RMSE':>10}")
    for band, v in reg["bands"].items():
        print(f"  {band:<10} {v['slope']:>10.4f} {v['intercept']:>12.4f} "
              f"{v['r2_sample']:>10.4f} {v['rmse_all']:>10.4f}")
    print("Impact               : larger residuals mean stronger spectral differences.")
    print("Claim limit          : season and moisture can also create residuals.")


def change(r):
    c = r["change_classification"]
    heading("5. CLASSIFICATION - POSSIBLE CHANGE / UNCHANGED")
    print(f"Method               : {c['method']}")
    print(f"Residual threshold   : {c['threshold']:.4f}")
    print(f"Possible-change px   : {c['possible_change_pixels']:,}")
    print(f"Unchanged px         : {c['unchanged_pixels']:,}")
    print(f"Possible-change rate : {c['possible_change_percent']:.2f}% of common valid grid")
    print(f"Result status        : {c['status']}")
    print("Accuracy             : not reported; independent ground truth is unavailable.")


def limitations(r):
    heading("6. LIMITATIONS")
    for i, item in enumerate(r["limitations"], 1):
        print(f"{i}. {item}")


def show_all(r):
    dataset(r); preprocessing(r); mle(r); regression(r); change(r); limitations(r)


def open_figure():
    if not FIGURE.exists():
        print(f"Figure missing: {FIGURE}")
        return
    if os.name == "nt":
        os.startfile(FIGURE)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(FIGURE)])
    else:
        subprocess.Popen(["xdg-open", str(FIGURE)])
    print(f"Opened: {FIGURE}")


def menu(r):
    actions = {"1": dataset, "2": preprocessing, "3": mle, "4": regression,
               "5": change, "6": limitations, "A": show_all}
    while True:
        heading("LANDSAT CHANGE-DETECTION TERMINAL DEMO")
        print("1  Dataset summary")
        print("2  Scaling, QA masking and alignment")
        print("3  Maximum Likelihood Estimation results")
        print("4  Linear Regression results")
        print("5  Possible-change classification")
        print("6  Limitations and accuracy statement")
        print("7  Open result figure")
        print("A  Run complete presentation")
        print("0  Exit")
        choice = input("\nSelect an option: ").strip().upper()
        if choice == "0":
            print("Demo finished.")
            return
        if choice == "7":
            open_figure()
        elif choice in actions:
            actions[choice](r)
        else:
            print("Invalid option. Choose 0-7 or A.")
        input("\nPress Enter to return to the menu...")


def main():
    parser = argparse.ArgumentParser(description="Terminal menu for real Landsat demo results")
    parser.add_argument("--all", action="store_true", help="Print every result without interaction")
    args = parser.parse_args()
    if not REPORT.exists():
        raise SystemExit(f"Cached result missing: {REPORT}\nRun build_demo_results.py once.")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if args.all:
        show_all(report)
    else:
        menu(report)


if __name__ == "__main__":
    main()
