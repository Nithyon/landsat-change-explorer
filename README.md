# Landsat Change Explorer

## Course assessment extension

The project now includes all supplied topics for **PBL 2 (20 marks)** and **PBL 3 (10 marks)**, together with the existing PBL 1 baseline. Start the app and open **Methods**. Saved results are bundled so class demonstration does not require retraining.

| Assessment | Implemented methods | Demo evidence |
| --- | --- | --- |
| PBL 1 | Existing provisional MLE, linear regression and residual change flags | History views and preserved baseline |
| PBL 2 | K-means, PCA, Hidden Markov Model, Sum Product Algorithm | Cluster maps, PCA variance, two-date HMM states, interactive exact-inference example |
| PBL 3 | CART, AdaBoost, random forest and soft voting | Tree rules, boosting progression, ensemble maps, spatial-holdout scores and confusion matrices |

Use [DEMO_GUIDE.md](DEMO_GUIDE.md) for the presentation order and viva questions. The full runnable notebook is [Landsat_PBL_1_2_3.ipynb](Landsat_PBL_1_2_3.ipynb), with actual execution outputs. The original corrected notebook remains available for the PBL 1 workflow. The consolidated submission report is [PDF](reports/Consolidated_Project_Report.pdf), with an [editable Markdown version](reports/Consolidated_Project_Report.md). Fill in student and institution details before submission and confirm that the existing baseline matches the instructor's PBL 1 assignment.

```powershell
python build_pbl_demo.py
python -m pip install -r requirements-report.txt
python build_project_report.py
python -m unittest discover -s tests -q
python -m streamlit run app.py
```

To rerun the notebook, install requirements-notebook.txt and use Jupyter Run All from the repository root. Alternatively, `python execute_pbl_notebook.py` executes this repository's trusted notebook and retains text/image outputs. `python make_pbl_notebook.py` recreates a clean notebook and clears previous outputs.

**Evaluation scope:** there are only two acquisition dates and no independent ground-truth labels. The HMM uses an explicitly assumed 0.90 transition persistence over the entire interval; forward-backward sum-product computes its exact marginals. It does not learn reliable temporal dynamics. Supervised targets are uniquely matched spectral rules using training-area thresholds. Evaluation uses an eastern Before spatial holdout with an eight-column guard strip. Reported agreement and F1 describe these weak labels, not independently measured land-cover accuracy. After predictions are unvalidated transfer maps. Cluster IDs and supervised class IDs are different label systems. PCA is exploratory; it is not claimed to improve classifier accuracy.

**Measured demonstration:** the first two PCs retain about 97.99% of standardized training variance. CART and soft voting agree with 99.97% of held-out rules, random forest 99.96%, and the fixed AdaBoost stump model 66.15%. High rule agreement is expected when models have the same indices that define their target rules; it is not evidence of real-world accuracy. The simple tree can outperform boosting on this task.

Implementation: `pbl_analysis.py` contains algorithms, `build_pbl_demo.py` computes figures and JSON/CSV outputs, `pbl_ui.py` provides the saved demo and interactive example, and `build_project_report.py` produces the consolidated report. Course outputs are separate in `outputs/pbl`; original PBL 1 reports and imagery are preserved. New uploads in New analysis continue to use PBL 1; the course demo explicitly uses the supplied saved scenes. Rebuilding course results overwrites only their generated artifacts.


A local satellite-image change-analysis project using **Landsat 8 Collection 2 Level-2** scenes from **13 December 2015** and **30 July 2024**, path/row **146040**. The supplied dataset is Landsat 8 only. The app parser also accepts the corresponding Landsat 9 product naming pattern, but no Landsat 9 data is supplied or verified here.

## Project aim

Compare the spectral appearance of two acquisitions, demonstrate maximum-likelihood classification (MLE), and identify pixels with unusually large regression residuals. The project includes an interactive Streamlit interface, a corrected runnable notebook, a terminal demonstration, reusable analysis functions, saved figures, and reproducibility checks. It runs locally without a GPU, paid API, or hosted database.

The scenes are used in a Noida-area demonstration, but they are full satellite scenes: no administrative Noida boundary or geographic city crop was provided.

## Clone and open the saved results

This repository is public. Install Git and Python 3.14 (the tested version), then run:

```powershell
git clone https://github.com/Nithyon/landsat-change-explorer.git
cd landsat-change-explorer
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. Open http://localhost:8501 and select History. Saved figures and metrics are included; viewing them does not rerun the analysis. Python package versions are pinned to the verified environment. Other Python versions/platforms have not been verified.

## Download the exact data and reproduce the analysis

**Dataset: [Google Drive folder - all ten GeoTIFF inputs](https://drive.google.com/drive/folders/1-mpfiI23HzrTgpsQpVCW8KUmGytMLkH-).** All input imagery is on Drive; GitHub contains the code, notebooks, saved outputs, and checksums.

Download the entire Drive folder (Drive may package it as a ZIP). Extract it, create a folder named `colab_test_data` inside this cloned repository, and put all downloaded `.TIF` files and `.TIF.partNN` transfer parts directly inside it. Avoid an extra nested folder. Keep the original filenames. The eight directly uploaded TIFFs and the parts for the final two bands together contain all ten original inputs. Total input size is **767,745,918 bytes**, about 768 MB. Drive access must be granted to your Google account by the owner if the folder is restricted.

```powershell
python prepare_data.py
python verify_data.py
python verify_landsat_app.py
```

`prepare_data.py` joins the final two bands from their numbered 20 MB transfer parts and verifies their original hashes. Parts are retained after assembly. These parts are needed because the connector repeatedly timed out on the complete files. You only need to run this preparation once.

`verify_data.py` checks every file against SHA-256 hashes and sizes. `verify_landsat_app.py` reruns the app pipeline and compares common valid pixels, seed counts, class counts, regression coefficients, and possible-change metrics with the saved baseline. Numerical comparisons allow small floating-point differences. Full raster processing takes time and several GB of available memory.

To regenerate terminal results and the notebook figures:

```powershell
python build_demo_results.py --data-dir colab_test_data
python build_saved_notebook_history.py
python terminal_demo.py
```

These commands overwrite the corresponding generated files in outputs. Git preserves the committed baseline. The app also accepts the five Before and five After GeoTIFFs from colab_test_data in New analysis.

## Run the notebook

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyter lab Corrected_Landsat_Change_Detection_Colab.ipynb
```

Start Jupyter from this repository and run all cells in order. The corrected notebook uses `colab_test_data` relative to the working directory. Set `LANDSAT_DATA_DIR` before starting Jupyter to use another data folder. The notebook's install cell installs the core analysis packages; the pinned requirements above should be installed first.

For Google Colab, open the corrected notebook and access the dataset through Google Drive. Add a code cell before the data-path cell:

```python
from google.colab import drive
import os
drive.mount('/content/drive')
os.environ['LANDSAT_DATA_DIR'] = '/content/drive/MyDrive/Landsat Change Explorer - Reproduction Dataset'
```

If your folder contains transfer parts, assemble them first: clone this public repository in Colab, put the Drive files/parts into its colab_test_data folder, and run `!python prepare_data.py` from that repository. Then set LANDSAT_DATA_DIR to that assembled folder. Alternatively, assemble locally and upload the restored TIFFs to your own Drive.

Use the actual location in your own Drive. For a shared folder, add a shortcut to My Drive or copy the files into your own data folder, then adjust the path. Run cells in order. Colab's changing runtime is not an exactly pinned replica of the tested local environment.

`archive/Change_Detection_Satellite_Imagery.ipynb` preserves the earlier notebook for reference. Use the corrected notebook for reproduction.

## Method and limits

Inputs are Landsat Collection 2 Level-2 SR_B2/B3/B4/B5 and QA_PIXEL for both scenes. Reflectance uses DN * 0.0000275 - 0.2; QA bits 0-5 are excluded. Analysis uses nearest-neighbor alignment to the Before B4 reference, a grid factor of 5 (about 150 m), random seed 42, provisional maximum-likelihood classes, and robust regression residuals.

Possible spectral change is not validated land-cover change. There is no independent ground truth or classification accuracy measurement. December and July introduce seasonal differences. The full scenes are not an administrative Noida crop. Earlier terminal class counts use each date's individual valid area; app comparison counts use common valid coverage.

The dataset includes the intact 2015 B4 copy recovered from the original download archive. No separate recovery folder or owner-specific Windows paths are needed. Sources: [USGS Collection 2 QA](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands) and [USGS Collection 2](https://pubs.usgs.gov/fs/2021/3002/fs20213002.pdf).


## Dataset details

| Item | Supplied value |
| --- | --- |
| Satellite/product | Landsat 8 Collection 2 Level-2 surface reflectance, L2SP |
| Before scene | LC08_L2SP_146040_20151213_20200908_02_T1 |
| After scene | LC08_L2SP_146040_20240730_20240801_02_T1 |
| Bands per date | SR_B2 (blue), SR_B3 (green), SR_B4 (red), SR_B5 (near infrared), QA_PIXEL |
| Number of files | 10 GeoTIFFs, five per date |
| Native grid | 7681 rows x 7531 columns, 30 m pixels |
| Coordinate system | EPSG:32643 (UTM zone 43N) |
| Analysis grid | 1537 rows x 1507 columns, approximately 150 m pixels |
| Registration detail | 2024 scene origin is 2100 m west of 2015; alignment is required |

## Processing and models

1. Validate product filenames, required bands, acquisition order, source-grid consistency, readability, and geographic overlap.
2. Apply the USGS surface-reflectance scale and offset. Exclude nodata/zero source pixels and QA_PIXEL bits 0-5 (fill, dilated cloud, cirrus, cloud, cloud shadow, snow). Water remains valid. QA_RADSAT is not included, so the supplied masks do not identify every saturated pixel.
3. Align to the Before red-band reference using nearest-neighbor resampling and grid factor 5. Compare both dates on their common valid footprint.
4. Display natural colour as red/green/blue = B4/B3/B2, with the same 0-0.30 reflectance display range.
5. Derive provisional class seeds from Before-image NDVI, NDWI, and visible brightness quantiles. The four classes are Water-like, Vegetation-like, Bright low-vegetation, and Bare/other-like. Fit class means and regularized covariance matrices using at most 5,000 samples per class. Apply Gaussian maximum-likelihood scores to both dates. These seeds are spectral rules, not independently labelled training data.
6. Fit one linear regression per band: After reflectance = slope x Before reflectance + intercept. Use up to 100,000 randomly selected common pixels; random seed is 42. Report slope, intercept, sample R-squared, and RMSE over the common pixels.
7. Scale each band's residuals by its median absolute residual plus a small stabilizer, combine the four bands as a root-mean-square score, and flag scores above `median(score) + 3 x 1.4826 x MAD(score)`. MAD is median absolute deviation. This is a possible spectral-change flag, not the difference between the MLE class maps.

## Verified reference results

| Measure | Value |
| --- | ---: |
| Common valid analysis pixels | 632,388 |
| Possible spectral-change pixels | 33,167 |
| Unflagged common pixels | 599,221 |
| Possible spectral change | 5.24% |
| Residual-score threshold | approximately 3.0085 |

These are scene-specific demonstration results. They do not establish that 5.24% of Noida changed land use. Regression R-squared measures model fit and is not classification accuracy.

## App workflow and outputs

- **History** opens the committed reference analysis with Overview, Satellite images, MLE classification, and Regression views. No new computation is needed.
- **New analysis** accepts five Before and five After GeoTIFFs. Select the acquisitions and click Run Analysis. JPG/PNG inputs and unrelated satellite products are unsupported.
- Uploads are processed on the computer running the server. The app uses temporary copies and removes those copies after processing. New results live in the browser session; download them before closing it. Changing the selected inputs clears stale results.
- New-analysis downloads include satellite images, classification maps, possible-change map, residual-score map, regression plots, class distribution, JSON summary, and a combined ZIP.
- The source inputs remain unchanged. Rebuild scripts write generated figures and JSON under outputs.

| File or folder | Purpose |
| --- | --- |
| app.py | Streamlit interface and downloads |
| landsat_analysis.py | Input validation, masking, alignment, MLE, regression, change score |
| landsat_uploads.py | Upload grouping and isolated temporary processing |
| Corrected_Landsat_Change_Detection_Colab.ipynb | Corrected step-by-step analysis and figures |
| build_demo_results.py | Regenerate terminal metrics and overview figure |
| build_saved_notebook_history.py | Execute corrected notebook analysis/plotting cells and regenerate History figures |
| terminal_demo.py | Present saved results in a terminal menu |
| visualize_landsat.py | Separate native-resolution natural-colour visualization; run with --data-dir colab_test_data --no-show |
| prepare_data.py | Restore the two split-band transfers and verify inputs |
| verify_data.py / data_manifest.json | Check exact input sizes and SHA-256 hashes |
| verify_landsat_app.py | Rerun analysis and compare against saved metrics |
| tests/test_landsat_app.py | Synthetic input-validation and session tests |
| outputs/notebook_history | Six saved notebook figures and numerical summary |
| outputs/terminal_demo_results.json | Reference metrics used by reproduction checks |
| archive | Earlier notebook retained as a historical reference |

## Verification and troubleshooting

The prepared package passed all **15 existing tests**, input checksums for all ten files, numerical regression against the saved baseline, regeneration of all six notebook figures, and Streamlit app loading from a fresh GitHub clone. Checks used the existing Windows Python 3.14 environment; a clean dependency installation on a second machine has not been tested.

```powershell
python -m unittest discover -s tests -q
```

If data is missing, check that the downloaded files and numbered parts are directly inside colab_test_data, then run python prepare_data.py. If a checksum fails, download that file again. If a dependency cannot install, use Python 3.14 in a fresh virtual environment and the supplied requirements. If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py` directly. If port 8501 is occupied, add `--server.port 8502`. Restart interrupted processing after ensuring sufficient memory is available.

## Interpretation limits and next steps

Season, clouds, remaining atmospheric effects, grid alignment, and coarse analysis resolution can influence the comparison. A high residual is not a verified physical change. MLE class names must remain provisional until evaluated using independent labels. No confusion matrix, validated accuracy, causal attribution, or administrative-area estimate is provided. A stronger study would use a defined study boundary, comparable seasons, additional dates, independently labelled validation data, and appropriate accuracy assessment.

## Preserved unit studies

The supplied updated project also includes u12_pipeline.py, u3_clustering_pca.py, u4_hmm.py, u5_ensembles.py, diagrams.py and the original template-based build_report.py. Their original figures and cached JSON results are preserved and available under Methods ? Additional analyses. See REPORT_ANALYSIS.md for their separate run order. The Unit 4 HMM is spatial along image rows and learns transitions using Baum?Welch; it is distinct from the main two-date temporal teaching example. These cached extra studies were not retrained during the current integration, and their scores must not be mixed with the new spatial-holdout experiment. The original Word report builder still requires its template; the new consolidated PDF has no template dependency.
