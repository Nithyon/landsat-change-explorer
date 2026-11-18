# Landsat Change Explorer

Local Streamlit app, reproducible analysis, and Jupyter notebook for Landsat scenes from 13 December 2015 and 30 July 2024 (path/row 146040).

## Clone and open the saved results

This repository is private. The owner must invite each teammate as a collaborator before they can clone or download the data. Install GitHub CLI and Python 3.14 (the tested version), then run:

```powershell
gh auth login
gh repo clone Nithyon/landsat-change-explorer
cd landsat-change-explorer
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. Open http://localhost:8501 and select History. Saved figures and metrics are included; viewing them does not rerun the analysis. Python package versions are pinned to the verified environment. Other Python versions/platforms have not been verified.

## Download the exact data and reproduce the analysis

The ten original GeoTIFF inputs total about 768 MB. They are supplied in a GitHub release asset to keep Git history small. Cloning alone includes code, notebooks, and saved outputs; this additional download supplies the full inputs.

```powershell
gh release download data-v1 --repo Nithyon/landsat-change-explorer --pattern landsat-inputs-v1.zip
python -m zipfile -e landsat-inputs-v1.zip .
python verify_data.py
python verify_landsat_app.py
```

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

For Google Colab, open the corrected notebook, upload the release ZIP using Colab's Files panel, and run `!unzip -q /content/landsat-inputs-v1.zip -d /content` before the data-path cell. Keep the default /content working directory. Alternatively, mount your own Drive and set `os.environ['LANDSAT_DATA_DIR']` to the extracted data folder before the path cell. Colab's changing runtime is not an exactly pinned replica of the tested local environment.

`archive/Change_Detection_Satellite_Imagery.ipynb` preserves the earlier notebook for reference. Use the corrected notebook for reproduction.

## Method and limits

Inputs are Landsat Collection 2 Level-2 SR_B2/B3/B4/B5 and QA_PIXEL for both scenes. Reflectance uses DN * 0.0000275 - 0.2; QA bits 0-5 are excluded. Analysis uses nearest-neighbor alignment to the Before B4 reference, a grid factor of 5 (about 150 m), random seed 42, provisional maximum-likelihood classes, and robust regression residuals.

Possible spectral change is not validated land-cover change. There is no independent ground truth or classification accuracy measurement. December and July introduce seasonal differences. The full scenes are not an administrative Noida crop. Earlier terminal class counts use each date's individual valid area; app comparison counts use common valid coverage.

The dataset includes the intact 2015 B4 copy recovered from the original download archive. No separate recovery folder or owner-specific Windows paths are needed. Sources: [USGS Collection 2 QA](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands) and [USGS Collection 2](https://pubs.usgs.gov/fs/2021/3002/fs20213002.pdf).
