"""Create the consolidated, portable PBL notebook without an nbformat dependency."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                  'source': text.splitlines(keepends=True)})


markdown('''# Landsat project PBL 1 PBL 2 and PBL 3

This notebook demonstrates the requested course methods on the same ten Landsat 8 inputs as the app.
Run from the repository root using its pinned requirements. Download and prepare the Drive data first.
On Colab, clone the public repository, install requirements appropriate for that runtime, mount Drive,
copy the supplied files and parts into colab_test_data, and run prepare_data.py. Set LANDSAT_DATA_DIR if needed.

The two acquisition dates are 2015-12-13 and 2024-07-30. No independent land-cover labels exist.
All supervised scores below are agreement with provisional spectral-rule labels.
''')
code('''from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image
from landsat_analysis import discover, analyze
from pbl_analysis import (load_course_data, fit_unsupervised, run_temporal,
                          fit_supervised, sum_product, transition_matrix)
from build_pbl_demo import save_course_results

ROOT = Path.cwd()
DATA_DIR = Path(os.environ.get('LANDSAT_DATA_DIR', str(ROOT / 'colab_test_data')))
assert DATA_DIR.is_dir(), 'Download the Drive data and run prepare_data.py first.'
OUTPUT = ROOT / 'outputs' / 'pbl'
''')
markdown('''## PBL 1 Existing baseline

Use surface-reflectance scaling, QA masking and nearest-neighbor alignment, then demonstrate provisional
maximum-likelihood classification and per-band linear regression. The robust residual threshold flags
possible spectral change. This maps PBL 1 to the existing project; confirm the formal PBL 1 assignment.
''')
code('''acquisitions = sorted(discover(str(DATA_DIR)).values(), key=lambda a: a.date)
baseline = analyze(acquisitions[0], acquisitions[1])
display(pd.DataFrame(baseline['report']['regression']['bands']).T)
display(baseline['report']['change_classification'])
del baseline  # Release the baseline display arrays before the course methods.
''')
markdown('''## Shared data and spatial evaluation design

The eastern quarter of common valid Before pixels is held out, with an eight-column guard strip.
Training-area pixels alone determine all fitted preprocessing and spectral-rule thresholds.
After is a separate, unvalidated transfer visualization; it is not a labelled test set.
''')
code('''data = load_course_data(DATA_DIR)
display(data['split'])
print('Common valid pixels:', int(data['common'].sum()))
''')
markdown('''## PBL 2 K-means clustering and PCA

Fit StandardScaler, KMeans(k=4, n_init=10) and full four-component PCA on a fixed sample of up to
20,000 Before training pixels. Reuse the centroids on both dates. Plot PC1 and PC2 without dropping
the other bands from the K-means or HMM models. Cluster IDs are arbitrary, not supervised class names.
''')
code('''unsupervised = fit_unsupervised(data)
display(pd.DataFrame(unsupervised['report']['centroids_reflectance'], columns=['B2', 'B3', 'B4', 'B5']))
variance = np.array(unsupervised['report']['pca_explained_variance_ratio'])
display(pd.DataFrame({'Component': range(1, 5), 'Variance ratio': variance, 'Cumulative': variance.cumsum()}))
''')
markdown('''## PBL 2 Hidden Markov model

For each pixel, z1 and z2 are hidden spectral states and x1 and x2 are observed four-band spectra.
The factorization is p(z1,z2,x1,x2) = pi(z1) e1(x1|z1) A(z1,z2) e2(x2|z2).
Initial frequencies and diagonal-Gaussian emissions come from the Before training clusters.
The transition prior assumes 0.90 same-state persistence over the entire nine-year interval.
Two dates support inference but do not identify a reliable learned transition model.
''')
code('''temporal = run_temporal(data, unsupervised, persistence=0.90)
display(pd.DataFrame(temporal['report']['transition']))
display(pd.DataFrame(temporal['report']['expected_transition_counts']))
''')
markdown('''## PBL 2 Sum-product algorithm

Forward-backward is exact sum-product on the HMM chain. Local factors are multiplied and hidden
states are summed out. The implementation uses scipy.special.logsumexp for numerical stability.
The state and adjacent-state marginals must each normalize to one. This is not Viterbi decoding.
Change the assumed persistence below to explore sensitivity for one actual pixel.
''')
code('''example = temporal['report']['examples'][2]
persistence = 0.90
posterior, joint, log_evidence = sum_product(
    np.array(example['log_emissions'])[None], temporal['report']['initial'],
    transition_matrix(4, persistence))
display(pd.DataFrame(posterior[0], index=data['dates'], columns=[f'State {i}' for i in range(4)]))
display(pd.DataFrame(joint[0, 0]))
print('Posterior of different latent states:', float(1 - np.trace(joint[0, 0])))
np.testing.assert_allclose(posterior.sum(axis=2), 1, atol=1e-10)
np.testing.assert_allclose(joint.sum(axis=(2, 3)), 1, atol=1e-10)
''')
markdown('''## PBL 3 CART Boosting and Ensemble Learning

Use four reflectance bands plus NDVI and NDWI. Spectral-rule labels use thresholds derived only
from the training area. Exclude pixels that match zero or multiple rules. CART uses Gini splits
and depth 6. AdaBoost uses 80 depth-one trees. Random forest uses 80 bootstrapped randomized trees.
Soft voting equally averages their probability outputs. Settings are fixed before evaluation.
''')
code('''supervised = fit_supervised(data)
display(pd.DataFrame([{k:v for k,v in m.items() if k != 'confusion_matrix'}
                      for m in supervised['report']['metrics']]))
display(supervised['report']['sample_class_counts'])
print(supervised['report']['note'])
''')
markdown('''## Save and inspect all figures

This cell writes course_results.json, model_comparison.csv, CART rules and all course figures.
The Streamlit Course demo opens these saved artifacts without retraining.
''')
code('''report = save_course_results(data, unsupervised, temporal, supervised, OUTPUT)
for name in ['kmeans_maps.png', 'pca_exploration.png', 'hmm_states.png', 'hmm_change.png',
             'sum_product_transitions.png', 'spatial_split.png', 'cart_tree.png',
             'pbl3_predictions.png', 'model_comparison.png', 'confusion_matrices.png', 'feature_importance.png']:
    display(Image(filename=str(OUTPUT / name), width=850))
''')
markdown('''## Conclusion and consolidated report

These experiments demonstrate the requested methods on real imagery. Provisional rule agreement
does not validate land-cover classes, seasonal change or urban growth. The HMM posterior depends on
an assumed transition prior. Independent labels, matched seasons, a geographic boundary and additional
acquisitions are needed for stronger conclusions. Run python build_project_report.py after generating
the course results to rebuild the consolidated PDF and editable Markdown report.

References: scikit-learn 1.8 user guides for clustering, PCA, trees and ensembles;
Michael Collins, The Forward-Backward Algorithm; USGS Landsat Collection 2 QA and surface reflectance documentation.
''')
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
             'language_info': {'name': 'python', 'version': '3.14'}}, 'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'pbl-cell-{i:02}'
(ROOT / 'Landsat_PBL_1_2_3.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print('Created Landsat_PBL_1_2_3.ipynb')
