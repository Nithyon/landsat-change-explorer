# Report analysis scripts (Units 1–5)

These scripts produce every number and figure in the 21CSC305P project report. They reuse `landsat_analysis.py` and the ten GeoTIFFs in `colab_test_data` (run `python prepare_data.py` first).

| Script | Unit | What it does |
| --- | --- | --- |
| `u12_pipeline.py` | 1–2 | Preprocessing, EDA, polynomial curve fitting, ridge/Bayesian regression, Gaussian MLE, logistic regression, SVM, regression residual change score. Caches arrays for later units. |
| `u3_clustering_pca.py` | 3 | K selection (elbow, silhouette, BIC), K-means, GMM (EM), hierarchical clustering, PCA and bi-temporal PCA |
| `u4_hmm.py` | 4 | Gaussian HMM written from scratch in NumPy (scaled forward–backward, Baum–Welch, Viterbi) on image-row transects |
| `u5_ensembles.py` | 5 | CART, Random Forest, AdaBoost, Gradient Boosting, GAM, Bayesian model averaging with spatial block hold-out |
| `diagrams.py` | – | Workflow diagrams |
| `build_report.py` | – | Builds the Word report from the cached results (needs the template .docx) |

## Run order (from the repository root)

```powershell
python prepare_data.py
mkdir work, outputs\report_figures
python u12_pipeline.py colab_test_data work outputs\report_figures
python u3_clustering_pca.py work outputs\report_figures
python u4_hmm.py work outputs\report_figures
python u5_ensembles.py work outputs\report_figures
python diagrams.py outputs\report_figures
```

Extra packages beyond `requirements.txt`: none (numpy, scipy, scikit-learn, matplotlib, rasterio). `build_report.py` also needs `python-docx`.
Runtime on a laptop: about 1 min (Units 1–2), 3 min (Unit 3), 1.5 min (Unit 4), 6.5 min (Unit 5). All runs use fixed random seeds.
`work/` holds cached arrays (~100 MB) and the JSON results; it is not committed.
