# Landsat project demonstration

## Before class

Run `python -m streamlit run app.py` and open http://localhost:8501. Open **Methods**. Saved results work without downloading or reprocessing the imagery during class. Keep the consolidated PDF report and notebook ready. Rebuild offline with `python build_pbl_demo.py` only when needed.

## Suggested presentation order

1. **Problem and dataset — 1 minute.** Compare Landsat 8 scenes from 13 December 2015 and 30 July 2024. Four reflectance bands and QA_PIXEL per date. Explain that seasons differ and no ground-truth labels were supplied.
2. **MLE and regression — 2 minutes.** Open History. Show matching satellite views, cloud masking, provisional MLE classes and regression residual flags. Explain the 5.24% result as possible spectral change on the common valid footprint. Confirm the original MLE and regression syllabus mapping with your instructor.
3. **K-means and PCA — 3 minutes.** Open Methods → Clustering and state inference. Show the four unsupervised groups, centroid spectra, PCA variance chart and two-component scatter plot. Cluster numbers do not name physical land-cover classes. PCA is used for exploration rather than replacing the four-band models.
4. **HMM and sum-product — 3 minutes.** Explain hidden states and Gaussian observations. There are two observations per pixel; the transition prior is assumed. In Sum-product walkthrough, choose an actual pixel and move the persistence slider. Read the state marginals and joint transition table. Explain that summation over all paths differs from Viterbi's single best path. The slider changes only the example; scene maps stay at the default prior.
5. **CART, boosting and ensembles — 3 minutes.** Show the CART tree, AdaBoost round curve, random forest, soft voting and spatial holdout map. Discuss macro F1, confusion matrices and per-class support. Scores measure agreement with provisional rules. A more complex ensemble need not outperform CART on rule-defined targets.
6. **Conclusion — 1 minute.** State what was implemented, what the experiment measures and what is needed next: independent labels, comparable seasons, a defined study boundary and more acquisition dates.

## Answers to likely questions

- **Why an HMM with only two dates?** Two-date exact inference is mathematically valid. We do not claim to learn temporal dynamics or predict future years from these data.
- **What is sum-product doing?** Multiplying local factors and summing over unobserved states to calculate normalized posterior marginals on the HMM chain. Forward-backward is the chain implementation.
- **Where do training labels come from?** Training-area NDVI, NDWI and brightness rules. Only uniquely matched pixels are used. These are weak labels, not surveyed land-cover truth.
- **How do you reduce leakage?** Rule thresholds, scaling, centroids, PCA and classifiers use only the Before training area. Evaluation uses the eastern Before holdout with a guard strip. Spatial dependence can still remain at larger scales.
- **Does PCA improve the classifier?** This demonstration does not test that claim. PCA provides an exploratory projection and variance explanation.
- **Are cluster IDs the same as class IDs?** No. Unsupervised cluster IDs and rule-defined supervised class IDs are separate label systems.
- **Is 5.24% an accuracy score?** No. It is the proportion of common valid pixels above the original regression-residual threshold.
- **Can we call this urban-growth measurement?** Not without validated labels, an explicit geographic boundary and controls for seasonality.

## Reproduction commands

```powershell
python prepare_data.py
python build_pbl_demo.py
python build_project_report.py
python -m unittest discover -s tests -q
python -m streamlit run app.py
```

Install `requirements-report.txt` to regenerate the PDF. The full notebook is `Landsat_PBL_1_2_3.ipynb`. The original corrected notebook remains available for the MLE and regression workflow.
