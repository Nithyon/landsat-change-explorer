"""Workflow diagrams for the report. Usage: python diagrams.py FIG_DIR"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

figs = Path(sys.argv[1])
plt.rcParams.update({"font.family": "serif"})
COL = {"in": "#dbe8f7", "pre": "#e2f0d9", "alg": "#fff2cc", "eval": "#fbe5d6", "out": "#e4dfec"}


def flow(name, stages, w=11, h=None):
    """stages: list of (title, body, colour) drawn left-to-right, wrapping into rows of 4."""
    per = 4
    nrows = (len(stages) + per - 1) // per
    h = h or 2.3 * nrows + 0.4
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, per * 3); ax.set_ylim(0, nrows * 2.6); ax.axis("off")
    pos = []
    for i, (t, body, c) in enumerate(stages):
        r, k = divmod(i, per)
        if r % 2:
            k = per - 1 - k
        x, y = k * 3 + 0.2, (nrows - 1 - r) * 2.6 + 0.3
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.9, boxstyle="round,pad=0.04,rounding_size=0.15", fc=COL[c], ec="#333", lw=1.2))
        ax.text(x + 1.25, y + 1.6, t, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + 1.25, y + 0.75, body, ha="center", va="center", fontsize=8.3, wrap=True, linespacing=1.3)
        pos.append((x, y))
    for (x0, y0), (x1, y1) in zip(pos, pos[1:]):
        if abs(y0 - y1) < 1e-6:
            s, e = ((x0 + 2.5, y0 + 0.95), (x1, y1 + 0.95)) if x1 > x0 else ((x0, y0 + 0.95), (x1 + 2.5, y1 + 0.95))
        else:
            s, e = (x0 + 1.25, y0), (x1 + 1.25, y1 + 1.9)
        ax.annotate("", e, s, arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333"))
    fig.tight_layout(); fig.savefig(figs / name, dpi=170); plt.close(fig)


flow("wf_overall.png", [
    ("Input", "Landsat 8 C2 L2SP\n2015-12-13 & 2024-07-30\nSR_B2..B5 + QA_PIXEL", "in"),
    ("Integrity", "Re-assemble split\nbands, SHA-256 check,\nproduct/date validation", "pre"),
    ("Preprocess", "DN x 2.75e-5 - 0.2\nQA bits 0-5 masked\nAlign to 2015 B4, 150 m", "pre"),
    ("Units 1-2", "EDA, curve fitting,\nGaussian MLE, LR/SVM,\nregression residual score", "alg"),
    ("Unit 3", "K-means, GMM-EM,\nhierarchical clustering,\nPCA / change PCA", "alg"),
    ("Unit 4", "Gaussian HMM on\nrow transects:\nBaum-Welch + Viterbi", "alg"),
    ("Unit 5", "CART, RF, AdaBoost,\nGBM, GAM, BMA\n(spatial block CV)", "alg"),
    ("Outputs", "Class / cluster /\nstate / change maps,\nmetrics, Streamlit app", "out"),
])
flow("wf_unit12.png", [
    ("Aligned stacks", "2015 & 2024\n4-band reflectance\n1537 x 1507 grid", "in"),
    ("EDA", "Band statistics,\nhistograms, correlation,\nNDVI, cloud masking %", "pre"),
    ("Seed labels", "NDVI / NDWI / brightness\nquantile rules ->\n4 provisional classes", "pre"),
    ("Gaussian MLE", "Class mean + covariance,\nlog-likelihood argmax\napplied to both dates", "alg"),
    ("Regression", "Per band: 2024 = a x 2015 + b\n100k samples; poly degree\n& ridge / Bayesian ridge", "alg"),
    ("Change score", "Residual / median|res|\nRMS over 4 bands\nthreshold med + 3*1.4826*MAD", "eval"),
    ("Evaluation", "R^2, RMSE, class counts,\nacc / F1 vs seed labels\n(LR, SVM baselines)", "eval"),
    ("Outputs", "MLE maps, change map\n(5.24 % flagged),\nregression plots", "out"),
])
flow("wf_unit3.png", [
    ("Features", "2015 valid pixels\n(1.17 M) 4 bands,\nz-score standardised", "in"),
    ("Model selection", "K = 2..8: elbow,\nsilhouette, GMM BIC", "pre"),
    ("Clustering", "K-means (K=4)\nGMM via EM (full cov)\nWard / average linkage", "alg"),
    ("PCA", "4-band PCA (scree)\n8-feature bi-temporal\nPCA + recon. error", "alg"),
    ("Evaluation", "Silhouette, DB, CH,\nARI / NMI vs MLE,\noverlap with regression flag", "eval"),
    ("Outputs", "Cluster maps,\ndendrogram, scree plots,\nPC scatter", "out"),
])
flow("wf_unit4.png", [
    ("Sequences", "East-west row transects,\nvalid runs >= 30 px\n(150 m steps)", "in"),
    ("Split", "Train rows r = 0 mod 15\nTest rows r = 7 mod 15\n(no shared rows)", "pre"),
    ("Initialise", "K-means -> means, covs\nA: 0.9 diagonal\npi uniform", "pre"),
    ("Baum-Welch", "Scaled forward-backward\nE-step: gamma, xi\nM-step: pi, A, mu, Sigma", "alg"),
    ("Viterbi", "Most probable state path\nfor every row of\n2015 and 2024", "alg"),
    ("Evaluation", "Held-out log-lik vs GMM,\nrun length, ARI,\n2015->2024 transitions", "eval"),
    ("Outputs", "State maps, transition\nmatrix, transect\nposterior plot", "out"),
])
flow("wf_unit5.png", [
    ("Data", "Common pixels: 2015\nB2-B5 + NDVI + NDWI\nTarget: 2024 band", "in"),
    ("Spatial split", "50x50-px blocks; 25 %\nblocks held out\n60k train / 30k test", "pre"),
    ("Tuning", "CART depth by\nGroupKFold (5) CV\non B5 RMSE", "pre"),
    ("Models", "CART, Random Forest,\nAdaBoost.R2, Hist-GBM,\nGAM splines", "alg"),
    ("Combining", "Bayesian model averaging\n(validation likelihood)\n+ simple averaging", "alg"),
    ("Evaluation", "Test R^2 / RMSE per band\nvs Unit 2 linear; classifier\nacc / F1 vs MLE/LR/SVM", "eval"),
    ("Outputs", "GBM-residual change map\nand agreement with\nlinear change map", "out"),
])
print("ok")
