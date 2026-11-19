"""Unit 3: K-means, Gaussian mixture (EM), hierarchical clustering, PCA (incl. bi-temporal change PCA).
Usage: python u3_clustering_pca.py WORK_DIR FIG_DIR"""
import sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

work, figs = map(Path, sys.argv[1:3])
plt.rcParams.update({"font.family": "serif", "font.size": 10})
A = np.load(work / "arrays.npz")
old, new, vold, common, mle_old, change = A["old"], A["new"], A["vold"], A["common"], A["mle_old"], A["change"]
rng = np.random.default_rng(42)
X_all = np.moveaxis(old, 0, -1)[vold]                      # 2015 valid pixels, 4 bands
mle_lab = mle_old[vold]
sc = StandardScaler().fit(X_all)
Z_all = sc.transform(X_all)
fit_idx = rng.choice(len(Z_all), 200000, replace=False)
eval_idx = rng.choice(len(Z_all), 20000, replace=False)
res = {"n_pixels": int(len(Z_all))}

# ---- choose k: elbow (inertia), silhouette, GMM BIC ----
ks = list(range(2, 9))
sel = {"k": ks, "inertia": [], "silhouette": [], "bic": []}
for k in ks:
    km = KMeans(k, n_init=4, random_state=42).fit(Z_all[fit_idx])
    sel["inertia"].append(float(km.inertia_))
    sel["silhouette"].append(float(silhouette_score(Z_all[eval_idx], km.predict(Z_all[eval_idx]))))
    gm = GaussianMixture(k, covariance_type="full", random_state=42).fit(Z_all[fit_idx[:60000]])
    sel["bic"].append(float(gm.bic(Z_all[fit_idx[:60000]])))
res["k_selection"] = sel
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(ks, sel["inertia"], "o-"); ax[0].set_title("K-means elbow (inertia)")
ax[1].plot(ks, sel["silhouette"], "o-", color="g"); ax[1].set_title("Silhouette coefficient")
ax[2].plot(ks, sel["bic"], "o-", color="m"); ax[2].set_title("GMM BIC")
for a in ax:
    a.set_xlabel("Number of clusters K"); a.axvline(4, ls=":", c="grey")
fig.tight_layout(); fig.savefig(figs / "u3_k_selection.png", dpi=150); plt.close(fig)

# ---- K = 4 models ----
K = 4
km = KMeans(K, n_init=10, random_state=42).fit(Z_all[fit_idx])
gm = GaussianMixture(K, covariance_type="full", random_state=42, max_iter=300).fit(Z_all[fit_idx])
hsub = rng.choice(len(Z_all), 8000, replace=False)
hc_ward = AgglomerativeClustering(K, linkage="ward").fit(Z_all[hsub])
hc_avg = AgglomerativeClustering(K, linkage="average").fit(Z_all[hsub])
lab_km, lab_gm = km.predict(Z_all), gm.predict(Z_all)
ev = Z_all[eval_idx]
out = {}
for name, pred_eval, pred_all in (("K-means", km.predict(ev), lab_km), ("GMM (EM)", gm.predict(ev), lab_gm)):
    out[name] = {"silhouette": float(silhouette_score(ev, pred_eval)),
                 "davies_bouldin": float(davies_bouldin_score(ev, pred_eval)),
                 "calinski_harabasz": float(calinski_harabasz_score(ev, pred_eval)),
                 "ARI_vs_MLE": float(adjusted_rand_score(mle_lab, pred_all)),
                 "NMI_vs_MLE": float(normalized_mutual_info_score(mle_lab, pred_all)),
                 "sizes": np.bincount(pred_all, minlength=K).tolist()}
for name, hc in (("Hierarchical (Ward)", hc_ward), ("Hierarchical (average)", hc_avg)):
    p = hc.labels_; zs = Z_all[hsub]
    out[name] = {"silhouette": float(silhouette_score(zs, p)), "davies_bouldin": float(davies_bouldin_score(zs, p)),
                 "calinski_harabasz": float(calinski_harabasz_score(zs, p)),
                 "ARI_vs_MLE": float(adjusted_rand_score(mle_lab[hsub], p)),
                 "NMI_vs_MLE": float(normalized_mutual_info_score(mle_lab[hsub], p)),
                 "sizes": np.bincount(p, minlength=K).tolist()}
res["clustering"] = out
res["gmm_converged"] = bool(gm.converged_); res["gmm_iter"] = int(gm.n_iter_)
res["gmm_loglik_per_sample"] = float(gm.score(Z_all[fit_idx]))
res["kmeans_centres_reflectance"] = sc.inverse_transform(km.cluster_centers_).tolist()
res["gmm_means_reflectance"] = sc.inverse_transform(gm.means_).tolist()
res["gmm_weights"] = gm.weights_.tolist()
# contingency K-means vs MLE
ct = np.zeros((K, 4), int); np.add.at(ct, (lab_km, mle_lab), 1); res["kmeans_vs_mle"] = ct.tolist()

# maps
rows, cols = np.where(common)
crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
for a, lab, t in ((ax[0], lab_km, "K-means (K=4)"), (ax[1], lab_gm, "GMM / EM (K=4)")):
    m = np.full(vold.shape, -1); m[vold] = lab
    a.imshow(np.ma.masked_less(np.where(common, m, -1), 0)[crop], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    a.set_title(t); a.axis("off")
from scipy.cluster.hierarchy import linkage, dendrogram
L = linkage(Z_all[hsub[:2000]], "ward")
dendrogram(L, ax=ax[2], truncate_mode="lastp", p=24, no_labels=True, color_threshold=L[-3, 2])
ax[2].set_title("Ward dendrogram (2,000-pixel sample)"); ax[2].set_ylabel("Linkage distance")
fig.tight_layout(); fig.savefig(figs / "u3_cluster_maps.png", dpi=150); plt.close(fig)

# ---- PCA on 2015 bands ----
pca4 = PCA().fit(Z_all[fit_idx])
res["pca4_explained"] = pca4.explained_variance_ratio_.tolist()
res["pca4_components"] = pca4.components_.tolist()
# ---- bi-temporal PCA (8 features: 2015 + 2024) ----
X8 = np.hstack([np.moveaxis(old, 0, -1)[common], np.moveaxis(new, 0, -1)[common]])
s8 = StandardScaler().fit(X8); Z8 = s8.transform(X8)
pca8 = PCA().fit(Z8[rng.choice(len(Z8), 200000, replace=False)])
res["pca8_explained"] = pca8.explained_variance_ratio_.tolist()
res["pca8_components"] = pca8.components_.tolist()
P8 = pca8.transform(Z8)
# change-like component: the component with most opposite-sign loadings between dates
contrast = [abs(np.dot(c[:4], c[4:])) / (np.linalg.norm(c[:4]) * np.linalg.norm(c[4:])) * np.sign(np.dot(c[:4], c[4:])) for c in pca8.components_]
res["pca8_date_alignment"] = [float(x) for x in contrast]
# reconstruction error with 2 PCs as anomaly score, compare with regression-change flag
k = 3
recon = P8[:, :k] @ pca8.components_[:k] + 0
err = np.sqrt(((Z8 - recon) ** 2).mean(1))
thr = np.median(err) + 3 * 1.4826 * np.median(np.abs(err - np.median(err)))
flag = err > thr
reg_flag = change[common] == 1
res["pca_change"] = {"k": k, "threshold": float(thr), "flag_pct": float(100 * flag.mean()),
                     "reg_flag_pct": float(100 * reg_flag.mean()),
                     "overlap_jaccard": float((flag & reg_flag).sum() / (flag | reg_flag).sum()),
                     "recall_of_reg_flags": float((flag & reg_flag).sum() / reg_flag.sum())}

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].bar(range(1, 5), pca4.explained_variance_ratio_, color="#4472c4", label="2015, 4 bands")
ax[0].plot(range(1, 5), np.cumsum(pca4.explained_variance_ratio_), "ro-", label="cumulative")
ax[0].set_title("Scree plot (2015 bands)"); ax[0].set_xlabel("Principal component"); ax[0].legend()
ax[1].bar(range(1, 9), pca8.explained_variance_ratio_, color="#70ad47")
ax[1].plot(range(1, 9), np.cumsum(pca8.explained_variance_ratio_), "ro-")
ax[1].set_title("Scree plot (bi-temporal, 8 features)"); ax[1].set_xlabel("Principal component")
sub = rng.choice(len(P8), 6000, replace=False)
ax[2].scatter(P8[sub, 0], P8[sub, 1], s=2, c=np.where(reg_flag[sub], "#e31a1c", "#9e9e9e"), alpha=.5)
ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].set_title("Bi-temporal PC1 vs PC2 (red = regression-flagged)")
fig.tight_layout(); fig.savefig(figs / "u3_pca.png", dpi=150); plt.close(fig)

json.dump(res, open(work / "u3_results.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("clustering", "pca4_explained", "pca8_explained", "pca_change", "gmm_iter")}, indent=1))
