"""Units 1-2: preprocessing, EDA, polynomial curve fitting, Gaussian MLE, regression, change score.
Usage: python u12_pipeline.py DATA_DIR WORK_DIR FIG_DIR"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import Ridge, BayesianRidge, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import landsat_analysis as la

data, work, figs = map(Path, sys.argv[1:4])
acq = la.discover(str(data))
before, after = sorted(acq.values(), key=lambda a: a.date)
la.validate_pair(before, after)
target = la.reference_grid(before)
old, vold = la.load_acquisition(before, target)
new, vnew = la.load_acquisition(after, target)
common = vold & vnew
rng = np.random.default_rng(42)
masks = la.seed_masks(old, vold)
models, seeds = la.fit_mle(old, masks, rng)
mle_old = la.classify_mle(old, vold, models)
mle_new = la.classify_mle(new, vnew, models)
reg, n, score, change, thr, so, sn = la.regression_and_change(old, new, common, rng)

# Seed label map: keep pixels that belong to exactly one provisional class
seed = np.full(vold.shape, -1, np.int8)
cnt = np.zeros(vold.shape, np.int8)
for i, m in enumerate(masks.values()):
    seed[m] = i
    cnt += m
seed[cnt != 1] = -1
np.savez_compressed(work / "arrays.npz", old=old, new=new, vold=vold, vnew=vnew, common=common,
                    mle_old=mle_old, mle_new=mle_new, score=score, change=change, seed=seed)

res = {"shape": list(vold.shape), "valid_old": int(vold.sum()), "valid_new": int(vnew.sum()),
       "common": int(common.sum()), "seeds": seeds,
       "seed_unique": {la.CLASS_NAMES[i]: int((seed == i).sum()) for i in range(4)},
       "regression": reg, "threshold": float(thr), "changed": int((change == 1).sum()),
       "unchanged": int((change == 0).sum())}
res["change_pct"] = 100 * res["changed"] / res["common"]
res["mle_common"] = {d: np.bincount(m[common], minlength=4).tolist() for d, m in (("2015", mle_old), ("2024", mle_new))}
res["mle_means"] = {m[1]: m[2].tolist() for m in models}
tt = np.zeros((4, 4), int)
np.add.at(tt, (mle_old[common], mle_new[common]), 1)
res["mle_transition"] = tt.tolist()

# ---------------- Unit 1: EDA ----------------
names = ["B2 Blue", "B3 Green", "B4 Red", "B5 NIR"]
stats = {}
for lab, st, v in (("2015", old, vold), ("2024", new, vnew)):
    px = np.moveaxis(st, 0, -1)[v]
    stats[lab] = {}
    for i in range(4):
        x = px[:, i]
        stats[lab][names[i]] = {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
                                "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
                                "q75": float(np.quantile(x, .75)), "max": float(x.max()),
                                "skew": float(((x - x.mean()) ** 3).mean() / x.std() ** 3)}
    stats[lab]["corr"] = np.corrcoef(px.T).tolist()
    nd = ((st[3] - st[2]) / (st[3] + st[2] + 1e-6))[v]
    stats[lab]["ndvi_mean"] = float(nd.mean())
    stats[lab]["ndvi_std"] = float(nd.std())
res["eda"] = stats
res["mask_pct"] = {"2015_invalid": 100 * (1 - vold.sum() / vold.size), "2024_invalid": 100 * (1 - vnew.sum() / vnew.size)}

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
for i, a in enumerate(ax.flat):
    for lab, st, v, c in (("2015-12-13", old, vold, "#1f5fa8"), ("2024-07-30", new, vnew, "#c0392b")):
        a.hist(st[i][v], bins=120, range=(-0.05, 0.6), density=True, alpha=.55, color=c, label=lab)
    a.set_title(names[i]); a.set_xlabel("Surface reflectance"); a.set_ylabel("Density")
ax[0, 0].legend(); fig.tight_layout(); fig.savefig(figs / "eda_histograms.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, lab in zip(ax, ("2015", "2024")):
    c = np.array(stats[lab]["corr"])
    im = a.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
    a.set_xticks(range(4)); a.set_yticks(range(4))
    a.set_xticklabels(["B2", "B3", "B4", "B5"]); a.set_yticklabels(["B2", "B3", "B4", "B5"])
    for x in range(4):
        for y in range(4):
            a.text(y, x, f"{c[x, y]:.2f}", ha="center", va="center", fontsize=9)
    a.set_title(f"Band correlation ({lab})")
fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(figs / "eda_correlation.png", dpi=150, bbox_inches="tight"); plt.close(fig)

rows, cols = np.where(common)
crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, lab, st in ((ax[0], "2015", old), (ax[1], "2024", new)):
    nd = np.where(common, (st[3] - st[2]) / (st[3] + st[2] + 1e-6), np.nan)
    im = a.imshow(nd[crop], cmap="RdYlGn", vmin=-0.2, vmax=0.8); a.set_title(f"NDVI {lab}"); a.axis("off")
    ax[2].hist(nd[common], bins=100, range=(-0.4, 0.9), alpha=.55, density=True, label=lab)
fig.colorbar(im, ax=ax[:2], shrink=.8); ax[2].legend(); ax[2].set_title("NDVI distribution (common pixels)"); ax[2].set_xlabel("NDVI")
fig.savefig(figs / "eda_ndvi.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------- Unit 1: polynomial curve fitting ----------------
op = np.moveaxis(old, 0, -1)[common]
npx = np.moveaxis(new, 0, -1)[common]
idx = rng.permutation(len(op)); tr, te = idx[:80000], idx[80000:120000]
poly = {}
for b in range(4):
    poly[names[b]] = {}
    for d in range(1, 7):
        X = np.vander(op[tr, b], d + 1)
        w, *_ = np.linalg.lstsq(X, npx[tr, b], rcond=None)
        e_tr = np.sqrt(np.mean((X @ w - npx[tr, b]) ** 2))
        e_te = np.sqrt(np.mean((np.vander(op[te, b], d + 1) @ w - npx[te, b]) ** 2))
        poly[names[b]][d] = {"train_rmse": float(e_tr), "test_rmse": float(e_te)}
res["poly"] = poly
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
for b, a in enumerate(ax):
    d = list(range(1, 7))
    a.plot(d, [poly[names[b]][k]["train_rmse"] for k in d], "o-", label="train")
    a.plot(d, [poly[names[b]][k]["test_rmse"] for k in d], "s--", label="test")
    a.set_title(names[b]); a.set_xlabel("Polynomial degree M")
ax[0].set_ylabel("RMSE"); ax[0].legend(); fig.tight_layout(); fig.savefig(figs / "u1_poly_degree.png", dpi=150); plt.close(fig)

# ---------------- Unit 2: multivariate / ridge / Bayesian regression for B4 ----------------
lin = {}
for nm, mdl in (("OLS (4 bands)", Ridge(alpha=1e-9)), ("Ridge (alpha=1)", Ridge(alpha=1.0)), ("Bayesian ridge", BayesianRidge())):
    mdl.fit(op[tr], npx[tr, 2]); p = mdl.predict(op[te]); y = npx[te, 2]
    lin[nm] = {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
               "r2": float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)), "coef": mdl.coef_.tolist()}
res["multi_reg_B4"] = lin

# ---------------- Unit 2: classifiers on seed labels (shared split reused in Unit 5) ----------------
lab_px = np.moveaxis(old, 0, -1)[seed >= 0]; lab_y = seed[seed >= 0]
take = np.random.default_rng(7).permutation(len(lab_y))[:60000]
Xa, ya = lab_px[take], lab_y[take]
Xtr, ytr, Xte, yte = Xa[:40000], ya[:40000], Xa[40000:], ya[40000:]
np.savez_compressed(work / "split.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

def summ(y, p):
    return {"acc": float(accuracy_score(y, p)), "f1": float(f1_score(y, p, average="macro")),
            "cm": confusion_matrix(y, p).tolist()}

cls = {}
params = []
for k in range(4):
    s = Xtr[ytr == k]; cov = np.cov(s, rowvar=False) + np.eye(4) * 1e-6
    params.append((s.mean(0), np.linalg.pinv(cov), np.linalg.slogdet(cov)[1]))
sc = np.stack([-0.5 * (ld + np.einsum("ij,jk,ik->i", Xte - m, ic, Xte - m)) for m, ic, ld in params], 1)
cls["Gaussian MLE"] = summ(yte, sc.argmax(1))
ss = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=2000).fit(ss.transform(Xtr), ytr)
cls["Logistic regression"] = summ(yte, lr.predict(ss.transform(Xte)))
sv = SVC(kernel="rbf", C=10, gamma="scale").fit(ss.transform(Xtr[:15000]), ytr[:15000])
cls["SVM (RBF)"] = summ(yte, sv.predict(ss.transform(Xte)))
res["u2_classifiers"] = cls
json.dump(res, open(work / "u12_results.json", "w"), indent=1)

# ---------------- Figures ----------------
cm = matplotlib.colors.ListedColormap(la.CLASS_COLORS); cm.set_bad("#d9d9d9")
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
for a, st, lab in ((ax[0], old, "13 Dec 2015"), (ax[1], new, "30 Jul 2024")):
    a.imshow(la.display_rgb(st, common)[crop]); a.set_title(f"Natural colour (B4/B3/B2) {lab}"); a.axis("off")
fig.tight_layout(); fig.savefig(figs / "rgb.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 5.8))
for a, m, lab in ((ax[0], mle_old, "2015"), (ax[1], mle_new, "2024")):
    a.imshow(np.ma.masked_less(np.where(common, m, -1), 0)[crop], cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    a.set_title(f"Gaussian MLE classes {lab}"); a.axis("off")
fig.legend(handles=[Patch(color=c, label=nm) for c, nm in zip(la.CLASS_COLORS, la.CLASS_NAMES)], loc="lower center", ncol=4)
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(figs / "mle_maps.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for b, a in enumerate(ax):
    a.scatter(so[:, b], sn[:, b], s=2, alpha=.3); r = reg[la.BANDS[b]]
    xx = np.linspace(so[:, b].min(), so[:, b].max(), 10)
    a.plot(xx, r["slope"] * xx + r["intercept"], "r-"); a.set_title(f"{names[b]}  R2={r['r2_sample']:.3f}")
    a.set_xlabel("2015 reflectance")
ax[0].set_ylabel("2024 reflectance"); fig.tight_layout(); fig.savefig(figs / "regression_scatter.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
im = ax[0].imshow(np.clip(score[crop], 0, 8), cmap="magma"); ax[0].set_title("Standardised residual score")
fig.colorbar(im, ax=ax[0], shrink=.8)
cc = matplotlib.colors.ListedColormap(["#202020", "#e31a1c"]); cc.set_bad("#d9d9d9")
ax[1].imshow(np.ma.masked_less(change, 0)[crop], cmap=cc, interpolation="nearest"); ax[1].set_title(f"Possible change ({res['change_pct']:.2f}%)")
for a in ax[:2]:
    a.axis("off")
ax[2].hist(np.clip(score[common], 0, 10), bins=150, color="#555")
ax[2].axvline(thr, color="r", ls="--", label=f"threshold = {thr:.3f}")
ax[2].set_yscale("log"); ax[2].set_xlabel("Residual score"); ax[2].legend(); ax[2].set_title("Score distribution")
fig.tight_layout(); fig.savefig(figs / "change_map.png", dpi=150); plt.close(fig)

print(json.dumps({k: res[k] for k in ("common", "changed", "change_pct", "threshold", "seed_unique", "multi_reg_B4")}, indent=1))
print({k: (v["acc"], v["f1"]) for k, v in cls.items()})
