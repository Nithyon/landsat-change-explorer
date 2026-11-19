"""Unit 5: CART, Random Forest, AdaBoost, Gradient Boosting, GAM (additive splines), Bayesian model averaging.
Task A (regression): predict 2024 reflectance of each band from 2015 features (spatial block hold-out),
and rebuild the change score from the best model's residuals.
Task B (classification): same split/labels as Unit 2 (provisional spectral classes).
Usage: python u5_ensembles.py WORK_DIR FIG_DIR"""
import sys, json, time
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

work, figs = map(Path, sys.argv[1:3])
plt.rcParams.update({"font.family": "serif", "font.size": 10})
A = np.load(work / "arrays.npz")
old, new, common, change = A["old"], A["new"], A["common"], A["change"]
rng = np.random.default_rng(42)
BN = ["B2 Blue", "B3 Green", "B4 Red", "B5 NIR"]

def feats(st):
    b, g, r, n = st
    return np.stack([b, g, r, n, (n - r) / (n + r + 1e-6), (g - n) / (g + n + 1e-6)], -1)
F_old = feats(old)[common]; Y = np.moveaxis(new, 0, -1)[common]
rr, cc = np.where(common)
block = (rr // 50) * 1000 + (cc // 50)                       # 50x50 px (7.5 km) spatial blocks
ub = np.unique(block); test_blocks = set(rng.choice(ub, len(ub) // 4, replace=False))
is_test = np.isin(block, list(test_blocks))
tr_all, te_all = np.flatnonzero(~is_test), np.flatnonzero(is_test)
tr = rng.choice(tr_all, 60000, replace=False); te = rng.choice(te_all, 30000, replace=False)
res = {"n_blocks": int(len(ub)), "n_test_blocks": len(test_blocks), "n_train": len(tr), "n_test": len(te)}

# CART depth selection with grouped (spatial) 5-fold CV on B5
cv = GroupKFold(5)
gs = GridSearchCV(DecisionTreeRegressor(random_state=42), {"max_depth": [2, 4, 6, 8, 10, 12, 15, 20], "min_samples_leaf": [20]},
                  cv=cv, scoring="neg_root_mean_squared_error").fit(F_old[tr], Y[tr, 3], groups=block[tr])
depths = [p["max_depth"] for p in gs.cv_results_["params"]]
res["cart_cv"] = {"depth": depths, "rmse": (-gs.cv_results_["mean_test_score"]).tolist(), "best_depth": gs.best_params_["max_depth"]}
best_depth = gs.best_params_["max_depth"]

def models():
    return {
        "Linear (Unit 2, single band)": None,
        "Linear (6 features)": LinearRegression(),
        "GAM (additive splines)": make_pipeline(SplineTransformer(n_knots=8, degree=3), Ridge(alpha=1.0)),
        "CART": DecisionTreeRegressor(max_depth=best_depth, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=42),
        "AdaBoost (R2)": AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=100, learning_rate=0.05, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42),
    }

reg = {}; preds_all = {}
fitted_b5 = {}
for b in range(4):
    for name, m in models().items():
        t0 = time.time()
        if m is None:
            m = LinearRegression(); Xtr_, Xte_ = F_old[tr][:, [b]], F_old[te][:, [b]]
        else:
            Xtr_, Xte_ = F_old[tr], F_old[te]
        m.fit(Xtr_, Y[tr, b]); p = m.predict(Xte_)
        reg.setdefault(name, {})[BN[b]] = {"r2": float(r2_score(Y[te, b], p)), "rmse": float(np.sqrt(mean_squared_error(Y[te, b], p))),
                                           "fit_s": round(time.time() - t0, 2)}
        preds_all.setdefault(BN[b], {})[name] = p
        if b == 3:
            fitted_b5[name] = m
res["regression"] = reg

# Bayesian model averaging over the candidate models (Gaussian likelihood on a validation split of train)
val = rng.choice(len(tr), 15000, replace=False); fit_part = np.setdiff1d(np.arange(len(tr)), val)
bma = {}
for b in range(4):
    lls, vpreds, tpreds = {}, {}, {}
    for name, m in models().items():
        if m is None:
            continue
        m.fit(F_old[tr][fit_part], Y[tr, b][fit_part]); pv = m.predict(F_old[tr][val])
        s2 = np.mean((Y[tr, b][val] - pv) ** 2)
        lls[name] = -0.5 * len(val) * (np.log(2 * np.pi * s2) + 1); tpreds[name] = preds_all[BN[b]][name]
    mx = max(lls.values()); w = {k: np.exp(v - mx) for k, v in lls.items()}; z = sum(w.values()); w = {k: v / z for k, v in w.items()}
    pb = sum(w[k] * tpreds[k] for k in w)
    # simple average ensemble of the four tree ensembles as a contrast to BMA
    pavg = np.mean([tpreds[k] for k in ("Random Forest", "AdaBoost (R2)", "Gradient Boosting", "GAM (additive splines)")], 0)
    bma[BN[b]] = {"weights": w, "r2": float(r2_score(Y[te, b], pb)), "rmse": float(np.sqrt(mean_squared_error(Y[te, b], pb))),
                  "avg_r2": float(r2_score(Y[te, b], pavg)), "avg_rmse": float(np.sqrt(mean_squared_error(Y[te, b], pavg)))}
res["bma"] = bma

# Feature importance (RF, B5)
names6 = ["B2", "B3", "B4", "B5", "NDVI", "NDWI"]
res["rf_importance_B5"] = dict(zip(names6, fitted_b5["Random Forest"].feature_importances_.tolist()))

# Change score from Gradient Boosting residuals on all common pixels (refit on train)
resid = np.empty_like(Y)
for b in range(4):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42).fit(F_old[tr], Y[tr, b])
    resid[:, b] = Y[:, b] - m.predict(F_old)
scale = np.median(np.abs(resid), 0) + 1e-6
score = np.sqrt(np.mean((resid / scale) ** 2, 1)); med = np.median(score)
thr = med + 3 * 1.4826 * np.median(np.abs(score - med)); flag = score > thr
lin_flag = change[common] == 1
res["gb_change"] = {"threshold": float(thr), "flag_pct": float(100 * flag.mean()), "lin_flag_pct": float(100 * lin_flag.mean()),
                    "jaccard_with_linear": float((flag & lin_flag).sum() / (flag | lin_flag).sum()),
                    "flagged_pixels": int(flag.sum())}

# ---- Task B: classification on Unit 2 split ----
S = np.load(work / "split.npz"); Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
clf = {"CART": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
       "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
       "AdaBoost (SAMME)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=42),
       "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=42)}
cres = {}
for n_, m in clf.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    cres[n_] = {"acc": float(accuracy_score(yte, p)), "f1": float(f1_score(yte, p, average="macro"))}
cdepth = []
for d in range(1, 13):
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, ytr); cdepth.append(float(accuracy_score(yte, m.predict(Xte))))
res["classification"] = cres; res["cart_clf_depth_acc"] = cdepth
json.dump(res, open(work / "u5_results.json", "w"), indent=1)

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(depths, res["cart_cv"]["rmse"], "o-"); ax[0].axvline(best_depth, ls=":", c="r"); ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("CV RMSE (B5)")
ax[0].set_title("CART depth selection (spatial 5-fold CV)")
mn = [k for k in reg]; x = np.arange(len(mn)); w = 0.2
for i, b in enumerate(BN):
    ax[1].bar(x + (i - 1.5) * w, [reg[k][b]["r2"] for k in mn], w, label=b)
ax[1].set_xticks(x); ax[1].set_xticklabels(mn, fontsize=7, rotation=35, ha="right"); ax[1].set_ylabel("Test R²"); ax[1].legend(fontsize=7)
ax[1].set_title("Spatial hold-out R² by model and band")
imp = res["rf_importance_B5"]; ax[2].barh(list(imp), list(imp.values()), color="#4472c4"); ax[2].set_title("Random Forest feature importance (target B5)")
fig.tight_layout(); fig.savefig(figs / "u5_regression.png", dpi=150); plt.close(fig)

rows, cols = np.where(common)
crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
gmap = np.full(common.shape, -1, np.int8); gmap[common] = flag
cmp_ = np.full(common.shape, -1, np.int8); cmp_[common] = lin_flag.astype(int) + 2 * flag.astype(int)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
cc1 = matplotlib.colors.ListedColormap(["#202020", "#e31a1c"]); cc1.set_bad("#d9d9d9")
ax[0].imshow(np.ma.masked_less(gmap, 0)[crop], cmap=cc1, interpolation="nearest"); ax[0].set_title(f"Gradient-boosting residual change ({100 * flag.mean():.2f}%)")
cc2 = matplotlib.colors.ListedColormap(["#202020", "#1f78b4", "#ff7f00", "#ffff33"]); cc2.set_bad("#d9d9d9")
ax[1].imshow(np.ma.masked_less(cmp_, 0)[crop], cmap=cc2, vmin=0, vmax=3, interpolation="nearest")
ax[1].set_title("Agreement: blue=linear only, orange=GB only, yellow=both")
for a in ax:
    a.axis("off")
fig.tight_layout(); fig.savefig(figs / "u5_change_compare.png", dpi=150); plt.close(fig)
print(json.dumps({k: res[k] for k in ("cart_cv", "regression", "bma", "gb_change", "classification", "rf_importance_B5")}, indent=1, default=float))
