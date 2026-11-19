"""Unit 4: Gaussian Hidden Markov Model implemented from scratch with NumPy.
Sequences are east-west pixel transects (image rows) of the 150 m grid; hidden states are
latent surface-cover regimes. Baum-Welch (scaled forward-backward) learns the parameters,
Viterbi decodes the most probable state path.
Usage: python u4_hmm.py WORK_DIR FIG_DIR"""
import sys, json, time
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

work, figs = map(Path, sys.argv[1:3])
plt.rcParams.update({"font.family": "serif", "font.size": 10})
A = np.load(work / "arrays.npz")
old, new, vold, vnew, common, mle_old, change = (A[k] for k in ("old", "new", "vold", "vnew", "common", "mle_old", "change"))
K, D, MINLEN = 4, 4, 30
mu0 = np.moveaxis(old, 0, -1)[vold].mean(0); sd0 = np.moveaxis(old, 0, -1)[vold].std(0)

def segments(stack, valid, rows):
    """Contiguous valid runs (length >= MINLEN) along each selected row -> list of (row, c0, Z)."""
    out = []
    for r in rows:
        v = valid[r].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, v, 0]))
        for c0, c1 in zip(edges[::2], edges[1::2]):
            if c1 - c0 >= MINLEN:
                out.append((r, c0, ((stack[:, r, c0:c1].T - mu0) / sd0).astype(np.float64)))
    return out

def log_gauss(X, means, covs):
    L = np.empty((len(X), K))
    for k in range(K):
        C = np.linalg.cholesky(covs[k]); d = np.linalg.solve(C, (X - means[k]).T)
        L[:, k] = -0.5 * (d ** 2).sum(0) - np.log(np.diag(C)).sum() - 0.5 * D * np.log(2 * np.pi)
    return L

def forward_backward(logB, pi, Atr):
    T = len(logB); m = logB.max(1, keepdims=True); B = np.exp(logB - m)
    alpha = np.empty((T, K)); c = np.empty(T)
    a = pi * B[0]; c[0] = a.sum(); alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ Atr) * B[t]; c[t] = a.sum(); alpha[t] = a / c[t]
    beta = np.empty((T, K)); beta[-1] = 1
    for t in range(T - 2, -1, -1):
        beta[t] = Atr @ (B[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    xi = (alpha[:-1, :, None] * Atr[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
    return gamma, xi.sum(0), float(np.log(c).sum() + m.sum())

def viterbi(logB, pi, Atr):
    T = len(logB); lA = np.log(Atr); delta = np.log(pi) + logB[0]; psi = np.empty((T, K), int)
    for t in range(1, T):
        s = delta[:, None] + lA; psi[t] = s.argmax(0); delta = s.max(0) + logB[t]
    path = np.empty(T, int); path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path

rows_all = np.arange(old.shape[1])
train = segments(old, vold, rows_all[::15]); test = segments(old, vold, rows_all[7::15])
Xtr = np.vstack([s[2] for s in train]); Xte = np.vstack([s[2] for s in test])
# init from K-means, ordered by NIR mean for readable state ids
km = KMeans(K, n_init=5, random_state=42).fit(Xtr)
order = np.argsort(km.cluster_centers_[:, 3]); lab = np.argsort(order)[km.labels_]
means = np.array([Xtr[lab == k].mean(0) for k in range(K)])
covs = np.array([np.cov(Xtr[lab == k], rowvar=False) + 1e-4 * np.eye(D) for k in range(K)])
pi = np.full(K, 1 / K); Atr = np.full((K, K), 0.1 / (K - 1)); np.fill_diagonal(Atr, 0.9)
hist = []; t0 = time.time()
for it in range(40):
    G, Xi, P0, LL = [], np.zeros((K, K)), np.zeros(K), 0.0
    for _, _, Z in train:
        g, xi, ll = forward_backward(log_gauss(Z, means, covs), pi, Atr)
        G.append(g); Xi += xi; P0 += g[0]; LL += ll
    hist.append(LL / len(Xtr))
    Gm = np.vstack(G); Nk = Gm.sum(0)
    pi = P0 / P0.sum(); Atr = Xi / Xi.sum(1, keepdims=True)
    means = (Gm.T @ Xtr) / Nk[:, None]
    covs = np.array([((Xtr - means[k]).T * Gm[:, k]) @ (Xtr - means[k]) / Nk[k] + 1e-4 * np.eye(D) for k in range(K)])
    if it > 2 and abs(hist[-1] - hist[-2]) < 1e-5:
        break
train_time = time.time() - t0

def total_ll(segs):
    return sum(forward_backward(log_gauss(Z, means, covs), pi, Atr)[2] for *_, Z in segs) / sum(len(s[2]) for s in segs)

hmm_test_ll = total_ll(test)
gmm = GaussianMixture(K, covariance_type="full", random_state=42).fit(Xtr)
gmm_test_ll = float(gmm.score(Xte))
# effective "parameters": HMM = GMM + (K*(K-1)) transitions
res = {"K": K, "n_train_seq": len(train), "n_train_obs": int(len(Xtr)), "n_test_seq": len(test), "n_test_obs": int(len(Xte)),
       "iterations": len(hist), "train_seconds": train_time, "ll_history": hist,
       "hmm_test_ll_per_obs": float(hmm_test_ll), "gmm_test_ll_per_obs": gmm_test_ll,
       "pi": pi.tolist(), "A": Atr.tolist(),
       "means_reflectance": (means * sd0 + mu0).tolist(),
       "expected_run_length_px": (1 / (1 - np.diag(Atr))).tolist()}

# ---- Viterbi decode both dates on all rows ----
def decode(stack, valid):
    m = np.full(valid.shape, -1, np.int8)
    for r, c0, Z in segments(stack, valid, rows_all):
        m[r, c0:c0 + len(Z)] = viterbi(log_gauss(Z, means, covs), pi, Atr)
    return m
vit15 = decode(old, vold); vit24 = decode(new, vnew)
both = common & (vit15 >= 0) & (vit24 >= 0)
# i.i.d. GMM labels for comparison (same emissions family, no Markov chain)
Zc = (np.moveaxis(old, 0, -1)[vit15 >= 0] - mu0) / sd0
gl = np.full(vold.shape, -1, np.int8); gl[vit15 >= 0] = gmm.predict(Zc)

def mean_run(m):
    lens = []
    for r in range(m.shape[0]):
        row = m[r][m[r] >= 0]
        if len(row) > 1:
            b = np.flatnonzero(np.diff(row)) ; lens.append(len(row) / (len(b) + 1))
    return float(np.mean(lens))
res["mean_run_viterbi"] = mean_run(vit15); res["mean_run_gmm"] = mean_run(gl)
sel = vit15 >= 0
res["ARI_viterbi_vs_mle"] = float(adjusted_rand_score(mle_old[sel], vit15[sel]))
res["ARI_gmm_vs_mle"] = float(adjusted_rand_score(mle_old[sel], gl[sel]))
res["state_sizes_2015"] = np.bincount(vit15[sel], minlength=K).tolist()
T = np.zeros((K, K), int); np.add.at(T, (vit15[both], vit24[both]), 1)
res["temporal_transition_counts"] = T.tolist()
changed_state = vit15[both] != vit24[both]; reg = change[both] == 1
res["state_change_pct"] = float(100 * changed_state.mean())
res["reg_flag_in_state_change_pct"] = float(100 * (reg & changed_state).sum() / reg.sum())
res["state_change_rate_flagged"] = float(100 * changed_state[reg].mean())
res["state_change_rate_unflagged"] = float(100 * changed_state[~reg].mean())
json.dump(res, open(work / "u4_results.json", "w"), indent=1)

# ---- figures ----
rows, cols = np.where(common)
crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].plot(range(1, len(hist) + 1), hist, "o-"); ax[0].set_xlabel("Baum-Welch iteration"); ax[0].set_ylabel("Log-likelihood per observation")
ax[0].set_title("EM (Baum-Welch) convergence")
im = ax[1].imshow(Atr, cmap="Blues", vmin=0, vmax=1)
for i in range(K):
    for j in range(K):
        ax[1].text(j, i, f"{Atr[i, j]:.3f}", ha="center", va="center", color="w" if Atr[i, j] > .5 else "k")
ax[1].set_xticks(range(K)); ax[1].set_yticks(range(K)); ax[1].set_xlabel("State at t+1"); ax[1].set_ylabel("State at t")
ax[1].set_title("Learned transition matrix A")
Tn = T / T.sum(1, keepdims=True)
for i in range(K):
    for j in range(K):
        ax[2].text(j, i, f"{Tn[i, j]:.2f}", ha="center", va="center", color="w" if Tn[i, j] > .5 else "k")
ax[2].imshow(Tn, cmap="Oranges", vmin=0, vmax=1); ax[2].set_xticks(range(K)); ax[2].set_yticks(range(K))
ax[2].set_xlabel("Viterbi state 2024"); ax[2].set_ylabel("Viterbi state 2015"); ax[2].set_title("2015 -> 2024 state transitions (row-normalised)")
fig.tight_layout(); fig.savefig(figs / "u4_hmm_training.png", dpi=150); plt.close(fig)

cmap = matplotlib.colors.ListedColormap(["#2b83ba", "#d8b365", "#a6611a", "#1a9850"]); cmap.set_bad("#d9d9d9")
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
for a, m, t in ((ax[0], gl, "i.i.d. GMM labels 2015"), (ax[1], vit15, "HMM Viterbi states 2015"), (ax[2], vit24, "HMM Viterbi states 2024")):
    a.imshow(np.ma.masked_less(np.where(common, m, -1), 0)[crop], cmap=cmap, vmin=0, vmax=3, interpolation="nearest"); a.set_title(t, pad=10); a.axis("off")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(figs / "u4_viterbi_maps.png", dpi=150); plt.close(fig)

# one transect illustration
r, c0, Z = max(train, key=lambda s: len(s[2]))
g, _, _ = forward_backward(log_gauss(Z, means, covs), pi, Atr); path = viterbi(log_gauss(Z, means, covs), pi, Atr)
fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
x = np.arange(len(Z)) * 0.15
ax[0].plot(x, Z[:, 2] * sd0[2] + mu0[2], label="B4 red"); ax[0].plot(x, Z[:, 3] * sd0[3] + mu0[3], label="B5 NIR"); ax[0].legend(); ax[0].set_ylabel("Reflectance")
ax[0].set_title(f"Transect along row {r} (2015)")
ax[1].stackplot(x, g.T, colors=["#2b83ba", "#d8b365", "#a6611a", "#1a9850"]); ax[1].set_ylabel("Posterior γ(z)")
ax[2].step(x, path, where="mid", color="k"); ax[2].set_yticks(range(K)); ax[2].set_ylabel("Viterbi state"); ax[2].set_xlabel("Distance along transect (km)")
fig.tight_layout(); fig.savefig(figs / "u4_transect.png", dpi=150); plt.close(fig)
print(json.dumps({k: v for k, v in res.items() if k != "ll_history"}, indent=1))
