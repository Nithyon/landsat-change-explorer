"""Rebuild saved PBL 2/3 figures and metrics from the intact local imagery."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.tree import plot_tree, export_text
from pbl_analysis import (load_course_data, fit_unsupervised, run_temporal, fit_supervised,
                          FEATURES, BANDS, CLASS_NAMES, SEED)

ROOT = Path(__file__).resolve().parent
COLORS = ['#2563eb', '#16a34a', '#eab308', '#9333ea']


def save_course_results(data, unsupervised, temporal, supervised, output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.savefig(destination / name, dpi=145, bbox_inches='tight', facecolor='white')
    def grid(values, missing=-1):
        image = np.full(data['common'].shape, missing, dtype=np.asarray(values).dtype)
        image[data['common']] = values
        rows, cols = data['rows'], data['cols']
        return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    def maps(arrays, titles, name, classes=True):
        fig = Figure(figsize=(5 * len(arrays), 4.8), layout='constrained')
        axes = fig.subplots(1, len(arrays), squeeze=False)[0]
        cmap = ListedColormap(COLORS) if classes else 'magma'
        if classes:
            cmap.set_bad('#e5e7eb')
        for ax, values, title in zip(axes, arrays, titles):
            image = grid(values, -1 if classes else np.nan)
            image = np.ma.masked_less(image, 0) if classes else np.ma.masked_invalid(image)
            im = ax.imshow(image, cmap=cmap, vmin=0, vmax=3 if classes else 1, interpolation='nearest')
            ax.set(title=title, xticks=[], yticks=[])
        if classes:
            fig.legend(handles=[Patch(color=color, label=f'State/cluster {i}') for i, color in enumerate(COLORS)],
                       loc='outside lower center', ncol=4, frameon=False)
        else:
            fig.colorbar(im, ax=axes, shrink=.7, label='Model-dependent posterior')
        save(fig, name)
    maps(unsupervised['clusters'], [f'K-means | {d}' for d in data['dates']], 'kmeans_maps.png')
    maps([temporal['labels'][:, i] for i in range(2)],
         [f'HMM marginal state | {d}' for d in data['dates']], 'hmm_states.png')
    maps([temporal['probabilities']], ['HMM posterior of a different latent state'], 'hmm_change.png', False)
    fig = Figure(figsize=(11, 4.3), layout='constrained')
    left, right = fig.subplots(1, 2)
    variance = np.array(unsupervised['report']['pca_explained_variance_ratio'])
    left.bar(range(1, 5), variance * 100, label='Individual component')
    left.plot(range(1, 5), variance.cumsum() * 100, 'o-', color='#dc2626', label='Cumulative')
    left.set(xlabel='Principal component', ylabel='Variance explained (%)', xticks=range(1, 5), ylim=(0, 105))
    left.legend()
    sample = unsupervised['sample']
    projected = unsupervised['pca'].transform(unsupervised['scaled'][0][sample])
    right.scatter(projected[:, 0], projected[:, 1], c=unsupervised['clusters'][0][sample],
                  cmap=ListedColormap(COLORS), vmin=0, vmax=3, s=4, alpha=.4)
    right.set(xlabel='PC1', ylabel='PC2', title='Before pixels projected into two dimensions')
    save(fig, 'pca_exploration.png')
    fig = Figure(figsize=(10, 4.4), layout='constrained')
    left, right = fig.subplots(1, 2)
    im = left.imshow(temporal['report']['transition'], vmin=0, vmax=1, cmap='Blues')
    left.set(title='Assumed transition prior', xlabel='After state', ylabel='Before state', xticks=range(4), yticks=range(4))
    for i in range(4):
        for j in range(4):
            left.text(j, i, f'{temporal["report"]["transition"][i][j]:.2f}', ha='center', va='center',
                      color='white' if i == j else 'black')
    right.imshow(temporal['report']['expected_transition_counts'], cmap='Blues')
    right.set(title='Posterior expected transition counts', xlabel='After state', ylabel='Before state', xticks=range(4), yticks=range(4))
    counts = np.array(temporal['report']['expected_transition_counts'])
    for i in range(4):
        for j in range(4):
            right.text(j, i, f'{counts[i,j]:,.0f}', ha='center', va='center',
                       color='white' if counts[i,j] > counts.max() * .55 else 'black', fontsize=8)
    fig.colorbar(im, ax=left, shrink=.7)
    save(fig, 'sum_product_transitions.png')
    from cart_highlight import export_highlighted_cart
    export_highlighted_cart(supervised['models']['CART'], destination, FEATURES, list(CLASS_NAMES))
    names = list(supervised['predictions'])
    fig = Figure(figsize=(12, 10), layout='constrained')
    cmap = ListedColormap(['#2b83ba', '#1a9850', '#d8b365', '#a6611a'])
    cmap.set_bad('#e5e7eb')
    for ax, name in zip(fig.subplots(2, 2).flat, names):
        ax.imshow(np.ma.masked_less(grid(supervised['predictions'][name]), 0), cmap=cmap, vmin=0, vmax=3)
        ax.set(title=f'{name} | After transfer prediction', xticks=[], yticks=[])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(cmap.colors, CLASS_NAMES)],
               loc='outside lower center', ncol=2, frameon=False)
    save(fig, 'pbl3_predictions.png')
    fig = Figure(figsize=(11, 4), layout='constrained')
    left, right = fig.subplots(1, 2)
    metrics = supervised['report']['metrics']
    x = np.arange(len(metrics))
    left.bar(x - .18, [m['rule_agreement'] for m in metrics], .36, label='Rule agreement')
    left.bar(x + .18, [m['macro_f1'] for m in metrics], .36, label='Macro F1 against rules')
    left.set(xticks=x, xticklabels=[m['model'] for m in metrics], ylim=(0, 1.05), title='Spatial holdout | provisional labels')
    left.tick_params(axis='x', rotation=15)
    left.legend(fontsize=8)
    right.plot(np.arange(1, 1 + len(supervised['report']['adaboost_holdout_agreement_by_round'])),
               supervised['report']['adaboost_holdout_agreement_by_round'], color='#2563eb')
    right.set(xlabel='Boosting round', ylabel='Holdout rule agreement', title='AdaBoost learning progression', ylim=(0, 1.05))
    save(fig, 'model_comparison.png')
    fig = Figure(figsize=(13, 10), layout='constrained')
    for ax, metric in zip(fig.subplots(2, 2).flat, metrics):
        matrix = np.asarray(metric['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues')
        ax.set(title=metric['model'], xlabel='Predicted rule class', ylabel='Reference rule class',
               xticks=range(4), yticks=range(4), xticklabels=range(4), yticklabels=range(4))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                        color='white' if matrix[i, j] > matrix.max() * .5 else 'black')
    save(fig, 'confusion_matrices.png')
    fig = Figure(figsize=(15, 6), layout='constrained')
    FigureCanvasAgg(fig)
    plot_tree(supervised['models']['CART'], max_depth=2, feature_names=FEATURES,
              class_names=list(CLASS_NAMES), filled=True, rounded=True, fontsize=7, ax=fig.subplots())
    fig.suptitle('CART | first three levels (complete fitted tree is deeper)')
    save(fig, 'cart_tree.png')
    fig = Figure(figsize=(8, 3.7), layout='constrained')
    FigureCanvasAgg(fig)
    plot_tree(supervised['models']['CART'], max_depth=1, feature_names=FEATURES,
              class_names=['Water-like', 'Vegetation', 'Bright low veg', 'Bare/other'],
              filled=True, rounded=True, fontsize=11, ax=fig.subplots())
    save(fig, 'cart_tree_report.png')
    fig = Figure(figsize=(10, 4), layout='constrained')
    ax = fig.subplots()
    y = np.arange(len(FEATURES))
    ax.barh(y - .18, supervised['report']['cart_feature_importances'], .36, label='CART')
    ax.barh(y + .18, supervised['report']['forest_feature_importances'], .36, label='Random forest')
    ax.set(yticks=y, yticklabels=FEATURES, xlabel='Impurity-based feature importance', title='Rule-reproduction models')
    ax.legend()
    save(fig, 'feature_importance.png')
    split_values = np.where(data['train'], 0, np.where(data['test'], 2, 1))
    fig = Figure(figsize=(6, 5), layout='constrained')
    ax = fig.subplots()
    cmap = ListedColormap(['#2563eb', '#fbbf24', '#dc2626']); cmap.set_bad('#e5e7eb')
    ax.imshow(np.ma.masked_less(grid(split_values), 0), cmap=cmap, vmin=0, vmax=2)
    ax.set(title='Before spatial split | no random pixel mixing', xticks=[], yticks=[])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(cmap.colors, ['Training area', 'Guard strip', 'Eastern holdout'])],
               loc='outside lower center', ncol=3, frameon=False)
    save(fig, 'spatial_split.png')
    report = {'version': 1, 'seed': SEED, 'input_manifest': data['input_manifest'], 'dataset': {'dates': data['dates'],
              'common_valid_pixels': int(data['common'].sum()), **data['grid']},
              'pbl2': {'unsupervised': unsupervised['report'], 'hmm_sum_product': temporal['report']},
              'pbl3': supervised['report'],
              'environment': {name: importlib.metadata.version(name) for name in
                              ['numpy', 'scipy', 'scikit-learn', 'rasterio', 'matplotlib', 'streamlit']}}
    (destination / 'course_results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (destination / 'model_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'rule_agreement', 'macro_f1', 'balanced_rule_agreement'])
        writer.writeheader()
        writer.writerows({k: m[k] for k in writer.fieldnames} for m in metrics)
    (destination / 'cart_rules.txt').write_text(export_text(supervised['models']['CART'], feature_names=FEATURES), encoding='utf-8')
    return report


def build(data_dir=ROOT / 'colab_test_data', output_dir=ROOT / 'outputs' / 'pbl'):
    print('Loading and aligning the original Landsat data', flush=True)
    data = load_course_data(data_dir)
    print('PBL 2: fitting K-means and PCA on Before training-area pixels', flush=True)
    unsupervised = fit_unsupervised(data)
    print('PBL 2: exact two-date HMM sum-product inference', flush=True)
    temporal = run_temporal(data, unsupervised)
    print('PBL 3: CART, AdaBoost, random forest and soft voting', flush=True)
    supervised = fit_supervised(data)
    print('Saving figures and numerical results', flush=True)
    report = save_course_results(data, unsupervised, temporal, supervised, output_dir)
    report['input_manifest'] = []
    for path in sorted(Path(data_dir).glob('*.TIF')):
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['input_manifest'].append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': checksum})
    (Path(output_dir) / 'course_results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps([{k: v for k, v in m.items() if k != 'confusion_matrix'} for m in report['pbl3']['metrics']], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'colab_test_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'pbl')
    args = parser.parse_args()
    build(args.data_dir, args.output_dir)
