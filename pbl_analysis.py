"""Course methods on real Landsat inputs; no independent ground truth is implied."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score
from landsat_analysis import (discover, validate_pair, reference_grid, load_acquisition,
                              BANDS, CLASS_NAMES)

FEATURES = list(BANDS) + ['NDVI', 'NDWI']
SEED = 42


def transition_matrix(states=4, persistence=0.9):
    if states < 2 or not 0 < persistence < 1:
        raise ValueError('Use at least two states and a persistence strictly between 0 and 1.')
    matrix = np.full((states, states), (1 - persistence) / (states - 1))
    np.fill_diagonal(matrix, persistence)
    return matrix


def sum_product(log_emissions, initial, transition):
    """Exact log-space forward/backward sum-product on batched HMM chains.

    log_emissions: [pixel, date, state]. Return state marginals, adjacent-state
    marginals, and log evidence. This sums paths; it is not Viterbi decoding.
    """
    emissions = np.asarray(log_emissions, dtype=float)
    initial, transition = np.asarray(initial, float), np.asarray(transition, float)
    if emissions.ndim != 3 or emissions.shape[1] < 1:
        raise ValueError('Emissions must have shape [pixel, date, state].')
    n, steps, states = emissions.shape
    if initial.shape != (states,) or transition.shape != (states, states):
        raise ValueError('Initial and transition shapes must match the state count.')
    if (not np.isfinite(emissions).all() or not np.isfinite(initial).all()
            or not np.isfinite(transition).all() or (initial <= 0).any()
            or (transition <= 0).any() or not np.isclose(initial.sum(), 1)
            or not np.allclose(transition.sum(axis=1), 1)):
        raise ValueError('Use finite emissions and strictly positive normalized probabilities.')
    log_initial, log_transition = np.log(initial), np.log(transition)
    forward = np.empty_like(emissions)
    backward = np.zeros_like(emissions)
    forward[:, 0] = log_initial + emissions[:, 0]
    for t in range(1, steps):
        forward[:, t] = emissions[:, t] + logsumexp(
            forward[:, t - 1, :, None] + log_transition, axis=1)
    evidence = logsumexp(forward[:, -1], axis=1)
    for t in range(steps - 2, -1, -1):
        backward[:, t] = logsumexp(log_transition + emissions[:, t + 1, None, :]
                                   + backward[:, t + 1, None, :], axis=2)
    posterior = np.exp(forward + backward - evidence[:, None, None])
    pairwise = np.empty((n, steps - 1, states, states))
    for t in range(steps - 1):
        pairwise[:, t] = np.exp(forward[:, t, :, None] + log_transition
                               + emissions[:, t + 1, None, :]
                               + backward[:, t + 1, None, :] - evidence[:, None, None])
    return posterior, pairwise, evidence


def spatial_split(rows, cols, buffer=8):
    """Eastern quarter held out, with an eight-analysis-pixel guard strip."""
    cutoff = int(np.quantile(cols, 0.75))
    train, test = cols < cutoff - buffer, cols >= cutoff
    if train.sum() < 100 or test.sum() < 100:
        raise ValueError('Not enough common valid pixels for a buffered spatial holdout.')
    return train, test, {'test_column_start': cutoff, 'buffer_pixels': buffer,
                         'training_pixels': int(train.sum()), 'test_pixels': int(test.sum()),
                         'excluded_guard_pixels': int((~(train | test)).sum()),
                         'method': 'Eastern quarter spatial holdout; eight-column guard strip'}


def spectral_features(pixels):
    blue, green, red, nir = np.asarray(pixels).T
    def ratio(a, b):
        denominator = a + b
        result = np.divide(a - b, denominator, out=np.zeros_like(a),
                           where=np.abs(denominator) > 1e-6)
        return np.clip(result, -5, 5)
    return np.column_stack([pixels, ratio(nir, red), ratio(green, nir)])


def rule_thresholds(train_features):
    return {'ndvi': np.quantile(train_features[:, 4], [.30, .55, .80]).tolist(),
            'ndwi': float(np.quantile(train_features[:, 5], .90)),
            'brightness': np.quantile(train_features[:, :3].mean(axis=1), [.45, .75]).tolist()}


def weak_labels(features, thresholds):
    """Only uniquely matched spectral rules receive provisional labels."""
    ndvi, ndwi = features[:, 4], features[:, 5]
    brightness = features[:, :3].mean(axis=1)
    q, b = thresholds['ndvi'], thresholds['brightness']
    rules = np.column_stack([
        (ndwi >= thresholds['ndwi']) & (ndvi <= q[0]),
        ndvi >= q[2],
        (brightness >= b[1]) & (ndvi <= q[1]),
        (brightness >= b[0]) & (ndvi > q[0]) & (ndvi < q[2])])
    return np.where(rules.sum(axis=1) == 1, rules.argmax(axis=1), -1)


def load_course_data(data_dir):
    acquisitions = sorted(discover(str(data_dir)).values(), key=lambda a: a.date)
    if len(acquisitions) != 2:
        raise ValueError('The course demo requires exactly two acquisitions.')
    before, after = acquisitions
    validate_pair(before, after)
    grid = reference_grid(before)
    old, old_valid = load_acquisition(before, grid)
    new, new_valid = load_acquisition(after, grid)
    common = old_valid & new_valid
    rows, cols = np.where(common)
    train, test, split = spatial_split(rows, cols)
    manifest = []
    for acquisition in acquisitions:
        for band, path in sorted(acquisition.files.items()):
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': checksum})
    return {'old': np.moveaxis(old, 0, -1)[common],
            'new': np.moveaxis(new, 0, -1)[common], 'common': common,
            'rows': rows, 'cols': cols, 'train': train, 'test': test,
            'split': split, 'dates': [before.date, after.date], 'input_manifest': manifest,
            'grid': {'shape': list(common.shape), 'crs': str(grid['crs']),
                     'resolution': [abs(grid['transform'].a), abs(grid['transform'].e)]}}


def fit_unsupervised(data, k=4, sample_limit=20000):
    rng = np.random.default_rng(SEED)
    indices = np.flatnonzero(data['train'])
    fit_indices = rng.choice(indices, min(sample_limit, len(indices)), replace=False)
    scaler = StandardScaler().fit(data['old'][fit_indices])
    training = scaler.transform(data['old'][fit_indices])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(training)
    pca = PCA(n_components=4, svd_solver='full').fit(training)
    transformed = [scaler.transform(data[key]) for key in ('old', 'new')]
    clusters = [kmeans.predict(x) for x in transformed]
    means, variance = [], []
    for state in range(k):
        points = training[kmeans.labels_ == state]
        means.append(points.mean(axis=0))
        variance.append(np.maximum(points.var(axis=0), .02))
    sample = rng.choice(len(data['old']), min(3000, len(data['old'])), replace=False)
    silhouette = float(silhouette_score(training, kmeans.labels_, sample_size=min(1500, len(training)),
                                        random_state=SEED))
    report = {'k': k, 'training_sample_pixels': len(fit_indices), 'n_init': 10,
              'inertia': float(kmeans.inertia_), 'silhouette_sample': silhouette,
              'centroids_reflectance': scaler.inverse_transform(kmeans.cluster_centers_).tolist(),
              'counts': {date: np.bincount(labels, minlength=k).tolist()
                         for date, labels in zip(data['dates'], clusters)},
              'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
              'pca_loadings': pca.components_.tolist(),
              'note': 'Scaler, PCA and K-means fit only Before training-area samples; cluster IDs are arbitrary.'}
    return {'report': report, 'scaler': scaler, 'pca': pca, 'kmeans': kmeans,
            'clusters': clusters, 'scaled': transformed, 'sample': sample,
            'means': np.asarray(means), 'variance': np.asarray(variance),
            'initial': (np.bincount(kmeans.labels_, minlength=k) + 1) / (len(training) + k)}


def gaussian_emissions(pixels, means, variance):
    return -.5 * (np.log(2 * np.pi * variance).sum(axis=1)
                  + ((pixels[:, None, :] - means) ** 2 / variance).sum(axis=2))


def run_temporal(data, unsupervised, persistence=.9):
    k = unsupervised['report']['k']
    transition = transition_matrix(k, persistence)
    n = len(data['old'])
    labels = np.empty((n, 2), dtype=np.int8)
    probabilities = np.empty(n, dtype=np.float32)
    expected = np.zeros((k, k))
    for start in range(0, n, 8192):
        stop = min(n, start + 8192)
        emissions = np.stack([gaussian_emissions(x[start:stop], unsupervised['means'],
                              unsupervised['variance']) for x in unsupervised['scaled']], axis=1)
        posterior, pairs, _ = sum_product(emissions, unsupervised['initial'], transition)
        labels[start:stop] = posterior.argmax(axis=2)
        probabilities[start:stop] = np.clip(1 - np.trace(pairs[:, 0], axis1=1, axis2=2), 0, 1)
        expected += pairs[:, 0].sum(axis=0)
    # Demonstration examples across the distribution, using actual observed pixels.
    order = np.argsort(probabilities)
    examples = []
    for quantile in (0.05, .25, .5, .75, .95):
        index = int(order[int(quantile * (n - 1))])
        emissions = np.stack([gaussian_emissions(x[index:index + 1], unsupervised['means'],
                              unsupervised['variance'])[0] for x in unsupervised['scaled']])
        examples.append({'row': int(data['rows'][index]), 'column': int(data['cols'][index]),
                         'before_reflectance': data['old'][index].tolist(),
                         'after_reflectance': data['new'][index].tolist(),
                         'log_emissions': emissions.tolist(),
                         'state_change_posterior': float(probabilities[index])})
    report = {'states': k, 'dates': data['dates'], 'persistence': persistence,
              'initial': unsupervised['initial'].tolist(), 'transition': transition.tolist(),
              'emission_means_standardized': unsupervised['means'].tolist(),
              'emission_variances_standardized': unsupervised['variance'].tolist(),
              'expected_transition_counts': expected.tolist(),
              'mean_state_change_posterior': float(probabilities.mean()),
              'examples': examples,
              'note': 'Two observations per pixel. Transition persistence is assumed, not learned. '
                      'Gaussian emissions fitted from Before K-means groups. '
                      'Posterior is model-dependent and is not a calibrated probability of real land-cover change.'}
    return {'report': report, 'labels': labels, 'probabilities': probabilities}


def fit_supervised(data, sample_per_class=4000, test_limit=12000):
    old = spectral_features(data['old'])
    new = spectral_features(data['new'])
    thresholds = rule_thresholds(old[data['train']])
    target = weak_labels(old, thresholds)
    rng = np.random.default_rng(SEED)
    training = []
    for c in range(4):
        indices = np.flatnonzero(data['train'] & (target == c))
        if len(indices) < 10:
            raise ValueError(f'Not enough unambiguous training rule labels for {CLASS_NAMES[c]}.')
        training.extend(rng.choice(indices, min(sample_per_class, len(indices)), replace=False))
    training = np.asarray(training)
    testing = np.flatnonzero(data['test'] & (target >= 0))
    if len(testing) < 10:
        raise ValueError('Not enough unambiguous rule labels in the spatial holdout.')
    testing = rng.choice(testing, min(test_limit, len(testing)), replace=False)
    models = {
        'CART': DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=25, random_state=SEED),
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
                                      n_estimators=80, learning_rate=.5, random_state=SEED),
        'Random forest': RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=10,
                                                n_jobs=2, random_state=SEED)}
    metrics, predictions, probabilities = [], {}, []
    for name, model in models.items():
        model.fit(old[training], target[training])
        probability = model.predict_proba(old[testing])
        probabilities.append(probability)
        predictions[name] = model.predict(new)
        prediction = model.classes_[probability.argmax(axis=1)]
        metrics.append(metric_row(name, target[testing], prediction))
    vote = np.mean(probabilities, axis=0)
    metrics.append(metric_row('Soft voting', target[testing], vote.argmax(axis=1)))
    # Chunk predictions to bound memory while building the second-date ensemble map.
    voting_predictions = np.empty(len(new), dtype=np.int8)
    for start in range(0, len(new), 20000):
        block = new[start:start + 20000]
        voting_predictions[start:start + len(block)] = np.mean(
            [model.predict_proba(block) for model in models.values()], axis=0).argmax(axis=1)
    predictions['Soft voting'] = voting_predictions
    stages = [float(accuracy_score(target[testing], p))
              for p in models['AdaBoost'].staged_predict(old[testing])]
    class_counts = {area: np.bincount(target[index], minlength=4).tolist()
                    for area, index in [('training', training), ('holdout', testing)]}
    report = {'features': FEATURES, 'class_names': list(CLASS_NAMES),
              'weak_label_thresholds_training_only': thresholds,
              'training_sample_pixels': len(training), 'holdout_sample_pixels': len(testing),
              'sample_class_counts': class_counts, 'split': data['split'], 'metrics': metrics,
              'cart_feature_importances': models['CART'].feature_importances_.tolist(),
              'forest_feature_importances': models['Random forest'].feature_importances_.tolist(),
              'adaboost_holdout_agreement_by_round': stages,
              'cart_depth': int(models['CART'].get_depth()), 'cart_leaves': int(models['CART'].get_n_leaves()),
              'settings': {'cart_max_depth': 6, 'adaboost_rounds': 80, 'adaboost_learning_rate': .5,
                           'forest_trees': 80, 'forest_max_depth': 10,
                           'voting': 'Equal average of the three fitted models predict_proba outputs'},
              'note': 'Scores are agreement with provisional spectral rules, not independent land-cover accuracy. '
                      'Only unique rule matches are evaluated. Thresholds and models fit Before training area only. '
                      'Eastern Before holdout is evaluation; After maps are unvalidated transfer predictions. '
                      'Do not choose hyperparameters using the displayed holdout round curve.'}
    return {'report': report, 'models': models, 'predictions': predictions, 'training_indices': training,
            'holdout_indices': testing, 'targets': target}


def metric_row(name, target, prediction):
    recalls = []
    for c in np.unique(target):
        recalls.append(float((prediction[target == c] == c).mean()))
    return {'model': name, 'rule_agreement': float(accuracy_score(target, prediction)),
            'macro_f1': float(f1_score(target, prediction, labels=np.arange(4), average='macro', zero_division=0)),
            'balanced_rule_agreement': float(np.mean(recalls)),
            'confusion_matrix': confusion_matrix(target, prediction, labels=np.arange(4)).tolist()}
