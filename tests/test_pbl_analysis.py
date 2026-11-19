"""Check inference mathematically and protect the spatial evaluation boundary."""
import itertools
import unittest
from pathlib import Path
import numpy as np
from pbl_analysis import (sum_product, transition_matrix, spatial_split, weak_labels,
                          rule_thresholds, fit_unsupervised)


class InferenceTests(unittest.TestCase):
    def test_sum_product_matches_enumeration_and_pairwise_marginals(self):
        initial = np.array([.4, .6])
        transition = np.array([[.85, .15], [.2, .8]])
        emissions = np.array([[.7, .2], [.3, .9], [.6, .4]])
        paths = list(itertools.product(range(2), repeat=3))
        weights = []
        for path in paths:
            weight = initial[path[0]] * emissions[0, path[0]]
            for t in range(1, 3):
                weight *= transition[path[t-1], path[t]] * emissions[t, path[t]]
            weights.append(weight)
        weights = np.array(weights)
        exact = np.zeros((3, 2))
        for path, weight in zip(paths, weights / weights.sum()):
            for t, state in enumerate(path):
                exact[t, state] += weight
        posterior, pairs, log_evidence = sum_product(np.log(emissions)[None], initial, transition)
        np.testing.assert_allclose(posterior[0], exact, atol=1e-12)
        np.testing.assert_allclose(np.exp(log_evidence[0]), weights.sum(), atol=1e-12)
        np.testing.assert_allclose(pairs.sum(axis=3), posterior[:, :-1], atol=1e-12)
        np.testing.assert_allclose(pairs.sum(axis=2), posterior[:, 1:], atol=1e-12)

    def test_extreme_log_emissions_remain_normalized(self):
        posterior, pairs, evidence = sum_product(np.full((2, 30, 4), -10000.),
                                                  np.full(4, .25), transition_matrix())
        self.assertTrue(np.isfinite(evidence).all())
        np.testing.assert_allclose(posterior.sum(axis=2), 1, atol=1e-9)
        np.testing.assert_allclose(pairs.sum(axis=(2, 3)), 1, atol=1e-9)

    def test_invalid_probability_model_rejected(self):
        with self.assertRaises(ValueError):
            sum_product(np.zeros((1, 2, 4)), np.ones(4), transition_matrix())


class EvaluationTests(unittest.TestCase):
    def test_split_has_disjoint_areas_and_guard(self):
        rows, cols = np.indices((30, 100)).reshape(2, -1)
        train, test, detail = spatial_split(rows, cols)
        self.assertFalse((train & test).any())
        self.assertGreater(cols[test].min() - cols[train].max(), 8)
        self.assertEqual(detail['training_pixels'] + detail['test_pixels'] + detail['excluded_guard_pixels'], len(cols))

    def test_rule_overlaps_are_excluded(self):
        thresholds = {'ndvi': [-.1, .2, .6], 'ndwi': .3, 'brightness': [.1, .3]}
        # Both water-like and bright-low-vegetation rules match this point.
        features = np.array([[.5, .5, .5, .1, -.2, .4], [.1, .1, .1, .7, .8, -.7]])
        self.assertEqual(weak_labels(features, thresholds).tolist(), [-1, 1])

    def test_unsupervised_fit_does_not_use_holdout_or_after(self):
        rng = np.random.default_rng(31)
        data = {'old': rng.normal(size=(500, 4)), 'new': rng.normal(size=(500, 4)),
                'train': np.arange(500) < 350, 'dates': ['before', 'after']}
        first = fit_unsupervised(data, sample_limit=250)
        changed = {**data, 'old': data['old'].copy(), 'new': data['new'] + 10000}
        changed['old'][~data['train']] += 10000
        second = fit_unsupervised(changed, sample_limit=250)
        np.testing.assert_allclose(first['scaler'].mean_, second['scaler'].mean_)
        np.testing.assert_allclose(first['kmeans'].cluster_centers_, second['kmeans'].cluster_centers_)
        np.testing.assert_allclose(first['pca'].components_, second['pca'].components_)


class CourseInterfaceTests(unittest.TestCase):
    def test_saved_course_page_and_downloads(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        at = AppTest.from_file(str(root / 'app.py')).run(timeout=45)
        self.assertFalse(at.exception)
        labels = [tab.label for tab in at.tabs]
        for label in ['Methods', 'K-means clustering', 'PCA', 'Hidden Markov model',
                      'Sum-product walkthrough', 'CART decision rules']:
            self.assertIn(label, labels)
        self.assertTrue(any('course_results.json' in button.label for button in at.get('download_button')))

    def test_persistence_slider_changes_only_example_without_training(self):
        from streamlit.testing.v1 import AppTest
        from unittest.mock import patch
        root = Path(__file__).resolve().parents[1]
        with patch('pbl_analysis.fit_supervised', side_effect=AssertionError('Must not train on rerun')):
            at = AppTest.from_file(str(root / 'app.py')).run(timeout=45)
            text_before = [item.value for item in at.markdown if 'Model-dependent probability' in item.value]
            at.slider(key='pbl_persistence').set_value(.30).run(timeout=45)
            self.assertFalse(at.exception)
            text_after = [item.value for item in at.markdown if 'Model-dependent probability' in item.value]
            self.assertNotEqual(text_before, text_after)


if __name__ == '__main__':
    unittest.main()
