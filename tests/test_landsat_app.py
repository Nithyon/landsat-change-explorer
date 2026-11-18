"""Small synthetic fixtures test input failures and session behavior, not claims."""
import json
import uuid
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import BytesIO
import zipfile

import numpy as np
import rasterio
from rasterio.transform import from_origin
from landsat_analysis import (AnalysisError, REQUIRED, discover, validate_pair,
                              replace_red_band, analyze, CLASS_NAMES)

from landsat_uploads import upload_catalog, staged_pair

class Upload(BytesIO):
    def __init__(self, name, data):
        super().__init__(data)
        self.name = name
        self.size = len(data)
        self.file_id = str(uuid.uuid4())

ROOT = Path(__file__).resolve().parents[1]


def create_acquisition(folder, date, x=0, cloud=False, constant=False):
    rng = np.random.default_rng(18)
    prefix = f'LC08_L2SP_146040_{date}_20240801_02_T1'
    for band in REQUIRED:
        data = (rng.integers(9000, 24000, size=(180,180), dtype=np.uint16)
                if not constant else np.full((180,180), 14000, dtype=np.uint16))
        if band == 'QA_PIXEL':
            data = np.full((180,180), 8 if cloud else 0, dtype=np.uint16)
        with rasterio.open(folder / f'{prefix}_{band}.TIF', 'w', driver='GTiff',
                           height=180,width=180,count=1,dtype='uint16',crs='EPSG:32643',
                           transform=from_origin(x, 6000, 30, 30),nodata=0 if band != 'QA_PIXEL' else None) as dst:
            dst.write(data, 1)
    return next(iter(discover(str(folder)).values()))


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.before_folder = self.root / 'before'
        self.after_folder = self.root / 'after'
        self.before_folder.mkdir()
        self.after_folder.mkdir()
        self.before = create_acquisition(self.before_folder, '20151213')
        self.after = create_acquisition(self.after_folder, '20240730')

    def tearDown(self):
        self.temp.cleanup()

    def test_date_discovery_and_missing_band(self):
        self.assertEqual(self.before.date, '2015-12-13')
        self.before.files['SR_B2'].unlink()
        before = next(iter(discover(str(self.before_folder)).values()))
        with self.assertRaisesRegex(AnalysisError, 'missing SR_B2'):
            validate_pair(before, self.after)

    def test_surface_reflectance_only_product(self):
        for path in self.before_folder.iterdir():
            path.rename(path.with_name(path.name.replace('_L2SP_', '_L2SR_')))
        before = next(iter(discover(str(self.before_folder)).values()))
        self.assertIn('_L2SR_', before.identifier)
        validate_pair(before, self.after)

    def test_duplicate_band(self):
        # Windows filenames are case-insensitive: inject a duplicate directory entry.
        path = self.before.files['SR_B2']
        with patch('landsat_analysis.Path.iterdir', return_value=iter([path,path])):
            with self.assertRaisesRegex(AnalysisError, 'Duplicate'):
                discover(str(self.before_folder))

    def test_unreadable(self):
        self.before.files['SR_B3'].write_bytes(b'not a GeoTIFF')
        with self.assertRaisesRegex(AnalysisError, 'Cannot read'):
            validate_pair(self.before, self.after)

    def test_no_overlap(self):
        self.after = create_acquisition(self.after_folder, '20240730', x=100000)
        with self.assertRaisesRegex(AnalysisError, 'no geographic overlap'):
            validate_pair(self.before, self.after)

    def test_clouds_leave_no_common_pixels(self):
        self.after = create_acquisition(self.after_folder, '20240730', cloud=True)
        with self.assertRaisesRegex(AnalysisError, 'common valid pixels'):
            analyze(self.before, self.after)

    def test_insufficient_seeds(self):
        self.before = create_acquisition(self.before_folder, '20151213', constant=True)
        with self.assertRaisesRegex(AnalysisError, 'Not enough provisional samples'):
            analyze(self.before, self.after)

    def test_reversed_dates_and_wrong_replacement(self):
        with self.assertRaisesRegex(AnalysisError, 'earlier'):
            validate_pair(self.after, self.before)
        with self.assertRaisesRegex(AnalysisError, 'same Landsat acquisition'):
            replace_red_band(self.before, str(self.after.files['SR_B4']))

    def test_grid_mismatch(self):
        with rasterio.open(self.before.files['SR_B2'], 'r+') as src:
            src.transform = from_origin(30,6000,30,30)
        with self.assertRaisesRegex(AnalysisError, 'same source grid'):
            validate_pair(self.before, self.after)

    def test_upload_staging_runs_and_cleans_up(self):
        before = [Upload(p.name,p.read_bytes()) for p in self.before_folder.iterdir()]
        after = [Upload(p.name,p.read_bytes()) for p in self.after_folder.iterdir()]
        with staged_pair(before, self.before.identifier, after, self.after.identifier) as (old,new):
            folder = old.files['SR_B2'].parent
            result = analyze(old,new)
            self.assertGreater(result['report']['preprocessing']['common_valid_analysis_pixels'], 0)
        self.assertFalse(folder.exists())
        self.assertTrue(self.before.files['SR_B2'].exists())
        with self.assertRaisesRegex(AnalysisError, 'missing'):
            with staged_pair(before[:-1], self.before.identifier, after, self.after.identifier):
                pass
        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
            with staged_pair(before, self.before.identifier, after, self.after.identifier) as (old,new):
                folder = old.files['SR_B2'].parent
                raise RuntimeError('simulated failure')
        self.assertFalse(folder.exists())

    def test_upload_duplicates_and_unsafe_names(self):
        upload = Upload(self.before.files['SR_B2'].name,b'dummy')
        with self.assertRaisesRegex(AnalysisError,'Duplicate'):
            upload_catalog([upload,upload])
        with self.assertRaisesRegex(AnalysisError,'individual GeoTIFF'):
            upload_catalog([Upload('../' + upload.name,b'dummy')])

    def test_reproducible_pipeline_and_common_counts(self):
        first, second = analyze(self.before, self.after), analyze(self.before, self.after)
        self.assertEqual(first['report'], second['report'])
        np.testing.assert_array_equal(first['change'], second['change'])
        report = first['report']
        total = report['preprocessing']['common_valid_analysis_pixels']
        for counts in report['mle']['class_counts'].values():
            self.assertEqual(sum(counts.values()), total)
        change = report['change_classification']
        self.assertEqual(change['possible_change_pixels'] + change['unchanged_pixels'], total)
        from app import build_artifacts
        artifacts, bundle = build_artifacts(first)
        self.assertEqual(json.loads(artifacts['analysis_summary.json']), report)
        for name, data in artifacts.items():
            if name.endswith('.png'):
                self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
        with zipfile.ZipFile(BytesIO(bundle)) as archive:
            self.assertEqual(set(archive.namelist()), set(artifacts))


class SessionTests(unittest.TestCase):
    def test_upload_widgets_and_initial_state(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get('file_uploader')), 2)
        self.assertEqual(len(at.text_input), 0)
        self.assertTrue(at.button[0].disabled)

    def test_existing_history_requires_no_analysis(self):
        from streamlit.testing.v1 import AppTest
        with patch('landsat_analysis.analyze') as mocked:
            at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
            self.assertFalse(at.exception)
            self.assertIn('History', [tab.label for tab in at.tabs])
            for label in ('Overview', 'Satellite images', 'MLE classification', 'Regression'):
                self.assertIn(label, [tab.label for tab in at.tabs])
            counts = at.dataframe[0].value
            self.assertEqual(int(counts.loc['Water-like', '2015']), 118461)
            self.assertEqual(int(counts.loc['Vegetation-like', '2024']), 587891)
            self.assertIn('Seed pixels', at.dataframe[2].value.columns)
            self.assertIn('R-squared (sample)', at.dataframe[3].value.columns)
            self.assertEqual(at.metric[0].value, '632,388')
            self.assertEqual(at.metric[2].value, '5.24%')
            self.assertEqual(len(at.get('download_button')), 9)
            expected = {'natural_colour_comparison.png', 'mle_classification_comparison.png',
                        'mle_comparison_with_numbers.png', 'regression_change_comparison.png',
                        'landsat_result_maps_clear.png', 'landsat_numerical_summary.png'}
            self.assertEqual({p.name for p in (ROOT / 'outputs/notebook_history').glob('*.png')}, expected)
            captions = '\n'.join(image.proto.imgs[0].caption for image in at.get('image'))
            self.assertIn('Four regression plots', captions)
            self.assertIn('Class distribution and possible change', captions)
            mocked.assert_not_called()

    def test_run_persistence_stale_clear_and_error(self):
        from streamlit.testing.v1 import AppTest
        # Synthetic result is used only to verify interface state, not real-data metrics.
        report = {'dataset': {'dates': ['2015-12-13','2024-07-30']},
                  'preprocessing': {'common_valid_analysis_pixels': 10},
                  'change_classification': {'possible_change_pixels': 2, 'unchanged_pixels': 8,
                                            'possible_change_percent': 20., 'threshold': 3.},
                  'mle': {'class_counts': {d: dict(zip(CLASS_NAMES,[2,3,3,2])) for d in ['2015-12-13','2024-07-30']}},
                  'regression': {'bands': {}}, 'settings': {}, 'limitations': ['Synthetic UI test fixture']}
        before_uploads = [Upload(f'LC08_L2SP_146040_20151213_20200908_02_T1_{band}.TIF', b'UI fixture') for band in REQUIRED]
        uploads = {'before_uploads': before_uploads,
                   'after_uploads': [Upload(f'LC08_L2SP_146040_20240730_20240801_02_T1_{band}.TIF', b'UI fixture') for band in REQUIRED]}
        with patch('streamlit.file_uploader', side_effect=lambda *args, **kwargs: uploads[kwargs['key']]), patch('landsat_analysis.analyze', return_value={'report': report}) as mocked:
            # The app renders real PNGs from this small, explicitly synthetic UI fixture.
            sample = {'report': report, 'rgb_before': np.ones((5,5,4)), 'rgb_after': np.ones((5,5,4)),
                      'mle_before': np.zeros((5,5),dtype=int), 'mle_after': np.zeros((5,5),dtype=int),
                      'change': np.zeros((5,5),dtype=int), 'score': np.ones((5,5)),
                      'scatter_before': np.ones((10,4)), 'scatter_after': np.ones((10,4))}
            report['regression']['bands'] = {b: {'slope': 1., 'intercept': 0., 'r2_sample': 1., 'rmse_all': 0.} for b in REQUIRED[:-1]}
            mocked.return_value = sample
            at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
            self.assertFalse(at.exception)
            at.button[0].click().run(timeout=30)
            self.assertFalse(at.exception)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(len(at.tabs), 9)
            self.assertEqual(at.metric[2].value, '20.00%')
            at.run(timeout=30)
            self.assertEqual(mocked.call_count, 1)
            uploads['before_uploads'] = []
            at.run(timeout=30)
            self.assertNotIn('result_report', at.session_state)
            self.assertEqual(len(at.metric), 3)
            uploads['before_uploads'] = before_uploads
            at.run(timeout=30)
            mocked.side_effect = AnalysisError('Fixture: unreadable GeoTIFF')
            at.button[0].click().run(timeout=30)
            self.assertIn('unreadable', at.error[0].value)
            self.assertNotIn('result_report', at.session_state)
