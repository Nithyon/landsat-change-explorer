"""Generate the corrected notebook's exact figures once for the History view.

Executes the notebook's analysis and plotting cells on the existing real inputs.
The notebook, original rasters and existing terminal reports are preserved.
"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'Corrected_Landsat_Change_Detection_Colab.ipynb'
DESTINATION = ROOT / 'outputs' / 'notebook_history'


def build():
    notebook_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    namespace = {'DATA_DIR': ROOT / 'colab_test_data'}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    figures = []
    try:
        os.chdir(DESTINATION)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code' or index < 6:
                continue
            print(f'Executing notebook cell {index}', flush=True)
            exec(compile(''.join(cell['source']), f'{NOTEBOOK.name}:cell{index}', 'exec'), namespace)
            if index == 10:
                plt.gcf().savefig('natural_colour_comparison.png', dpi=200, bbox_inches='tight')
            plt.close('all')
        old_report = json.loads((ROOT / 'outputs' / 'terminal_demo_results.json').read_text())
        common = int(namespace['total_valid'])
        changed = int(namespace['changed_count'])
        assert common == old_report['preprocessing']['common_valid_analysis_pixels']
        assert changed == old_report['change_classification']['possible_change_pixels']
        np.testing.assert_allclose(namespace['threshold'], old_report['change_classification']['threshold'], rtol=1e-6)
        expected = ['natural_colour_comparison.png', 'mle_classification_comparison.png',
                    'mle_comparison_with_numbers.png', 'regression_change_comparison.png',
                    'landsat_result_maps_clear.png', 'landsat_numerical_summary.png']
        for name in expected:
            path = DESTINATION / name
            if not path.is_file():
                raise RuntimeError(f'Notebook did not generate {name}')
            figures.append(name)
        summary = {
            'notebook': NOTEBOOK.name,
            'notebook_sha256': hashlib.sha256(notebook_bytes).hexdigest(),
            'dates': ['2015-12-13', '2024-07-30'],
            'figures': figures,
            'common_valid_pixels': common,
            'class_counts_common': {
                '2015': namespace['counts_2015'].astype(int).tolist(),
                '2024': namespace['counts_2024'].astype(int).tolist()},
            'possible_change_pixels': changed,
            'possible_change_percent': float(namespace['changed_percentage']),
            'threshold': float(namespace['threshold']),
            'recovered_red_band': str(namespace['FILES']['2015']['SR_B4']),
            'footprint_note': 'Notebook MLE comparison-with-numbers uses each date individually; final numerical summary uses common valid pixels.'}
        (DESTINATION / 'notebook_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Saved all {len(figures)} notebook figures to {DESTINATION}', flush=True)
    finally:
        plt.close('all')
        os.chdir(original_directory)


if __name__ == '__main__':
    build()
