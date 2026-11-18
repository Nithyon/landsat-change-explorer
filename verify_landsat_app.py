"""Read-only real-data regression check against the preserved terminal report."""
import json
import time
from pathlib import Path
import numpy as np
from landsat_analysis import analyze, discover, replace_red_band

ROOT = Path(__file__).resolve().parent


def verify():
    acquisitions = sorted(discover(str(ROOT / 'colab_test_data')).values(), key=lambda a: a.date)
    before = acquisitions[0]
    started = time.perf_counter()
    result = analyze(before, acquisitions[1], lambda f,m: print(f'{f:.0%} {m}', flush=True))
    report = result['report']
    saved = json.loads((ROOT / 'outputs/terminal_demo_results.json').read_text())
    assert report['preprocessing']['common_valid_analysis_pixels'] == saved['preprocessing']['common_valid_analysis_pixels']
    assert report['mle']['seed_counts'] == saved['mle']['seed_counts']
    individual = report['mle']['class_counts_individual_footprints']
    assert individual['2015-12-13'] == saved['mle']['class_counts']['2015']
    assert individual['2024-07-30'] == saved['mle']['class_counts']['2024']
    for band, detail in report['regression']['bands'].items():
        for key, value in detail.items():
            np.testing.assert_allclose(value, saved['regression']['bands'][band][key], rtol=1e-6, atol=1e-7)
    change = report['change_classification']
    for key in ('threshold', 'possible_change_pixels', 'unchanged_pixels', 'possible_change_percent'):
        np.testing.assert_allclose(change[key], saved['change_classification'][key], rtol=1e-6, atol=1e-7)
    total = report['preprocessing']['common_valid_analysis_pixels']
    assert change['possible_change_pixels'] + change['unchanged_pixels'] == total
    for counts in report['mle']['class_counts'].values():
        assert sum(counts.values()) == total
    assert np.array_equal(result['mle_before'] < 0, result['mle_after'] < 0)
    assert np.array_equal(result['mle_before'] < 0, result['change'] < 0)
    print(f"PASS: matched saved metrics; {total:,} common pixels; {change['possible_change_pixels']:,} possible change ({change['possible_change_percent']:.2f}%); {time.perf_counter()-started:.1f}s", flush=True)
    return result


if __name__ == '__main__':
    verify()
