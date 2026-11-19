"""Minimal local browser interface for real Landsat acquisitions."""
from pathlib import Path
from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from landsat_analysis import (AnalysisError, BANDS, CLASS_NAMES, CLASS_COLORS,
                              REQUIRED, analyze)
from landsat_uploads import upload_catalog, upload_signature, staged_pair
from pbl_ui import show_course_demo

ROOT = Path(__file__).resolve().parent


def png(fig):
    out = BytesIO()
    fig.savefig(out, format="png", dpi=130, facecolor="white", bbox_inches="tight")
    return out.getvalue()


def map_pair(arrays, titles, kind):
    fig = Figure(figsize=(12, 5.8), layout="constrained")
    axes = fig.subplots(1, len(arrays), squeeze=False)[0]
    if kind == 'class':
        cmap = ListedColormap(CLASS_COLORS)
        cmap.set_bad('#e5e7eb')
    elif kind == 'change':
        cmap = ListedColormap(['#252a32', '#e34c4c'])
        cmap.set_bad('#e5e7eb')
    for ax, image, title in zip(axes, arrays, titles):
        if kind == 'rgb':
            ax.set_facecolor('#e5e7eb')
            ax.imshow(image)
        elif kind in ('class', 'change'):
            ax.imshow(np.ma.masked_less(image, 0), cmap=cmap, vmin=0,
                      vmax=3 if kind == 'class' else 1, interpolation='nearest')
        else:
            im = ax.imshow(image, cmap='magma', vmin=0, vmax=max(float(np.nanpercentile(image, 98)), 1e-6))
            fig.colorbar(im, ax=ax, fraction=.035, label='Standardized residual score')
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if kind in ('class', 'change'):
        colors = list(CLASS_COLORS) if kind == 'class' else ['#252a32', '#e34c4c']
        names = list(CLASS_NAMES) if kind == 'class' else ['Below threshold', 'Possible change']
        fig.legend(handles=[Patch(color=c, label=n) for c,n in zip(colors+['#e5e7eb'], names+['Masked'])],
                   loc='outside lower center', ncol=3, frameon=False)
    return png(fig)


def build_artifacts(result):
    dates = result['report']['dataset']['dates']
    artifacts = {
        'satellite_images.png': map_pair([result['rgb_before'], result['rgb_after']], dates, 'rgb'),
        'classification_maps.png': map_pair([result['mle_before'], result['mle_after']], dates, 'class'),
        'possible_change.png': map_pair([result['change']], ['Possible spectral change'], 'change'),
        'residual_score.png': map_pair([result['score']], ['Regression residual score'], 'score')}
    fig = Figure(figsize=(12, 3.5), layout='constrained')
    for index, (ax, band) in enumerate(zip(fig.subplots(1, 4), BANDS)):
        x, y = result['scatter_before'][:, index], result['scatter_after'][:, index]
        detail = result['report']['regression']['bands'][band]
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, y, s=4, alpha=.15, color='#3977a8')
        ax.plot(line_x, detail['slope'] * line_x + detail['intercept'], color='#e34c4c')
        ax.set(title=band, xlabel='Before reflectance', ylabel='After reflectance')
        ax.grid(alpha=.15)
    artifacts['regression_plots.png'] = png(fig)
    fig = Figure(figsize=(10, 4), layout='constrained')
    ax = fig.subplots()
    counts = result['report']['mle']['class_counts']
    total = result['report']['preprocessing']['common_valid_analysis_pixels']
    x = np.arange(4)
    for date, offset, color in zip(dates, [-.18,.18], ['#3977a8','#e17c3a']):
        ax.bar(x+offset, [counts[date][n]/total*100 for n in CLASS_NAMES], .36, label=date, color=color)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel('Percentage of common valid pixels')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.15)
    artifacts['class_distribution.png'] = png(fig)
    artifacts['analysis_summary.json'] = json.dumps(result['report'], indent=2, allow_nan=False).encode()
    bundle = BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in artifacts.items():
            archive.writestr(name, data)
    return artifacts, bundle.getvalue()


def clear_results():
    for key in ('result_report', 'artifacts', 'bundle', 'result_signature'):
        st.session_state.pop(key, None)


def select_acquisition(uploads, label):
    try:
        acquisitions = upload_catalog(uploads)
    except AnalysisError as exc:
        st.error(str(exc))
        return None
    if not acquisitions:
        st.caption('Upload the five required bands for this date.')
        return None
    keys = sorted(acquisitions, key=lambda k: (acquisitions[k].date, k))
    index = 0 if label == 'Before' else len(keys)-1
    key = st.selectbox(f'{label} acquisition', keys, index=index,
                      format_func=lambda k: f"{acquisitions[k].date} | {k}",
                      key=f'{label.lower()}_acquisition', on_change=clear_results)
    acquisition = acquisitions[key]
    missing = set(REQUIRED) - set(acquisition.files)
    st.caption(f"{len(acquisition.files)} / 5 required bands uploaded")
    if missing:
        st.caption('Still needed: ' + ', '.join(sorted(missing)))
        return None
    return acquisition



def show_history():
    report_path = ROOT / 'outputs' / 'terminal_demo_results.json'
    notebook_folder = ROOT / 'outputs' / 'notebook_history'
    if not report_path.is_file():
        st.caption('No saved analysis available.')
        return
    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        total = report['preprocessing']['common_valid_analysis_pixels']
        change = report['change_classification']
    except (OSError, ValueError, KeyError):
        st.error('Could not open the saved analysis.')
        return
    files = sorted(notebook_folder.glob('*.png'))
    bundle = BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr(path.name, path.read_bytes())
        summary_path = notebook_folder / 'notebook_summary.json'
        if summary_path.is_file():
            archive.writestr(summary_path.name, summary_path.read_bytes())

    heading, action = st.columns([3, 1], vertical_alignment='center')
    with heading:
        st.subheader('Noida | 2015 to 2024')
        st.caption('13 Dec 2015  /  30 Jul 2024')
    with action:
        st.download_button('Download results', bundle.getvalue(), 'saved_notebook_figures.zip',
                           'application/zip', key='history_notebook_zip', icon=':material/download:', width='stretch')
    c1, c2, c3 = st.columns(3)
    c1.metric('Comparable pixels', f'{total:,}', border=True, icon=':material/grid_on:')
    c2.metric('Flagged pixels', f"{change['possible_change_pixels']:,}", border=True, icon=':material/location_searching:')
    c3.metric('Possible change', f"{change['possible_change_percent']:.2f}%", border=True, icon=':material/compare_arrows:')

    def figure(filename, caption):
        path = notebook_folder / filename
        if path.is_file():
            with st.container(border=True):
                st.image(path.read_bytes(), caption=caption, width='stretch')
        else:
            st.caption('Figure unavailable: ' + filename)

    overview, satellite, mle, regression = st.tabs([
        'Overview', 'Satellite images', 'MLE classification', 'Regression'])
    with overview:
        figure('landsat_numerical_summary.png', 'Class distribution and possible change | common valid area')
    with satellite:
        figure('natural_colour_comparison.png', 'Before / After | matching valid area')
        with st.expander('All result maps', icon=':material/map:'):
            figure('landsat_result_maps_clear.png', 'Satellite images, classifications, residuals and possible change')
    with mle:
        figure('mle_comparison_with_numbers.png', 'Provisional MLE classes | percentages and pixel counts')
        with st.expander('Classification maps', icon=':material/layers:'):
            figure('mle_classification_comparison.png', '2015 / 2024 | class legend')
        with st.expander('Class counts and training samples', icon=':material/table_chart:'):
            counts = pd.DataFrame(report['mle']['class_counts']).reindex(CLASS_NAMES)
            counts.index.name = 'Class'
            st.dataframe(counts, width='stretch')
            st.dataframe((counts.div(counts.sum(axis=0), axis=1)*100).round(2), width='stretch')
            st.caption("This earlier notebook comparison uses each date's valid area. Overview uses the common area.")
            st.dataframe(pd.DataFrame.from_dict(report['mle']['seed_counts'], orient='index', columns=['Seed pixels']), width='stretch')
    with regression:
        figure('regression_change_comparison.png', 'Four regression plots, residual scores and possible-change results')
        with st.expander('Band statistics', icon=':material/table_chart:'):
            st.caption(f"{report['regression']['training_sample_pixels']:,} sampled pixels | threshold {change['threshold']:.4f}")
            st.dataframe(pd.DataFrame(report['regression']['bands']).T.rename(columns={
                'slope': 'Slope', 'intercept': 'Intercept', 'r2_sample': 'R-squared (sample)',
                'rmse_all': 'RMSE (all common pixels)'}), width='stretch')
            st.caption('R-squared describes regression fit, not classification accuracy.')
    with st.expander('Method and data details', icon=':material/info:'):
        dataset, preprocessing = report['dataset'], report['preprocessing']
        st.write('QA exclusions: ' + ', '.join(preprocessing['qa_exclusions']))
        st.write('Reference grid: ' + preprocessing['reference_grid'])
        st.dataframe(pd.DataFrame({'Pixels': [preprocessing['valid_2015_analysis_pixels'], preprocessing['valid_2024_analysis_pixels'], total]},
                     index=['Before valid pixels', 'After valid pixels', 'Common valid pixels']), width='stretch')
    with st.expander('Individual downloads', icon=':material/download:'):
        st.download_button('JSON summary', report_bytes, 'saved_landsat_summary.json', 'application/json', key='history_summary')
        original_map = ROOT / 'outputs' / 'terminal_demo_results.png'
        if original_map.is_file():
            st.download_button('Original result map', original_map.read_bytes(), 'saved_landsat_maps.png', 'image/png', key='history_maps')
        names = {
            'natural_colour_comparison.png': 'Satellite images',
            'mle_classification_comparison.png': 'MLE maps',
            'mle_comparison_with_numbers.png': 'MLE maps and numbers',
            'regression_change_comparison.png': 'Regression and change',
            'landsat_result_maps_clear.png': 'All result maps',
            'landsat_numerical_summary.png': 'Numerical summary'}
        for path in files:
            st.download_button(names.get(path.name, path.stem), path.read_bytes(), path.name, 'image/png', key='history_' + path.stem)

def main():
    st.set_page_config(page_title='Landsat Change Explorer', page_icon=':material/public:', layout='wide')
    st.title('Landsat Change Explorer')
    upload_view, history_view, course_view = st.tabs(['New analysis', 'History', 'Course demo'], default='History')
    with history_view:
        show_history()
    with course_view:
        show_course_demo(ROOT)
    with upload_view:
        with st.container(border=True):
            columns = st.columns(2)
            acquisitions = []
            upload_sets = []
            for column, label in zip(columns, ('Before', 'After')):
                with column:
                    st.subheader(label)
                    uploads = st.file_uploader(f'{label} GeoTIFF files', type=['tif', 'tiff'],
                        accept_multiple_files=True, max_upload_size=512,
                        key=f'{label.lower()}_uploads', on_change=clear_results,
                        help='Select SR_B2, SR_B3, SR_B4, SR_B5 and QA_PIXEL together. Keep the original Landsat filenames.')
                    upload_sets.append(uploads)
                    acquisitions.append(select_acquisition(uploads, label))
            st.caption('5 bands per date: B2, B3, B4, B5 and QA_PIXEL. Maximum 512 MB per file.')
            run = st.button('Run Analysis', type='primary', disabled=any(a is None for a in acquisitions))
        signature = (upload_signature(upload_sets[0]), upload_signature(upload_sets[1]),
                     *(a.identifier if a else None for a in acquisitions))
        if st.session_state.get('result_signature') != signature:
            clear_results()
        if run:
            clear_results()
            progress = st.progress(0, text='Starting analysis')
            try:
                progress.progress(.02, text='Preparing uploaded GeoTIFFs')
                with staged_pair(upload_sets[0], acquisitions[0].identifier,
                                 upload_sets[1], acquisitions[1].identifier) as (before, after):
                    result = analyze(before, after, lambda fraction, message: progress.progress(fraction, text=message))
                # Downloads retain original filenames, not temporary paths that no longer exist.
                for acquisition in result['report']['dataset'].get('acquisitions', []):
                    acquisition['files'] = {band: Path(path).name for band, path in acquisition['files'].items()}
                result['report']['dataset']['input_origin'] = 'Browser uploads, processed on this local computer'
                progress.progress(1.0, text='Preparing maps and downloads')
                artifacts, bundle = build_artifacts(result)
                st.session_state.update(result_report=result['report'], artifacts=artifacts,
                                        bundle=bundle, result_signature=signature)
            except (AnalysisError, MemoryError) as exc:
                st.error(str(exc) if isinstance(exc, AnalysisError) else 'Not enough memory for this scene. Close other applications or use smaller, consistently cropped inputs.')
            except Exception:
                import logging
                logging.exception('Landsat analysis failed')
                st.error('Analysis could not complete. Check that all inputs are intact Landsat GeoTIFFs. Technical details are in the server terminal.')
            finally:
                progress.empty()
        if 'result_report' not in st.session_state:
            st.caption('Add both dates to start.')
            return
        report, artifacts = st.session_state.result_report, st.session_state.artifacts
        dates = report['dataset']['dates']
        st.subheader(f'{dates[0]} to {dates[1]}')
        total = report['preprocessing']['common_valid_analysis_pixels']
        change = report['change_classification']
        c1,c2,c3 = st.columns(3)
        c1.metric('Comparable pixels', f'{total:,}', border=True)
        c2.metric('Flagged pixels', f"{change['possible_change_pixels']:,}", border=True)
        c3.metric('Possible change', f"{change['possible_change_percent']:.2f}%", border=True)
        satellite, classification, possible = st.tabs(['Satellite Images', 'Classification', 'Possible Change'])
        with satellite:
            st.image(artifacts['satellite_images.png'], width='stretch')
            st.caption('Before / After | matching valid area')
        with classification:
            st.image(artifacts['classification_maps.png'], width='stretch')
            st.image(artifacts['class_distribution.png'], width='stretch')
            counts = report['mle']['class_counts']
            st.dataframe(pd.DataFrame({'Class': CLASS_NAMES, **{date: [counts[date][n] for n in CLASS_NAMES] for date in dates}}), hide_index=True, width='stretch')
            st.caption('Provisional classes | common valid area')
        with possible:
            st.image(artifacts['possible_change.png'], width='stretch')
            st.caption(f"Red: above the residual threshold. Dark: below threshold. Grey: masked. Threshold: {change['threshold']:.4f}")
            st.bar_chart(pd.DataFrame({'Pixels': [change['unchanged_pixels'], change['possible_change_pixels']]}, index=['Below threshold', 'Possible change']))
            with st.expander('Regression details'):
                st.image(artifacts['residual_score.png'], width='stretch')
                st.image(artifacts['regression_plots.png'], width='stretch')
                st.dataframe(pd.DataFrame(report['regression']['bands']).T, width='stretch')
                st.caption('R-squared is regression fit on the sampled pixels, not classification accuracy. RMSE uses all common valid pixels.')
        with st.expander('Method and data details'):
            st.write('Reflectance scaling -> cloud and quality masking -> geographic alignment -> provisional MLE classes -> band-wise regression -> robust residual threshold.')
        st.download_button('Download all results (.zip)', st.session_state.bundle, 'landsat_results.zip', 'application/zip')
        with st.expander('Individual maps and summary'):
            for name, data in artifacts.items():
                st.download_button(name, data, name, 'application/json' if name.endswith('.json') else 'image/png', key=name)


if __name__ == '__main__':
    main()
