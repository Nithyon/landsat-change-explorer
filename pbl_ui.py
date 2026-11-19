"""Saved, fast course demonstration plus an interactive exact-inference example."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pbl_analysis import sum_product, transition_matrix


def show_course_demo(root):
    root = Path(root)
    folder = root / 'outputs' / 'pbl'
    summary = folder / 'course_results.json'
    st.subheader('Landsat analysis methods')
    st.caption('Saved results use the supplied Landsat 8 scenes. New uploads in New analysis use the original MLE and regression pipeline.')
    if not summary.is_file():
        st.info('Build the course results first: python build_pbl_demo.py')
        return
    report = json.loads(summary.read_text(encoding='utf-8'))
    st.info('The class labels are provisional spectral rules. Model scores measure agreement with those rules; independent land-cover accuracy is unavailable.')
    labels = ['MLE and regression', 'Clustering and state inference', 'Trees and ensembles']
    supplied_results = root / 'outputs' / 'report_results'
    has_supplied = supplied_results.is_dir()
    views = st.tabs(labels + (['Additional analyses'] if has_supplied else []))
    pbl1, pbl2, pbl3 = views[:3]
    def image(name, caption):
        path = folder / name
        if path.is_file():
            st.image(str(path), caption=caption, width='stretch')
    with pbl1:
        st.markdown('**MLE and regression:** quality masking, geographic alignment, provisional MLE classification, band-wise linear regression and robust residual change flags.')
        st.write('Open History for the original satellite images, MLE maps, regression details and the 5.24% possible spectral-change result.')
        st.caption('The consolidated report explains the dataset, methods, measured results and interpretation limits.')
    with pbl2:
        clustering, pca, hmm, messages = st.tabs(['K-means clustering', 'PCA', 'Hidden Markov model', 'Sum-product walkthrough'])
        unsupervised = report['pbl2']['unsupervised']
        temporal = report['pbl2']['hmm_sum_product']
        with clustering:
            st.markdown('**Question:** Which pixels have similar four-band spectra?')
            st.write('Standardize the four reflectance bands, then fit four K-means clusters on a sample from the Before training area. Reuse the same centroids on both dates so cluster IDs stay comparable.')
            image('kmeans_maps.png', 'Unsupervised spectral groups. Cluster numbers are not land-cover names.')
            st.dataframe(pd.DataFrame(unsupervised['centroids_reflectance'], columns=['Blue', 'Green', 'Red', 'Near infrared']).rename_axis('Cluster'), width='stretch')
            st.caption(f"Training sample: {unsupervised['training_sample_pixels']:,} pixels · sampled silhouette: {unsupervised['silhouette_sample']:.3f}. Silhouette describes cluster separation, not accuracy.")
        with pca:
            st.markdown('**Question:** How much spectral variation can two components retain?')
            st.write('PCA rotates standardized training spectra into orthogonal directions ordered by variance. It is fitted on the Before training sample; it does not use class labels.')
            image('pca_exploration.png', 'Explained variance and a two-dimensional view of actual Before pixels.')
            variance = np.array(unsupervised['pca_explained_variance_ratio'])
            st.dataframe(pd.DataFrame({'Component': ['PC1', 'PC2', 'PC3', 'PC4'], 'Explained variance (%)': variance * 100,
                                      'Cumulative (%)': variance.cumsum() * 100}), hide_index=True, width='stretch')
            st.dataframe(pd.DataFrame(unsupervised['pca_loadings'], columns=['Blue', 'Green', 'Red', 'Near infrared'],
                                      index=['PC1', 'PC2', 'PC3', 'PC4']), width='stretch')
            st.caption('The 2D plot is exploratory; K-means and the HMM use all four standardized bands. PCA reconstruction error is not a classification score.')
        with hmm:
            st.markdown('**Question:** How do a persistence assumption and both observations affect latent spectral states?')
            st.write('Each pixel has two observations: 2015 and 2024. Hidden states are the four spectral groups. Gaussian emissions come from the Before training clusters. The transition prior assumes 90% persistence across the whole interval.')
            image('hmm_states.png', 'Most probable state at each date from the smoothed marginal posterior.')
            image('hmm_change.png', 'Posterior probability that the two latent states differ, conditional on the assumed model.')
            st.warning('Two dates do not support learning a reliable temporal transition model. The 90% persistence is an assumption, not an estimated annual probability. These posteriors are not calibrated real-world change probabilities.')
        with messages:
            st.markdown('**Sum-product is the inference algorithm used inside this HMM.** It sums all possible hidden-state paths to compute posterior probabilities; it does not choose one best Viterbi path.')
            st.code('p(z1, z2 | x1, x2) ∝ π(z1) × e1(z1) × A(z1,z2) × e2(z2)\nforward[t,j] = e[t,j] × sum_i(forward[t-1,i] × A[i,j])\nbackward[t,i] = sum_j(A[i,j] × e[t+1,j] × backward[t+1,j])\nposterior[t] = normalize(forward[t] × backward[t])', language='text')
            selected = st.selectbox('Actual pixel example', range(len(temporal['examples'])),
                                    format_func=lambda i: f"Example {i+1}: row {temporal['examples'][i]['row']}, column {temporal['examples'][i]['column']}",
                                    key='pbl_example')
            persistence = st.slider('Assumed state persistence across 2015–2024', .25, .99, .90, .01, key='pbl_persistence')
            example = temporal['examples'][selected]
            posterior, pairwise, evidence = sum_product(np.array(example['log_emissions'])[None],
                                                        temporal['initial'], transition_matrix(4, persistence))
            st.dataframe(pd.DataFrame({'Band': ['Blue', 'Green', 'Red', 'Near infrared'],
                                      'Before reflectance': example['before_reflectance'],
                                      'After reflectance': example['after_reflectance']}), hide_index=True, width='stretch')
            st.dataframe(pd.DataFrame(posterior[0], index=report['dataset']['dates'],
                                      columns=[f'State {i}' for i in range(4)]), width='stretch')
            st.dataframe(pd.DataFrame(pairwise[0, 0], index=[f'Before {i}' for i in range(4)],
                                      columns=[f'After {i}' for i in range(4)]), width='stretch')
            changed = float(1 - np.trace(pairwise[0, 0]))
            st.write(f'Model-dependent probability of different states: **{changed:.4f}**. Joint posterior sum: {pairwise.sum():.6f}.')
            st.caption('The slider recomputes only this example pixel. Saved scene maps above remain at 0.90 persistence. All message calculations use log space to avoid underflow.')
            image('sum_product_transitions.png', 'Assumed transitions and the sum of posterior transition probabilities across valid pixels.')
    with pbl3:
        methods, comparison, tree = st.tabs(['CART, boosting and ensembles', 'Spatial holdout evaluation', 'CART decision rules'])
        supervised = report['pbl3']
        with methods:
            st.markdown('**CART:** one Gini-based decision tree. **AdaBoost:** 80 sequential weak decision stumps. **Random forest:** 80 randomized, bootstrapped trees. **Soft voting:** equal averaging of the three models’ class probabilities.')
            st.write('Inputs are blue, green, red, near infrared, NDVI and NDWI. Training targets come from unambiguous spectral rules; unmatched or overlapping rules are excluded. These models learn to reproduce the rules.')
            image('pbl3_predictions.png', 'Unvalidated transfer predictions on the After date. These class labels are separate from K-means cluster IDs.')
            image('feature_importance.png', 'Feature importance explains the fitted rule-reproduction models, not causes of land-cover change.')
        with comparison:
            st.write('All models use the same Before training samples and the same eastern spatial holdout, separated by an eight-column guard strip. Rule thresholds are fitted only in the training area. No model is trained on the holdout or on After pixels.')
            image('spatial_split.png', 'Training area, guard strip and spatial holdout on the common valid footprint.')
            st.dataframe(pd.DataFrame([{k: v for k, v in m.items() if k != 'confusion_matrix'} for m in supervised['metrics']]), hide_index=True, width='stretch')
            image('model_comparison.png', 'Agreement against provisional rule labels. The fixed 80 boosting rounds were not selected using this holdout curve.')
            image('confusion_matrices.png', 'Rows are reference rule classes; columns are predicted classes. Class IDs follow the four names shown below.')
            st.dataframe(pd.DataFrame({'ID': range(4), 'Provisional rule class': supervised['class_names'],
                                      'Training sample': supervised['sample_class_counts']['training'],
                                      'Holdout sample': supervised['sample_class_counts']['holdout']}), hide_index=True, width='stretch')
            st.caption(supervised['note'])
        with tree:
            image('cart_tree_highlighted.png' if (folder / 'cart_tree_highlighted.png').is_file() else 'cart_tree.png', 'Bold lines highlight the Gini impurity and split threshold. Gold boxes are decision nodes.')
            if supervised.get('cart_nodes'):
                nodes = pd.DataFrame(supervised['cart_nodes'])
                st.dataframe(nodes.style.format({'Gini index': '{:.3f}', 'Split threshold': '{:.6f}'}, na_rep='Leaf').set_properties(subset=['Gini index'], **{'background-color': '#fff0b3', 'color': '#111111', 'font-weight': 'bold'}).set_properties(subset=['Split threshold'], **{'background-color': '#dbeafe', 'color': '#111111', 'font-weight': 'bold'}), hide_index=True, width='stretch')
                st.caption('Yellow: Gini impurity (0 means a pure node). Blue: threshold used in the feature <= threshold split. Leaf nodes have no split threshold.')
            st.write(f"Fitted depth: {supervised['cart_depth']} · terminal leaves: {supervised['cart_leaves']}")
            with st.expander('Full decision rules'):
                st.code((folder / 'cart_rules.txt').read_text(encoding='utf-8'), language='text')
    if has_supplied:
        with views[3]:
            st.caption('Additional experiments explore the same Landsat imagery using the methods below. Their training and evaluation designs differ from the main comparisons.')
            units = st.tabs(['Data exploration and regression', 'Clustering and PCA', 'Spatial HMM', 'Ensemble regression'])
            figures = root / 'outputs' / 'report_figures'
            entries = [
                ('u12_results.json', ['eda_histograms.png', 'eda_correlation.png', 'eda_ndvi.png', 'u1_poly_degree.png']),
                ('u3_results.json', ['u3_k_selection.png', 'u3_cluster_maps.png', 'u3_pca.png']),
                ('u4_results.json', ['u4_hmm_training.png', 'u4_viterbi_maps.png', 'u4_transect.png']),
                ('u5_results.json', ['u5_regression.png', 'u5_change_compare.png'])]
            for index, (view, (filename, names)) in enumerate(zip(units, entries)):
                with view:
                    if index == 2:
                        st.info('This HMM learns transitions between adjacent pixels along image-row transects using Baum–Welch, then decodes spatial paths with Viterbi. It is a spatial model, distinct from the two-date temporal HMM in the two-date HMM demonstration. Spatial transition probabilities must not be described as probabilities between years.')
                    for name in names:
                        if (figures / name).is_file():
                            st.image(str(figures / name), caption=name.replace('_', ' ').replace('.png', ''), width='stretch')
                    source = supplied_results / filename
                    if source.is_file():
                        with st.expander('Supplied cached numerical results'):
                            st.json(json.loads(source.read_text(encoding='utf-8')))
                        st.download_button('Download ' + filename, source.read_bytes(), filename, key='existing_' + filename)
