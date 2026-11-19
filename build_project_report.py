"""Create a consolidated academic PDF and editable Markdown from measured results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, Preformatted, KeepTogether)

ROOT = Path(__file__).resolve().parent


def build():
    report = json.loads((ROOT / 'outputs/pbl/course_results.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT / 'outputs/terminal_demo_results.json').read_text())
    unsup = report['pbl2']['unsupervised']
    hmm = report['pbl2']['hmm_sum_product']
    supervised = report['pbl3']
    folder = ROOT / 'reports'
    folder.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportBody', fontName='Times-Roman', fontSize=10.5, leading=14.5, spaceAfter=9))
    styles.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=23, leading=29, spaceAfter=18, alignment=TA_CENTER))
    styles.add(ParagraphStyle('ReportHeading', fontName='Helvetica-Bold', fontSize=15, leading=19, spaceAfter=13))
    styles.add(ParagraphStyle('ReportSmall', fontName='Helvetica', fontSize=8.5, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle('ReportCode', fontName='Courier', fontSize=8, leading=11, spaceAfter=10))
    flow, markdown = [], []

    def heading(title, first=False):
        if not first:
            flow.append(PageBreak())
        flow.append(Paragraph(escape(title), styles['ReportTitle' if first else 'ReportHeading']))
        markdown.append(('# ' if first else '## ') + title + '\n')

    def paragraph(text):
        flow.append(Paragraph(escape(text), styles['ReportBody']))
        markdown.append(text + '\n')

    def table(headers, rows, widths=None):
        content = [[Paragraph(escape(str(x)), styles['ReportSmall']) for x in row] for row in [headers] + rows]
        t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5e7eb')),
                              ('GRID', (0, 0), (-1, -1), .35, colors.HexColor('#9ca3af')),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                              ('LEFTPADDING', (0, 0), (-1, -1), 7),
                              ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                              ('TOPPADDING', (0, 0), (-1, -1), 5),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        flow.extend([t, Spacer(1, 10)])
        markdown.extend(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        markdown.append('')

    def figure(relative_path, caption, max_height=3.3):
        path = ROOT / relative_path
        image = Image(str(path))
        scale = min(6.7 * inch / image.imageWidth, max_height * inch / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        flow.append(KeepTogether([image, Spacer(1, 5), Paragraph(escape(caption), styles['ReportSmall']), Spacer(1, 9)]))
        markdown.append(f'![{caption}](../{relative_path})\n')

    def code(text):
        flow.append(Preformatted(text, styles['ReportCode']))
        markdown.append('```text\n' + text + '\n```\n')

    heading('Landsat Spectral Change Analysis', first=True)
    paragraph('Consolidated project report covering PBL 1, PBL 2 and PBL 3. Prepared for the project demonstration requirements supplied on 19 November 2026.')
    paragraph('Student names, register numbers, institution, course title and faculty details should be completed before submission. No student identity or institutional approval is assumed in this report.')
    paragraph('Abstract. This project compares two Landsat 8 Collection 2 Level-2 acquisitions using a common quality-masked geographic grid. The existing baseline demonstrates provisional maximum-likelihood classification and linear-regression residual analysis. The extension adds K-means, principal component analysis, a two-date hidden Markov model with exact sum-product inference, CART, AdaBoost, random forests and soft voting. Supervised evaluation uses a buffered spatial holdout and provisional spectral-rule targets. The work demonstrates the assigned algorithms on real imagery; it does not establish independent land-cover accuracy or validated urban-growth estimates.')
    table(['Component', 'Assigned coverage', 'Assessment'], [
        ['PBL 1', 'Existing MLE and regression baseline; confirm original assignment wording', 'Weight not supplied'],
        ['PBL 2', 'K-means, PCA, Hidden Markov Model and Sum Product Algorithm', '20 marks'],
        ['PBL 3', 'Boosting, Ensemble Learning and CART', '10 marks'],
        ['Report', 'Consolidated methodology, results, evaluation and reproduction', 'PBL 1 to PBL 3']], [65, 320, 95])
    paragraph(f"The preserved baseline flags {baseline['change_classification']['possible_change_pixels']:,} of {report['dataset']['common_valid_pixels']:,} common valid pixels, or {baseline['change_classification']['possible_change_percent']:.2f}%, as possible spectral change. This percentage is a threshold exceedance rate, not an accuracy score.")

    heading('1 Introduction Objectives and Dataset')
    paragraph('The problem is to explore differences between two multispectral acquisitions while showing how several machine-learning and probabilistic-inference methods operate on the same inputs. The objectives are to preserve the original data and baseline, align both dates, explore spectral groups, summarize variance, infer latent states, compare supervised rule-reproduction models and provide a reproducible class demonstration.')
    table(['Property', 'Verified value'], [
        ['Satellite', 'Landsat 8 only in the supplied dataset'],
        ['Dates and path row', '13 December 2015 and 30 July 2024; path/row 146040'],
        ['Product', 'Collection 2 Level-2 L2SP surface reflectance'],
        ['Inputs', 'SR_B2 blue, SR_B3 green, SR_B4 red, SR_B5 near infrared and QA_PIXEL per date'],
        ['Size', '10 original GeoTIFF files totaling 767,745,918 bytes'],
        ['Native grid', '7681 rows by 7531 columns; 30 m pixels; EPSG:32643'],
        ['Analysis grid', f"{report['dataset']['shape'][0]} by {report['dataset']['shape'][1]}; approximately 150 m pixels"],
        ['Study extent', 'Full scene footprint; no administrative Noida boundary was supplied']], [115, 365])
    paragraph('Both dates are aligned to the Before red-band reference. Native scene origins differ by 2100 m in the east-west direction. Reflectance is calculated as DN times 0.0000275 minus 0.2. QA_PIXEL bits 0 through 5 exclude fill, dilated cloud, cirrus, cloud, cloud shadow and snow. Zero/nodata reflectance samples are excluded. Water remains valid. No QA_RADSAT band is available, so the mask does not identify every saturated pixel.')
    paragraph('A grid factor of five reduces computation for the class demonstration. Nearest-neighbor resampling preserves categorical quality flags. All comparison maps use the common valid footprint. December and July differ seasonally; the analysis cannot attribute every spectral difference to physical land-cover change.')

    heading('2 PBL 1 Existing Baseline')
    paragraph('The original project supplies the PBL 1 baseline. It derives provisional spectral-rule samples, estimates multivariate Gaussian class means and covariances, and assigns the maximum-likelihood class to valid spectra. The labels Water-like, Vegetation-like, Bright low-vegetation and Bare/other-like are demonstration labels rather than independently verified land cover.')
    paragraph('For each band, ordinary least-squares regression predicts After reflectance from Before reflectance using up to 100,000 sampled common pixels. The residual vector is standardized by each band\'s median absolute residual. Its root-mean-square score is compared with a robust threshold: median score plus three times 1.4826 times the median absolute deviation. Large residuals indicate possible spectral change.')
    rows = [[band, f"{detail['slope']:.4f}", f"{detail['intercept']:.4f}", f"{detail['r2_sample']:.4f}", f"{detail['rmse_all']:.4f}"]
            for band, detail in baseline['regression']['bands'].items()]
    table(['Band', 'Slope', 'Intercept', 'Sample R squared', 'Common pixel RMSE'], rows, [75, 85, 85, 120, 115])
    figure('outputs/notebook_history/landsat_numerical_summary.png', 'Baseline class distribution and possible-change summary on the common valid footprint.', 3.1)
    paragraph('Some earlier class-count figures use each date\'s own valid footprint; direct comparisons should use common-area summaries. No confusion matrix against independently collected labels exists for this baseline. The formal PBL 1 topic list was not supplied and must be checked against the instructor\'s original assignment.')

    heading('3 PBL 2 K Means Clustering and PCA')
    paragraph('K-means minimizes the within-cluster sum of squared distances from each standardized four-band spectrum to its assigned centroid. The demonstration fixes k at four, uses k-means++ initialization with ten restarts and random seed 42, and fits at most 20,000 Before training-area spectra. Applying those same centroids to both dates retains a common cluster-ID system. Cluster IDs are arbitrary and must not be interpreted as supervised class IDs.')
    paragraph(f"The fitted training inertia is {unsup['inertia']:.2f}; the silhouette score on a sample of up to 1,500 training points is {unsup['silhouette_sample']:.4f}. These describe spectral grouping under the chosen scale and k, not classification accuracy. No claim of optimal k is made.")
    paragraph('PCA is fitted to the same standardized Before training sample. It finds orthogonal directions of decreasing variance. The two-dimensional scatter plot is exploratory; the four-band K-means and HMM do not discard components based on this projection. Thus the demonstration does not claim that PCA improves classification performance.')
    ratios = unsup['pca_explained_variance_ratio']
    cumulative = 0
    rows = []
    for i, ratio in enumerate(ratios):
        cumulative += ratio
        rows.append([f'PC{i+1}', f'{100*ratio:.2f}%', f'{100*cumulative:.2f}%'])
    table(['Component', 'Variance explained', 'Cumulative'], rows, [150, 165, 165])
    figure('outputs/pbl/pca_exploration.png', 'PCA explained variance and actual Before pixels coloured by their K-means cluster.', 2.6)
    table(['Cluster', 'Blue', 'Green', 'Red', 'Near infrared'],
          [[str(i)] + [f'{v:.4f}' for v in row] for i, row in enumerate(unsup['centroids_reflectance'])],
          [80, 100, 100, 100, 100])

    heading('4 PBL 2 Hidden Markov Model')
    paragraph('For each valid geographic pixel, the two four-band spectra are observations x1 and x2. Hidden variables z1 and z2 represent four latent spectral groups. The factorization is p(z1,z2,x1,x2) = pi(z1) e1(x1|z1) A(z1,z2) e2(x2|z2). Initial state frequencies use smoothed Before cluster frequencies. Emissions use diagonal Gaussian means and variances estimated from the Before training clusters, with a variance floor of 0.02 in standardized units.')
    paragraph('The transition matrix assumes a 0.90 probability of retaining the same state over the entire 2015-to-2024 interval. Each alternative state receives 0.10 divided by three. This is a user-inspectable prior, not a learned transition probability, an annual persistence rate or evidence that 90% of the landscape stayed physically unchanged. With only two acquisitions, reliable temporal dynamics cannot be learned from this demonstration.')
    table(['Before to After', 'State 0', 'State 1', 'State 2', 'State 3'],
          [[f'State {i}'] + [f'{x:.4f}' for x in row] for i, row in enumerate(hmm['transition'])], [120, 90, 90, 90, 90])
    figure('outputs/pbl/hmm_states.png', 'Smoothed marginal-state modes for the two actual observation dates.', 3.0)
    paragraph(f"The mean model-dependent posterior probability of different latent states is {hmm['mean_state_change_posterior']:.4f}. It is not directly comparable to the baseline 5.24% threshold flag: the two methods define and model different quantities. The HMM posteriors are not calibrated probabilities of real land-cover change.")

    heading('5 PBL 2 Sum Product Inference')
    paragraph('Sum-product computes exact marginal probabilities on the chain factor graph. The forward message multiplies the current emission by the sum of previous forward-message and transition products. The backward message sums the transition, next emission and next backward-message products. Their normalized product gives the smoothed state marginal. Combining adjacent messages and factors gives the joint posterior over the two states. The implementation operates in log space with logsumexp to prevent numerical underflow.')
    code('alpha[t,j] = e[t,j] * sum_i(alpha[t-1,i] * A[i,j])\nbeta[t,i] = sum_j(A[i,j] * e[t+1,j] * beta[t+1,j])\ngamma[t] = normalize(alpha[t] * beta[t])\nP(different states | observations) = 1 - sum_i P(z1=i,z2=i | x1,x2)')
    paragraph('The procedure sums all possible hidden-state paths. It does not perform Viterbi max-product decoding, and the independently most probable marginal states need not form the most probable joint path. On a chain, forward-backward is the sum-product implementation; HMM and sum-product are shown as a model and its inference algorithm rather than falsely presented as unrelated models.')
    figure('outputs/pbl/sum_product_transitions.png', 'Assumed transition matrix and summed posterior transition counts over the common valid pixels.', 2.9)
    paragraph('The app includes actual pixel examples with reflectance values, marginal probabilities and the full four-by-four joint posterior. A persistence slider recomputes one selected pixel only. Saved scene maps remain at the default prior. Unit checks compare a three-step, two-state chain with exhaustive enumeration and verify both state and pairwise normalization, including extreme log-emission values.')

    heading('6 PBL 3 CART Boosting and Ensembles')
    paragraph('The supervised experiment uses blue, green, red, near-infrared, NDVI and NDWI features. Training-area quantiles define Water-like, Vegetation-like, Bright low-vegetation and Bare/other-like rules. Only pixels matching exactly one rule receive a weak label; overlaps and unmatched pixels are excluded. Rule thresholds are computed before evaluation using the training area alone. A maximum of 4,000 training examples per class is sampled using seed 42.')
    table(['Model', 'Implementation and fixed settings'], [
        ['CART', 'DecisionTreeClassifier; binary Gini splits; maximum depth 6; minimum leaf size 25'],
        ['AdaBoost', '80 sequential decision stumps; learning rate 0.5; sample weights emphasize difficult examples'],
        ['Random forest', '80 bootstrapped randomized trees; maximum depth 10; minimum leaf size 10'],
        ['Soft voting', 'Equal average of the three fitted class-probability outputs, then choose the largest']], [100, 380])
    paragraph('CART supplies an interpretable single-tree baseline. AdaBoost demonstrates boosting, while the random forest demonstrates bagging and feature randomization. Soft voting demonstrates combination across different fitted learners. Class probabilities are not externally calibrated. Hyperparameters are fixed rather than selected by maximizing displayed holdout scores. A simpler model can score better when the target is itself a deterministic spectral rule.')
    figure('outputs/pbl/cart_tree_report.png', 'CART root and next decision level. Class names are shortened in the figure; complete rules are exported separately.', 2.8)

    heading('7 Evaluation Design and Measured Results')
    split = supervised['split']
    paragraph(f"The eastern quarter of common valid Before pixels is held out, with an eight-analysis-column guard strip separating it from the training area. The spatial masks contain {split['training_pixels']:,} training-area pixels, {split['test_pixels']:,} test-area pixels and {split['excluded_guard_pixels']:,} excluded guard pixels. The final weak-labelled samples contain {supervised['training_sample_pixels']:,} training points and {supervised['holdout_sample_pixels']:,} held-out points. This reduces immediate pixel mixing but does not eliminate all large-scale spatial autocorrelation.")
    paragraph('Evaluation compares predictions with the held-out provisional rules, not independently surveyed land cover. Rule agreement is the matching fraction; macro F1 averages the four class F1 values with zero for undefined values; balanced rule agreement averages recall over reference classes present in the holdout. All models use the same samples. After maps are unvalidated transfer predictions and are not used to report labelled test performance.')
    table(['Model', 'Rule agreement', 'Macro F1', 'Balanced agreement'],
          [[m['model'], f"{m['rule_agreement']:.4f}", f"{m['macro_f1']:.4f}", f"{m['balanced_rule_agreement']:.4f}"] for m in supervised['metrics']], [120, 120, 110, 130])
    table(['Class', 'Training examples', 'Holdout examples'],
          [[name, supervised['sample_class_counts']['training'][i], supervised['sample_class_counts']['holdout'][i]]
           for i, name in enumerate(supervised['class_names'])], [220, 130, 130])
    figure('outputs/pbl/model_comparison.png', 'Fixed-model comparison and AdaBoost progression against held-out provisional labels.', 2.3)

    heading('8 Interpretation of Predictions and Errors')
    paragraph('The following CART confusion matrix compares held-out provisional rules with predictions. Rows are reference rule classes and columns are predicted classes. All four model matrices are available at full size in the app and course_results.json.')
    table(['Reference to predicted', 'Water-like', 'Vegetation-like', 'Bright low veg', 'Bare other'],
          [[supervised['class_names'][i]] + row for i, row in enumerate(supervised['metrics'][0]['confusion_matrix'])],
          [160, 80, 80, 80, 80])
    paragraph('Confusion counts reveal which provisional rules are difficult for each fitted model to reproduce in the eastern holdout. High scores can arise because the features include the same NDVI, NDWI and brightness information used to construct the targets. Such agreement demonstrates implementation and approximation of a weak-label task, not independent scientific confirmation of land cover.')
    paragraph('Impurity-based feature importances and CART rules explain the fitted decision process but are not causal evidence. The four After prediction maps reveal how models transfer to a different season and year; they are not validated by the Before holdout scores. Likewise, K-means cluster IDs are not interchangeable with the four supervised class IDs. Both are explicitly labelled in the app.')
    paragraph('The report does not claim that ensembles must beat a single tree, that PCA improves supervised accuracy, that the two-date HMM has learned environmental dynamics, or that the baseline change percentage measures urban growth. Independent validation would be necessary for each stronger claim.')

    heading('9 Reproduction and Project Demonstration')
    paragraph('The public repository is https://github.com/Nithyon/landsat-change-explorer. The README links to the Drive folder holding all source imagery. Eight bands are direct GeoTIFF downloads; two bands use numbered transfer parts because complete-file uploads timed out. prepare_data.py restores the original bytes and validates SHA-256 hashes. The tested environment is Windows with Python 3.14 and the pinned package versions below.')
    table(['Package', 'Measured version'], list(report['environment'].items()), [240, 240])
    code('python prepare_data.py\npython -m pip install -r requirements.txt\npython build_pbl_demo.py\npython -m pip install -r requirements-report.txt\npython build_project_report.py\npython -m unittest discover -s tests -q\npython -m streamlit run app.py')
    paragraph('Open History to present PBL 1. Open Methods for K-means, PCA, HMM and sum-product, then show CART, AdaBoost, random forest, soft voting and the spatial holdout. Adjust the example-pixel persistence slider and explain that the saved scene maps do not change. The combined notebook exposes these stages in execution order. Saved figures allow demonstration without retraining during class.')
    paragraph('Verification checks include baseline numerical regression, exact HMM inference versus enumerated paths, normalized marginals, isolation of unsupervised fitting from held-out and After data, disjoint buffered split masks, exclusion of ambiguous weak labels, and app interaction without retraining. Generated figures and the report should be inspected before submission. Cross-platform clean installation and independent ground-truth validation are outside the verified scope.')

    heading('10 Limitations Conclusion and References')
    paragraph('The supplied inputs span different seasons and nearly nine years. Cloud screening, residual atmospheric effects, the coarser demonstration grid and full-scene extent constrain interpretation. There are only two observations per pixel. Transition persistence is assumed; emissions use a simplified diagonal Gaussian model. The four latent groups may not correspond to stable semantic land-cover types. The supervised target is a spectral rule, and evaluation excludes ambiguous or unmatched pixels, so scores cannot be generalized to every pixel or to real land-cover accuracy.')
    paragraph('The project now demonstrates every supplied PBL 2 and PBL 3 topic within one Landsat workflow and preserves the existing PBL 1 baseline. K-means provides spectral groups, PCA provides variance-based exploration, the HMM defines a two-date latent-state model, sum-product provides exact chain inference, CART provides decision rules, AdaBoost provides sequential boosting and the forest and voting models provide ensemble examples. The measurements support a reproducible teaching demonstration rather than a validated environmental assessment.')
    paragraph('Future work should define an explicit study boundary, acquire comparable-season observations and more dates, collect independent labelled validation samples, assess spatially separated generalization and uncertainty, and test sensitivity to cluster count, transition assumptions and label quality. Confirm the exact PBL 1 rubric and complete the student details before final submission.')
    references = [
        ('USGS Landsat Collection 2 Quality Assessment Bands', 'https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands'),
        ('USGS Landsat Collection 2 Fact Sheet', 'https://pubs.usgs.gov/fs/2021/3002/fs20213002.pdf'),
        ('scikit-learn 1.8 Clustering user guide', 'https://scikit-learn.org/1.8/modules/clustering.html'),
        ('scikit-learn 1.8 PCA and decomposition user guide', 'https://scikit-learn.org/1.8/modules/decomposition.html'),
        ('scikit-learn 1.8 Decision Trees user guide', 'https://scikit-learn.org/1.8/modules/tree.html'),
        ('scikit-learn 1.8 Ensembles user guide', 'https://scikit-learn.org/1.8/modules/ensemble.html'),
        ('Michael Collins The Forward Backward Algorithm', 'https://www.cs.columbia.edu/~mcollins/fb.pdf')]
    for i, (title, url) in enumerate(references, 1):
        flow.append(Paragraph(f'{i}. <link href="{escape(url)}" color="black">{escape(title)}</link>', styles['ReportSmall']))
        markdown.append(f'{i}. [{title}]({url})\n')
    paragraph('Numerical evidence comes from outputs/terminal_demo_results.json and outputs/pbl/course_results.json generated from the supplied imagery. Implementation is in landsat_analysis.py and pbl_analysis.py. No claims of independent accuracy are imported from the references.')

    def page_frame(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#4b5563'))
        canvas.drawString(45, 813, 'Landsat Spectral Change Analysis | Consolidated Project Report')
        canvas.drawRightString(550, 28, f'{doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(str(folder / 'Consolidated_Project_Report.pdf'), pagesize=(595.28, 841.89),
                            rightMargin=45, leftMargin=45, topMargin=55, bottomMargin=45,
                            title='Landsat Spectral Change Analysis Consolidated Project Report',
                            author='Landsat project team')
    doc.build(flow, onFirstPage=page_frame, onLaterPages=page_frame)
    (folder / 'Consolidated_Project_Report.md').write_text('\n'.join(markdown), encoding='utf-8')
    print('Created PDF and editable Markdown report in reports/')


if __name__ == '__main__':
    build()
