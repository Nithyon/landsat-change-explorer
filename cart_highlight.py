"""Readable CART nodes with emphasized impurity and split thresholds."""
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.tree import plot_tree

def export_highlighted_cart(model, destination, features, classes):
    destination = Path(destination)
    tree = model.tree_
    nodes = []
    for i in range(tree.node_count):
        leaf = tree.children_left[i] == tree.children_right[i]
        nodes.append({'Node': i, 'Feature': 'Leaf' if leaf else features[tree.feature[i]],
                      'Gini index': float(tree.impurity[i]),
                      'Split threshold': None if leaf else float(tree.threshold[i]),
                      'Samples': int(tree.n_node_samples[i])})
    fig = Figure(figsize=(14, 6.5), layout='constrained')
    FigureCanvasAgg(fig)
    annotations = plot_tree(model, max_depth=2, feature_names=features,
                            class_names=classes, filled=True, rounded=True,
                            node_ids=True, fontsize=10, precision=3, ax=fig.subplots())
    for annotation in annotations:
        lines = annotation.get_text().splitlines()
        decision = False
        for index, line in enumerate(lines):
            if '<=' in line:
                feature, threshold = line.split('<=', 1)
                feature = feature.strip().replace('_', r'\_')
                lines[index] = r'$\mathbf{' + feature + r' \leq ' + threshold.strip() + '}$'
                decision = True
            elif line.startswith('gini ='):
                lines[index] = r'$\mathbf{gini = ' + line.split('=',1)[1].strip() + '}$'
        annotation.set_text('\n'.join(lines))
        if decision:
            annotation.get_bbox_patch().set_facecolor('#fff0b3')
            annotation.get_bbox_patch().set_edgecolor('#d97706')
            annotation.get_bbox_patch().set_linewidth(2)
    fig.suptitle('CART decision tree | bold lines show split thresholds and Gini impurity', fontsize=14)
    fig.savefig(destination / 'cart_tree_highlighted.png', dpi=170, bbox_inches='tight', facecolor='white')
    return nodes
